import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import os

import gymnasium as gym
import arrow
import numpy as np
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from application.ant.ant_v4_brs_wrapper import AntBRSRewardWrapperV1,AntBRSRewardWrapperV2
from application.wrappers.original_reward_info_wrapper import OriginalRewardInfoWrapper
from configs.base_args import get_root_path
import swanlab
from utils.swanlab_callback import SwanLabCallback
import tyro
import envs.mujoco.ant_v4_tasks
os.environ["SDL_VIDEODRIVER"] = "dummy"

@dataclass
class Args:
    env_id: str = "MyMujoco/AntStand-v0"
    total_timesteps: int = int(1e4)
    repeat: int = 1
    seed: int = -1
    track: bool = False
    r_wrapper_version: int = 2 # 1 for AntBRSRewardWrapperV1, 2 for AntBRSRewardWrapperV2
    enable_brave:bool = True
    swanlab_project: str = "Brave_Antv4"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "td3_ant_standerd"
    root_path: str = get_root_path()
    n_eval_episodes: int = 3
    model_dir: str = get_root_path()+"/results/checkpoints/Ant_v3"
    video_dir: str = get_root_path()+"/results/videos/Ant_v3"
    tags: list[str] = field(default_factory=list)

    # TD3 Hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    learning_starts: int = 10000
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: list[int] = field(default_factory=lambda: [400, 300])
    noise_std: float = 0.1
    noise_type: str = 'normal'

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        self.experiment_name = self.env_id + '_' + arrow.now().format('MMDD_HHmm')
        if self.enable_brave:
            self.experiment_name += '_brave'
        else:
            self.experiment_name += '_standerd'
        if self.seed == -1:
            self.reset_seed()
        print(f"Using seed: {self.seed}")
        if self.tags:
            parsed_tags = []
            for tag in self.tags:
                parsed_tags.extend([t.strip() for t in tag.split(',') if t.strip()])
            self.tags = parsed_tags
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

def save_model(model: TD3, path: str) -> None:
    model.save(path)

def load_model(path: str, env) -> TD3:
    model = TD3.load(path, env=env)
    return model


def train_and_evaluate():
    args_dict = asdict(args)
    def make_env():
        env = gym.make("MyMujoco/AntSpeed-v0", reward_type="dense", target_speed=3.0)
        env = OriginalRewardInfoWrapper(env)
        if args.enable_brave:
            print(f'use reward wrapper v{args.r_wrapper_version}')
            if args.r_wrapper_version == 1:
                env = AntBRSRewardWrapperV1(env)
            elif args.r_wrapper_version == 2:
                env = AntBRSRewardWrapperV2(env)
            else:
                env = AntBRSRewardWrapperV1(env)
        return env
    # 创建训练环境
    #env = make_vec_env(make_env, n_envs=1, seed=args.seed)
    env = make_env()


    # 配置 Action Noise
    n_actions = env.action_space.shape[-1]
    if args.noise_type == 'normal':
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise_std * np.ones(n_actions))
    else:
        action_noise = None # 或者实现 OrnsteinUhlenbeckActionNoise

    hyperparams = dict(
        #policy="MlpPolicy",
        policy="MultiInputPolicy",
        learning_rate=args.learning_rate,
        buffer_size=1000000,  # 默认通常较大
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=args.net_arch),
        verbose=1,
        seed=args.seed,
    )

    model = TD3(env=env, **hyperparams)

    # 训练
    if args.track:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=SwanLabCallback(
                        project=args.swanlab_project,
                        experiment_name=args.experiment_name,
                        workspace=args.swanlab_workspace,
                        group=args.swanlab_group,
                        verbose=2,
                        **args_dict
                    ),
        )
    else:
        model.learn(total_timesteps=args.total_timesteps)

    # 保存模型
    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    save_model(model, str(model_path))

    def make_eval_env():
        base_env = gym.make(args.env_id, render_mode="rgb_array")
        base_env = OriginalRewardInfoWrapper(base_env)
        video_env = RecordVideo(
            base_env,
            video_folder=str(args.video_dir),
            episode_trigger=lambda ep_id: True, # 录制所有评估 episode
            name_prefix=f"eval_{args.experiment_name}",
        )
        return video_env

    eval_env = make_eval_env()

    # 评估
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    if args.track:
        swanlab.finish()

    eval_env.close()
    env.close()
    print(f"Evaluation mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Model saved to: {str(model_path)}")
    print(f"Videos saved to: {str(args.video_dir)}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.finalize()
    for i in range(args.repeat):
        train_and_evaluate()
        args.reset_seed()
        time.sleep(10)
