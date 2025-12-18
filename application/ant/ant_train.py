import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import os
from swanlab.env import is_windows

from application.ant.brs_wrapper.v1 import AntBRSRewardWrapperV1
from application.ant.brs_wrapper.v2 import AntBRSRewardWrapperV2
from application.ant.brs_wrapper.v3 import AntBRSRewardWrapperV3
from application.ant.brs_wrapper.v4 import AntBRSRewardWrapperV4

if not is_windows():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import arrow
import numpy as np
from gymnasium.wrappers import RecordVideo
import torch
import torch.optim as optim
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

from application.ant.info_wrapper import OriginalRewardInfoWrapper
from configs.base_args import get_root_path
import swanlab
from utils.swanlab_callback import SwanLabCallback
import tyro

os.environ["SDL_VIDEODRIVER"] = "dummy"



@dataclass
class Args:
    env_id: str = "MyMujoco/AntSpeed-v0"
    total_timesteps: int = int(1e4)
    repeat: int = 1
    seed: int = -1
    track: bool = True
    r_wrapper_ver: int = -1 # 1 for AntBRSRewardWrapperV1, 2 for AntBRSRewardWrapperV2
    enable_brave:bool = True
    swanlab_project: str = "Brave_Antv4_speed"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "td3_ant_standerd"
    root_path: str = get_root_path()
    n_eval_episodes: int = 1
    model_dir: str = get_root_path()+"/results/checkpoints/Ant"
    video_dir: str = get_root_path()+"/results/videos/Ant"
    tags: list[str] = field(default_factory=list)

    # TD3 Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 256
    learning_starts: int = 10000
    train_freq: int = 1
    gradient_steps: int = 1
    net_arch: list[int] = field(default_factory=lambda: [400, 300])
    noise_std: float = 0.1
    noise_type: str = 'normal'
    optimizer: str = "adam"
    optimizer_eps: float = 1e-7

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + '_' + arrow.now().format('MMDD_HHmm')
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
            self.tags.append(f'warpperv{self.r_wrapper_ver}')
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

_WRAPPER_MAP = {
    1: AntBRSRewardWrapperV1,
    2: AntBRSRewardWrapperV2,
    3: AntBRSRewardWrapperV3,
    4: AntBRSRewardWrapperV4,
}

_OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "rmsprop": optim.RMSprop,
}

def resolve_optimizer(name: str):
    try:
        return _OPTIMIZER_MAP[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported optimizer '{name}'. Available: {list(_OPTIMIZER_MAP)}") from exc

def make_ant_brs_wrapper(env, version: int):
    """根据版本号返回对应的 Ant BRS reward wrapper。"""
    wrapper_cls = _WRAPPER_MAP.get(version)
    print(f'use reward wrapper v{str(wrapper_cls)}')
    if wrapper_cls is None:
        raise ValueError(f"未知的 AntBRSRewardWrapper 版本: {version}")
    return wrapper_cls(env)

def save_model(model: TD3, path: str) -> None:
    model.save(path)

def load_model(path: str, env) -> TD3:
    model = TD3.load(path, env=env)
    return model

def add_reward_wrapper(env, args):
    if args.enable_brave:
        env = make_ant_brs_wrapper(env, args.r_wrapper_ver)
    return env

def train_and_evaluate():
    args_dict = asdict(args)
    def make_env():
        env = gym.make(args.env_id, reward_type="sparse", target_speed=4.5)
        env = OriginalRewardInfoWrapper(env)
        env = add_reward_wrapper(env, args)
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
        #tau=0.005,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        # policy_kwargs=dict(
        #     #net_arch=args.net_arch,
        #     optimizer_class=resolve_optimizer(args.optimizer),
        #     optimizer_kwargs=dict(eps=args.optimizer_eps),
        # ),
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
        base_env = add_reward_wrapper(base_env, args)
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
