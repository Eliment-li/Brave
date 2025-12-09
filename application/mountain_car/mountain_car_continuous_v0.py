#depend on stablebaselines3, gymnasium
import argparse
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
import arrow
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from configs.base_args import get_root_path
import swanlab
from utils.swanlab_callback import SwanLabCallback

# 新增：导入自定义 wrapper
from core.brs_mountaincar_wrapper import BRSRewardWrapper

#batch_size 必须整除 n_steps × n_envs（对 PPO/A2C 等），否则会报错或被内部调整。
@dataclass
class Args:
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = 500_000
    seed: int = -1
    track: bool = True
    enable_brave = True
    swanlab_project: str = "Brave"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "ppo_mountain_car_sb"
    root_path: str = get_root_path()
    video_freq: int = 1  #
    n_eval_episodes: int = 5
    model_dir: str = get_root_path()+"/results/checkpoints/MountainCarContinuous_v0"
    video_dir: str = get_root_path()+"/results/videos/MountainCarContinuous_v0"
    def finalize(self):
        self.experiment_name = self.env_id + '_' + arrow.now().format('MMDD_HHmm')
        if self.enable_brave:
            self.experiment_name += '_brave'
        #set seed to random value if seed is -1
        if self.seed == -1:
            self.seed = torch.randint(0, 10000, (1,)).item()
        print(f"Using seed: {self.seed}")
        # make dir if model_dir or video_dir not exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

    #make dir if model_dir or video_dir not exist

def save_model(model: PPO, path: str) -> None:
    model.save(path)

def load_model(path: str, env) -> PPO:
    model = PPO.load(path, env=env)
    return model

def train_and_evaluate():
    args = Args()
    args.finalize()
    args_dict = asdict(args)

    def make_env():
        env = gym.make(args.env_id)
        if args.enable_brave:
            env = BRSRewardWrapper(env)
        return env

    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    # use PPO with default hyper-paras
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    # train
    if args.track:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=SwanLabCallback(
                        project=args.swanlab_project,
                        experiment_name=args.experiment_name,
                        workspace=args.swanlab_workspace,
                        group = args.swanlab_group,
                        verbose=2,
                        # **kwargs
                        **args_dict
                    ),
        )
    else:
        model.learn(total_timesteps=args.total_timesteps)
    # save model

    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    save_model(model, str(model_path))


    def make_eval_env():
        env_ = gym.make(args.env_id, render_mode="rgb_array")
        env_ = BRSRewardWrapper(env_)  # 包裹自定义 wrapper
        env_ = RecordVideo(
            env_,
            video_folder=str(args.video_dir),
            episode_trigger=lambda ep_id: ep_id % args.video_freq == 0,
            name_prefix=f"eval_{args.experiment_name}",
        )
        return env_

    eval_env = make_eval_env()

    # evaluate：use evaluate_policy from sb3
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="repeat train_and_evaluate")
    args = parser.parse_args()
    for _ in range(args.repeat):
        train_and_evaluate()
