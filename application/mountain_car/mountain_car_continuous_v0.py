#depend on stablebaselines3, gymnasium

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import swanlab
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from configs.base_args import get_root_path
import swanlab
from swanlab.integration.sb3 import SwanLabCallback

#batch_size 必须整除 n_steps × n_envs（对 PPO/A2C 等），否则会报错或被内部调整。
@dataclass
class Args:
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = 10000
    seed: int = 1
    track: bool = True
    swanlab_project: str = "Brave"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "ppo_mountain_car_sb"
    experiment_name: str = "ppo_mountain_car_continuous"
    root_path: str = get_root_path()
    video_freq: int = 1  # 评估时每次都录视频
    n_eval_episodes: int = 5
    model_dir: str = get_root_path()+"/results/checkpoints"
    video_dir: str = get_root_path()+"/results/videos"



def save_model(model: PPO, path: Path) -> None:
    #"保存 SB3 模型到指定路径。#
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path.as_posix())


def load_model(path: Path, env) -> PPO:
    #从指定路径加载 SB3 模型，并绑定到给定 env。
    model = PPO.load(path.as_posix(), env=env)
    return model


def train_and_evaluate():
    args = Args()
    run_name = f"{args.env_id}__{args.experiment_name}__{args.seed}__{int(time.time())}"

    # SwanLab 初始化，config=asdict(args)，包含全部超参数
    # 训练环境（向量化）
    def make_env():
        return gym.make(args.env_id)

    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    # 使用默认参数的 PPO
    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)

    # 训练
    if args.track:
        model.learn(total_timesteps=args.total_timesteps,
                    callback=SwanLabCallback(
                        project=args.swanlab_project,
                        experiment_name=args.experiment_name,
                        group = args.swanlab_group,
                        workspace=args.swanlab_workspace,
                        verbose=2,
                        ),
                    )
    else:
        model.learn(total_timesteps=args.total_timesteps)
    # 保存模型
    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    save_model(model, model_path)

    # 评估环境（带视频记录）
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    def make_eval_env():
        env_ = gym.make(args.env_id, render_mode="rgb_array")
        env_ = RecordVideo(
            env_,
            video_folder=video_dir.as_posix(),
            episode_trigger=lambda ep_id: ep_id % args.video_freq == 0,
            name_prefix=f"eval_{args.experiment_name}",
        )
        return env_

    eval_env = make_eval_env()

    # 评估：使用 SB3 提供的 evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    swanlab.finish()
    eval_env.close()
    env.close()
    print(f"Evaluation mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Model saved to: {model_path.as_posix()}")
    print(f"Videos saved to: {video_dir.as_posix()}")


if __name__ == "__main__":
    train_and_evaluate()
