import os

import gymnasium_robotics
import torch
import tyro
import  gymnasium as gym

from train.point.point_maze_train import Args, train_and_evaluate
from gymnasium.envs.registration import register,register_envs
if __name__ == "__main__":
    #register_envs(gymnasium_robotics)
    args = tyro.cli(Args)
    args.track = False  # smoke test 不启用日志记录
    args.total_timesteps = 5000
    args.reward_mode = 'brave'
    args.finalize()
    register(id="PointMazeEnv-v0", entry_point="envs.maze.point_maze:PointMazeEnv",
             max_episode_steps=args.max_episode_steps)

    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    train_and_evaluate(args)