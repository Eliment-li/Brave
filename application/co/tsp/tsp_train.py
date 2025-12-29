from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, field
from gc import callbacks
from pathlib import Path
from typing import Literal

import arrow
import gymnasium as gym
import numpy as np
import swanlab
import torch
import tyro
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from application.co.tsp.tsp_env import TSP2OptEnv
from configs.base_args import get_root_path
from utils.swanlab_callback import SwanLabCallback


def make_env(env_name: str, *, n: int, k: int, max_steps: int, reward_mode: str, seed: int):
    def _thunk():
        if env_name == "tsp":
            return TSP2OptEnv(n=n, k=k, max_steps=max_steps, reward_mode=reward_mode, seed=seed)
        # if env_name == "cvrp":
        #     return CVRPRelocateSwapEnv(n=n, k=k, max_steps=max_steps, reward_mode=reward_mode, seed=seed)
        # if env_name == "jssp":
        #     # map n to jobs/machines for quick control
        #     j = max(4, int(np.sqrt(n)))
        #     m = j
        #     return JSSPSwapEnv(num_jobs=j, num_machines=m, k=k, max_steps=max_steps, reward_mode=reward_mode, seed=seed)
        # if env_name == "maxcut":
        #     p = 0.05
        #     return MaxCutFlipEnv(n=n, p=p, k=k, max_steps=max_steps, reward_mode=reward_mode, seed=seed)
        # raise ValueError(f"Unknown env: {env_name}")

    return _thunk


@dataclass
class TrainArgs:
    env_id: Literal["tsp", "cvrp", "jssp", "maxcut"] = "tsp"
    n: int = 100
    k: int = 16
    max_steps: int = 200
    reward_mode: Literal["delta", "terminal", "improve_only", "normalized_delta", "brave", "potential"] = "delta"
    total_timesteps: int = 200_00
    repeat: int = 1
    seed: int = -1
    track: bool = False
    swanlab_project: str = "Brave_tsp"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = env_id
    root_path: str = get_root_path()
    n_eval_episodes: int = 1
    model_dir: str = get_root_path()+"/results/checkpoints/"+env_id
    tags: list[str] = field(default_factory=list)

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        self.experiment_name = self.env_id + '_' + arrow.now().format('MMDD_HHmm')
        self.experiment_name += self.reward_mode
        if self.seed == -1:
            self.reset_seed()
        print(f"Using seed: {self.seed}")
        print(f"reward_mode: {self.reward_mode}")
        if self.tags:
            parsed_tags = []
            for tag in self.tags:
                parsed_tags.extend([t.strip() for t in tag.split(',') if t.strip()])
            self.tags = parsed_tags
        self.tags.append(self.env_id)
        self.tags.append(self.reward_mode)

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)


def main(args):
    # run_dir = os.path.join(args.run_dir, args.env_id, args.reward_mode)
    # os.makedirs(run_dir, exist_ok=True)
    env = DummyVecEnv([make_env(args.env_id, n=args.n, k=args.k, max_steps=args.max_steps, reward_mode=args.reward_mode, seed=args.seed)])
    eval_env = DummyVecEnv([make_env(args.env_id, n=args.n, k=args.k, max_steps=args.max_steps, reward_mode=args.reward_mode, seed=args.seed + 1)])

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
        # tensorboard_log=run_dir,
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.model_dir,
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    args_dict = asdict(args)
    swanlabcallback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        workspace=args.swanlab_workspace,
        group=args.swanlab_group,
        verbose=2,
        **args_dict
    )
    callbacks_list = [eval_cb]
    if args.track:
        callbacks_list.append(swanlabcallback)

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks_list)
    model.save(os.path.join(args.model_dir, "final_model"))
    if args.track:
        swanlab.finish()

if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    args.finalize()
    for i in range(args.repeat):
        main(args)
        args.reset_seed()
        time.sleep(10)
