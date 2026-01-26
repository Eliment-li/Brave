"""Smoke test for ReLara on Fetch envs.

Goal: verify env creation + dict->flat observation + a short learn loop.
"""

import tyro
from gymnasium import register

from train.fetch.relara.fetch_train_relara import Args, main


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Run a tiny training loop
    args.task = "push"
    args.total_timesteps = 1000
    # args.pa_learning_starts = 10
    # args.ra_learning_starts = 10
    args.track = True
    # args.seed = 0
    args.reward_type = "sparse"
    # args.transform_sparse_reward = False
    args.max_episode_steps = 100

    args.finalize()
    register(id="Reach", entry_point="envs.fetch.reach:MujocoFetchReachEnv", max_episode_steps=args.max_episode_steps)
    register(id="Push", entry_point="envs.fetch.push:MujocoFetchPushEnv", max_episode_steps=args.max_episode_steps)
    main(args)
