"""Smoke test for ReLara on Fetch envs.

Goal: verify env creation + dict->flat observation + a short learn loop.
"""

import os
import sys

# Allow running as: `python train/fetch/relara/smoke_test_relara.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gymnasium_robotics
import tyro
from gymnasium.envs.registration import register_envs

from train.fetch.relara.fetch_train_relara import Args, main


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Run a tiny training loop
    args.task = "reach"
    args.total_timesteps = 200
    args.pa_learning_starts = 10
    args.ra_learning_starts = 10
    args.track = False
    args.seed = 0
    args.reward_type = "sparse"
    args.transform_sparse_reward = False
    args.max_episode_steps = 50

    args.finalize()
    register_envs(gymnasium_robotics)

    main(args)
