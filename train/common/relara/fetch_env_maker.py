from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten_space


def make_fetch_relara_env(
    env_id: str,
    *,
    seed: int,
    reward_type: str = "sparse",
    transform_sparse_reward: bool = False,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
):
    """Create a Fetch env compatible with ReLara's assumptions.

    Contract (what ReLara needs):
    - env.observation_space is a flat Box (1D vector)
    - env.action_space is Box
    - step returns Gymnasium style, and we wrap RecordEpisodeStatistics so info["episode"] exists on done

    Notes:
    - Fetch envs use Dict observations: {"observation", "achieved_goal", "desired_goal"}
      We flatten them into a single vector in a deterministic order.
    - reward_type is passed through to the underlying env.
    """

    kwargs: dict = {"reward_type": reward_type}
    if max_episode_steps is not None and int(max_episode_steps) > 0:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)

    # Optional: match author's robotics_env_maker behavior for sparse tasks.
    # Fetch sparse reward is typically {0,-1} (success=0). Transform to {1,0} by +1.
    if transform_sparse_reward:
        env = gym.wrappers.TransformReward(env, lambda r: float(r) + 1.0)

    # Flatten Dict obs -> vector [observation, achieved_goal, desired_goal]
    def _flatten_obs(obs):
        # robust to envs missing some keys (keep order stable)
        parts = []
        if isinstance(obs, dict):
            if "observation" in obs:
                parts.append(np.asarray(obs["observation"], dtype=np.float32).ravel())
            if "achieved_goal" in obs:
                parts.append(np.asarray(obs["achieved_goal"], dtype=np.float32).ravel())
            if "desired_goal" in obs:
                parts.append(np.asarray(obs["desired_goal"], dtype=np.float32).ravel())
        else:
            parts.append(np.asarray(obs, dtype=np.float32).ravel())
        return np.concatenate(parts, dtype=np.float32)

    orig_space = env.observation_space
    flat_obs_space = flatten_space(orig_space)
    env = gym.wrappers.TransformObservation(env, _flatten_obs, observation_space=flat_obs_space)

    # Ensure observation_space matches the flattened actual output.
    sample_obs, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(sample_obs).ravel().shape[0])
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # ReLara expects episode stats in info on done
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Ensure dtype float32 for SB3 ReplayBuffer
    env.observation_space.dtype = np.float32
    return env
