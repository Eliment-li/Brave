from __future__ import annotations

import gymnasium as gym
import numpy as np


def make_humanoid_relara_env(
    env_id: str,
    *,
    seed: int,
    reward_type: str = "dense",  # dense | sparse
    height_th: float = 0.6,
    transform_sparse_reward: bool = False,
    max_episode_steps: int | None = None,
    render_mode: str | None = None,
):
    """Create a HumanoidStandup env compatible with ReLara's assumptions.

    Contract (what ReLara needs):
    - env.observation_space is a flat 1D Box
    - env.action_space is Box
    - step/reset follow Gymnasium API
    - env is wrapped with RecordEpisodeStatistics (info["episode"] exists when episode ends)

    Notes:
    - HumanoidStandup already returns a vector observation, so unlike Fetch/Ant we don't need to flatten dict obs.
    - Some Gymnasium versions require TransformObservation(..., observation_space=...).
    """

    kwargs: dict = {
        "reward_type": reward_type,
        "height_th": float(height_th),
    }
    if max_episode_steps is not None and int(max_episode_steps) > 0:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)

    # Optional: match author's robotics_env_maker behavior for sparse tasks.
    # Humanoid sparse reward is typically {0,-1} (success=0). Transform to {1,0} by +1.
    if transform_sparse_reward:
        env = gym.wrappers.TransformReward(env, lambda r: float(r) + 1.0)

    # Determine the true flat obs dim first (important for wrappers that require observation_space).
    sample_obs, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(sample_obs).ravel().shape[0])
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # Ensure observations are float32 and flat.
    def _to_float32(obs):
        return np.asarray(obs, dtype=np.float32).ravel()

    env = gym.wrappers.TransformObservation(env, _to_float32, observation_space=obs_space)

    # Keep the wrapper-computed observation_space (some envs expose float64 by default).
    env.observation_space = obs_space

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Crucial: ReLara code expects info["episode"] on done
    env = gym.wrappers.RecordEpisodeStatistics(env)

    env.observation_space.dtype = np.float32
    return env
