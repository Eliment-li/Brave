import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
import numpy as np


def make_ant_relara_env(
    env_id: str,
    *,
    seed: int,
    reward_type: str,
    target_height: float | None = None,
    target_speed: float | None = None,
    target_dist: float | None = None,
    transform_sparse_reward: bool = False,
):
    """
    Create an env compatible with ReLara author's assumptions:
    - observation_space is a flat Box (vector), not Dict
    - reward can be optionally shifted from {-1,0} to {0,1} (like author's robotics_env_maker)
    - env is wrapped with RecordEpisodeStatistics to provide info["episode"]["r"], ["l"]

    This maker is intended for AntTaskEnv (Dict obs).
    """
    env = gym.make(
        env_id,
        reward_type=reward_type,
        target_height=target_height,
        target_speed=target_speed,
        target_dist=target_dist,
    )

    # Optional: match author's robotics_env_maker behavior for sparse tasks.
    # Your sparse reward is {0,-1}. Transform to {1,0} by +1.
    if transform_sparse_reward:
        env = gym.wrappers.TransformReward(env, lambda r: float(r) + 1.0)

    # Flatten Dict obs -> vector [observation, achieved_goal, desired_goal]
    def _flatten_obs(obs):
        return np.concatenate(
            [
                np.asarray(obs["observation"], dtype=np.float32).ravel(),
                np.asarray(obs["achieved_goal"], dtype=np.float32).ravel(),
                np.asarray(obs["desired_goal"], dtype=np.float32).ravel(),
            ],
            dtype=np.float32,
        )

    orig_space = env.observation_space
    flat_obs_space = flatten_space(orig_space)
    env = gym.wrappers.TransformObservation(env, _flatten_obs, observation_space=flat_obs_space)

    # Redefine observation_space to match flattened shape
    # (author code relies on env.observation_space.shape[0])
    sample_obs, _ = env.reset(seed=seed)
    obs_dim = int(np.asarray(sample_obs).ravel().shape[0])
    env.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Crucial: ReLara code expects info["episode"] on done
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Ensure dtype float32 (author does: self.env.observation_space.dtype = np.float32)
    env.observation_space.dtype = np.float32
    return env