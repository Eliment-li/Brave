from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np


class OriginalRewardInfoWrapper(gym.Wrapper):
    """
    记录最原始的 reward，并维护与 stable-baselines3 Monitor 相同含义的 ep_len_mean。
    """
    def __init__(self, env: gym.Env, ep_len_buffer_size: int = 100):
        super().__init__(env)
        self._ep_len_buffer: Deque[int] = deque(maxlen=ep_len_buffer_size)
        self._ep_rew_buffer: Deque[float] = deque(maxlen=ep_len_buffer_size)
        self._current_len = 0
        self._current_rew = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_len = 0
        self._current_rew = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_len += 1
        self._current_rew += float(reward)
        info = dict(info)

        # 记录未被修改过的最原始 reward
        info.setdefault("original_reward", reward)

        episode_done = terminated or truncated
        if episode_done:
            self._ep_len_buffer.append(self._current_len)
            self._ep_rew_buffer.append(self._current_rew)
            self._current_len = 0
            self._current_rew = 0.0

        if self._ep_len_buffer:
            info[r"original/ep_len_mean"] = float(np.mean(self._ep_len_buffer))
        if self._ep_rew_buffer:
            info[r"original/ep_rew_mean"] = float(np.mean(self._ep_rew_buffer))

        return obs, reward, terminated, truncated, info

