from collections import deque
from typing import Deque, Optional

import gymnasium as gym
import numpy as np


class AntTaskInfoWrapper(gym.Wrapper):
    """
    记录最原始的 reward，并维护与 stable-baselines3 Monitor 相同含义的 ep_len_mean。
    并记录相关 metrics 到 info
    """
    def __init__(self, env: gym.Env, ep_len_buffer_size: int = 100,):
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

        # 记录未被修改过的最原始 reward, 如果使用 relara 那么这里已经被存入了对应的正确值 不要再覆盖
        if info.get(r"original/standerd_reward") is None:
            info.setdefault(r"original/standerd_reward", reward)

        episode_done = terminated or truncated
        if episode_done:
            self._ep_len_buffer.append(self._current_len)
            self._ep_rew_buffer.append(self._current_rew)
            self._current_len = 0
            self._current_rew = 0.0

        if self._ep_len_buffer:
            info[r"original/ep_len_mean"] = float(np.mean(self._ep_len_buffer))
        if self._ep_rew_buffer:
            # 如果使用 relara 那么这里已经被存入了对应的正确值 不要再覆盖
            if info.get(r'original/ep_rew_mean') is None:
                info[r"original/ep_rew_mean"] = float(np.mean(self._ep_rew_buffer))

        #put metrics into info
        for m in ["stand", "speed", "far"]:
            info.setdefault(r'metric/'+m, self._current_metric()[["stand", "speed", "far"].index(m)])
        return obs, reward, terminated, truncated, info

    def _current_metric(self) -> float:
        data = self.env.unwrapped.data
        qpos = data.qpos.ravel()
        qvel = data.qvel.ravel()
        stand = float(qpos[2])
        speed = float(qvel[0])
        far = float(np.linalg.norm(qpos[:2]))

        return stand, speed, far

class AntMazeInfoWrapper(gym.Wrapper):
    """
    记录最原始的 reward，并维护与 stable-baselines3 Monitor 相同含义的 ep_len_mean。
    并记录相关 metrics 到 info
    """
    def __init__(self, env: gym.Env, ep_len_buffer_size: int = 100,):
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

        # 记录未被修改过的最原始 reward, 如果使用 relara 那么这里已经被存入了对应的正确值 不要再覆盖
        if info.get(r"original/standerd_reward") is None:
            info.setdefault(r"original/standerd_reward", reward)

        episode_done = terminated or truncated
        if episode_done:
            self._ep_len_buffer.append(self._current_len)
            self._ep_rew_buffer.append(self._current_rew)
            self._current_len = 0
            self._current_rew = 0.0

        if self._ep_len_buffer:
            info[r"original/ep_len_mean"] = float(np.mean(self._ep_len_buffer))
        if self._ep_rew_buffer:
            # 如果使用 relara 那么这里已经被存入了对应的正确值 不要再覆盖
            if info.get(r'original/ep_rew_mean') is None:
                info[r"original/ep_rew_mean"] = float(np.mean(self._ep_rew_buffer))

        return obs, reward, terminated, truncated, info

