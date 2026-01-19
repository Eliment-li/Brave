from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class FetchExploRSConfig:
    # --- exploration bonus (ExploB-style) ---
    lmbd: float = 1.0
    max_bonus: float = 1.0
    explore_scale: float = 1.0

    # --- exploitation shaping (progress) ---
    exploit_scale: float = 1.0
    exploit_clip: float = 1.0

    # --- phi discretization ---
    # Fetch goal 空间通常是米级，0.02~0.05 的粒度更常见；这里给保守默认值
    bin_goal: float = 0.05

    # 当观测不是 dict/或缺字段时的兜底
    obs_fallback_k: int = 6
    bin_obs: float = 0.5


def _bin(x: float, width: float) -> int:
    if width <= 0:
        return int(np.floor(float(x)))
    return int(np.floor(float(x) / float(width)))


class FetchExploRSRewardWrapper(gym.Wrapper):
    """
    ExploRS-like reward shaping wrapper for Fetch goal-conditioned envs.

    Expected obs(dict): achieved_goal, desired_goal, observation
    - phi(): discretize achieved_goal (preferred) for visitation counts
    - goal_gap(): ||ag - dg|| for progress shaping
    """

    def __init__(self, env: gym.Env, config: FetchExploRSConfig | None = None):
        super().__init__(env)
        self.cfg = config or FetchExploRSConfig()
        self._counts: Dict[Hashable, int] = {}
        self._prev_goal_gap: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_gap = self._goal_gap(obs)

        info = dict(info or {})
        info["ExploRS/count_table_size"] = int(len(self._counts))
        info["ExploRS/prev_goal_gap"] = float(self._prev_goal_gap)
        return obs, info

    def step(self, action):
        obs, r_orig, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})

        # ---- exploration bonus ----
        key = self.phi(obs)
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n

        denom = np.sqrt((self.cfg.lmbd / self.cfg.max_bonus) ** 2 + float(n))
        r_explore = float(self.cfg.lmbd / denom) * float(self.cfg.explore_scale)

        # ---- exploitation shaping (progress) ----
        gap = self._goal_gap(obs)
        if self._prev_goal_gap is None:
            r_exploit = 0.0
        else:
            progress = float(self._prev_goal_gap - gap)
            r_exploit = float(np.clip(progress, -self.cfg.exploit_clip, self.cfg.exploit_clip)) * float(
                self.cfg.exploit_scale
            )
        self._prev_goal_gap = gap

        r_hat = float(r_orig) + r_explore + r_exploit

        # ---- logging ----
        info.setdefault("ExploRS/r_orig", float(r_orig))
        info["ExploRS/r_explore"] = float(r_explore)
        info["ExploRS/r_exploit"] = float(r_exploit)
        info["ExploRS/r_hat"] = float(r_hat)
        info["ExploRS/phi_n"] = int(n)
        info["ExploRS/count_table_size"] = int(len(self._counts))
        info["ExploRS/goal_gap"] = float(gap)

        return obs, r_hat, terminated, truncated, info

    def phi(self, obs: Any) -> Tuple:
        # 优先：Fetch dict obs 里的 achieved_goal
        if isinstance(obs, dict) and "achieved_goal" in obs:
            ag = np.asarray(obs["achieved_goal"], dtype=np.float64).ravel()
            return tuple(_bin(float(x), self.cfg.bin_goal) for x in ag.tolist())

        # fallback：从 observation 向量取前 k 维
        vec = obs.get("observation") if isinstance(obs, dict) else obs
        vec = np.asarray(vec, dtype=np.float64).ravel()
        k = int(max(0, min(self.cfg.obs_fallback_k, vec.size)))
        return tuple(_bin(float(vec[i]), self.cfg.bin_obs) for i in range(k))

    @staticmethod
    def _goal_gap(obs: Any) -> float:
        if not (isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs):
            return 0.0
        ag = np.asarray(obs["achieved_goal"], dtype=np.float64).ravel()
        dg = np.asarray(obs["desired_goal"], dtype=np.float64).ravel()
        if ag.size == 0 or dg.size == 0:
            return 0.0
        m = min(ag.size, dg.size)
        return float(np.linalg.norm(ag[:m] - dg[:m], ord=2))


# 别名（跟 ant 保持一致的 import 习惯）
FetchExploRSWrapper = FetchExploRSRewardWrapper

