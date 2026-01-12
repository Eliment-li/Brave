from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Hashable, Any

import gymnasium as gym
import numpy as np


@dataclass
class ExploRSConfig:
    # --- exploration bonus (ExploB-style) ---
    lmbd: float = 1.0
    max_bonus: float = 1.0  # 用于 (lmbd/max_bonus)^2 的平滑项
    explore_scale: float = 1.0  # alpha：整体缩放探索奖励

    # --- exploitation shaping (简化版) ---
    exploit_scale: float = 1.0  # beta：整体缩放“进展”奖励
    exploit_clip: float = 1.0   # 防止过大

    # --- phi discretization (通用) ---
    # 注意：粒度越小(更细)，桶越多，bonus 越像噪声；粒度越大(更粗)，bonus 衰减快。
    bin_xy: float = 0.5     # x,y 网格（米）
    bin_z: float = 0.05     # 身体高度 z 网格（米）
    bin_v: float = 0.5      # 速度网格（m/s）

    # 只要你用 exclude_xy=True，那么观测里不含 x,y；因此这里建议直接从 unwrapped.data.qpos/qvel 取
    use_mujoco_state: bool = True

    # 当 fallback 到 obs 向量时，最多取前 k 维做分桶（避免维度太大导致 key 爆炸）
    obs_fallback_k: int = 6


def _bin(x: float, width: float) -> int:
    return int(np.floor(float(x) / float(width)))


class ExploRSRewardWrapper(gym.Wrapper):
    """
    ExploRS-like reward shaping wrapper (generic).

    Assumptions (best-effort):
    - obs is goal-conditioned dict with keys: achieved_goal, desired_goal (any shape)
    - MuJoCo envs expose env.unwrapped.data.qpos/qvel (optional, preferred for phi)
    """

    def __init__(self, env: gym.Env, config: ExploRSConfig | None = None):
        super().__init__(env)
        self.cfg = config or ExploRSConfig()
        self._counts: Dict[Hashable, int] = {}
        self._prev_goal_gap: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_goal_gap = self._goal_gap(obs)

        info = dict(info or {})
        info["ExploRS/count_table_size"] = len(self._counts)
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
        if self.cfg.use_mujoco_state and hasattr(self.env.unwrapped, "data"):
            data = self.env.unwrapped.data
            qpos = np.asarray(data.qpos, dtype=np.float64).ravel()
            qvel = np.asarray(data.qvel, dtype=np.float64).ravel()

            # 通用：取前几个维度做分桶（对 PointMaze: qpos[0:2]=xy, qvel[0:2]=vxy）
            x = float(qpos[0]) if qpos.size > 0 else 0.0
            y = float(qpos[1]) if qpos.size > 1 else 0.0
            z = float(qpos[2]) if qpos.size > 2 else 0.0
            vx = float(qvel[0]) if qvel.size > 0 else 0.0
            vy = float(qvel[1]) if qvel.size > 1 else 0.0

            return (
                _bin(x, self.cfg.bin_xy),
                _bin(y, self.cfg.bin_xy),
                _bin(z, self.cfg.bin_z),
                _bin(vx, self.cfg.bin_v),
                _bin(vy, self.cfg.bin_v),
            )

        # fallback：直接从 observation 向量取前 k 维分桶（信息弱，但通用）
        vec = obs.get("observation") if isinstance(obs, dict) else obs
        vec = np.asarray(vec, dtype=np.float64).ravel()
        k = int(max(0, min(self.cfg.obs_fallback_k, vec.size)))
        # 用不同宽度：前两维按 xy，第三按 z，其余按 v（经验性兜底）
        bins = []
        for i in range(k):
            w = self.cfg.bin_xy if i < 2 else (self.cfg.bin_z if i == 2 else self.cfg.bin_v)
            bins.append(_bin(float(vec[i]), w))
        return tuple(bins)

    @staticmethod
    def _goal_gap(obs: Any) -> float:
        if not (isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs):
            return 0.0
        achieved = np.asarray(obs["achieved_goal"], dtype=np.float64).ravel()
        desired = np.asarray(obs["desired_goal"], dtype=np.float64).ravel()
        if achieved.size == 0 or desired.size == 0:
            return 0.0
        m = min(achieved.size, desired.size)
        return float(np.linalg.norm(achieved[:m] - desired[:m], ord=2))


# 兼容旧名字（避免其他地方 import 失败）
AntExploRSRewardWrapper = ExploRSRewardWrapper
