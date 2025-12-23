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


def _bin(x: float, width: float) -> int:
    return int(np.floor(float(x) / float(width)))


class AntExploRSRewardWrapper(gym.Wrapper):
    """
    ExploRS-like wrapper for AntTaskEnv (goal-conditioned dict obs).

    - Exploration bonus: count-based bonus on phi(s')
    - Exploitation shaping (simplified): progress on goal gap (achieved vs desired)

    Works for both reward_type='dense' and 'sparse' because we treat env reward as r_orig and add bonuses.
    """

    def __init__(self, env: gym.Env, config: ExploRSConfig | None = None):
        super().__init__(env)
        self.cfg = config or ExploRSConfig()

        # Count table: key -> N(key)
        self._counts: Dict[Hashable, int] = {}

        # For exploitation shaping (progress-based)
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

        # ExploB-style: lmbd / sqrt((lmbd/max)^2 + N)
        denom = np.sqrt((self.cfg.lmbd / self.cfg.max_bonus) ** 2 + float(n))
        r_explore = float(self.cfg.lmbd / denom) * float(self.cfg.explore_scale)

        # ---- exploitation shaping (simplified progress) ----
        gap = self._goal_gap(obs)
        if self._prev_goal_gap is None:
            r_exploit = 0.0
        else:
            # progress: gap decreases => positive shaping
            progress = float(self._prev_goal_gap - gap)
            r_exploit = float(np.clip(progress, -self.cfg.exploit_clip, self.cfg.exploit_clip)) * float(
                self.cfg.exploit_scale
            )
        self._prev_goal_gap = gap

        # ---- total shaped reward ----
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

    # -----------------
    # phi: 通用分桶
    # -----------------
    def phi(self, obs: Any) -> Tuple:
        """
        A task-agnostic discretization of the physical state.
        Uses MuJoCo qpos/qvel by default (more reliable than guessing indices in obs_vec).
        """
        if self.cfg.use_mujoco_state and hasattr(self.env.unwrapped, "data"):
            data = self.env.unwrapped.data
            qpos = data.qpos.ravel()
            qvel = data.qvel.ravel()
            x, y, z = float(qpos[0]), float(qpos[1]), float(qpos[2])
            vx, vy = float(qvel[0]), float(qvel[1])
        else:
            # fallback: try from obs["observation"] (may exclude xy)
            vec = obs["observation"] if isinstance(obs, dict) and "observation" in obs else np.asarray(obs)
            vec = np.asarray(vec, dtype=np.float64).ravel()
            # 这里无法可靠拿到 x,y，因此仅用 z 和速度做分桶（通用但信息少）
            z = float(vec[0])  # if exclude_xy=True, qpos[2] is first element
            vx = float(vec[-14])  # qvel[0]的位置取决于拼接方式，这里只是兜底
            vy = 0.0
            x = 0.0
            y = 0.0

        return (
            _bin(x, self.cfg.bin_xy),
            _bin(y, self.cfg.bin_xy),
            _bin(z, self.cfg.bin_z),
            _bin(vx, self.cfg.bin_v),
            _bin(vy, self.cfg.bin_v),
        )

    # -----------------
    # simplified exploitation shaping: goal progress
    # -----------------
    @staticmethod
    def _goal_gap(obs: Any) -> float:
        """
        Generic goal gap for all tasks:
        gap = |desired - achieved|
        - speed task: desired is target_speed, achieved is xvel
        - stand task: desired is target_height, achieved is z
        - far task:   desired is target_dist, achieved is xy_norm
        """
        if not (isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs):
            return 0.0
        achieved = float(np.asarray(obs["achieved_goal"]).ravel()[0])
        desired = float(np.asarray(obs["desired_goal"]).ravel()[0])
        return abs(desired - achieved)