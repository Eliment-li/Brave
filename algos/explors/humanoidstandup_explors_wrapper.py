from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class HumanoidStandupExploRSConfig:
    # --- exploration bonus (ExploB-style) ---
    lmbd: float = 1.0
    max_bonus: float = 1.0
    explore_scale: float = 1.0

    # --- exploitation shaping (progress) ---
    # For standup, "progress" is primarily going up (torso/pelvis z).
    exploit_scale: float = 1.0
    exploit_clip: float = 1.0

    # --- phi discretization ---
    # Humanoid is high-dim. Keep phi low-dim & stable.
    # We use (x, y, z, vx, vy, vz) bins.
    bin_xy: float = 0.5
    bin_z: float = 0.05
    bin_v: float = 0.5

    # Fallback when mujoco state isn't available
    obs_fallback_k: int = 6
    bin_obs: float = 0.5


def _bin(x: float, width: float) -> int:
    if width <= 0:
        return int(np.floor(float(x)))
    return int(np.floor(float(x) / float(width)))


class HumanoidStandupExploRSRewardWrapper(gym.Wrapper):
    """ExploRS-like reward shaping wrapper for HumanoidStandup.

    Reward:
      r_hat = r_orig + r_explore + r_exploit

    - r_explore: count-based bonus using discretized MuJoCo state (qpos/qvel)
    - r_exploit: progress shaping using delta-height (z_t - z_{t-1})

    Notes
    -----
    * HumanoidStandup isn't goal-conditioned, so we don't use goal gap.
    * We keep phi() low-dimensional to avoid exploding count tables.
    """

    def __init__(self, env: gym.Env, config: HumanoidStandupExploRSConfig | None = None):
        super().__init__(env)
        self.cfg = config or HumanoidStandupExploRSConfig()
        self._counts: Dict[Hashable, int] = {}
        self._prev_z: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_z = self._get_z(obs)

        info = dict(info or {})
        info["ExploRS/count_table_size"] = int(len(self._counts))
        info["ExploRS/prev_z"] = float(self._prev_z)
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
        z = self._get_z(obs)
        if self._prev_z is None:
            r_exploit = 0.0
        else:
            progress = float(z - self._prev_z)
            r_exploit = float(np.clip(progress, -self.cfg.exploit_clip, self.cfg.exploit_clip)) * float(
                self.cfg.exploit_scale
            )
        self._prev_z = z

        r_hat = float(r_orig) + r_explore + r_exploit

        # ---- logging ----
        info.setdefault("ExploRS/r_orig", float(r_orig))
        info["ExploRS/r_explore"] = float(r_explore)
        info["ExploRS/r_exploit"] = float(r_exploit)
        info["ExploRS/r_hat"] = float(r_hat)
        info["ExploRS/phi_n"] = int(n)
        info["ExploRS/count_table_size"] = int(len(self._counts))
        info["ExploRS/z"] = float(z)

        return obs, r_hat, terminated, truncated, info

    def phi(self, obs: Any) -> Tuple:
        # Preferred: MuJoCo state from unwrapped env
        if hasattr(self.env.unwrapped, "data"):
            data = self.env.unwrapped.data
            qpos = np.asarray(getattr(data, "qpos", []), dtype=np.float64).ravel()
            qvel = np.asarray(getattr(data, "qvel", []), dtype=np.float64).ravel()

            x = float(qpos[0]) if qpos.size > 0 else 0.0
            y = float(qpos[1]) if qpos.size > 1 else 0.0
            z = float(qpos[2]) if qpos.size > 2 else float(self._get_z(obs))

            vx = float(qvel[0]) if qvel.size > 0 else 0.0
            vy = float(qvel[1]) if qvel.size > 1 else 0.0
            vz = float(qvel[2]) if qvel.size > 2 else 0.0

            return (
                _bin(x, self.cfg.bin_xy),
                _bin(y, self.cfg.bin_xy),
                _bin(z, self.cfg.bin_z),
                _bin(vx, self.cfg.bin_v),
                _bin(vy, self.cfg.bin_v),
                _bin(vz, self.cfg.bin_v),
            )

        # Fallback: discretize first k dims of obs vector
        vec = obs.get("observation") if isinstance(obs, dict) else obs
        vec = np.asarray(vec, dtype=np.float64).ravel()
        k = int(max(0, min(self.cfg.obs_fallback_k, vec.size)))
        return tuple(_bin(float(vec[i]), self.cfg.bin_obs) for i in range(k))

    def _get_z(self, obs: Any) -> float:
        # best-effort: take z from mujoco qpos[2]
        if hasattr(self.env.unwrapped, "data"):
            data = self.env.unwrapped.data
            qpos = np.asarray(getattr(data, "qpos", []), dtype=np.float64).ravel()
            if qpos.size > 2:
                return float(qpos[2])

        # fallback to obs vector layout (HumanoidStandup default obs starts with z)
        vec = obs.get("observation") if isinstance(obs, dict) else obs
        vec = np.asarray(vec, dtype=np.float64).ravel()
        return float(vec[0]) if vec.size > 0 else 0.0


# alias
HumanoidStandupExploRSWrapper = HumanoidStandupExploRSRewardWrapper
