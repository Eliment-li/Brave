import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


def _make_thresholds(start: float, end: float, step: float) -> Tuple[float, ...]:
    values: List[float] = []
    current = start
    while current <= end + 1e-12:
        values.append(round(current, 2))
        current += step
    return tuple(values)


@dataclass(frozen=True)
class NovelRewardConfig:
    # Achievement thresholds
    height_thresholds: Tuple[float, ...] = _make_thresholds(0.6, 0.8, 0.01)
    speed_thresholds: Tuple[float, ...] = _make_thresholds(1.0, 4.0, 0.01)

    # Anti-noise: require condition to hold for N consecutive steps
    hold_steps_height: int = 5
    hold_steps_speed: int = 5

    # Stand definition
    stand_height: float = 0.65
    stand_max_roll: float = 0.35   # rad
    stand_max_pitch: float = 0.35  # rad
    stand_max_xy_speed: float = 0.3
    hold_steps_stand: int = 10

    # Bonus shaping
    # bonus = weight * bonus_fn(delta_steps)
    weight_height: float = 0
    weight_speed: float = 3
    weight_stand: float = 0

    # bonus_fn options:
    # - "const": always 1.0 when record broken
    # - "log": log(1 + delta_steps)
    # - "linear": delta_steps
    bonus_fn: str = "log"


def _quat_to_euler_xyz(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw) in radians.
    MuJoCo uses (w, x, y, z).
    """
    w, x, y, z = q
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class AntBRSRewardWrapperV1(gym.Wrapper):
    """
    Adds an extra bonus reward when the agent reaches a "never achieved before good state",
    formalized as: within an episode, the FIRST time an achievement condition is met, the
    step index is smaller than the historical best step index across episodes.

    - Maintains cross-episode records: best_steps[event_key] (minimum steps to achieve).
    - Resets per-episode state on reset().
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[NovelRewardConfig] = None,
        initial_records: Optional[Dict[str, int]] = None,
    ):
        super().__init__(env)
        self.cfg = config or NovelRewardConfig()
        # Cross-episode records (persist across resets)
        self.best_steps: Dict[str, int] = dict(initial_records or {})
        self._init_missing_records()

        # Per-episode
        self.episode_step: int = 0
        self.global_step: int = 0
        self.episode_triggered: set[str] = set()

        # Hold counters per event (per-episode)
        self._height_hold: Dict[float, int] = {h: 0 for h in self.cfg.height_thresholds}
        self._speed_hold: Dict[float, int] = {v: 0 for v in self.cfg.speed_thresholds}
        self._stand_hold: int = 0

    def _init_missing_records(self) -> None:
        big = 10**12
        for h in self.cfg.height_thresholds:
            self.best_steps.setdefault(self._key_height(h), big)
        for v in self.cfg.speed_thresholds:
            self.best_steps.setdefault(self._key_speed(v), big)
        self.best_steps.setdefault(self._key_stand(), big)

    @staticmethod
    def _key_height(h: float) -> str:
        return f"height>={h:g}"

    @staticmethod
    def _key_speed(v: float) -> str:
        return f"speed_x>={v:g}"

    @staticmethod
    def _key_stand() -> str:
        return "stand"

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset per-episode state
        self.episode_step = 0
        self.episode_triggered.clear()
        for h in self._height_hold:
            self._height_hold[h] = 0
        for v in self._speed_hold:
            self._speed_hold[v] = 0
        self._stand_hold = 0

        # Optional: expose records for logging
        info = dict(info)
        info["novel_records_best_steps"] = dict(self.best_steps)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        self.episode_step += 1
        self.global_step += 1

        bonus, bonus_info = self._compute_bonus()
        total_reward = float(reward) + float(bonus+10)

        info = dict(info)
        info["brs_bonus"] = float(bonus)
        info["brs_bonus_detail"] = bonus_info
        info["brs_records_best_steps"] = dict(self.best_steps)

        return obs, total_reward, terminated, truncated, info

    # ---------- Achievement computation ----------

    def _compute_bonus(self) -> Tuple[float, Dict]:
        """
        Returns (bonus_reward, detail_info).
        """
        data = getattr(self.env.unwrapped, "data", None)
        # Ant torso is usually the root body:
        # qpos: [x, y, z, qw, qx, qy, qz, ...]
        qpos = np.asarray(data.qpos).copy()
        qvel = np.asarray(data.qvel).copy()

        z = float(qpos[2])
        quat = np.asarray(qpos[3:7], dtype=np.float64)
        roll, pitch, _yaw = _quat_to_euler_xyz(quat)

        # Root linear velocities are typically first 3 of qvel: vx, vy, vz
        vx = float(qvel[0])
        vy = float(qvel[1])
        vxy = float(math.sqrt(vx * vx + vy * vy))

        bonus = 0.0
        triggered = []

        # Height achievements
        for h in self.cfg.height_thresholds:
            key = self._key_height(h)
            if key in self.episode_triggered:
                continue

            if z >= h:
                self._height_hold[h] += 1
            else:
                self._height_hold[h] = 0

            r1=0
            r2=0
            r3=0
            # Check hold, the height must be maintained for several steps to get bonus to avoid reward hacking
            if self._height_hold[h] >= self.cfg.hold_steps_height:
                b = self._maybe_break_record_and_bonus(key, self.episode_step, self.cfg.weight_height)
                r1=b
                if b > 0:
                    bonus += b
                    triggered.append({"event": key, "bonus": b})

                self.episode_triggered.add(key)

        # Speed achievements (x direction)
        for v in self.cfg.speed_thresholds:
            key = self._key_speed(v)
            if key in self.episode_triggered:
                continue

            if vx >= v:
                self._speed_hold[v] += 1
            else:
                self._speed_hold[v] = 0

            if self._speed_hold[v] >= self.cfg.hold_steps_speed:
                b = self._maybe_break_record_and_bonus(key, self.episode_step, self.cfg.weight_speed)
                r2 = b
                if b > 0:
                    bonus += b
                    triggered.append({"event": key, "bonus": b})

                self.episode_triggered.add(key)

        # Stand achievement
        stand_key = self._key_stand()
        if stand_key not in self.episode_triggered:
            stand_ok = (
                (z >= self.cfg.stand_height)
                and (abs(roll) <= self.cfg.stand_max_roll)
                and (abs(pitch) <= self.cfg.stand_max_pitch)
                and (vxy <= self.cfg.stand_max_xy_speed)
            )
            if stand_ok:
                self._stand_hold += 1
            else:
                self._stand_hold = 0

            if self._stand_hold >= self.cfg.hold_steps_stand:
                b = self._maybe_break_record_and_bonus(stand_key, self.episode_step, self.cfg.weight_stand)
                r3 = b
                if b > 0:
                    bonus += b
                    triggered.append({"event": stand_key, "bonus": b})

                self.episode_triggered.add(stand_key)

        detail = {
            "triggered": triggered,
            "measures": {
                "z": z,
                "vx": vx,
                "vy": vy,
                "vxy": vxy,
                "roll": roll,
                "pitch": pitch,
            },
            "episode_step": self.episode_step,
        }
        if bonus >0:
            bonus = math.log(1+bonus)+5
        return bonus, detail

    def _maybe_break_record_and_bonus(self, key: str, steps_now: int, weight: float) -> float:
        """
        If steps_now breaks the historical record best_steps[key], update it and return bonus.
        Otherwise return 0.
        """
        best_old = self.best_steps.get(key, 10**12)
        if steps_now >= best_old:
            return 0.0

        # Break record
        self.best_steps[key] = steps_now

        delta = best_old - steps_now
        if best_old >= 10**11:
            # first time ever: treat as delta=1 for stability
            delta = 1

        if self.cfg.bonus_fn == "const":
            base = 1.0
        elif self.cfg.bonus_fn == "linear":
            base = float(delta)
        else:
            base = float(math.log(1.0 + float(delta)))

        return float(weight * base)


class AntBRSRewardWrapperV2(AntBRSRewardWrapperV1):
    """
    Ant bonus wrapper that keeps V1's best-step bonuses while also granting
    a one-time per-episode reward as soon as each threshold is first met.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[NovelRewardConfig] = None,
        initial_records: Optional[Dict[str, int]] = None,
    ):
        super().__init__(env, config=config, initial_records=initial_records)
        self._episode_bonus_paid: set[str] = set()
        self.bouns_cnt = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        self._episode_bonus_paid.clear()
        return obs, info

    def _maybe_break_record_and_bonus(self, key: str, steps_now: int, weight: float) -> float:
        episode_bonus = 0.0
        if weight > 0 and key not in self._episode_bonus_paid:
            episode_bonus = float(weight)
            self._episode_bonus_paid.add(key)
            self.bouns_cnt +=1
            # if self.bouns_cnt %100==0:
            #     print(f"episode bonus cnt:{self.bouns_cnt} key:{key} step:{steps_now}")
        record_bonus = super()._maybe_break_record_and_bonus(key, steps_now, weight)
        return episode_bonus + record_bonus



class AntBRSRewardWrapperV3(gym.Wrapper):
    """给在当前 episode 内刷新任务指标纪录的行为额外奖励。"""

    def __init__(self, env, bonus: float = 5.0):
        super().__init__(env)
        self._bonus = float(bonus)
        self._task = getattr(self.env.unwrapped, "task", None)
        if self._task not in {"stand", "speed", "far"}:
            raise ValueError(f"Unsupported task: {self._task}")
        self.episode_max = -np.inf

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_max = self._current_metric()
        #self.episode_max == -np.inf
        info = info or {}
        info["episode_max_metric"] = self.episode_max
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metric = self._current_metric()
        bonus = 0.0
        if  self.episode_max == -np.inf:
            self.episode_max = metric
        if metric > self.episode_max:
            self.episode_max = metric
            bonus = self._bonus
        reward += bonus
        info = info or {}
        info["episode_max_"+self._task] = self.episode_max
        #info["episode_max_metric"] = self.episode_max
        info[str(self._task)] = metric
        info["brs_bonus"] = bonus
        return obs, reward, terminated, truncated, info

    def _current_metric(self) -> float:
        data = self.env.unwrapped.data
        qpos = data.qpos.ravel()
        qvel = data.qvel.ravel()
        if self._task == "stand":
            return float(qpos[2])
        if self._task == "speed":
            return float(qvel[0])
        else:  # far
            return float(np.linalg.norm(qpos[:2]))
