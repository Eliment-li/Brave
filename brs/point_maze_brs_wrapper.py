from typing import Optional

import numpy as np
import gymnasium as gym

from brs.brs_wrapper import BRSRewardWrapperBaseV2


class PointMazeBRSRewardWrapper(BRSRewardWrapperBaseV2):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        beta: float = 1.01,
        min_bonus: float = 1,
        use_global_max_bonus: bool = None,
        global_bonus: Optional[float] = None,
    ):
        def _last_obs(e: gym.Env):
            return getattr(e.unwrapped, "_last_obs", None)

        def agent_xy(e: gym.Env) -> np.ndarray:
            # 优先：GoalEnv dict obs（你的 PointMazeEnv 保证有 achieved_goal）
            obs = _last_obs(e)
            if isinstance(obs, dict) and "achieved_goal" in obs:
                arr = np.asarray(obs["achieved_goal"], dtype=np.float64).ravel()
                if arr.size >= 2:
                    return arr[:2]

            # 兜底：mujoco state
            u = e.unwrapped
            if hasattr(u, "data") and hasattr(u.data, "qpos"):
                qpos = np.asarray(u.data.qpos).ravel()
                if qpos.size >= 2:
                    return qpos[:2].astype(np.float64)

            # 最后兜底：observation 前两维（不推荐，但避免直接炸）
            if isinstance(obs, dict) and "observation" in obs:
                arr = np.asarray(obs["observation"], dtype=np.float64).ravel()
                if arr.size >= 2:
                    return arr[:2]

            raise AttributeError("Cannot locate agent xy for PointMaze environment.")

        def goal_xy(e: gym.Env) -> np.ndarray:
            # 优先：GoalEnv dict obs（你的 PointMazeEnv 保证有 desired_goal）
            obs = _last_obs(e)
            if isinstance(obs, dict) and "desired_goal" in obs:
                arr = np.asarray(obs["desired_goal"], dtype=np.float64).ravel()
                if arr.size >= 2:
                    return arr[:2]

            # 如果上面的方法拿不到，再尝试从 env 属性中找
            u = e.unwrapped
            for attr in ("goal_xy", "goal_pos", "goal", "target_goal"):
                if hasattr(u, attr):
                    v = getattr(u, attr)
                    if v is not None:
                        arr = np.asarray(v, dtype=np.float64).ravel()
                        if arr.size >= 2:
                            return arr[:2]

            raise AttributeError("Cannot locate goal xy for PointMaze environment.")

        def distance(e: gym.Env) -> float:
            return float(np.linalg.norm(agent_xy(e) - goal_xy(e)))

        def metric_fn(e: gym.Env) -> float:
            return -distance(e)

        def info_fn(e: gym.Env, metric: float, metric_max: float):
            d = -metric
            return {
                "distance_to_goal": d,
                "metric_max_distance": metric_max,
            }

        super().__init__(
            env=env,
            metric_fn=metric_fn,
            metric_name="distance_metric",
            info_fn=info_fn,
            gamma=gamma,
            beta=beta,
            min_bonus=min_bonus,
            use_global_max_bonus = use_global_max_bonus,
            global_bonus = global_bonus,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        try:
            self.unwrapped._last_obs = obs
        except Exception:
            pass
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        try:
            self.unwrapped._last_obs = obs
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

