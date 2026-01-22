import numpy as np
import gymnasium as gym

from brs.brs_wrapper import BRSRewardWrapperBaseV2


class AntMazeBRSRewardWrapper(BRSRewardWrapperBaseV2):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
        use_global_max_bonus:bool=True,
        global_bonus: float | None = None,

    ):
        def agent_xy(e: gym.Env) -> np.ndarray:
            data = e.unwrapped.data
            qpos = np.asarray(data.qpos).ravel()
            if qpos.size < 2:
                raise ValueError("AntMaze environment qpos must contain at least 2 elements.")
            return qpos[:2]

        def goal_xy(e: gym.Env) -> np.ndarray:
            u = e.unwrapped
            for attr in ("goal_xy", "goal_pos", "goal", "target_goal"):
                if hasattr(u, attr):
                    v = getattr(u, attr)
                    if v is not None:
                        arr = np.asarray(v, dtype=np.float64).ravel()
                        if arr.size >= 2:
                            return arr[:2]
            raise AttributeError("Cannot locate goal position on the AntMaze environment.")

        def distance(e: gym.Env) -> float:
            return float(np.linalg.norm(agent_xy(e) - goal_xy(e)))

        def metric_fn(e: gym.Env) -> float:
            # 越大越好：离目标越近 metric 越大
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
            use_global_max_bonus=use_global_max_bonus,
            global_bonus=global_bonus,
        )