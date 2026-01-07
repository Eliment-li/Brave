import numpy as np
import gymnasium as gym

from core.brs_wrapper import BRSRewardWrapperBase


class AntTaskBRSRewardWrapper(BRSRewardWrapperBase):
    def __init__(
        self,
        env: gym.Env,
        task: str,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
    ):
        task = str(task)
        if task not in {"stand", "speed", "far"}:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task

        def metric_fn(e: gym.Env) -> float:
            data = e.unwrapped.data
            qpos = np.asarray(data.qpos).ravel()
            qvel = np.asarray(data.qvel).ravel()
            if task == "stand":
                return float(qpos[2])
            if task == "speed":
                return float(qvel[0])
            # task == "far"
            return float(np.linalg.norm(qpos[:2]))

        def info_fn(e: gym.Env, metric: float, metric_max: float):
            return {
                task: metric,
                f"metric_max_{task}": metric_max,
            }

        super().__init__(
            env=env,
            metric_fn=metric_fn,
            metric_name=task,
            info_fn=info_fn,
            gamma=gamma,
            beta=beta,
            min_bonus=min_bonus,
        )