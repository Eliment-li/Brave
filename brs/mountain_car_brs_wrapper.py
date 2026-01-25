import gymnasium as gym
from typing import Callable, Dict, Any, Optional

from brs.brs_wrapper import BRSRewardWrapperBase

MetricFn = Callable[[gym.Env], float]
InfoFn = Callable[[gym.Env, float, float], Dict[str, Any]]


def _get_state(env: gym.Env) -> tuple[float, float]:
    """Return (position, velocity) from the underlying MountainCar env."""
    pos, vel = env.unwrapped.state
    return float(pos), float(vel)


def default_metric_fn(env: gym.Env) -> float:
    """Metric: current position (higher is better for reaching the hill)."""
    pos, _ = _get_state(env)
    return pos


def default_info_fn(env: gym.Env, metric: float, metric_max: float) -> Dict[str, Any]:
    goal_pos = float(getattr(env.unwrapped, "goal_position", 0.45))
    distance = goal_pos - float(metric)
    return {
        "goal_position": goal_pos,
        "distance_to_goal": distance,
        "metric_max_pos": metric_max,
    }


class MountainCarBRSRewardWrapper(BRSRewardWrapperBase):
    """BRS reward shaping for MountainCarContinuous.

    - Metric uses car position so higher is better.
    - Info includes goal position and distance for logging.
    """

    def __init__(
        self,
        env: gym.Env,
        metric_fn: MetricFn | None = None,
        info_fn: InfoFn | None = None,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
        use_global_max_bonus: bool = True,
        global_bonus: Optional[float] = 5.0,
    ):
        super().__init__(
            env=env,
            metric_fn=metric_fn or default_metric_fn,
            metric_name="pos",
            info_fn=info_fn or default_info_fn,
            gamma=gamma,
            beta=beta,
            min_bonus=min_bonus,
            use_global_max_bonus=use_global_max_bonus,
            global_bonus=global_bonus,
        )

