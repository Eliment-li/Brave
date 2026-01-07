import gymnasium as gym
from typing import Callable, Dict, Optional, Tuple, Any


MetricFn = Callable[[gym.Env], float]
InfoFn = Callable[[gym.Env, float, float], Dict[str, Any]]
# InfoFn(env, metric, metric_max) -> extra_info


class BRSRewardWrapperBase(gym.Wrapper):
    """
    通用 BRS wrapper：只处理 rdcr/rdcr_max、metric_max 与 bonus 的更新。
    任务差异通过 metric_fn (越大越好) 与 info_fn 注入。
    """

    def __init__(
        self,
        env: gym.Env,
        metric_fn: MetricFn,
        metric_name: str = "metric",
        info_fn: Optional[InfoFn] = None,
        gamma: float = 0.99,
        beta: float = 1.1,
        min_bonus: float = 0.01,
    ):
        super().__init__(env)
        self.metric_fn = metric_fn
        self.metric_name = str(metric_name)
        self.info_fn = info_fn

        self.gamma = float(gamma)
        self.beta = float(beta)
        self.min_bonus = float(min_bonus)

        self.metric_max = 0.0
        self.rdcr = 0.0
        self.rdcr_max = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        metric = float(self.metric_fn(self.env))
        self.metric_max = metric
        self.rdcr = 0.0
        self.rdcr_max = 0.0

        info = dict(info or {})
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info[f"metric_max_{self.metric_name}"] = self.metric_max

        if self.info_fn is not None:
            info.update(dict(self.info_fn(self.env, metric, self.metric_max) or {}))

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        metric = float(self.metric_fn(self.env))
        bonus = 0.0

        if metric > self.metric_max:
            self.metric_max = metric
            bonus = self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus
            reward = bonus
            self.rdcr = self.gamma * self.rdcr + bonus
            assert self.rdcr > self.rdcr_max, f"rdcr did not increase: {self.rdcr} <= {self.rdcr_max}"
        else:
            self.rdcr = self.gamma * self.rdcr + reward

        self.rdcr_max = max(self.rdcr_max, self.rdcr)

        info = dict(info or {})
        info["brs_bonus"] = bonus
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info[self.metric_name] = metric
        info[f"metric_max_{self.metric_name}"] = self.metric_max

        if self.info_fn is not None:
            info.update(dict(self.info_fn(self.env, metric, self.metric_max) or {}))

        return obs, reward, terminated, truncated, info
