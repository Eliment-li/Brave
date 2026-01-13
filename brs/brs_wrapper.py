import gymnasium as gym
from typing import Callable, Dict, Optional, Tuple, Any

from utils.calc_util import SlideWindow

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
        use_global_max_bonus: bool = None,
        global_bonus: Optional[float] = None,
    ):
        super().__init__(env)
        self.metric_fn = metric_fn
        self.metric_name = str(metric_name)
        self.info_fn = info_fn

        self.gamma = float(gamma)
        self.beta = float(beta)
        self.min_bonus = float(min_bonus)

        self.use_global_max_bonus = bool(use_global_max_bonus)
        # 若不传，则默认复用 episode 提升的同款奖励（v5 里是额外加成；这里给通用默认）
        self.global_bonus = global_bonus
        self.global_max = None  # float
        self.episode_max= None
        self.rdcr = 0.0
        self.rdcr_max = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        metric = float(self.metric_fn(self.env))
        # 第一次 reset 时初始化；之后每个 episode 刷新 episode_max，global_max 跨 episode 保留
        if self.global_max is None:
            self.global_max = metric
        self.episode_max = metric
        # 注意：global_max 不在 reset 时清零（跨 episode）
        self.rdcr = 0.0
        self.rdcr_max = 0.0

        info = dict(info or {})
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max

        info[f"metric_episode_max_{self.metric_name}"] = self.episode_max
        info[f"metric_global_max_{self.metric_name}"] = self.global_max
        # 兼容旧字段：historical `metric_max_*` 实际是 episode max
        info[f"metric_max_{self.metric_name}"] = self.episode_max

        if self.info_fn is not None:
            # 兼容旧签名：第三个参数仍传 episode max（原 metric_max 语义）
            info.update(dict(self.info_fn(self.env, metric, self.episode_max) or {}))

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        metric = float(self.metric_fn(self.env))
        bonus = 0.0

        if metric > self.episode_max:
            self.episode_max = metric

            # episode 提升奖励（原逻辑）
            bonus += self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus

            # global 提升奖励（v5 风格：在 episode 提升的前提下叠加）
            if self.use_global_max_bonus and metric > self.global_max:
                self.global_max = metric
                if self.global_bonus is None:
                    bonus += self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus
                else:
                    bonus += float(self.global_bonus)

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

        info[f"metric_episode_max_{self.metric_name}"] = self.episode_max
        info[f"metric_global_max_{self.metric_name}"] = self.global_max
        # 兼容旧字段
        info[f"metric_max_{self.metric_name}"] = self.episode_max

        if self.info_fn is not None:
            info.update(dict(self.info_fn(self.env, metric, self.episode_max) or {}))

        return obs, reward, terminated, truncated, info


class BRSRewardWrapperBaseV2(gym.Wrapper):
    """
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
        use_global_max_bonus: bool = None,
        global_bonus: Optional[float] = None,
    ):
        super().__init__(env)
        self.metric_fn = metric_fn
        self.metric_name = str(metric_name)
        self.info_fn = info_fn

        self.gamma = float(gamma)
        self.beta = float(beta)
        self.min_bonus = float(min_bonus)

        self.use_global_max_bonus = bool(use_global_max_bonus)
        # 若不传，则默认复用 episode 提升的同款奖励（v5 里是额外加成；这里给通用默认）
        self.global_bonus = global_bonus
        self.global_max = None  # float
        self.episode_max= None
        self.rdcr = 0.0
        self.rdcr_max = 0.0

        self.recent_metrics= SlideWindow(size=1000)


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        metric = float(self.metric_fn(self.env))
        # 第一次 reset 时初始化；之后每个 episode 刷新 episode_max，global_max 跨 episode 保留
        if self.global_max is None:
            self.global_max = metric
        self.episode_max = metric
        # 注意：global_max 不在 reset 时清零（跨 episode）
        self.rdcr = 0.0
        self.rdcr_max = 0.0

        info = dict(info or {})
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max

        info[f"metric_episode_max_{self.metric_name}"] = self.episode_max
        info[f"metric_global_max_{self.metric_name}"] = self.global_max
        # 兼容旧字段：historical `metric_max_*` 实际是 episode max
        info[f"metric_max_{self.metric_name}"] = self.episode_max

        if self.info_fn is not None:
            # 兼容旧签名：第三个参数仍传 episode max（原 metric_max 语义）
            info.update(dict(self.info_fn(self.env, metric, self.episode_max) or {}))

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metric = float(self.metric_fn(self.env))
        bonus = 0.0
        self.recent_metrics.add(metric)
        if metric > self.self.recent_metrics.average:
            self.episode_max = metric

            # episode 提升奖励（原逻辑）
            bonus += self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus

            # global 提升奖励（v5 风格：在 episode 提升的前提下叠加）
            if self.use_global_max_bonus and metric > self.global_max:
                self.global_max = metric
                if self.global_bonus is None:
                    bonus += self.beta * (self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus
                else:
                    bonus += float(self.global_bonus)

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

        info[f"metric_episode_max_{self.metric_name}"] = self.episode_max
        info[f"metric_global_max_{self.metric_name}"] = self.global_max
        # 兼容旧字段
        info[f"metric_max_{self.metric_name}"] = self.episode_max

        if self.info_fn is not None:
            info.update(dict(self.info_fn(self.env, metric, self.episode_max) or {}))

        return obs, reward, terminated, truncated, info
