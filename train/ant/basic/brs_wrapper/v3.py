import gymnasium as gym
import numpy as np

from utils.calc_util import SlideWindow


class AntBRSRewardWrapperV3(gym.Wrapper):
    """给在当前 episode 内刷新任务指标纪录的行为额外奖励。"""

    def __init__(self, env, bonus: float = 10):
        super().__init__(env)
        self._bonus = float(bonus)
        self._task = getattr(self.env.unwrapped, "task", None)
        if self._task not in {"stand", "speed", "far"}:
            raise ValueError(f"Unsupported task: {self._task}")
        self.episode_max = -np.inf
        self.slidewindow = SlideWindow(size=200)
        self.episode_metric = SlideWindow(size=99999)
        self.episode_metric_mean_max=0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_max = self._current_metric()
        #self.episode_max == -np.inf
        info = info or {}
        info["episode_max_metric"] = self.slidewindow.average

        self.episode_metric.reset()

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metric = self._current_metric()
        self.episode_metric.next(metric)
        self.slidewindow.next(metric)
        bonus = 50
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

        if  terminated or truncated:
            mean_metric = self.episode_metric.average
            if self.episode_metric_mean_max ==0:
                self.episode_metric_mean_max = mean_metric
            if mean_metric > self.episode_metric_mean_max:
                self.episode_metric_mean_max = mean_metric
            reward +=100
            info["stander_episode_reward_mean"] = mean_metric

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
