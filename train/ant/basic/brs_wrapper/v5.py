import math
import gymnasium as gym
import numpy as np
from utils.calc_util import SlideWindow
# V5 use global rdcr_max and global best so far performance  to compute bonus
# V4 use spisode rdcr_max and global best so far performance  to compute bonus
class AntBRSRewardWrapperV5(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        beta: float = 1.001,
        min_bonus: float = 1,
        global_bonus: float = 0,
    ):
        super().__init__(env)
        self.global_bonus = global_bonus
        self.task = getattr(self.env.unwrapped, "task", None)
        if self.task not in {"stand", "speed", "far"}:
            raise ValueError(f"Unsupported task: {self.task}")
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.min_bonus = float(min_bonus)
        self.global_max = self._current_metric()
        self.episode_max = self._current_metric()
        self.rdcr = 0.0
        self.rdcr_max = 0.0
        print(f'global_bonus set to {self.global_bonus}')
        print(f'beta set to {self.beta}')


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.brs_trigger_count=1
        info = dict(info or {})
        self.episode_max = self._current_metric()
        self.rdcr = 0.0
        self.rdcr_max = 0.0

        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info["global_max_"+self.task] = self.global_max
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        metric = self._current_metric()
        info[str(self.task)] = metric

        bonus = 0.0
        if metric - self.episode_max>0.01:
            self.episode_max = metric
            beta = 1+(1/2**(self.brs_trigger_count))
            bonus =(beta*(self.rdcr_max - self.gamma * self.rdcr) + self.min_bonus/self.brs_trigger_count)
            #bonus = beta*(self.rdcr_max - self.gamma * self.rdcr)
            bonus = max(reward,bonus)
            #global
            if metric > self.global_max:
                self.global_max = metric
                bonus +=self.global_bonus

            reward = bonus
            self.rdcr = self.gamma * self.rdcr + reward
            self.brs_trigger_count += 1
            #assert self.rdcr >= self.rdcr_max, f"rdcr did not increase: {self.rdcr} <= {self.rdcr_max}"
        else:
            self.rdcr = self.gamma * self.rdcr + reward


        self.rdcr_max = max(self.rdcr_max, self.rdcr)

        # Be defensive: some envs/algos may return immutable mappings.

        info["brs_bonus"] = bonus
        info["rdcr"] = self.rdcr
        info["rdcr_max"] = self.rdcr_max
        info[str(self.task)] = metric
        info[r"metric/global_max"] = self.global_max
        info[r"metric/episode_max"] = self.episode_max
        return obs, reward, terminated, truncated, info

    def _current_metric(self) -> float:
        data = self.env.unwrapped.data
        qpos = data.qpos.ravel()
        qvel = data.qvel.ravel()
        if self.task == "stand":
            return float(qpos[2])
        if self.task == "speed":
            return float(qvel[0])
        if self.task == "far":
            return float(np.linalg.norm(qpos[:2]))

    @staticmethod
    def _signed_log(x: float) -> float:
        return math.copysign(math.log1p(abs(x)), x)
