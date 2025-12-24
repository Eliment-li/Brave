
import numpy as np
import gymnasium as gym
from utils.calc_util import SlideWindow

class BRSRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma: float = 0.95, beta_min: float = 1.1):
        super().__init__(env)
        self.gamma = gamma
        self.beta_min = beta_min
        self.goal_pos = self.env.unwrapped.goal_position
        # 这些变量按 episode / 全局维护
        self.C0 = None          # 当前 episode 的初始 cost
        self.C_Last = None       # c_{t-1}
        self.C_min = None     # 历史最小 cost（全局）
        self.R_t = 0.0           # 当前 episode RDCR
        self.R_max = 0.0     # 历史最大 RDCR
        self.sw = SlideWindow(50)
        self.num_steps = 0
        self.min_num_steps=200


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos,vel = self._get_state()
        self.C0 = self._cost(pos,vel)
        self.sw.next(self.C0)
        self.C_min = self.C0
        self.C_Last = self.C0
        self.R_t = 0.0
        self.R_max = 0.0
        self.num_steps = 0
        self.min_num_steps = 200

        return obs, info

    #return ln(1+|x|) with sign of x
    def signed_log(self,x):
        return np.sign(x) * np.log1p(abs(x))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info["reward"] = reward
        # pos,vel= self._get_state()
        # C_t = self._cost(pos,vel)
        # self.sw.next(C_t)
        #
        # # 判断是否产生新的 global minimum cost
        # if C_t < self.C_min:
        #     # 计算 bonus，保证 RDCR 超过历史最大
        #     bonus =  self.beta_min * (self.R_max - self.gamma * self.R_t)+0.01
        #     bonus = self.signed_log(bonus)
        #
        #
        #     self.C_min = C_t
        #     reward = bonus
        # else:
        #     reward = self.reward_function()
        #
        # self.C_Last = C_t
        # self.R_t = self.gamma * self.R_t + reward
        # self.R_max = max(self.R_max, self.R_t)
        #
        # info["brs_reward"] = reward
        # info["cost"] = C_t
        # info["C_min"] = self.C_min
        # info["RDCR"] = self.R_t
        # info["R_max"] = self.R_max
        self.num_steps+=1
        if terminated or truncated:
            if self.num_steps < self.min_num_steps:
                reward=10
                self.min_num_steps = self.num_steps
                #print(f'self.min_num_steps ={self.min_num_steps}')
        return obs, reward, terminated, truncated, info


    def reward_function(self):
        # 原始 MountainCar 的 reward 是 -1 每步；我们这里不用它，改用 cost-aware r_t
        pos,vel = self._get_state()
        C_t = self._cost(pos,vel)
        delta_0 = (C_t - self.C0) / self.sw.average
        delta_last = (C_t - self.C_Last) / self.sw.average

        if delta_last < 0:
            # agent 在上一步减少了 cost
            reward = ((1 - delta_last) ** 2 - 1) * (1+abs(delta_0)) + 0.01
        elif delta_last > 0:
            # agent 在上一步增加了 cost
            reward = -((1 + delta_last) ** 2 - 1) * (1+abs(delta_0))*1.2 -0.01
        else:
            reward = -0.001

        return reward

    def _get_state(self):
        # MountainCar obs = [position, velocity]
        pos, vel = self.env.unwrapped.state
        return pos, vel

    # def _cost(self, position):
    #     # 越接近 goal cost 越小，用距离作为 cost
    #     return abs(self.goal_pos - position)

    def _cost(self, position, velocity=None):
        if velocity is None:
            velocity = self.env.unwrapped.state[1]

        # mass = 1.0
        # gravity = 1.0
        # kinetic_energy = 0.5 * mass * (velocity ** 2)
        #
        # height = np.sin(3.0 * position)
        # potential_energy = gravity * (height + 1.0)
        cost = abs(self.goal_pos - position) - 20 * abs(velocity)
        #cost =  - 10 * abs(velocity)

        return cost
