import numpy as np
import gymnasium as gym

class BRSRewardWrapper(gym.Wrapper):
    def __init__(self, env, gamma: float = 0.99, beta_min: float = 1.01):
        super().__init__(env)
        self.gamma = gamma
        self.beta_min = beta_min
        self.goal_pos = self.env.unwrapped.goal_position  # MountainCar 自带
        # 这些变量按 episode / 全局维护
        self.C0 = None           # 当前 episode 的初始 cost
        self.C_Last = None       # c_{t-1}
        self.C_star = np.inf     # 历史最小 cost（全局）
        self.R_t = 0.0           # 当前 episode RDCR
        self.R_max = -np.inf     # 历史最大 RDCR


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = self._get_position()
        self.C0 = self._cost(pos)
        self.C_star = np.inf

        self.R_t = 0.0
        self.R_max = -np.inf

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["reward"] = reward
        # 原始 MountainCar 的 reward 是 -1 每步；我们这里不用它，改用 cost-aware r_t
        pos = self._get_position()
        C_t = self._cost(pos)
        delta_C_t0 = (C_t - self.C0) / (self.C0 + 1e-8)
        # 这里没有 C_{t-1}，为了简单，用单变量版本：只看相对 C_t0
        if delta_C_t0 >= 0:
            r_t = (1 - delta_C_t0) ** 2 - 1
        else:
            r_t = -((1 + delta_C_t0) ** 2 - 1)

        # 先用 r_t 更新 RDCR
        self.R_t = self.gamma * self.R_t + r_t

        # 判断是否产生新的 global minimum cost
        brs_reward = r_t
        if C_t < self.C_star:
            # 计算 bonus，保证 RDCR 超过历史最大
            beta = self.beta_min#max(self.beta_min, (self.C_star - C_t) / (self.C_star + 1e-8))
            bonus = beta * (self.R_max - self.gamma * self.R_t)
            brs_reward = bonus
            if brs_reward >10:
                brs_reward =10.0

            # 更新 C* 与 R_max, R_t
            self.C_star = C_t
            self.R_t = self.gamma * self.R_t + bonus
            self.R_max = max(self.R_max, self.R_t)
            # 用 BRS 后的 reward 替代原 reward
            reward = float(max(brs_reward,reward,1e-5))  # 保证 reward 不小于原始 reward
        info["brs_reward"] = reward
        info["cost"] = C_t
        info["C_star"] = self.C_star
        info["RDCR"] = self.R_t
        info["R_max"] = self.R_max

        return obs, reward, terminated, truncated, info

    def _get_position(self):
        # MountainCar obs = [position, velocity]
        obs = self.env.unwrapped.state
        pos = obs[0]
        return pos

    def _cost(self, position):
        # 越接近 goal cost 越小，用距离作为 cost
        return abs(self.goal_pos - position)
