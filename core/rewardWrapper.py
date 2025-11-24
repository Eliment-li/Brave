# python
import gymnasium as gym
import numpy as np
from typing import Callable, Tuple, Any

class CustomRewardWrapper(gym.Wrapper):
    """
    reward_fn 的签名示例:
      def reward_fn(obs, action, reward, next_obs, terminated, truncated, info) -> float
    """
    def __init__(self, env: gym.Env, reward_fn: Callable[..., float]):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 允许 reward_fn 使用旧 obs/next_obs/other 信息返回新 reward
        try:
            new_reward = self.reward_fn(None, action, reward, obs, terminated, truncated, info)
        except TypeError:
            # 兼容只传 (reward,) 或 (reward, info) 的简单函数
            new_reward = self.reward_fn(reward)
        return obs, new_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def Brave_reward():
    pass




# demo
if __name__ == "__main__":

    # 示例 reward 函数：Atari 常见做法（奖励取符号）
    def atari_sign_reward(_obs, _action, reward, _next_obs, _terminated, _truncated, _info):
        return float(np.sign(reward))

    # 示例 reward 函数：CartPole 的简单形状化（根据杆子角度奖励更接近竖直）
    def cartpole_shaped_reward(_obs, _action, reward, next_obs, _terminated, _truncated, _info):

        # next_obs 是 ndarray；CartPole 返回 [x, x_dot, theta, theta_dot]
        theta = float(next_obs[2])
        shaped = 1.0 - (abs(theta) / (np.pi / 2))  # 范围约在 0..1，越接近竖直越大
        return float(reward) + 0.5 * shaped


    env_id = "PongNoFrameskip-v4"  # 示例 Atari
    env = gym.make(env_id, render_mode=None)
    # 包装为自定义 reward（例如替代 ClipRewardEnv）
    env = CustomRewardWrapper(env, reward_fn=atari_sign_reward)

    obs, _ = env.reset(seed=0)
    done = False
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward)
        if terminated or truncated:
            break
    env.close()