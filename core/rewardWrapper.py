# python
import gymnasium as gym
import numpy as np
from typing import Callable, Tuple, Any
from configs.args import PpoAtariArgs

args = PpoAtariArgs().finalize()
class CustomRewardWrapper(gym.Wrapper):
    """
    reward_fn 的签名示例:
      def reward_fn(obs, action, reward, next_obs, terminated, truncated, info) -> float
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_recorder = []
        self.score_recorder = []
        self.best_score = 0
        self.score = 0

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.score += reward
        new_reward = reward
        if self.score > self.best_score:
            self.best_score = self.score
            new_reward *=4
        self.score_recorder.append(self.score)
        self.reward_recorder.append(new_reward)

        return obs, new_reward, terminated, truncated, info


    def close(self):
        # plot reward_recorder and score_recorder on a graph and save it
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(self.reward_recorder, label='Modified Reward', alpha=0.7)
        plt.plot(self.score_recorder, label='Cumulative Score', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Reward and Score over Time')
        plt.legend()
        plt.grid()
        plt.savefig(args.root_path / 'results' / 'reward_score_plot.png')
        plt.close()



    def reset(self, **kwargs):
        self.score = 0
        self.best_score = 0
        self.reward_recorder = []
        self.score_recorder = []
        return self.env.reset(**kwargs)







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