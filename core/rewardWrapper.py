# python
from pathlib import Path

import gymnasium as gym
import numpy as np
from typing import Callable, Tuple, Any
from configs.args import PpoAtariArgs
import matplotlib.pyplot as plt
args = PpoAtariArgs().finalize()
class BreakoutRewardWrapper(gym.Wrapper):
    """
    reward_fn 的签名示例:
      def reward_fn(obs, action, reward, next_obs, terminated, truncated, info) -> float
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.discount_factor = 0.95
        self.discounted_return = 0.0
        self.peak_discounted_return = 0.0
        self.reward_history = []
        self.score_history = []
        self.discounted_return_history = []
        self.best_score_record = 0
        self.current_score = 0
        self.life_loss_penalty = getattr(args, "life_loss_penalty", -2.0)
        self.last_lives = None

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.sign(float(reward))  # bin reward to +1 -1 or 0

        lives = info.get("lives")
        if lives is not None:
            if self.last_lives is not None and lives < self.last_lives:
                reward += self.life_loss_penalty
            self.last_lives = lives
        elif terminated or truncated:
            self.last_lives = None

        if self.discounted_return > self.peak_discounted_return:
            self.peak_discounted_return = self.discounted_return
        self.discounted_return_history.append(self.discounted_return)
        self.current_score += reward

        if (self.current_score > self.best_score_record
                and reward > 0
                and args.enable_brave):
            self.best_score_record = self.current_score
            reward = (self.peak_discounted_return - self.discounted_return) * 1.2

        self.discounted_return = self.discounted_return * self.discount_factor + reward
        self.score_history.append(self.current_score)
        self.reward_history.append(reward)
        return obs, reward, terminated, truncated, info

    def save_plots(self):
        # plot reward_recorder and score_recorder on a graph and save it
        plt.figure(figsize=(12, 6))
        plt.plot(self.reward_history, label='Reward', alpha=0.7)
        plt.plot(self.discounted_return_history, label='discounted_return_history', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Reward and Score over Time')
        plt.legend()
        plt.grid()
        print(args.root_path / 'results' / Path(args.experiment_name + '_reward_score_plot.png'))
        plt.savefig(args.root_path / 'results' / Path(args.experiment_name + '_reward_score_plot.png'))
        plt.close()

    def reset(self, **kwargs):
        self.current_score = 0
        self.best_score_record = 0
        self.reward_history = []
        self.score_history = []
        self.discounted_return_history = []
        self.last_lives = None
        self.discounted_return = 0.0
        self.peak_discounted_return = 0.0
        obs, info = self.env.reset(**kwargs)
        self.last_lives = info.get("lives")
        return obs, info

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
    env = BreakoutRewardWrapper(env, reward_fn=atari_sign_reward)

    obs, _ = env.reset(seed=0)
    done = False
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward:", reward)
        if terminated or truncated:
            break
    env.close()