#depend on stablebaselines3, gymnasium
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import gymnasium as gym
import arrow
import numpy as np
from gymnasium.wrappers import RecordVideo
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from configs.base_args import get_root_path
import swanlab

from utils.calc_util import SlideWindow
from utils.swanlab_callback import SwanLabCallback
from utils.plot.plot_lines import plot_lines
import tyro
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
#batch_size 必须整除 n_steps × n_envs（对 PPO/A2C 等），否则会报错或被内部调整。
@dataclass
class Args:
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = 3000
    repeat: int = 1
    seed: int = -1
    track: bool = True
    enable_brave:bool = True
    normalize: bool = True
    swanlab_project: str = "Brave"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "ppo_mountain_car_sb"
    root_path: str = get_root_path()
    video_freq: int = 1  #
    n_eval_episodes: int = 2
    model_dir: str = get_root_path()+"/results/checkpoints/MountainCarContinuous_v0"
    video_dir: str = get_root_path()+"/results/videos/MountainCarContinuous_v0"
    tags: list[str] = field(default_factory=list)
    def finalize(self):
        self.experiment_name = self.env_id + '_' + arrow.now().format('MMDD_HHmm')
        if self.enable_brave:
            self.experiment_name += '_brave'
        #set seed to random value if seed is -1
        if self.seed == -1:
            self.seed = torch.randint(0, 10000, (1,)).item()
        print(f"Using seed: {self.seed}")
        if self.tags:
            parsed_tags = []
            for tag in self.tags:
                parsed_tags.extend([t.strip() for t in tag.split(',') if t.strip()])
            self.tags = parsed_tags
        # make dir if model_dir or video_dir not exist
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

    #make dir if model_dir or video_dir not exist

class BRSRewardWrapperV1(gym.Wrapper):
    def __init__(self, env,gamma: float = 0.99, beta_min: float = 1.1, evaluate: bool = False, plot_save_dir = None):
        super().__init__(env)
        self.cost = 0
        self.min_cost = 0
        self.stander_episode_reward = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cost = 0
        self.min_cost = 0
        self.num_steps = 0
        return obs, info
    def _get_state(self):
        # MountainCar obs = [position, velocity]
        pos, vel = self.env.unwrapped.state
        return pos, vel
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.stander_episode_reward +=reward
        self.cost+=(0.1* action**2)
        if terminated or truncated:
            if self.min_cost==0:
                self.min_cost = self.cost
                print(f'self.min_cost ={self.min_cost}')
            elif self.cost < self.min_cost:
                reward=20
                self.min_cost = self.cost
                print(f'self.min_cost ={self.min_cost}')
        self.num_steps += 1
        if terminated or truncated:
            info["stander_episode_reward_mean"] = self.stander_episode_reward/self.num_steps  # 更新累计 reward 到 info
            self.stander_episode_reward = 0
            pos, vel = self._get_state()
            print(f'num_steps = {self.num_steps}, term={terminated},trun={truncated},pos ={pos},vel={vel}reward ={reward}')
        return obs, reward, terminated, truncated, info

class BRSRewardWrapperV2(gym.Wrapper):
    def __init__(self, env, gamma: float = 0.95, beta_min: float = 1.1, evaluate: bool = False, plot_save_dir = None):
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
        self.evaluate = evaluate
        self.plot_save_dir = Path(plot_save_dir) if plot_save_dir is not None else None
        if self.plot_save_dir:
            self.plot_save_dir.mkdir(parents=True, exist_ok=True)
        self.eval_episode_idx = 1
        self._reset_eval_buffers()
        self.stander_episode_reward = 0

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
        if self.evaluate:
            self._reset_eval_buffers()
        return obs, info

    #return ln(1+|x|) with sign of x
    def signed_log(self,x):
        return np.sign(x) * np.log1p(abs(x))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["stander_reward"] = reward
        self.stander_episode_reward+=reward
        pos,vel= self._get_state()
        C_t = self._cost(pos,vel)
        self.sw.next(C_t)
        if self.evaluate:
            self.eval_costs.append(C_t)

        # 判断是否产生新的 global minimum cost
        if C_t < self.C_min:
            # 计算 bonus，保证 RDCR 超过历史最大
            bonus =  self.beta_min * (self.R_max - self.gamma * self.R_t)+0.01
            bonus = self.signed_log(bonus)
            self.C_min = C_t
            reward = bonus
        else:
            reward = self.reward_function()

        self.C_Last = C_t
        self.R_t = self.gamma * self.R_t + reward
        self.R_max = max(self.R_max, self.R_t)

        if self.evaluate:
            self.eval_rewards.append(reward)
            self.eval_rmax.append(self.R_max)

        info["brs_reward"] = reward
        info["cost"] = C_t
        info["C_min"] = self.C_min
        info["RDCR"] = self.R_t
        info["R_max"] = self.R_max
        self.num_steps+=1
        if terminated or truncated:
            info["stander_episode_reward_mean"] = self.stander_episode_reward/self.num_steps  # 更新累计 reward 到 info
            self.stander_episode_reward = 0
        if self.evaluate and (terminated or truncated):
            self._plot_eval_episode()
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
            reward = -((1 + delta_last) ** 2 - 1) * (1+abs(delta_0))*1.1 -0.01
        else:
            reward = -0.001

        return reward

    def _get_state(self):
        # MountainCar obs = [position, velocity]
        pos, vel = self.env.unwrapped.state
        return pos, vel

    def _cost(self, position, velocity=None):
        return abs(self.goal_pos - position)

    def _cost(self, position, velocity=None):
        if velocity is None:
            velocity = self.env.unwrapped.state[1]

        # mass = 1.0
        # gravity = 1.0
        # kinetic_energy = 0.5 * mass * (velocity ** 2)
        #
        # height = np.sin(3.0 * position)
        # potential_energy = gravity * (height + 1.0)
        if velocity == 0:
            velocity = 0.01  # prevent division by zero
        cost = abs(self.goal_pos - position) /( 10 * abs(velocity))
        #cost =  - 10 * abs(velocity)

        return cost

    def _reset_eval_buffers(self):
        self.eval_rewards = []
        self.eval_costs = []
        self.eval_rmax = []

    def _plot_eval_episode(self):
        if not self.eval_rewards:
            return
        save_path = None
        if self.plot_save_dir:
            save_path = self.plot_save_dir / f"episode_{self.eval_episode_idx:04d}.png"
        plot_lines(
            [self.eval_rewards, self.eval_costs, self.eval_rmax],
            names=["reward", "C_t", "R_max"],
            xlabel="Step",
            ylabel="Value",
            title=f"Eval Episode {self.eval_episode_idx}",
            show=False,
            save_path=str(save_path) if save_path else None
        )
        print(f'Saved eval episode plot to {save_path}')
        # log to swanlab
        image = swanlab.Image(str(save_path),file_type='png', caption=f"Eval Episode {self.eval_episode_idx}")
        swanlab.log({f"Eval Episode {self.eval_episode_idx}": image})
        self.eval_episode_idx += 1
        self._reset_eval_buffers()

def save_model(model: PPO, path: str) -> None:
    model.save(path)

def load_model(path: str, env) -> PPO:
    model = PPO.load(path, env=env)
    return model

def train_and_evaluate():
    args_dict = asdict(args)
    def make_env():
        env = gym.make(args.env_id)
        if args.enable_brave:
            env = BRSRewardWrapperV1(env)
        return env

    # 使用超参数
    hyperparams = dict(
        policy="MlpPolicy",
        batch_size=args.n_steps,
        n_steps=args.n_steps,
        gamma=0.9999,
        learning_rate=7.77e-5,
        ent_coef=0.00429,
        clip_range=0.1,
        n_epochs=10,
        gae_lambda=0.9,
        max_grad_norm=5,
        vf_coef=0.19,
        use_sde=True,
        policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
        verbose=1,
        seed=args.seed,
    )

    # 创建环境
    env = make_vec_env(make_env, n_envs=1, seed=args.seed)
    # 是否归一化
    if args.normalize:  # normalize: true
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 创建PPO模型
    model = PPO(**hyperparams, env=env)

    # 训练
    if args.track:
        model.learn(total_timesteps=args.total_timesteps,  # n_timesteps: 20000
                    callback=SwanLabCallback(
                        project=args.swanlab_project,
                        experiment_name=args.experiment_name,
                        workspace=args.swanlab_workspace,
                        group = args.swanlab_group,
                        verbose=2,
                        # **kwargs
                        **args_dict
                    ),
        )
    else:
        model.learn(total_timesteps=args.total_timesteps)
    # 保存模型
    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    save_model(model, str(model_path))


    def make_eval_env():
        plot_dir = Path(args.video_dir) / "plots"
        env_ = gym.make(args.env_id, render_mode="rgb_array")
        env_ = BRSRewardWrapperV1(env_, evaluate = True, plot_save_dir = plot_dir)
        env_ = RecordVideo(
            env_,
            video_folder=str(args.video_dir),
            episode_trigger=lambda ep_id: ep_id % args.video_freq == 0,
            name_prefix=f"eval_{args.experiment_name}",
        )
        return env_

    eval_env = make_eval_env()

    # evaluate：use evaluate_policy from sb3
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    if args.track:
        swanlab.finish()
    eval_env.close()
    env.close()
    print(f"Evaluation mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Model saved to: {str(model_path)}")
    print(f"Videos saved to: {str(args.video_dir)}")

'''
  "python -m application.mountain_car.mountain_car_continuous_v0 "
    "--repeat 3 --tags brave,debug --total_timesteps 10000"
'''
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.finalize()
    for i in range(args.repeat):
        args.n_steps = 2**(i+3)  # ensure batch_size divides n_steps * n_envs
        train_and_evaluate()
        time.sleep(60)
