#depend on stablebaselines3, gymnasium
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
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from application.mountain_car.mountain_car_info_wrapper import MountainCarRewardStatsWrapper
from bak.configs.base_args import get_root_path
import swanlab

from utils.calc_util import SlideWindow
from utils.swanlab_callback import SwanLabCallback
from utils.plot.plot_lines import plot_lines
import tyro
import os
import matplotlib.pyplot as plt
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
    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()
    def finalize(self):
        self.experiment_name = self.env_id + '_' + arrow.now().format('MMDD_HHmm')
        if self.enable_brave:
            self.experiment_name += '_brave'
        #set seed to random value if seed is -1
        if self.seed == -1:
            self.reset_seed()
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
                reward=80
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
        env = MountainCarRewardStatsWrapper(env)
        return env

    # tuned hyperparams from official stable-baselines3 https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
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

    # 创建训练环境
    env = make_vec_env(make_env, n_envs=1, seed=args.seed)
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

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
    #plot reward stats
    try:
        stats_list = env.env_method("stats_dict")
        if stats_list:
            reward_stats = stats_list[0]
            rewards_by_bin = reward_stats["rewards_by_bin"]
            bin_size = reward_stats["bin_size"]
            x_min = reward_stats["x_min"]
            x_max = reward_stats["x_max"]

            x_centers, avg_rewards = [], []
            for idx, rewards in enumerate(rewards_by_bin):
                if not rewards:
                    continue
                x_center = min(x_min + (idx + 0.5) * bin_size, x_max)
                x_centers.append(x_center)
                avg_rewards.append(float(np.mean(rewards)))

            if avg_rewards:
                stats_plot_path = Path(args.model_dir) / f"{args.experiment_name}_reward_stats.png"
                plt.figure(figsize=(10, 4))
                plt.plot(x_centers, avg_rewards, label="Avg reward by x")
                plt.xlabel("Position (x)")
                plt.ylabel("Average reward")
                plt.title("MountainCar Reward Mean vs Position")
                plt.legend()
                plt.tight_layout()
                plt.savefig(stats_plot_path)
                plt.close()
                print(f"Reward stats plot saved to: {stats_plot_path}")
    except AttributeError:
        pass
    # 保存模型
    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    save_model(model, str(model_path))

    # 如果使用了 VecNormalize，把归一化统计单独保存出来，评估时复用
    if args.normalize and isinstance(env, VecNormalize):
        vecnorm_path = Path(args.model_dir) / f"{args.experiment_name}_vecnormalize.pkl"
        env.save(str(vecnorm_path))
    else:
        vecnorm_path = None

    def make_eval_env():
        """
        评估环境构造：
        - 未归一化：RecordVideo(gym.Env 或 BRSRewardWrapperV1)
        - 归一化：在 RecordVideo 包裹的 env 外，再用 VecNormalize.load 创建 VecEnv
        """
        plot_dir = Path(args.video_dir) / "plots"

        def _make_single_eval_env():
            # 基础环境：render_mode="rgb_array" 以便 RecordVideo 录制
            base_env = gym.make(args.env_id, render_mode="rgb_array")
            if args.enable_brave:
                base_env = BRSRewardWrapperV1(base_env, evaluate=True, plot_save_dir=plot_dir)

            # 先在 gym.Env 上包 RecordVideo
            video_env = RecordVideo(
                base_env,
                video_folder=str(args.video_dir),
                episode_trigger=lambda ep_id: ep_id % args.video_freq == 0,
                name_prefix=f"eval_{args.experiment_name}",
            )
            return video_env

        if args.normalize and vecnorm_path is not None:
            # 评估时也需要 VecNormalize：
            # 1. 用 DummyVecEnv 把单个 env 包成 VecEnv
            # 2. 用 VecNormalize.load 载入训练时的统计，设置 training=False
            eval_vec_env = DummyVecEnv([_make_single_eval_env])
            eval_vec_env = VecNormalize.load(str(vecnorm_path), eval_vec_env)
            eval_vec_env.training = False
            eval_vec_env.norm_reward = True
            eval_vec_env.norm_obs = True
            return eval_vec_env
        else:
            # 不使用归一化：直接返回 gym.Env（已带 RecordVideo）
            return _make_single_eval_env()

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
  "python -m train.mountain_car.mountain_car_continuous_v0 "
    "--repeat 3 --tags brave,debug --total_timesteps 10000"
'''
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.finalize()
    for i in range(args.repeat):
        args.n_steps = 8  # ensure batch_size divides n_steps * n_envs
        train_and_evaluate()
        args.reset_seed()
        #time.sleep(30)
