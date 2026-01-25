import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import arrow
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from bak.configs.base_args import get_root_path
from brs.mountain_car_brs_wrapper import MountainCarBRSRewardWrapper
from train.mountain_car.mountain_car_info_wrapper import MountainCarRewardStatsWrapper
from utils.screen import set_screen_config
from utils.swanlab_callback import SwanLabCallback

import swanlab
import tyro


@dataclass
class Args:
    # env
    env_id: str = "MountainCarContinuous-v0"
    total_timesteps: int = int(1000000)
    repeat: int = 1
    seed: int = -1
    track: bool = False
    reward_mode: str = "brave"  # brave | standard

    # allow override episode length; <=0 means use env default spec
    max_episode_steps: int = -1

    # eval
    n_eval_episodes: int = 3

    # path
    root_path: str = get_root_path()
    model_dir: str = get_root_path() + "/results/checkpoints/MountainCarContinuous"
    video_dir: str = get_root_path() + "/results/videos/MountainCarContinuous"

    # logging
    swanlab_project: str = "Brave_MountainCar"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = ""
    tags: list[str] = field(default_factory=list)

    # brave
    global_bonus: float = 5.0
    use_global_max_bonus: bool = True
    gamma: float = 0.99
    beta: float = 1.1
    min_bonus: float = 0.01

    # TD3 hyperparameters
    learning_rate: float = 1e-3
    buffer_size: int = 500_000
    batch_size: int = 256
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1
    noise_std: float = 0.1
    noise_type: str = "normal"

    # others
    num_threads: int = -1

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(2)

        if self.seed == -1:
            self.reset_seed()

        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + "_" + arrow.now().format("MMDD_HHmm")
        self.experiment_name += ("_" + self.reward_mode)
        self.swanlab_group = self.reward_mode

        if self.tags:
            parsed = []
            for t in self.tags:
                parsed.extend([x.strip() for x in t.split(",") if x.strip()])
            self.tags = parsed
        self.tags.extend(["td3", self.noise_type])

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

        print(f"Using seed: {self.seed}")


def add_reward_wrapper(env: gym.Env, args: Args) -> gym.Env:
    print(f"Adding reward wrapper: {args.reward_mode}")
    match args.reward_mode:
        case "brave":
            env = MountainCarBRSRewardWrapper(
                env,
                gamma=args.gamma,
                beta=args.beta,
                min_bonus=args.min_bonus,
                use_global_max_bonus=args.use_global_max_bonus,
                global_bonus=args.global_bonus,
            )
        case "standard":
            pass
        case _:
            raise ValueError(f"Unknown reward_mode: {args.reward_mode}")
    return env


def _make_mountain_car_env(args: Args, *, render_mode=None) -> gym.Env:
    kwargs = {}
    if args.max_episode_steps and args.max_episode_steps > 0:
        kwargs["max_episode_steps"] = args.max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(args.env_id, **kwargs)
    env = add_reward_wrapper(env, args)
    env = MountainCarRewardStatsWrapper(env)
    env = Monitor(env, filename=None, allow_early_resets=True)
    env.reset()
    return env


def train_and_evaluate(args: Args):
    args_dict = asdict(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # train env (VecEnv)
    env = make_vec_env(lambda: _make_mountain_car_env(args), n_envs=1, seed=args.seed)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = None
    if args.noise_type == "normal":
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise_std * np.ones(n_actions))

    hyperparams = dict(
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        verbose=1,
        seed=args.seed,
    )

    model = TD3(env=env, **hyperparams)

    if args.track:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=SwanLabCallback(
                project=args.swanlab_project,
                experiment_name=args.experiment_name,
                workspace=args.swanlab_workspace,
                group=args.swanlab_group,
                verbose=2,
                **args_dict,
            ),
        )
    else:
        model.learn(total_timesteps=args.total_timesteps)

    # save
    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    model.save(str(model_path))

    # eval env + record video
    base_env = _make_mountain_car_env(args, render_mode="rgb_array")
    video_env = RecordVideo(
        base_env,
        video_folder=str(args.video_dir),
        episode_trigger=lambda ep_id: True,
        name_prefix=f"eval_{args.experiment_name}",
    )

    mean_reward, std_reward = evaluate_policy(
        model,
        video_env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    if args.track:
        swanlab.finish()

    video_env.close()
    env.close()

    print(f"Evaluation mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Model saved to: {str(model_path)}")
    print(f"Videos saved to: {str(args.video_dir)}")


if __name__ == "__main__":
    set_screen_config()

    args = tyro.cli(Args)
    args.finalize()

    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    for _ in range(args.repeat):
        train_and_evaluate(args)
        args.reset_seed()
        time.sleep(5)

