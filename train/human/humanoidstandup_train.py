import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import arrow
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from bak.configs.base_args import get_root_path
from brs.humanoidstandup_brs import HumanoidStandupBRSRewardWrapper
from algos.explors.humanoidstandup_explors_wrapper import (
    HumanoidStandupExploRSRewardWrapper,
    HumanoidStandupExploRSConfig,
)
from utils.screen import set_screen_config
from utils.swanlab_callback import SwanLabCallback
import swanlab
import tyro


@dataclass
class Args:
    # env
    env_id: str = "HumanoidStandup-v5"
    total_timesteps: int = int(1e5)
    repeat: int = 1
    seed: int = -1
    track: bool = False
    reward_mode: str = "standard"  # standard|brave|explors
    reward_type: str = "dense"     # dense|sparse
    sparse_height_th: float = 0.2

    # allow override episode length
    max_episode_steps: int = -1

    # eval
    n_eval_episodes: int = 3

    # path
    root_path: str = get_root_path()
    model_dir: str = get_root_path() + "/results/checkpoints/HumanoidStandup"
    video_dir: str = get_root_path() + "/results/videos/HumanoidStandup"

    # logging
    swanlab_project: str = "Brave_HumanoidStandup"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = ""
    tags: list[str] = field(default_factory=list)

    # brave
    global_bonus: float = 100
    use_global_max_bonus: bool = True

    # explors
    explors_lmbd: float = 1.0
    explors_max_bonus: float = 1.0
    explors_explore_scale: float = 1.0
    explors_exploit_scale: float = 1.0
    explors_exploit_clip: float = 1.0
    explors_bin_xy: float = 0.5
    explors_bin_z: float = 0.05
    explors_bin_v: float = 0.5

    # PPO hyperparameters (reasonable defaults for MuJoCo continuous control)
    # learning_rate: float = 3e-4
    # n_steps: int = 2048
    # batch_size: int = 64
    # n_epochs: int = 10
    # gamma: float = 0.99
    # gae_lambda: float = 0.95
    # clip_range: float = 0.2
    # ent_coef: float = 0.0
    # vf_coef: float = 0.5
    # max_grad_norm: float = 0.5

    # TD3 Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 1024
    learning_starts: int = 50_000
    train_freq: int = 1
    gradient_steps: int = 2
    noise_std: float = 0.2

    noise_type: str = "normal"

    # TD3 network architecture (MlpPolicy)
    # 例如: [512, 512] 表示 2 层，每层 512；[] 表示不加隐藏层
    td3_net_arch: list[int] = field(default_factory=lambda: [512,512])

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

        self.experiment_name = 'human_stand' + "_" + arrow.now().format("MMDD_HHmm")
        self.experiment_name += ("_" + self.reward_mode)
        self.swanlab_group = self.reward_mode

        if self.tags:
            parsed = []
            for t in self.tags:
                parsed.extend([x.strip() for x in t.split(",") if x.strip()])
            self.tags = parsed

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)

        print(f"Using seed: {self.seed}")


def _make_humanoid_env(args: Args, *, render_mode=None):
    kwargs = {}
    if args.max_episode_steps and args.max_episode_steps > 0:
        kwargs["max_episode_steps"] = args.max_episode_steps
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    kwargs["reward_type"] = args.reward_type
    kwargs["height_th"] = args.sparse_height_th

    env = gym.make(args.env_id, **kwargs)
    env = add_reward_wrapper(env, args)
    env = Monitor(env, filename=None, allow_early_resets=True)
    env.reset()
    return env


def add_reward_wrapper(env: gym.Env, args: Args):
    print(f"Adding reward wrapper: {args.reward_mode}")
    match args.reward_mode:
        case "brave":
            env = HumanoidStandupBRSRewardWrapper(
                env,
                use_global_max_bonus=args.use_global_max_bonus,
                global_bonus=args.global_bonus,
            )
        case "explors":
            env = HumanoidStandupExploRSRewardWrapper(
                env,
                config=HumanoidStandupExploRSConfig(
                    lmbd=args.explors_lmbd,
                    max_bonus=args.explors_max_bonus,
                    explore_scale=args.explors_explore_scale,
                    exploit_scale=args.explors_exploit_scale,
                    exploit_clip=args.explors_exploit_clip,
                    bin_xy=args.explors_bin_xy,
                    bin_z=args.explors_bin_z,
                    bin_v=args.explors_bin_v,
                ),
            )
        case "standard":
            pass
    return env


def train_and_evaluate(args: Args):
    args_dict = asdict(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_vec_env(lambda: _make_humanoid_env(args), n_envs=1, seed=args.seed)

    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = None
    if args.noise_type == "normal":
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise_std * np.ones(n_actions))

    policy_kwargs = dict(net_arch=list(args.td3_net_arch))

    hyperparams = dict(
        policy="MlpPolicy",
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
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

    model_path = Path(args.model_dir) / f"{args.experiment_name}_seed{args.seed}.zip"
    model.save(str(model_path))

    base_env = _make_humanoid_env(args, render_mode="rgb_array")
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
    from gymnasium.envs.registration import register

    args = tyro.cli(Args)
    args.finalize()

    # Register local env to ensure we use repo's implementation
    register(
        id="HumanoidStandup-v5-local",
        entry_point="envs.human.humanoidstandup_v5:HumanoidStandupEnv",
        max_episode_steps=args.max_episode_steps if args.max_episode_steps > 0 else 1000,
    )
    # Use local by default
    args.env_id = "HumanoidStandup-v5-local"

    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    for _ in range(args.repeat):
        train_and_evaluate(args)
        args.reset_seed()
        time.sleep(5)
