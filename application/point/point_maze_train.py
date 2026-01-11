import time
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

import arrow
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import numpy as np
import torch
import tyro
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import swanlab
from swanlab.env import is_windows

from configs.base_args import get_root_path
from core.brs.point_maze_brs_wrapper import PointMazeBRSRewardWrapper
# + ExploRS (generic)
from application.ant.basic.wrappers.ant_explors_wrapper import ExploRSRewardWrapper, ExploRSConfig
from envs.maze.point_maze import PointMazeEnv
from utils.camera import FixedMujocoOffscreenRender
from utils.swanlab_callback import SwanLabCallback
import envs.maze.point_maze

if not is_windows():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["SDL_VIDEODRIVER"] = "dummy"


@dataclass
class Args:
    # env
    env_id: str = "PointMazeEnv-v0"
    reward_type: str = "dense"  # sparse / dense（前提：对应 env_id 注册了 Dense）
    total_timesteps: int = int(2e3)
    repeat: int = 1
    seed: int = -1
    reward_mode: str = "standerd"  # standerd / brave / explors

    # --- ExploRS config (reward shaping) ---
    explors_lmbd: float = 1.0
    explors_max_bonus: float = 1.0
    explors_explore_scale: float = 1.0
    explors_exploit_scale: float = 1.0
    explors_exploit_clip: float = 1.0
    explors_bin_xy: float = 0.5
    explors_bin_z: float = 0.05
    explors_bin_v: float = 0.5
    explors_use_mujoco_state: bool = True
    explors_obs_fallback_k: int = 6

    # logging
    track: bool = False
    swanlab_project: str = "PointMaze_UMaze"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "td3"
    tags: list[str] = field(default_factory=list)

    # path
    root_path: str = get_root_path()
    n_eval_episodes: int = 1
    model_dir: str = get_root_path() + "/results/checkpoints/PointMaze"
    video_dir: str = get_root_path() + "/results/videos/PointMaze"

    # TD3
    learning_rate: float = 1e-4
    batch_size: int = 256
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    noise_std: float = 0.1

    # others
    num_threads: int = -1

    # eval / video
    record_video: bool = True  # 排障：如果卡住，先改成 False 看 evaluation 是否能正常结束
    eval_render_mode: str | None = "rgb_array"  # 排障：设为 None 将完全跳过渲染链路

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(2)

        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + "_" + arrow.now().format("MMDD_HHmm")
        if self.seed == -1:
            self.reset_seed()

        if self.tags:
            parsed = []
            for tag in self.tags:
                parsed.extend([t.strip() for t in tag.split(",") if t.strip()])
            self.tags = parsed
        else:
            self.tags = []
        self.tags.append(self.reward_type)

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.video_dir).mkdir(parents=True, exist_ok=True)


def add_reward_wrapper(env, args: Args):
    print(f"Adding reward wrapper: {args.reward_mode}")
    match args.reward_mode:
        case "brave":
            env = PointMazeBRSRewardWrapper(env)
        case "explors":
            cfg = ExploRSConfig(
                lmbd=args.explors_lmbd,
                max_bonus=args.explors_max_bonus,
                explore_scale=args.explors_explore_scale,
                exploit_scale=args.explors_exploit_scale,
                exploit_clip=args.explors_exploit_clip,
                bin_xy=args.explors_bin_xy,
                bin_z=args.explors_bin_z,
                bin_v=args.explors_bin_v,
                use_mujoco_state=args.explors_use_mujoco_state,
                obs_fallback_k=args.explors_obs_fallback_k,
            )
            env = ExploRSRewardWrapper(env, cfg)
        case "standerd":
            pass
        case _:
            pass
    return env


def save_model(model: TD3, path: str) -> None:
    model.save(path)


def train_and_evaluate(args: Args):
    args_dict = asdict(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def make_env():
        # 用自实现 PointMazeEnv（不再用 gym.make(args.env_id, ...)）
        env = gym.make(args.env_id, reward_type=args.reward_type)
        env = add_reward_wrapper(env, args)
        env = Monitor(env, filename=None, allow_early_resets=True)
        env.reset(seed=args.seed)
        return env

    env = make_vec_env(make_env, n_envs=1, seed=args.seed)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=args.noise_std * np.ones(n_actions),
    )

    model = TD3(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=1_000_000,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.99,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        action_noise=action_noise,
        verbose=1,
        seed=args.seed,
    )

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
    save_model(model, str(model_path))

    def make_eval_env():
        # 注意：PointMaze 自实现 env 若 rgb_array 渲染实现有阻塞，evaluate_policy 会卡死在 render()
        base_env = gym.make(args.env_id,
            render_mode=args.eval_render_mode,
            reward_type=args.reward_type,
        )
        base_env = add_reward_wrapper(base_env, args)
        base_env = Monitor(base_env, filename=None, allow_early_resets=True)

        # 只有在明确需要录制且启用 rgb_array 时才走离屏渲染+录制
        if args.record_video and args.eval_render_mode == "rgb_array":
            base_env = FixedMujocoOffscreenRender(base_env, None, width=480, height=480)
            base_env = RecordVideo(
                base_env,
                video_folder=str(args.video_dir),
                episode_trigger=lambda ep_id: ep_id == 0,  # 避免每个 episode 都录制
                name_prefix=f"eval_{args.experiment_name}",
            )
        return base_env

    eval_env = make_eval_env()
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


if __name__ == "__main__":
    gym.register_envs(gymnasium_robotics)
    args = tyro.cli(Args)
    args.finalize()

    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    for _ in range(args.repeat):
        train_and_evaluate(args)
        args.reset_seed()
        time.sleep(10)

# python -m application.point.train_point_maze --total_timesteps 1000000 --track --reward-mode explors
