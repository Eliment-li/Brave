import time
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import arrow
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import tyro
from gymnasium import register
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import swanlab

from brs.ant_maze_brs_wrapper import AntMazeBRSRewardWrapper
from info_wrapper.ant_info_wrapper import AntMazeInfoWrapper
from bak.configs.base_args import get_root_path
from utils.camera import FixedMujocoOffscreenRender
from utils.screen import set_screen_config
from utils.swanlab_callback import SwanLabCallback

# from train.ant.basic.algos.ant_info_wrapper import OriginalRewardInfoWrapper

@dataclass
class Args:
    # env
    env_id: str = "AntMaze_UMaze-v5"  # 例如：AntMaze_UMaze-v5 / AntMaze_BigMaze-v5 / AntMaze_HardestMaze-v5
    #如果使用 dense rewrad,只需修改 env_id 即可 例如 AntMaze_UMazeDense-v5
    reward_type: str = "sparse"       # sparse / dense（取决于你注册的 id 是否支持 Dense；否则用 reward_type 参数）
    total_timesteps: int = int(2e3)
    repeat: int = 1
    seed: int = -1
    reward_mode: str = "standard"
    max_episode_steps:int = 200

    # brave
    global_bonus: float = 5.0
    use_global_max_bonus: bool = True

    # maze map
    maze_map_name: str = "U_MAZE"              # 训练用地图名（见 envs/maze/maps.py: MAPS）
    eval_maze_map_name: str = ""              # 评估用地图名；空字符串表示复用训练地图

    # logging
    track: bool = False
    swanlab_project: str = "AntMaze_UMaze"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "td3"
    tags: list[str] = field(default_factory=list)

    # path
    root_path: str = get_root_path()
    n_eval_episodes: int = 1
    model_dir: str = get_root_path() + "/results/checkpoints/AntMaze"
    video_dir: str = get_root_path() + "/results/videos/AntMaze"

    # TD3
    learning_rate: float = 1e-4
    batch_size: int = 256
    learning_starts: int = 10000
    train_freq: int = 1
    gradient_steps: int = 1
    noise_std: float = 0.1

    # others
    num_threads: int = -1
    # eval / video
    record_video: bool = True
    eval_render_mode: Optional[str] = "rgb_array"

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(2)

        #self.swanlab_project = self.swanlab_project + '_' + self.task

        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + "_" + arrow.now().format("MMDD_HHmm") + "_" + self.reward_mode
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

def brs_wrapper(env, version: int):
    """根据版本号返回对应的 Ant BRS reward wrapper。"""
    # wrapper_cls = _WRAPPER_MAP.get(version)
    # print(f'use reward wrapper v{str(wrapper_cls)}')
    # if wrapper_cls is None:
    #     raise ValueError(f"未知的 AntBRSRewardWrapper 版本: {version}")
    # return wrapper_cls(env)
def add_reward_wrapper(env, args):
    print(f'Adding reward wrapper: {args.reward_mode}')
    match args.reward_mode:
        case 'brave':
            env = AntMazeBRSRewardWrapper(
                env,
                use_global_max_bonus=args.use_global_max_bonus,
                global_bonus=args.global_bonus,
            )
        case 'standard':
            pass
    return env
def save_model(model: TD3, path: str) -> None:
    model.save(path)


def train_and_evaluate(args: Args):
    args_dict = asdict(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from envs.maze import maps as maze_maps
    train_maze_map = maze_maps.get_map(args.maze_map_name)
    eval_maze_map = maze_maps.get_map(args.eval_maze_map_name) if args.eval_maze_map_name else train_maze_map

    def make_env():
        env = gym.make(args.env_id,reward_type=args.reward_type, maze_map=train_maze_map)
        # 如需把原始 reward 写入 info，可在这里包 wrapper（若你有对应实现）
        # env = OriginalRewardInfoWrapper(env)
        env = AntMazeInfoWrapper(env)
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
        base_env = gym.make(
            args.env_id,
            render_mode=args.eval_render_mode,
            reward_type=args.reward_type,
            maze_map=eval_maze_map
        )
        base_env = add_reward_wrapper(base_env, args)
        base_env = Monitor(base_env, filename=None, allow_early_resets=True)
        if args.record_video and args.eval_render_mode == "rgb_array":
            base_env = FixedMujocoOffscreenRender(base_env, None, width=480, height=480)
            base_env = RecordVideo(
                base_env,
                video_folder=str(args.video_dir),
                episode_trigger=lambda ep_id: ep_id == 0,
                name_prefix=f"eval_{args.experiment_name}",
            )
        base_env.reset(seed=args.seed)
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

    set_screen_config()
    args = tyro.cli(Args)
    args.finalize()
    register(id="AntMaze", entry_point="envs.maze.ant_maze:AntMazeEnv",
             max_episode_steps=args.max_episode_steps)

    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    for _ in range(args.repeat):
        train_and_evaluate(args)
        args.reset_seed()
        time.sleep(10)

#python -m train.ant.maze.ant_maze_train   --total_timesteps 1000000 --track --reward-mode brave