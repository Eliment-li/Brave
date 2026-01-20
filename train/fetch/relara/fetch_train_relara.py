import time
from dataclasses import dataclass, field
from pathlib import Path

import arrow
from gymnasium.envs.registration import register
import gymnasium_robotics
import numpy as np
import swanlab
import torch
import tyro
from gymnasium.envs.registration import register_envs
from stable_baselines3.common.monitor import Monitor

from bak.configs.base_args import get_root_path
from algos.relara.relara_algo import ReLaraAlgo, ReLaraConfig
from algos.relara.relara_networks import BasicActor, ActorResidual, BasicQNetwork, QNetworkResidual
from train.common.relara.fetch_env_maker import make_fetch_relara_env


@dataclass
class Args:
    task: str = "reach"  # reach | push
    env_id: str = ""

    total_timesteps: int = int(10000)
    seed: int = -1
    track: bool = False

    # env
    reward_type: str = "sparse"  # sparse | dense
    transform_sparse_reward: bool = False
    max_episode_steps: int = 50

    # ReLara hyperparams
    beta: float = 0.2
    gamma: float = 0.99
    proposed_reward_scale: float = 1.0
    pa_learning_starts: int = 10_000
    ra_learning_starts: int = 5_000

    # output
    root_path: str = get_root_path()
    save_dir: str = get_root_path() + "/results/relara_checkpoints/Fetch"
    tags: list[str] = field(default_factory=list)

    swanlab_project: str = "Brave_Fetch_ReLara"
    swanlab_workspace: str = "Eliment-li"

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if self.seed == -1:
            self.reset_seed()

        if self.task == 'reach':
            self.env_id = 'Reach'
        elif self.task == 'push':
            self.env_id = 'Push'
        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + "_" + arrow.now().format("MMDD_HHmm")
        self.experiment_name += ('_' + 'relara')
        self.swanlab_group = 'relara'
        self.swanlab_project = self.swanlab_project + '_' + self.task
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


def main(args: Args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.track:
        swanlab.init(project=args.swanlab_project,
                     swanlab_workspace="Eliment-li",
                     name=args.experiment_name,
                     config=vars(args))
    env = make_fetch_relara_env(
        args.env_id,
        seed=args.seed,
        reward_type=args.reward_type,
        transform_sparse_reward=args.transform_sparse_reward,
        max_episode_steps=args.max_episode_steps,
    )

    # Keep the same monitoring stack as other trainers
    env = Monitor(env, filename=None, allow_early_resets=True)
    env.reset(seed=args.seed)

    cfg = ReLaraConfig(
        exp_name=args.experiment_name,
        seed=args.seed,
        gamma=args.gamma,
        proposed_reward_scale=args.proposed_reward_scale,
        beta=args.beta,
        save_folder=str(Path(args.save_dir) / args.experiment_name),
        save_frequency=10_000,
        track=args.track,
    )

    agent = ReLaraAlgo(
        env=env,
        pa_actor_class=BasicActor,
        pa_critic_class=BasicQNetwork,
        ra_actor_class=ActorResidual,
        ra_critic_class=QNetworkResidual,
        cfg=cfg,
        track=args.track
    )

    agent.learn(
        total_timesteps=args.total_timesteps,
        pa_learning_starts=args.pa_learning_starts,
        ra_learning_starts=args.ra_learning_starts,
    )
    agent.save(indicator="final")
    time.sleep(1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.finalize()
    # Ensure Gymnasium robotics envs (Fetch*) are registered
    # register_envs(gymnasium_robotics)

    register(id="Reach", entry_point="envs.fetch.reach:MujocoFetchReachEnv", max_episode_steps=args.max_episode_steps)
    register(id="Push", entry_point="envs.fetch.push:MujocoFetchPushEnv", max_episode_steps=args.max_episode_steps)


    main(args)
