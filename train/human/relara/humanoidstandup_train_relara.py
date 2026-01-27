import time
from dataclasses import dataclass, field
from pathlib import Path

import arrow
import numpy as np
import swanlab
import torch
import tyro
from gymnasium.envs.registration import register
from stable_baselines3.common.monitor import Monitor

from bak.configs.base_args import get_root_path
from algos.relara.relara_algo import ReLaraAlgo, ReLaraConfig
from algos.relara.relara_networks import (
    BasicActor,
    ActorResidual,
    BasicQNetwork,
    QNetworkResidual,
)
from train.common.relara.humanoid_env_maker import make_humanoid_relara_env


@dataclass
class Args:
    env_id: str = "HumanoidStandup-v5-local"

    total_timesteps: int = int(1000000)
    seed: int = -1
    track: bool = False
    repeat: int = 1

    # env
    reward_type: str = "sparse"  # dense | sparse
    sparse_height_th: float = 0.4
    transform_sparse_reward: bool = False
    max_episode_steps: int = 200

    # ReLara hyperparams
    beta: float = 0.2
    gamma: float = 0.99
    proposed_reward_scale: float = 1.0
    pa_learning_starts: int = 10_000
    ra_learning_starts: int = 5_000

    # output
    root_path: str = get_root_path()
    save_dir: str = get_root_path() + "/results/relara_checkpoints/HumanoidStandup"
    tags: list[str] = field(default_factory=list)

    swanlab_project: str = "Brave_HumanoidStandup"
    swanlab_workspace: str = "Eliment-li"

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if self.seed == -1:
            self.reset_seed()

        safe_env = self.env_id.replace("/", "_")
        self.experiment_name = safe_env + "_" + arrow.now().format("MMDD_HHmm")
        self.experiment_name += "_relara"
        self.swanlab_group = "relara"

        if self.tags:
            parsed = []
            for t in self.tags:
                parsed.extend([x.strip() for x in t.split(",") if x.strip()])
            self.tags = parsed

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


def main(args: Args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.track:
        swanlab.init(
            project=args.swanlab_project,
            swanlab_workspace=args.swanlab_workspace,
            name=args.experiment_name,
            config=vars(args),
        )

    env = make_humanoid_relara_env(
        args.env_id,
        seed=args.seed,
        reward_type=args.reward_type,
        height_th=args.sparse_height_th,
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
        track=args.track,
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

    # Register local env to ensure we use repo's implementation
    register(
        id="HumanoidStandup-v5-local",
        entry_point="envs.human.humanoidstandup_v5:HumanoidStandupEnv",
        max_episode_steps=args.max_episode_steps if args.max_episode_steps and args.max_episode_steps > 0 else 1000,
    )
    args.env_id = "HumanoidStandup-v5-local"

    for _ in range(args.repeat):
        args.reset_seed()
        time.sleep(1)
        main(args)
