import os
import time
from dataclasses import dataclass, field
from pathlib import Path
import arrow
import numpy as np
import swanlab
import torch
import tyro
from stable_baselines3.common.monitor import Monitor

from application.ant.basic.wrappers.ant_info_wrapper import AntTaskInfoWrapper

from application.ant.basic.relara.relara_env_maker import make_ant_relara_env
from application.ant.basic.relara.relara_algo import ReLaraAlgo, ReLaraConfig
from application.ant.basic.relara.relara_networks import BasicActor, BasicQNetwork, ActorResidual, QNetworkResidual
from configs.base_args import get_root_path
import  application.ant.basic.ant_tasks

@dataclass
class Args:
    env_id: str = ""#AntSpeed-v0, AntFar-v0, AntStand-v0
    total_timesteps: int = int(100_0000)
    seed: int = -1
    track: bool = False
    task: str = ""  # speed, far, stand
    # task config
    reward_type: str = "dense"
    speed_target: float = 4.0
    height_target: float = 0.9
    dist_target: float = 5.0
    terminate_when_unhealthy: bool = True
    ctrl_cost_weight: float = 0.5
    early_break: bool = True
    healthy_reward: float = 1.0

    # ReLara hyperparams
    beta: float = 0.2
    gamma: float = 0.99
    proposed_reward_scale: float = 1.0
    pa_learning_starts: int = 10000
    ra_learning_starts: int = 5000

    # if reward_type == "sparse", you may want to shift {-1,0}->{0,1}
    transform_sparse_reward: bool = False

    # output
    root_path: str = get_root_path()
    save_dir: str = get_root_path() + "/results/relara_checkpoints/Ant"
    tags: list[str] = field(default_factory=list)

    swanlab_project: str = "Brave_Antv4"  # final project name = swanlab_project+task
    repeat:int = 1

    num_threads:int = -1

    def reset_seed(self):
        self.seed = torch.randint(0, 10000, (1,)).item()

    def finalize(self):
        if args.num_threads > 0:
            # Set number of threads for torch, to control CPU usage
            torch.set_num_threads(16)
            torch.set_num_interop_threads(2)
        task_map = {
            'stand': "AntStand-v0",
            'far': 'AntFar-v0',
            'speed': "AntSpeed-v0"

        }
        self.env_id = task_map.get(self.task)
        self.swanlab_project = self.swanlab_project + '_' + self.task
        if self.seed == -1:
            self.reset_seed()
        self.experiment_name = self.env_id.replace("/", "_") + "_" + arrow.now().format("MMDD_HHmm") + "_relara"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


def main(args: Args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make_ant_relara_env(
        args.env_id,
        seed=args.seed,
        reward_type=args.reward_type,
        target_speed=args.speed_target ,
        target_height=args.height_target ,
        target_dist=args.dist_target ,
        terminate_when_unhealthy=args.terminate_when_unhealthy,
        ctrl_cost_weight = args.ctrl_cost_weight,
        early_break=args.early_break,
        healthy_reward=args.healthy_reward,
        transform_sparse_reward=args.transform_sparse_reward,
    )
    env = AntTaskInfoWrapper(env)
    env = Monitor(env, filename=None, allow_early_resets=True)
    cfg = ReLaraConfig(
        exp_name=args.experiment_name,
        seed=args.seed,
        gamma=args.gamma,
        proposed_reward_scale=args.proposed_reward_scale,
        beta=args.beta,
        save_folder=str(Path(args.save_dir) / args.experiment_name),
        save_frequency=10000,
        track=args.track,
    )

    if args.track:
        swanlab.init(project=args.swanlab_project,
                     swanlab_workspace="Eliment-li",
                     name=args.experiment_name,
                     config=vars(args))

    agent = ReLaraAlgo(
        env=env,
        pa_actor_class=BasicActor,
        pa_critic_class=BasicQNetwork,
        ra_actor_class=ActorResidual,
        ra_critic_class=QNetworkResidual,
        cfg=cfg,
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
    print("torch num_threads:", torch.get_num_threads())
    print("torch interop:", torch.get_num_interop_threads())
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    for i in range(args.repeat):
        args.reset_seed()
        main(args)
        time.sleep(60)