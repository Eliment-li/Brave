from dataclasses import field, dataclass
from pathlib import Path

import arrow

from bak.configs.base_args import get_root_path, BaseArgs

M = 1000*1000
K = 1000
@dataclass
class DqnArgs(BaseArgs):
    exp_name: str = "dqn_cleanrl"
    env_id: str = "MountainCar-v0"
    num_envs: int = 1
    track: bool = True
    enable_brave: bool = True
    total_timesteps: int = 2*M
    buffer_size: int = 10 * K
    gamma: float = 0.99
    target_network_frequency: int = 500
    max_grad_norm: float = 0.5
    dqn_batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.8
    learning_starts: int = 10000
    train_frequency: int = 1
    swanlab_project: str = "Brave"
    swanlab_workspace: str = "Eliment-li"
    swanlab_group: str = "dqn_mountain_car"
    checkpoint_dir: Path = field(default_factory=lambda: Path(get_root_path()) / "results" / "checkpoints")

    def finalize(self):
        super().finalize()
        self.batch_size = self.dqn_batch_size
        self.minibatch_size = self.batch_size
        self.num_iterations = self.total_timesteps // max(1, self.batch_size)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = self.env_id+'_' + arrow.now().format('MMDD_HHmm')
        if self.enable_brave:
            self.experiment_name += '_brave'
        return self