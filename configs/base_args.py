import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import arrow
import torch

from configs.private.private_args import PrivateArgs
from utils.str_util import get_animals_name


def get_root_path():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = root_dir[:-8]
    return root_dir

K=1000
M=1000*K
@dataclass
class BaseArgs:
    seed: int = -1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    env_id: str = "UnknownEnv"
    learning_rate: float = 1e-4
    num_envs: int = 10
    num_steps: int = 1024
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    batch_size: int = field(init=False, default=10240)
    minibatch_size: int = field(init=False, default=128)
    num_iterations: int = field(init=False, default=0)

    root_path:Path = Path(get_root_path())

    def finalize(self):
        #set seed to random value if seed is -1
        if self.seed == -1:
            self.seed = torch.randint(0, 10000, (1,)).item()
        print(f"Using seed: {self.seed}")
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size
        self.experiment_name = get_animals_name() +'_'+ arrow.now().format('MMDD_HHmm')

        return self


if __name__ == '__main__':
    print(get_root_path())
    print(torch.version.cuda)
    print(torch.randint(0, 10000, (1,)).item())