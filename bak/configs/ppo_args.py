from dataclasses import dataclass
from bak.configs.base_args import BaseArgs
from bak.configs.private.private_args import PrivateArgs


@dataclass
class PpoAtariArgs(BaseArgs):
    exp_name: str = "ppo_atari"
    env_id: str = "BreakoutNoFrameskip-v4"
    swanlab_key = PrivateArgs.swanlab_key
    swanlab_project= 'Brave'
    swanlab_workspace = 'Eliment-li'
    swanlab_group = 'PPOAtari'
    track: bool = True
    enable_brave: bool = False
    total_timesteps: int = 10*1000*1000

    def finalize(self):
        super().finalize()
        if self.enable_brave:
            self.experiment_name += '_brave'
        #get git version number
        return self