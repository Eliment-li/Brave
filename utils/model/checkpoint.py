import os
import time
from pathlib import Path

import torch

from configs.args import get_root_path


def save_checkpoint(run_name, iteration, global_step, agent, optimizer, args):
    ckpt_dir = Path(get_root_path())/"results"/"checkpoints"/run_name
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_iter{iteration}_step{global_step}.pt")
    try:
        args_data = vars(args)
    except Exception:
        args_data = getattr(args, "__dict__", str(args))
    torch.save(
        {
            "iteration": iteration,
            "global_step": global_step,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args_data,
            "time": time.time(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")

def load_checkpoint(ckpt_path, agent, optimizer, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location,weights_only=False)
    agent.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"], ckpt["global_step"], ckpt.get("args")