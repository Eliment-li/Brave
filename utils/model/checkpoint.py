import os
import time
from pathlib import Path

import torch

from configs.args import get_root_path


def _ensure_ckpt_dir(run_name: str) -> Path:
    ckpt_dir = Path(get_root_path()).expanduser() / "results" / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_checkpoint(run_name, iteration, global_step, agent, optimizer, args):
    ckpt_dir = _ensure_ckpt_dir(run_name)
    ckpt_path = ckpt_dir / f"checkpoint_iter{iteration}_step{global_step}.pt"
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
        str(ckpt_path),
    )
    print(f"Saved checkpoint: {ckpt_path}")


def _resolve_map_location(map_location):
    if map_location is not None:
        return map_location
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(ckpt_path, agent, optimizer, map_location=None):
    ckpt_file = Path(ckpt_path).expanduser()
    ckpt = torch.load(str(ckpt_file), map_location=_resolve_map_location(map_location), weights_only=False)
    agent.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["iteration"], ckpt["global_step"], ckpt.get("args")