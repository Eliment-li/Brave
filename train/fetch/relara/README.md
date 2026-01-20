# ReLara on Fetch (Reach/Push)

This folder provides a **thin Fetch adapter** so you can reuse the existing ReLara implementation (`train/ant/basic/relara/*`) on Fetch tasks.

## What’s included

- `fetch_train_relara.py`: training entrypoint for Fetch + ReLara
- `smoke_test_relara.py`: a tiny run (requires MuJoCo)
- Shared env maker: `train/common/relara/fetch_env_maker.py`

## Key idea

ReLara only assumes:
- `env.observation_space` is a flat `Box` and observations are 1D float32 arrays
- `env.action_space` is a `Box`
- `info["episode"]` exists on episode end (provided by `RecordEpisodeStatistics`)

Fetch envs output `Dict` observations (`observation/achieved_goal/desired_goal`), so we flatten them in the env maker.

## Quick start

### Option A: Run smoke test without MuJoCo (wrapper only)

This validates the flattening/env maker behavior using a fake Fetch-like env:

```powershell
python D:\project\Brave\train\common\relara\smoke_test_fetch_env_maker.py
```

### Option B: Train on real Fetch envs (requires MuJoCo)

```powershell
python -m train.fetch.relara.fetch_train_relara --task reach --reward_type sparse --max_episode_steps 50
python -m train.fetch.relara.fetch_train_relara --task push  --reward_type sparse --max_episode_steps 50
```

If you prefer running as a script:

```powershell
python D:\project\Brave\train\fetch\relara\fetch_train_relara.py --task reach
```

## Notes

- If you see MuJoCo errors, install/configure MuJoCo according to your `gymnasium_robotics` backend (mujoco / mujoco-py).
- For sparse rewards, you can optionally shift `{0,-1} -> {1,0}` with `--transform_sparse_reward true`.
