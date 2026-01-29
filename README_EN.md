# Brave

A reinforcement learning experiment repo built on **Gymnasium / MuJoCo / Stable-Baselines3**. It includes multiple environments (Ant / Fetch / PointMaze / MountainCarContinuous / HumanoidStandup, etc.) and several reward shaping / exploration approaches (e.g., BRS/Brave, RND, ExploRS, ReLara).

> Note: training entrypoints mainly live under `train/`. CLI arguments are parsed with `tyro`, so you can always run `--help` to see all available options.

## Project layout (quick view)

- `train/`: training & evaluation entrypoints (grouped by environment)
  - `train/ant/basic/ant_train.py`: Ant task training
  - `train/fetch/fetch_train.py`: Fetch Reach/Push/Slide/PickAndPlace training
  - `train/point/point_maze_train.py`: PointMaze training
  - `train/mountain_car/mountain_car_train.py`: MountainCarContinuous training
  - various `smoke_test*.py`: small smoke tests
- `brs/`: Brave/BRS reward wrappers
- `algos/`: algorithm / exploration modules (RND, ExploRS, ReLara, ...)
- `envs/`: custom envs and maps (maze, ant tasks, ...)
- `info_wrapper/`: wrappers for logging/stats in `info`
- `utils/`: utilities (logging, plotting, checkpointing, rendering, ...)
- `results/`: outputs (checkpoints, videos, figures, ...)

## Requirements

- Python 3.10+ (recommended)
- Windows / Linux
- Tasks that require MuJoCo: Ant / Fetch / Humanoid / Maze, etc. (depends on what you run)

Dependencies are listed in `requirements.txt`. Core ones include:
- `gymnasium` / `gymnasium-robotics`
- `mujoco`
- `stable-baselines3[extra]`
- `tyro`

## Installation

It’s recommended to use a virtual environment:

```powershell
cd D:\project\Brave
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

If you want to import the repo as a package (so `python -m train.xxx` works everywhere), optionally run:

```powershell
pip install -e .
```

## Quick start (recommended: run with `-m`)

### 1) Fetch (Reach/Push/Slide/Pick)

Training (default algorithm: TD3):

```powershell
python -m train.fetch.fetch_train --task reach --reward_mode standerd --total_timesteps 10000
python -m train.fetch.fetch_train --task push  --reward_mode standerd --total_timesteps 10000
```

Common args:
- `--task`: `reach | push | slide | pick`
- `--reward_mode`: reward mode defined by the script (e.g., `standerd`/`brave`/`rnd`/`explors`; see code for the exact options)
- `--max_episode_steps`: override episode length (`<=0` means don’t pass it to `gym.make`, use the env’s default spec)

### 2) Ant (stand / far / speed)

```powershell
python -m train.ant.basic.ant_train --task stand --reward_mode standerd --total_timesteps 1000000
```

Common args:
- `--task`: `stand | far | speed`
- `--reward_mode`: `standerd | brave | rnd | ...`
- `--r_wrapper_ver`: Brave wrapper version (used when `reward_mode=brave`)

### 3) PointMaze

```powershell
python -m train.point.point_maze_train --reward_mode standard --reward_type dense --total_timesteps 2000
```

Tip: PointMaze supports map selection via `--maze_map_name` / `--eval_maze_map_name`.

### 4) MountainCarContinuous

```powershell
python -m train.mountain_car.mountain_car_train --reward_mode brave --total_timesteps 1000
```

## Smoke tests

There are small scripts to quickly validate that envs/wrappers run end-to-end, for example:

```powershell
python -m train.fetch.smoke_test
python -m train.point.smoke_test
python -m train.ant.basic.smoke_test
```

If you’re working on Fetch + ReLara, also see: `train/fetch/relara/README.md`.

## Logging & outputs

Typical output locations:
- Models: `results/checkpoints/...`
- Videos: `results/videos/...`

Some scripts support experiment tracking via `swanlab` (see `--track`, `--swanlab_project`, etc.).

## FAQ

1) **MuJoCo / gymnasium-robotics installation errors**
- Make sure your Python version matches the dependencies.
- Fetch/Humanoid tasks rely on a MuJoCo backend. If you see DLL/driver errors, it’s often related to GPU drivers or MuJoCo setup.

2) **Not sure what arguments are available**
- Add `--help` to the entrypoint:

```powershell
python -m train.fetch.fetch_train --help
```


