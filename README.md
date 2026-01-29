# Brave

一个基于 **Gymnasium / MuJoCo / Stable-Baselines3** 的强化学习实验仓库，包含多种环境（Ant / Fetch / PointMaze / MountainCarContinuous / HumanoidStandup 等）以及多种奖励改造/探索奖励方案（如 BRS/Brave、RND、ExploRS、ReLara）。

> 说明：本仓库的训练脚本主要位于 `train/` 下，命令行参数使用 `tyro` 解析（所以可以直接 `--help` 查看全部参数）。

## 目录结构（简版）

- `train/`：训练与评估入口脚本（按环境划分）
  - `train/ant/basic/ant_train.py`：Ant 系列任务训练
  - `train/fetch/fetch_train.py`：Fetch Reach/Push/Slide/PickAndPlace 训练
  - `train/point/point_maze_train.py`：PointMaze 训练
  - `train/mountain_car/mountain_car_train.py`：MountainCarContinuous 训练
  - 多个 `smoke_test*.py`：小规模冒烟测试
- `brs/`：Brave/BRS 奖励包装器（reward wrapper）
- `algos/`：探索奖励/算法相关封装（如 RND、ExploRS、ReLara）
- `envs/`：自定义环境与地图（maze、ant tasks 等）
- `info_wrapper/`：用于记录/统计 info 的 wrapper
- `utils/`：通用工具（日志、绘图、保存、屏幕渲染等）
- `results/`：训练产物（checkpoints、videos、fig 等）

## 环境要求

- Python 3.10+（建议）
- Windows / Linux 均可（你当前环境是 Windows PowerShell）
- 需要 MuJoCo 的任务：Ant / Fetch / Humanoid / Maze 等（取决于你跑的环境）

依赖在 `requirements.txt` 中。核心依赖包括：
- `gymnasium` / `gymnasium-robotics`
- `mujoco`
- `stable-baselines3[extra]`
- `tyro`

## 安装

建议使用虚拟环境：

```powershell
cd D:\project\Brave
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果你希望以包的形式导入（支持 `python -m train.xxx`），可选执行：

```powershell
pip install -e .
```

## 快速开始（推荐用 `-m` 模式运行）

### 1) Fetch（Reach/Push/Slide/Pick）

训练（默认 TD3）：

```powershell
python -m train.fetch.fetch_train --task reach --reward_mode standerd --total_timesteps 10000
python -m train.fetch.fetch_train --task push  --reward_mode standerd --total_timesteps 10000
```

常用参数：
- `--task`：`reach | push | slide | pick`
- `--reward_mode`：脚本里定义的奖励模式（例如 `standerd`/`brave`/`rnd`/`explors`，以代码为准）
- `--max_episode_steps`：覆盖 episode 长度（`<=0` 表示不传给 env，使用默认 spec）

### 2) Ant（stand / far / speed）

```powershell
python -m train.ant.basic.ant_train --task stand --reward_mode standerd --total_timesteps 1000000
```

常用参数：
- `--task`：`stand | far | speed`
- `--reward_mode`：`standerd | brave | rnd | ...`
- `--r_wrapper_ver`：使用 Brave wrapper 的版本号（当 `reward_mode=brave` 时）

### 3) PointMaze

```powershell
python -m train.point.point_maze_train --reward_mode standard --reward_type dense --total_timesteps 2000
```

提示：PointMaze 支持地图参数（`--maze_map_name` / `--eval_maze_map_name`）。

### 4) MountainCarContinuous

```powershell
python -m train.mountain_car.mountain_car_train --reward_mode brave --total_timesteps 1000
```

## 冒烟测试（smoke tests）

仓库内有一些小脚本用于快速验证环境和 wrapper 是否能跑通，例如：

```powershell
python -m train.fetch.smoke_test
python -m train.point.smoke_test
python -m train.ant.basic.smoke_test
```

如果你在 Fetch + ReLara 方向工作，可以看：`train/fetch/relara/README.md`。

## 日志与结果输出

通常会输出到：
- 模型：`results/checkpoints/...`
- 视频：`results/videos/...`

部分脚本支持 `swanlab` 记录实验（见参数 `--track`、`--swanlab_project` 等）。

## 常见问题

1) **MuJoCo / gymnasium-robotics 安装报错**
- 先确保你的 Python 版本与依赖匹配。
- Fetch/Humanoid 等任务依赖 MuJoCo 后端；如果报 DLL/驱动错误，通常与显卡驱动或 MuJoCo 安装有关。

2) **不确定有哪些参数**
- 直接对脚本加 `--help` 查看：

```powershell
python -m train.fetch.fetch_train --help
```


