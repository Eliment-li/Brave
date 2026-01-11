"""
Docs:https://docs.swanlab.cn/guide_cloud/integration/integration-sb3.html

For adaptation to the stable_baseline3 framework. Detailed usage are as follows:
------trian.py in stable_baseline3------
import swanlab
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from swanlab.integration.sb3 import SwanLabCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}

def make_env():
    env = gym.make(config["env_name"], render_mode="rgb_array")
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])
model = PPO(
    config["policy_type"],
    env,
    verbose=1,
)

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=SwanLabCallback(
        project="PPO",
        experiment_name="MlpPolicy",
        verbose=2,
    ),
)

swanlab.finish()
---------------------------------
"""
import numbers

import swanlab
from typing import Optional, Dict, Any, Union, Tuple, List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, Logger


class SwanLabOutputFormat(KVWriter):
    def __init__(self, swanlab_callback):
        self.swanlab_callback = swanlab_callback

    def write(
            self,
            key_values: Dict[str, Any],
            key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
            step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):
            # 如果是标量指标
            if isinstance(value, (int, float)):
                # 记录指标
                self.swanlab_callback.experiment.log({key: value}, step=step)


class SwanLabCallback(BaseCallback):
    def __init__(
            self,
            project: Optional[str] = None,
            workspace: Optional[str] = None,
            experiment_name: Optional[str] = None,
            description: Optional[str] = None,
            logdir: Optional[str] = None,
            mode: Optional[bool] = None,
            verbose: int = 0,
            tags: Optional[List[str]] = None,
            **kwargs: Any,
    ):
        super().__init__(verbose)
        self._run = None

        tags = tags or []
        #tags.append("🤖stable_baselines3") if "🤖stable_baselines3" not in tags else None

        self._swanlab_init: Dict[str, Any] = {
            "project": project,
            "workspace": workspace,
            "experiment_name": experiment_name,
            "description": description,
            "logdir": logdir,
            "mode": mode,
            "tags": tags,
        }

        self._swanlab_init.update(**kwargs)
        self._kwargs = kwargs  # 新增：保存原始 kwargs

        self._project = self._swanlab_init.get("project")
        self._workspace = self._swanlab_init.get("workspace")
        self._experiment_name = self._swanlab_init.get("experiment_name")
        self._description = self._swanlab_init.get("decsription")
        self._logdir = self._swanlab_init.get("logdir")
        self._mode = self._swanlab_init.get("mode")

    def _init_callback(self) -> None:
        kwargs_from_init = self._kwargs
        args = {"algo": type(self.model).__name__}
        for key in self.model.__dict__:
            if type(self.model.__dict__[key]) in [float, int, str]:
                args[key] = self.model.__dict__[key]
            else:
                args[key] = str(self.model.__dict__[key])

        #merge args and kwargs_from_init
        args.update(kwargs_from_init)
        self.setup(config=args)

        loggers = Logger(
            folder=None,
            output_formats=[SwanLabOutputFormat(self)],
        )

        self.model.set_logger(loggers)

    def update_config(self, config: Dict[str, Any]):
        swanlab.config.update(config)

    @property
    def experiment(self):
        if swanlab.get_run() is None:
            self.setup()
        return self._run

    def setup(self, config=None):
        swanlab.config["FRAMEWORK"] = "🤖stable_baselines3"
        if swanlab.get_run() is None:
            print('swanlab.init(**self._swanlab_init)')
            self._run = swanlab.init(**self._swanlab_init,  settings = swanlab.Settings(
                    backup = False
                ))
        else:
            self._run = swanlab.get_run()

        if config:
            print('swanlab update config')
            self._run.config.update(config)

    def get_lr(self):
        opt = getattr(self.model.policy, "optimizer", None)
        if opt is not None and getattr(opt, "param_groups", None):
            lr = opt.param_groups[0].get("lr", None)
            if lr is not None:
                return {"train/learning_rate", float(lr)}
            else:
                return None

    def _on_step(self) -> bool:
        # 从 rollout 的 locals 拿到 infos（VecEnv: List[Dict]）
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        extra_keys =[
            'rdcr',
            'rdcr_max',
            'brs_bonus',
            'ctrl_cost',
            'healthy_reward',
             'is_success',
            'achieved_goal',
            r'metric/global_max',
            r'metric/episode_max',
            r'metric/stand',
            r'metric/speed',
            r'metric/far',
            r'original/standerd_reward',
           # r'original/stander_episode_reward_mean',
            r'original/ep_rew_mean',
            #r'original/ep_len_mean',
            r'rollout/episodic_return',
            #r'charts/episodic_length',
            'cost','best_cost','delta','improved'
        ]

        if dones[0]:
            extra_keys.append('stander_episode_reward_mean')

        if infos:
            step = int(self.num_timesteps)
            episode_logs = []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if isinstance(episode, dict):
                    payload = {}
                    if "r" in episode:
                        payload["rollout/episodic_return"] = episode["r"]
                    if "l" in episode:
                        payload["charts/episodic_length"] = episode["l"]
                    if payload:
                        episode_logs.append(payload)
            for payload in episode_logs:
                self.experiment.log(payload, step=step)
            # 合并/过滤每个 env 的 info
            aggregated: Dict[str, float] = {}
            for info in infos:
                if not isinstance(info, dict):
                    continue
                for k in extra_keys:
                    v = info.get(k, None)
                    if isinstance(v, numbers.Number):
                        # 多环境简单做平均
                        aggregated[k] = aggregated.get(k, 0.0) + float(v)
            if aggregated:
                n_envs = len(infos)
                for k in aggregated:
                    aggregated[k] /= max(1, n_envs)
                #aggregated.update(self.get_lr() or {})
                # 上报到 SwanLab
                self.experiment.log(aggregated, step=step)
        return True
