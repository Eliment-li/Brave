from __future__ import annotations

from logging import raiseExceptions
from typing import Optional

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.registration import register

from utils.calc_util import SlideWindow

DEFAULT_CAMERA_CONFIG = {"distance": 4.0}


class AntTaskEnv(MujocoEnv, utils.EzPickle):
    """
    A compact, gymnasium-compatible Ant with 3 tasks:
      - "stand": reach torso height >= target_height
      - "speed": reach x_speed >= target_speed
      - "far":   reach xy_distance_from_origin >= target_dist

    reward_type controls dense vs sparse:
      - reward_type="sparse": reward in {-1, 0} (0 if success else -1)
      - reward_type="dense":  shaped reward based on task metric, plus healthy_reward, minus ctrl_cost
    Observation is goal-conditioned Dict:
      {"observation", "achieved_goal", "desired_goal"}
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    def __init__(
        self,
        xml_file: str = "ant.xml",
        task: str = None,  # "stand" | "speed" | "far"
        reward_type: str = None,  # "sparse" | "dense"  <-- single switch
        threshold: float = 0.1,
        random_goal: bool = False,
        target_height: float = None,#0.9
        target_speed: float = None,
        target_dist: float = None,#4.0
        # costs/health (kept close to Gymnasium Ant defaults)
        ctrl_cost_weight: float = None, #0.5,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        early_break:bool = True,
        **kwargs,
    ):
        assert task in ("stand", "speed", "far")
        assert reward_type in ("dense", "sparse")

        utils.EzPickle.__init__(
            self,
            xml_file,
            task,
            reward_type,
            threshold,
            random_goal,
            target_height,
            target_speed,
            target_dist,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self.task = task
        self.reward_type = reward_type
        self.threshold = float(threshold)
        self.random_goal = bool(random_goal)

        self.target_height = float(target_height)
        self.target_speed = float(target_speed)
        self.target_dist = float(target_dist)

        self._ctrl_cost_weight = float(ctrl_cost_weight)
        self._healthy_reward = float(healthy_reward)
        self._terminate_when_unhealthy = bool(terminate_when_unhealthy)
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = float(reset_noise_scale)
        self._exclude_xy = bool(exclude_current_positions_from_observation)
        self._early_break = bool(early_break)

        # Ant: nq=15, nv=14. Excluding xy in qpos -> 13 + 14 = 27. Otherwise 29.
        obs_dim = 27 if self._exclude_xy else 29
        base_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)

        # frame_skip=5 matches gymnasium Ant-v4
        MujocoEnv.__init__(self, xml_file, frame_skip=5, observation_space=base_space, **kwargs)

        # Wrap to goal-conditioned Dict
        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype=np.float64),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype=np.float64),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=obs["desired_goal"].shape, dtype=np.float64),
            }
        )
        print(f"AntTaskEnv initialized with task={self.task}, reward_type={self.reward_type}")

        self.metric_sw = SlideWindow(size=100)
        self.state_sw = SlideWindow(size=10)

    # ----------------- helpers -----------------
    @property
    def is_healthy(self) -> bool:
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        return np.isfinite(state).all() and (min_z <= state[2] <= max_z)

    @property
    def terminated(self) -> bool:
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    @property
    def healthy_reward(self) -> float:
        # If terminate_when_unhealthy=False, still give healthy_reward always
        return float(self.is_healthy or (not self._terminate_when_unhealthy)) * self._healthy_reward

    def control_cost(self, action: np.ndarray) -> float:
        return self._ctrl_cost_weight * float(np.sum(np.square(action)))

    def _sample_goal(self):
        if not self.random_goal:
            return
        if self.task == "stand":
            self.target_height = float(self.np_random.uniform(0.6, 1.2))
        elif self.task == "speed":
            self.target_speed = float(self.np_random.uniform(0.5, 6.0))
        else:  # far
            self.target_dist = float(self.np_random.uniform(1.0, 8.0))

    def _desired_goal(self) -> np.ndarray:
        match self.task:
            case "stand":
                return np.array([self.target_height], dtype=np.float64)
            case "speed":
                return np.array([self.target_speed], dtype=np.float64)
            case "far":
                return np.array([self.target_dist], dtype=np.float64)
            case _:
                raise ValueError(f"Unknown task: {self.task}")


    def _achieved_goal(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        match self.task:
            case "stand":
                return np.array([qpos[2]], dtype=np.float64)  # torso z
            case "speed":
                return np.array([qvel[0]], dtype=np.float64)  # x velocity
            case "far":
                return np.array([np.linalg.norm(qpos[:2])], dtype=np.float64)  # xy distance
            case _:
                raise ValueError(f"Unknown task: {self.task}")

    def _is_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        # reach-at-least task: achieved >= desired - threshold
        return bool(achieved >= desired - self.threshold)

    def _task_metric(self, achieved: np.ndarray, desired: np.ndarray) -> float:
        """
        Dense shaping per task:
          - speed:  -abs(achieved - desired)  (peak at target speed, penalize偏差)
          - stand/far: achieved - desired    (超过目标继续加分)
        """
        diff = float(achieved[0] - desired[0])
        if self.task == "speed":
            return -abs(diff)
        return diff

    # ----------------- gym API -----------------
    def reset(self,*,seed: Optional[int] = None,options: Optional[dict] = None,):
        self.init_metric = None
        self.prev_metric = None
        self.state_sw.reset()
        return  super().reset(seed=seed)

    def _get_obs(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        if self._exclude_xy:
            obs_vec = np.concatenate([qpos[2:], qvel], dtype=np.float64)
        else:
            obs_vec = np.concatenate([qpos, qvel], dtype=np.float64)

        achieved = self._achieved_goal(qpos, qvel)
        desired = self._desired_goal()
        return {"observation": obs_vec, "achieved_goal": achieved, "desired_goal": desired}

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        terminated = self.terminated
        truncated = False  # handled by TimeLimit wrapper / max_episode_steps in registration

        achieved, desired = obs["achieved_goal"], obs["desired_goal"]
        self.state_sw.next(achieved[0])

        metric_now = self._task_metric(achieved, desired)
        #self.metric_sw.next(metric_now)

        success = self._is_success(achieved[0], desired[0])

        ctrl_cost = self.control_cost(action)

        #compute reward
        match self.reward_type:
            case "sparse":
                reward = 0.0 if success else -1.0
            case "dense":
                reward = float(self._task_metric(achieved, desired) + self.healthy_reward - ctrl_cost)
            case _:
                raiseExceptions(f"Unknown reward_type: {self.reward_type}")

        info = {
            "is_success": success,
            "ctrl_cost": ctrl_cost,
            "healthy_reward": self.healthy_reward,
            "achieved_goal": achieved.copy(),
            "desired_goal": desired.copy(),
        }
        if self._early_break:
            if success:
                truncated = True
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self._sample_goal()
        noise_low, noise_high = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(noise_low, noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        for k, v in DEFAULT_CAMERA_CONFIG.items():
            setattr(self.viewer.cam, k, v)


# -------- convenience wrappers --------
class AntStand(AntTaskEnv):
    def __init__(self, **kwargs):
        super().__init__(task="stand", **kwargs)


class AntSpeed(AntTaskEnv):
    def __init__(self, **kwargs):
        super().__init__(task="speed", **kwargs)


class AntFar(AntTaskEnv):
    def __init__(self, **kwargs):
        super().__init__(task="far", **kwargs)


# -------- registration (import this module once) --------
register(id="AntStand-v0", entry_point="application.ant.ant_tasks:AntStand", max_episode_steps=200)
register(id="AntSpeed-v0", entry_point="application.ant.ant_tasks:AntSpeed", max_episode_steps=200)
register(id="AntFar-v0",   entry_point="application.ant.ant_tasks:AntFar",   max_episode_steps=200)