from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class EvalCameraConfig:
    eval_fixed_camera: bool = True
    eval_camera_name: str = "none"
    eval_cam_auto_lookat: bool = True
    eval_cam_lookat_x: float = 0.0
    eval_cam_lookat_y: float = 0.0
    eval_cam_lookat_z: float = 0.0
    eval_cam_distance: float = 30.0
    eval_cam_elevation: float = -80.0
    eval_cam_azimuth: float = 90.0


class FixedMujocoOffscreenRender(gym.Wrapper):
    """
    用 MuJoCo 官方 mujoco.Renderer + 自定义 MjvCamera 接管 rgb_array 渲染。
    这样渲染用的是“free camera”（MjvCamera），不会再跟随 ant。
    """

    def __init__(self, env, camera_config: EvalCameraConfig=None, width: int = 640, height: int = 480):
        super().__init__(env)
        if camera_config is None:
            camera_config = EvalCameraConfig()
        self.camera_config = camera_config

        u = self.env.unwrapped
        self.model = getattr(u, "model", None)
        self.data = getattr(u, "data", None)
        if self.model is None or self.data is None:
            raise RuntimeError("env.unwrapped.model/data 不存在，无法使用 mujoco.Renderer 接管渲染")

        # clamp 到 XML offscreen framebuffer 上限
        max_w = max_h = None
        try:
            max_w = int(self.model.vis.global_.offwidth)
            max_h = int(self.model.vis.global_.offheight)
        except Exception:
            pass

        self.width = int(width)
        self.height = int(height)
        if max_w is not None and self.width > max_w:
            print(f"[FixedMujocoOffscreenRender] clamp width {self.width} -> {max_w} (model offwidth)")
            self.width = max_w
        if max_h is not None and self.height > max_h:
            print(f"[FixedMujocoOffscreenRender] clamp height {self.height} -> {max_h} (model offheight)")
            self.height = max_h

        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        # 关键：自定义 free camera（MjvCamera）
        self.cam = mujoco.MjvCamera()
        self._sync_camera_from_config()

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass
        return self.env.close()

    def _get_lookat(self) -> np.ndarray:
        lookat = np.array(
            [self.camera_config.eval_cam_lookat_x, self.camera_config.eval_cam_lookat_y,
             self.camera_config.eval_cam_lookat_z],
            dtype=np.float64,
        )
        if self.camera_config.eval_cam_auto_lookat and hasattr(self.model, "geom_pos"):
            gp = np.asarray(self.model.geom_pos)
            if gp.ndim == 2 and gp.shape[1] >= 3 and gp.shape[0] > 0:
                lookat = gp[:, :3].mean(axis=0)
        return lookat

    def _sync_camera_from_config(self):
        if not self.camera_config.eval_fixed_camera:
            return

        lookat = self._get_lookat()

        # MjvCamera 的参数就是你想要的：lookat/distance/azimuth/elevation
        # 注意：这里 elevation/azimuth 单位就是“度”
        self.cam.lookat[:] = lookat
        self.cam.distance = float(self.camera_config.eval_cam_distance)
        self.cam.azimuth = float(self.camera_config.eval_cam_azimuth)
        self.cam.elevation = float(self.camera_config.eval_cam_elevation)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._sync_camera_from_config()
        return obs

    def step(self, action):
        return self.env.step(action)

    def render(self):
        if getattr(self.env, "render_mode", None) != "rgb_array":
            return self.env.render()

        # 每次都同步一次，防止你后面动态调参
        self._sync_camera_from_config()

        # 关键：update_scene 使用我们自定义的 free camera，而不是 camera=0
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()