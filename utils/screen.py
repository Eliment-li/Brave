#
# if not is_windows():
#     os.environ.setdefault("MUJOCO_GL", "egl")
#     os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import os

from swanlab.env import is_windows


def set_screen_config():
    # --- replace the current env var block with this ---
    if not is_windows():
        # Desktop Ubuntu + monitor: prefer GLFW/GLX path to keep offscreen rgb_array working
        os.environ.setdefault("MUJOCO_GL", "glfw")
        # Only set PYOPENGL_PLATFORM when using egl/osmesa, not glfw
        if os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
            os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)

    # Do not force dummy; it can break window/context init on desktop machines
    os.environ.pop("SDL_VIDEODRIVER", None)