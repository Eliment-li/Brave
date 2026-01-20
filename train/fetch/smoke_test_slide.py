"""Minimal smoke test for Fetch Slide env.

Goals:
- Create the Fetch Slide environment implemented in this repo.
- Optionally wrap it with the BRS reward wrapper.
- Run a short random rollout and print a few key fields.

This file is intentionally lightweight and independent from SB3 training,
so you can quickly verify the environment + wrapper stack works.

Usage examples (PowerShell):
    python -m train.fetch.smoke_test_slide
    python -m train.fetch.smoke_test_slide --steps 200 --reward_mode brave
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (needed for register_envs)
import tyro

from brs.fetch_task_brs import FetchTaskBRSRewardWrapper


@dataclass
class Args:
    seed: int = 0
    steps: int = 200
    max_episode_steps: int = 50

    # reward wrapper: standerd/brave
    reward_mode: str = "standerd"

    # BRS params (only used when reward_mode == "brave")
    task: str = "slide"
    use_global_max_bonus: bool = False
    global_bonus: float = 10.0

    # rendering: None | "human" | "rgb_array"
    render_mode: str | None = None


def _make_env(args: Args) -> gym.Env:
    from gymnasium.envs.registration import register, register_envs

    # Make sure gymnasium_robotics default envs are registered (not strictly required
    # for our custom id, but keeps behavior consistent in this repo).
    register_envs(gymnasium_robotics)

    # Register our local Slide implementation with a short id.
    # Note: if you change id, also change the gym.make call below.
    register(
        id="Slide",
        entry_point="envs.fetch.slide:MujocoFetchSlideEnv",
        max_episode_steps=args.max_episode_steps if args.max_episode_steps > 0 else None,
    )

    kwargs = {}
    if args.max_episode_steps and args.max_episode_steps > 0:
        kwargs["max_episode_steps"] = args.max_episode_steps
    if args.render_mode is not None:
        kwargs["render_mode"] = args.render_mode

    env = gym.make("Slide", **kwargs)

    if args.reward_mode == "brave":
        env = FetchTaskBRSRewardWrapper(
            env,
            task=args.task,
            use_global_max_bonus=args.use_global_max_bonus,
            global_bonus=args.global_bonus,
        )

    return env


def main() -> None:
    args = tyro.cli(Args)

    env = _make_env(args)

    # Basic API checks
    assert hasattr(env, "action_space"), "env has no action_space"
    assert env.action_space.shape == (4,), f"unexpected action shape: {env.action_space.shape}"

    obs, info = env.reset(seed=args.seed)
    assert isinstance(obs, dict), f"expected dict obs, got: {type(obs)}"
    for k in ("observation", "achieved_goal", "desired_goal"):
        assert k in obs, f"missing key in obs: {k}"

    ep_return = 0.0
    for t in range(int(args.steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)

        if t < 10 or (t + 1) % 50 == 0:
            # distance_to_goal exists when using FetchTaskBRSRewardWrapper
            dist = info.get("distance_to_goal", None)
            bonus = info.get("brs_bonus", None)
            is_success = info.get("is_success", None)
            print(
                f"t={t:04d} r={reward:+.4f} return={ep_return:+.4f} "
                f"dist={dist} bonus={bonus} success={is_success}"
            )

        if terminated or truncated:
            obs, info = env.reset()
            ep_return = 0.0

    env.close()


if __name__ == "__main__":
    main()
