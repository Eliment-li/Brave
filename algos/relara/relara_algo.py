"""
ReLara algorithm adapted for Brave's AntTaskEnv.

This is based on:
https://github.com/mahaozhe/ReLara/blob/main/ReLara/Algorithms.py

Main differences:
- We assume the env passed in already has a flat Box observation_space (see ant_env_maker.py).
- Keeps the original SAC-style PA/RA training logic intact.
"""

from __future__ import annotations

import datetime
import os
import random
import time
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import swanlab
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ReLaraConfig:
    exp_name: str = "ReLara-Ant"
    seed: int = 1
    cuda: int = 0

    gamma: float = 0.99
    proposed_reward_scale: float = 1.0
    beta: float = 0.2  # weight of proposed reward in PA's target

    pa_buffer_size: int = int(1e6)
    ra_buffer_size: int = int(1e6)
    pa_batch_size: int = 256
    ra_batch_size: int = 256

    pa_actor_lr: float = 3e-4
    pa_critic_lr: float = 1e-3
    pa_alpha_lr: float = 1e-4

    ra_actor_lr: float = 3e-4
    ra_critic_lr: float = 3e-4
    ra_alpha_lr: float = 1e-4

    pa_policy_frequency: int = 2
    pa_target_frequency: int = 1
    ra_policy_frequency: int = 2
    ra_target_frequency: int = 1

    pa_tau: float = 0.005
    ra_tau: float = 0.005

    pa_alpha: float = 0.2
    pa_alpha_autotune: bool = True
    ra_alpha: float = 0.2
    ra_alpha_autotune: bool = True

    write_frequency: int = 100
    save_frequency: int = 100000
    save_folder: str = "./results/relara/"

    track: bool = False


class ReLaraAlgo:
    """
    ReLara algorithm for SINGLE environment:
      - PA: policy agent (SAC)
      - RA: reward agent that proposes r_p(s,a) (SAC)
    """

    def __init__(self, env: gym.Env, pa_actor_class, pa_critic_class, ra_actor_class, ra_critic_class, cfg: ReLaraConfig,track:bool):
        self.cfg = cfg
        self.track = track
        # seeds
        self.seed = int(cfg.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device(f"cuda:{cfg.cuda}" if torch.cuda.is_available() else "cpu")
        self.env = env

        # proposed reward space
        self.proposed_reward_space = gym.spaces.Box(
            low=-float(cfg.proposed_reward_scale),
            high=float(cfg.proposed_reward_scale),
            shape=(1,),
            dtype=np.float32,
            seed=self.seed,
        )

        # RA observation: concat (s, a)
        assert hasattr(env.observation_space, "shape") and env.observation_space.shape is not None
        assert hasattr(env.action_space, "shape") and env.action_space.shape is not None
        self.ra_obs_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(int(env.observation_space.shape[0] + env.action_space.shape[0]),),
            dtype=np.float32,
            seed=self.seed,
        )

        self.beta = float(cfg.beta)
        self.gamma = float(cfg.gamma)

        # tensorboard
        # run_name = "{}-{}-{}-{}".format(
        #     cfg.exp_name,
        #     getattr(env.unwrapped, "spec", None).id if getattr(env.unwrapped, "spec", None) else "AntTask",
        #     self.seed,
        #     datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S"),
        # )
        # os.makedirs("./runs/", exist_ok=True)
        #self.writer = SummaryWriter(os.path.join("./runs/", run_name))

        self.write_frequency = int(cfg.write_frequency)
        self.save_folder = str(cfg.save_folder)
        self.save_frequency = int(cfg.save_frequency)
        os.makedirs(self.save_folder, exist_ok=True)

        # --- PA networks ---
        self.pa_actor = pa_actor_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_1 = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_2 = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_1_target = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_2_target = pa_critic_class(env.observation_space, env.action_space).to(self.device)
        self.pa_qf_1_target.load_state_dict(self.pa_qf_1.state_dict())
        self.pa_qf_2_target.load_state_dict(self.pa_qf_2.state_dict())

        self.pa_actor_optimizer = optim.Adam(self.pa_actor.parameters(), lr=float(cfg.pa_actor_lr))
        self.pa_critic_optimizer = optim.Adam(
            list(self.pa_qf_1.parameters()) + list(self.pa_qf_2.parameters()),
            lr=float(cfg.pa_critic_lr),
        )

        self.pa_alpha_autotune = bool(cfg.pa_alpha_autotune)
        if self.pa_alpha_autotune:
            self.pa_target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.pa_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.pa_alpha = self.pa_log_alpha.exp().item()
            self.pa_alpha_optimizer = optim.Adam([self.pa_log_alpha], lr=float(cfg.pa_alpha_lr))
        else:
            self.pa_alpha = float(cfg.pa_alpha)

        # --- RA networks ---
        self.ra_actor = ra_actor_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_1 = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_2 = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_1_target = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_2_target = ra_critic_class(self.ra_obs_space, self.proposed_reward_space).to(self.device)
        self.ra_qf_1_target.load_state_dict(self.ra_qf_1.state_dict())
        self.ra_qf_2_target.load_state_dict(self.ra_qf_2.state_dict())

        self.ra_actor_optimizer = optim.Adam(self.ra_actor.parameters(), lr=float(cfg.ra_actor_lr))
        self.ra_critic_optimizer = optim.Adam(
            list(self.ra_qf_1.parameters()) + list(self.ra_qf_2.parameters()),
            lr=float(cfg.ra_critic_lr),
        )

        self.ra_alpha_autotune = bool(cfg.ra_alpha_autotune)
        if self.ra_alpha_autotune:
            self.ra_target_entropy = -torch.prod(torch.Tensor(self.proposed_reward_space.shape).to(self.device)).item()
            self.ra_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.ra_alpha = self.ra_log_alpha.exp().item()
            self.ra_alpha_optimizer = optim.Adam([self.ra_log_alpha], lr=float(cfg.ra_alpha_lr))
        else:
            self.ra_alpha = float(cfg.ra_alpha)

        # --- replay buffers ---
        self.env.observation_space.dtype = np.float32

        self.pa_replay_buffer = ReplayBuffer(
            cfg.pa_buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
        )
        self.ra_replay_buffer = ReplayBuffer(
            cfg.ra_buffer_size,
            self.ra_obs_space,
            self.proposed_reward_space,
            self.device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
        )

        self.pa_batch_size = int(cfg.pa_batch_size)
        self.pa_policy_frequency = int(cfg.pa_policy_frequency)
        self.pa_target_frequency = int(cfg.pa_target_frequency)
        self.pa_tau = float(cfg.pa_tau)

        self.ra_batch_size = int(cfg.ra_batch_size)
        self.ra_policy_frequency = int(cfg.ra_policy_frequency)
        self.ra_target_frequency = int(cfg.ra_target_frequency)
        self.ra_tau = float(cfg.ra_tau)
        self.success_buffer: deque[float] = deque(maxlen=100)

        # episode stats buffers (window=100, SB3-like)
        self._current_episode_reward_env: float = 0.0
        self._ep_rew_env_buffer: deque[float] = deque(maxlen=100)

        self._current_episode_reward_pro: float = 0.0
        self._ep_rew_pro_buffer: deque[float] = deque(maxlen=100)

        self._current_episode_len: int = 0
        self._ep_len_buffer: deque[int] = deque(maxlen=100)

    def learn(self, total_timesteps: int = int(1e6), pa_learning_starts: int = int(1e4), ra_learning_starts: int = int(5e3)):
        obs, _info = self.env.reset(seed=self.seed)

        self._current_episode_reward_env = 0.0
        self._current_episode_reward_pro = 0.0
        self._current_episode_len = 0

        for global_step in range(int(total_timesteps)):
            # first action
            if global_step == 0:
                action = self.env.action_space.sample()

            # RA obs = [s, a]
            obs_ra = np.hstack((obs, action)).astype(np.float32, copy=False)

            # sample / propose reward
            if global_step < int(ra_learning_starts):
                reward_pro = self.proposed_reward_space.sample()
            else:
                reward_pro, _, _ = self.ra_actor.get_action(torch.tensor(np.expand_dims(obs_ra, axis=0)).to(self.device))
                reward_pro = reward_pro.detach().cpu().numpy()[0]

            next_obs, reward_env, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            # per-step accumulation (env + pro + len)
            self._current_episode_reward_env += float(reward_env)
            self._current_episode_reward_pro += float(np.asarray(reward_pro).reshape(-1)[0])
            self._current_episode_len += 1

            log_data = {
                r"original/standerd_reward": reward_env,
                r"reward_pro": reward_pro,
                # r"metric/speed": speed,
                # r"metric/far": far,
                # r"metric/stand": stand,
            }

            # means from completed episodes
            original_ep_rew_mean = float(np.mean(self._ep_rew_env_buffer)) if self._ep_rew_env_buffer else None
            rollout_ep_rew_mean = float(np.mean(self._ep_rew_pro_buffer)) if self._ep_rew_pro_buffer else None
            rollout_ep_len_mean = float(np.mean(self._ep_len_buffer)) if self._ep_len_buffer else None

            if original_ep_rew_mean is not None:
                log_data[r"original/ep_rew_mean"] = original_ep_rew_mean
            if rollout_ep_rew_mean is not None:
                log_data[r"rollout/ep_rew_mean"] = rollout_ep_rew_mean
            if rollout_ep_len_mean is not None:
                log_data[r"rollout/ep_len_mean"] = rollout_ep_len_mean

            if self.track:
                swanlab.log(log_data, step=global_step)

            # PA stores env reward only
            self.pa_replay_buffer.add(obs, next_obs, action, reward_env, done, info)

            if not done:
                obs = next_obs
            else:
                # push completed episode stats into buffers (ONCE)
                self._ep_rew_env_buffer.append(float(self._current_episode_reward_env))
                self._ep_rew_pro_buffer.append(float(self._current_episode_reward_pro))
                self._ep_len_buffer.append(int(self._current_episode_len))

                self._current_episode_reward_env = 0.0
                self._current_episode_reward_pro = 0.0
                self._current_episode_len = 0

                success = self._extract_success(info)
                if success is not None:
                    self.success_buffer.append(success)
                    success_rate = float(np.mean(self.success_buffer))
                    if self.track:
                        swanlab.log({"rollout/success_rate": success_rate}, step=global_step)

                # episode stats expected from RecordEpisodeStatistics
                if isinstance(info, dict) and "episode" in info:
                    # self.writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    # self.writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    if self.track:
                        swanlab.log(
                            {
                                "charts/episodic_return": info["episode"]["r"],
                                "charts/episodic_length": info["episode"]["l"],
                            },
                            step=global_step,
                        )
                obs, _ = self.env.reset()

            # next action from PA
            if global_step < int(pa_learning_starts):
                action = self.env.action_space.sample()
            else:
                action, _, _ = self.pa_actor.get_action(torch.tensor(np.expand_dims(obs, axis=0)).to(self.device))
                action = action.detach().cpu().numpy()[0]

            next_obs_ra = np.hstack((obs, action)).astype(np.float32, copy=False)

            # RA stores (s,a)->r_p, reward is env reward
            self.ra_replay_buffer.add(obs_ra, next_obs_ra, reward_pro, reward_env, done, info)

            # train
            if global_step > int(pa_learning_starts):
                self.optimize_pa(global_step)
            if global_step > int(ra_learning_starts):
                self.optimize_ra(global_step)

            if (global_step + 1) % self.save_frequency == 0:
                self.save(indicator=f"{global_step // 1000}k")

        self.env.close()
        if self.cfg.track:
            swanlab.finish()
        #self.writer.close()

    def optimize_pa(self, global_step: int):
        data = self.pa_replay_buffer.sample(self.pa_batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.pa_actor.get_action(data.next_observations)
            qf_1_next_target = self.pa_qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.pa_qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.pa_alpha * next_state_log_pi

            obs_ra = torch.cat((data.observations, data.actions), dim=1)
            reward_pro, _, _ = self.ra_actor.get_action(obs_ra)

            next_q_value = data.rewards.flatten() + self.beta * reward_pro.flatten() + (
                1 - data.dones.flatten()
            ) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.pa_qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.pa_qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.pa_critic_optimizer.zero_grad()
        qf_loss.backward()
        self.pa_critic_optimizer.step()

        if global_step % self.pa_policy_frequency == 0:
            for _ in range(self.pa_policy_frequency):
                pi, log_pi, _ = self.pa_actor.get_action(data.observations)
                qf_1_pi = self.pa_qf_1(data.observations, pi)
                qf_2_pi = self.pa_qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.pa_alpha * log_pi) - min_qf_pi).mean()

                self.pa_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.pa_actor_optimizer.step()

                if self.pa_alpha_autotune:
                    with torch.no_grad():
                        _, log_pi2, _ = self.pa_actor.get_action(data.observations)
                    alpha_loss = (-self.pa_log_alpha.exp() * (log_pi2 + self.pa_target_entropy)).mean()

                    self.pa_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.pa_alpha_optimizer.step()
                    self.pa_alpha = self.pa_log_alpha.exp().item()

        if global_step % self.pa_target_frequency == 0:
            for param, target_param in zip(self.pa_qf_1.parameters(), self.pa_qf_1_target.parameters()):
                target_param.data.copy_(self.pa_tau * param.data + (1 - self.pa_tau) * target_param.data)
            for param, target_param in zip(self.pa_qf_2.parameters(), self.pa_qf_2_target.parameters()):
                target_param.data.copy_(self.pa_tau * param.data + (1 - self.pa_tau) * target_param.data)

        # if global_step % self.write_frequency == 0:
        #     self.writer.add_scalar("losses/pa_qf_1_values", qf_1_a_values.mean().item(), global_step)
        #     self.writer.add_scalar("losses/pa_qf_2_values", qf_2_a_values.mean().item(), global_step)
        #     self.writer.add_scalar("losses/pa_qf_1_loss", qf_1_loss.item(), global_step)
        #     self.writer.add_scalar("losses/pa_qf_2_loss", qf_2_loss.item(), global_step)
        #     self.writer.add_scalar("losses/pa_qf_loss", qf_loss.item() / 2.0, global_step)
        #     self.writer.add_scalar("losses/pa_actor_loss", actor_loss.item(), global_step)
        #     self.writer.add_scalar("losses/pa_alpha", float(self.pa_alpha), global_step)
        #     if self.pa_alpha_autotune:
        #         self.writer.add_scalar("losses/pa_alpha_loss", alpha_loss.item(), global_step)

    def optimize_ra(self, global_step: int):
        data = self.ra_replay_buffer.sample(self.ra_batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.ra_actor.get_action(data.next_observations)
            qf_1_next_target = self.ra_qf_1_target(data.next_observations, next_state_actions)
            qf_2_next_target = self.ra_qf_2_target(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf_1_next_target, qf_2_next_target) - self.ra_alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_qf_next_target.view(-1)

        qf_1_a_values = self.ra_qf_1(data.observations, data.actions).view(-1)
        qf_2_a_values = self.ra_qf_2(data.observations, data.actions).view(-1)
        qf_1_loss = F.mse_loss(qf_1_a_values, next_q_value)
        qf_2_loss = F.mse_loss(qf_2_a_values, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        self.ra_critic_optimizer.zero_grad()
        qf_loss.backward()
        self.ra_critic_optimizer.step()

        if global_step % self.ra_policy_frequency == 0:
            for _ in range(self.ra_policy_frequency):
                pi, log_pi, _ = self.ra_actor.get_action(data.observations)
                qf_1_pi = self.ra_qf_1(data.observations, pi)
                qf_2_pi = self.ra_qf_2(data.observations, pi)
                min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
                actor_loss = ((self.ra_alpha * log_pi) - min_qf_pi).mean()

                self.ra_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.ra_actor_optimizer.step()

                if self.ra_alpha_autotune:
                    with torch.no_grad():
                        _, log_pi2, _ = self.ra_actor.get_action(data.observations)
                    alpha_loss = (-self.ra_log_alpha.exp() * (log_pi2 + self.ra_target_entropy)).mean()

                    self.ra_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.ra_alpha_optimizer.step()
                    self.ra_alpha = self.ra_log_alpha.exp().item()

        if global_step % self.ra_target_frequency == 0:
            for param, target_param in zip(self.ra_qf_1.parameters(), self.ra_qf_1_target.parameters()):
                target_param.data.copy_(self.ra_tau * param.data + (1 - self.ra_tau) * target_param.data)
            for param, target_param in zip(self.ra_qf_2.parameters(), self.ra_qf_2_target.parameters()):
                target_param.data.copy_(self.ra_tau * param.data + (1 - self.ra_tau) * target_param.data)

        # if global_step % self.write_frequency == 0:
        #     self.writer.add_scalar("losses/ra_qf_1_values", qf_1_a_values.mean().item(), global_step)
        #     self.writer.add_scalar("losses/ra_qf_2_values", qf_2_a_values.mean().item(), global_step)
        #     self.writer.add_scalar("losses/ra_qf_1_loss", qf_1_loss.item(), global_step)
        #     self.writer.add_scalar("losses/ra_qf_2_loss", qf_2_loss.item(), global_step)
        #     self.writer.add_scalar("losses/ra_qf_loss", qf_loss.item() / 2.0, global_step)
        #     self.writer.add_scalar("losses/ra_actor_loss", actor_loss.item(), global_step)
        #     self.writer.add_scalar("losses/ra_alpha", float(self.ra_alpha), global_step)
        #     if self.ra_alpha_autotune:
        #         self.writer.add_scalar("losses/ra_alpha_loss", alpha_loss.item(), global_step)

    def save(self, indicator: str = "final"):
        torch.save(
            self.pa_actor.state_dict(),
            os.path.join(self.save_folder, f"pa-actor-{self.cfg.exp_name}-{indicator}-{self.seed}.pth"),
        )
        torch.save(
            self.ra_actor.state_dict(),
            os.path.join(self.save_folder, f"ra-actor-{self.cfg.exp_name}-{indicator}-{self.seed}.pth"),
        )

    def _extract_success(self, info):
        if not isinstance(info, dict):
            return None
        for key in ("success", "is_success", "goal_achieved"):
            val = info.get(key)
            if isinstance(val, (bool, int, float)):
                return float(val)
        episode_info = info.get("episode")
        if isinstance(episode_info, dict):
            for key in ("success", "is_success"):
                val = episode_info.get(key)
                if isinstance(val, (bool, int, float)):
                    return float(val)
        return None
