# core/brs_mountaincar_wrapper.py
from pathlib import Path
from typing import Any

# ppo_mountaincar_brs.py
import random
import time
import traceback

import gymnasium as gym
import numpy as np
import swanlab
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

from configs.ppo_args import PpoAtariArgs
from core.brs_mountaincar_wrapper import BRSRewardWrapper

from core.model.checkpoint import save_checkpoint
from configs.base_args import get_root_path

args = tyro.cli(PpoAtariArgs)
args.finalize()
def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=20000)
            env = gym.wrappers.RecordVideo(env, Path(get_root_path()) / "results" / "videos"/  run_name)
        else:
            env = gym.make(env_id, max_episode_steps=20000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.enable_brave:
            env = BRSRewardWrapper(env, gamma=gamma)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def get_info_val(infos: Any, key: str, env_idx: int = 0):
    """从 vector env 返回的 infos 中安全提取字段 key（优先常规位置，再查 final_info / _final_info）。"""
    # 情况 A: infos 是 list/tuple/np.ndarray，每项是 dict
    if isinstance(infos, (list, tuple, np.ndarray)):
        try:
            info_i = infos[env_idx]
            if isinstance(info_i, dict) and key in info_i:
                return info_i[key]
        except Exception:
            pass

    # 情况 B: infos 是 dict（常见于 gym.vector 返回），可能包含 per-env arrays 或 final_info
    if isinstance(infos, dict):
        # 直接作为 per-env array/dict 的键
        if key in infos:
            entry = infos[key]
            try:
                return entry[env_idx]
            except Exception:
                return entry

        # 查 final_info / _final_info（数组，每项可能为 None 或 dict）
        for final_key in ("final_info", "_final_info"):
            if final_key in infos:
                try:
                    fin_entry = infos[final_key][env_idx]
                    if isinstance(fin_entry, dict) and key in fin_entry:
                        return fin_entry[key]
                except Exception:
                    pass

    return None


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape[0]
        act_dim = envs.single_action_space.n
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def train(args, envs, run_name):
    if args.track:
        swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            group=args.swanlab_group,
            config=vars(args),
            experiment_name=args.experiment_name,
            settings=swanlab.Settings(
                backup=False
            )
        )
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    last_progress_bucket = -1
    brs_reward = []
    cost = []
    RDCR = []
    ori_reward = []
    for iteration in range(1, args.num_iterations + 1):
        progress_bucket = int((iteration * 20) / args.num_iterations)
        if progress_bucket > last_progress_bucket:
            last_progress_bucket = progress_bucket
            print(f"Training process: {progress_bucket * 5}%")
            # save checkpoint
            try:
                save_checkpoint(run_name, iteration, global_step, agent, optimizer, args)
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
            print(f"Training process: {progress_bucket * 5}%")
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        C_min = 0
        cost_val = 0
        rdcr_val = 0
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            ep_info = get_info_val(infos, "episode")
            if ep_info is not None and "r" in ep_info:
                swanlab.log(data={
                    "charts/episodic_return": ep_info["r"],
                    "charts/episodic_length": ep_info["l"],
                }, step=global_step)

            env_idx = 0
            if args.enable_brave:
                brs_val = get_info_val(infos, "brs_reward", env_idx)
                C_min = get_info_val(infos, "C_min", env_idx)
                cost_val = get_info_val(infos, "cost", env_idx)
                rdcr_val = get_info_val(infos, "RDCR", env_idx)
                ori_reward_val = get_info_val(infos, "reward", env_idx)
                if global_step <50000:
                    if args.track:
                        swanlab.log(
                            data={
                                "debug/brs_reward": brs_val,
                                "debug/cost": cost_val,
                                "debug/rdcr": rdcr_val,
                                "debug/ori_reward": ori_reward_val,
                                "debug/C_min": C_min,
                            },
                            step=global_step
                        )
            # brs_reward.append(float(brs_val))
            # cost.append(float(cost_val))
            # RDCR.append(float(rdcr_val))
            # ori_reward.append(float(ori_reward_val))

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = (
                torch.as_tensor(next_obs_np, dtype=torch.float32).to(device),
                torch.as_tensor(next_done, dtype=torch.float32).to(device),
            )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        # print(
        #     f"global_step={global_step}, "
        #     f"return_mean={y_true.mean():.3f}, "
        #     f"SPS={sps}, "
        #     f"explained_var={explained_var:.3f}"
        # )
        if args.track:
            swanlab.log(
                data={
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/return_mean": y_true.mean(),
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(sps),
                    "debug/C_min": C_min,
                    "debug/cost": cost_val,
                    "debug/rdcr": rdcr_val,
                },
                step=global_step
            )

def main():
    # 覆盖为 MountainCar 的设置（也可以在 args 里配置）
    args.env_id = "MountainCar-v0"
    args.swanlab_group = "MountainCarV0"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    envs = None
    try:
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
        train(args, envs, run_name)
    except KeyboardInterrupt:
        print("Interrupted by user. Exiting gracefully...")
    except Exception:
        print("Unhandled exception in main:")
        traceback.print_exc()
    finally:
        if envs is not None:
            envs.close()
        if args.track:
            swanlab.finish()

if __name__ == "__main__":
    main()