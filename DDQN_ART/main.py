import argparse
import json
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch

from agent import DDQNAgent
from buffer import ReplayBuffer
from bias_diagnostics import ARTDiagnosticWrapper

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Atari environment factory
# ---------------------------------------------------------------------------

def make_env(env_id: str, seed: int, render_mode=None):
    """Create a preprocessed Atari environment."""
    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,               # frame skip handled by AtariPreprocessing
        repeat_action_probability=0.0,
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,           # keep uint8; network normalises internally
    )
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------------------------

def epsilon_schedule(t: int, warmup: int, eps_start=1.0, eps_end=0.1, decay=1_000_000):
    if t < warmup:
        return eps_start
    progress = min((t - warmup) / decay, 1.0)
    return eps_start + progress * (eps_end - eps_start)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent: DDQNAgent, eval_env, eval_eps: int = 5) -> float:
    total = 0.0
    for _ in range(eval_eps):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action = agent.select_action(np.array(obs), epsilon=0.05)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total += reward
            done = terminated or truncated
    return total / eval_eps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ALE/Pong-v5", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--art", action="store_true", help="enable ART bias correction")
    parser.add_argument("--max_timesteps", default=10_000_000, type=int)
    parser.add_argument("--warmup", default=50_000, type=int, help="random exploration steps before training")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--buffer_size", default=1_000_000, type=int)
    parser.add_argument("--lr", default=6.25e-5, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--target_update_freq", default=1000, type=int, help="gradient steps between target updates")
    parser.add_argument("--train_freq", default=4, type=int, help="env steps between gradient updates")
    parser.add_argument("--eval_freq", default=50_000, type=int)
    parser.add_argument("--eval_eps", default=5, type=int)
    parser.add_argument("--bias_ema_alpha", default=0.005, type=float)
    parser.add_argument("--task", default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ---- Result directory (mirrors TD3/TD7 naming convention) ----------
    algo_parts = ["ddqn_art" if args.art else "ddqn"]
    if args.task:
        algo_parts.append(args.task)
    algo_dir = "_".join(algo_parts)

    env_tag = args.env.replace("/", "_").replace("-", "_")
    file_stem = f"{env_tag}_seed_{args.seed}"
    data_dir = os.path.join("results", algo_dir)
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, f"{file_stem}_config.json"), "w") as cf:
        json.dump(vars(args), cf, indent=2)

    log_file = open(os.path.join(data_dir, f"{file_stem}.csv"), "w")
    log_file.write("eval_reward,expl_reward,value,be_error\n")

    # ---- Seeding -------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ---- Environments --------------------------------------------------
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)

    obs_shape = env.observation_space.shape   # (4, 84, 84)
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  Env: {args.env}  Actions: {n_actions}  ART: {args.art}")

    # ---- Agent & Buffer ------------------------------------------------
    agent = DDQNAgent(
        n_actions=n_actions,
        obs_shape=obs_shape,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
        art=args.art,
        bias_ema_alpha=args.bias_ema_alpha,
    )
    replay = ReplayBuffer(obs_shape, capacity=args.buffer_size, device=device)

    # ---- ART diagnostics (only when --art) -----------------------------
    diag = None
    if args.art:
        diag_path = os.path.join(data_dir, f"{file_stem}_art_diagnostics.csv")
        diag = ARTDiagnosticWrapper(agent, gamma=args.gamma, log_path=diag_path)
        # Hook in BEFORE the sync so we measure Q1 ≠ Q2
        agent.pre_sync_callback = lambda step: diag.record_sync(step, replay, batch_size=1024)
        print(f"ART diagnostics → {diag_path}")

    # ---- Training loop -------------------------------------------------
    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    last_ep_reward = 0.0
    value_log = None
    be_log = None
    start_time = time.time()

    for t in range(1, args.max_timesteps + 1):

        epsilon = epsilon_schedule(t, args.warmup)
        action = agent.select_action(np.array(obs), epsilon)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # store raw done (not truncated) so bootstrap is not cut on timeout
        replay.add(np.array(obs), action, np.array(next_obs), reward, terminated)
        obs = next_obs

        if done:
            last_ep_reward = ep_reward
            ep_reward = 0.0
            obs, _ = env.reset()

        # ---- Gradient update -----------------------------------------
        if t >= args.warmup and t % args.train_freq == 0 and len(replay) >= args.batch_size:
            batch = replay.sample(args.batch_size)
            value_log, be_log = agent.train(batch, args.batch_size)

            # (diagnostics are recorded via pre_sync_callback inside agent.train)

        # ---- Evaluation & logging ------------------------------------
        if t % args.eval_freq == 0:
            eval_rew = evaluate(agent, eval_env, args.eval_eps)
            elapsed = time.time() - start_time
            bias_str = f"  bias:{agent.bias:.4f}" if args.art else ""
            print(
                f"T:{t:.2e}  eval:{eval_rew:.1f}  expl:{last_ep_reward:.1f}"
                f"  val:{value_log}  be:{be_log}{bias_str}"
                f"  eps:{epsilon:.3f}  time:{elapsed:.0f}s"
            )
            log_file.write(f"{eval_rew},{last_ep_reward},{value_log},{be_log}\n")
            log_file.flush()
            start_time = time.time()

    log_file.close()
    env.close()
    eval_env.close()

    if diag:
        diag.close()
        print("\n=== ART Bias Diagnostic Summary ===")
        print(diag.summary())
        print(f"Theoretical prediction: be_ratio ≈ gamma = {args.gamma}")
