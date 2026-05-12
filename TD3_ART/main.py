import json
import time
from argparse import ArgumentParser
import numpy as np
import torch
import os
import gymnasium as gym
from Trainer import Trainer
from utils import eval_policy
from replaybuffer import ReplayBuffer


def parse_args():
    parser = ArgumentParser(description='train args')
    parser.add_argument('-en', '--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('-mt', '--max_timesteps', type=int, default=1_000_000)
    parser.add_argument('-b', '--batch_size', type=int, default=256)
    parser.add_argument('-seed', '--seed', type=int, default=1)
    parser.add_argument('-pbts', '--perturbations', nargs='+', default=['0.0'],
                        help='reward perturbation list (e.g. 0.0 0.1 0.2)')
    parser.add_argument('--mrt', action='store_true')
    parser.add_argument('--mrt_direct', action='store_true', help='Direct bias estimate (no EMA smoothing)')
    parser.add_argument('--bias_ema_alpha', type=float, default=0.005, help='EMA decay for bias estimate')
    parser.add_argument('--reward_centering', action='store_true')
    parser.add_argument('--gpl', action='store_true')
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    torch.set_num_threads(1)

    # Build experiment directory name
    algo_parts = ['td3']
    if args.mrt:
        algo_parts.append(f'mrt_a{args.bias_ema_alpha:g}')
    if args.mrt_direct:
        algo_parts.append('mrt_direct')
    if args.reward_centering:
        algo_parts.append('rc')
    if args.gpl:
        algo_parts.append('gpl')
    if args.task:
        algo_parts.append(args.task)
    algo_dir = '_'.join(algo_parts)

    file_stem = f"{args.env}_seed_{args.seed}"
    data_dir = os.path.join('results', algo_dir)
    os.makedirs(data_dir, exist_ok=True)

    config_path = os.path.join(data_dir, f"{file_stem}_config.json")
    with open(config_path, 'w') as cf:
        json.dump(vars(args), cf, indent=2)

    f = open(os.path.join(data_dir, f"{file_stem}.csv"), 'w')
    f.write("eval_reward,expl_reward,value,be_error\n")

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    perturbations = np.array(args.perturbations).astype(float)
    len_of_ptb = len(perturbations)

    policy = Trainer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        ptbs=perturbations,
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        policy_freq=2,
        mrt=args.mrt,
        mrt_direct=args.mrt_direct,
        bias_ema_alpha=args.bias_ema_alpha,
        reward_centering=args.reward_centering,
        gpl=args.gpl,
    )
    buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset(seed=args.seed)
    rewards = []
    expl_rew = 0.0
    str_time = time.time()
    q_idx = np.random.randint(0, len_of_ptb)
    value_timestep = 0
    update = False
    value = None
    be_error = None

    for t in range(int(args.max_timesteps)):

        if t < 25000:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * 0.1, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_done = terminated or truncated
        rewards.append(reward)

        done_bool = float(terminated) if len(rewards) < 1000 else 0.0
        buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if ep_done:
            state, _ = env.reset()
            expl_rew = sum(rewards)
            value_timestep += len(rewards)
            if value_timestep >= 1000:
                update = True
                value_timestep = 0
                q_idx = np.random.randint(0, len_of_ptb)
                rewards = []

        if t >= 25000 and update:
            for _ in range(t - 25000 - policy.total_it):
                value, be_error = policy.train(buffer, args.batch_size, q_idx=q_idx)
            update = False

        if t >= 25000 and (t + 1) % 1000 == 0:
            eval_rew = eval_policy(policy, env=eval_env)
            cur_time = time.time()
            print(f"T:{t+1:.2e}  expl:{expl_rew:.1f}  eval:{eval_rew:.1f}  "
                  f"val:{value}  be:{be_error}  time:{cur_time-str_time:.1f}s")
            str_time = time.time()
            f.write(",".join(map(str, [eval_rew, expl_rew, value, be_error])) + '\n')
            f.flush()

    f.close()
