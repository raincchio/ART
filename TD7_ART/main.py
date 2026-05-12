import argparse
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch

import TD7_MRT as TD7

# import TR as TD7

def train_online(RL_agent, env, eval_env, args, f):

	start_time = time.time()
	allow_train = False
	value = None
	tdd = None
	last_expl_reward = None
	state, _ = env.reset(seed=args.seed)
	ep_finished = False
	ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

	for t in range(int(args.max_timesteps+1)):

		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, reward, terminated, truncated, _ = env.step(action)
		ep_finished = terminated or truncated

		ep_total_reward += reward
		ep_timesteps += 1

		done = float(terminated) if ep_timesteps < env._max_episode_steps else 0
		RL_agent.replay_buffer.add(state, action, next_state, reward, done)

		state = next_state

		if allow_train and not args.use_checkpoints:
			value, tdd = RL_agent.train()

		if allow_train and t % args.eval_freq == 0:
			maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, args, last_expl_reward, value, tdd, f)

		if t >= args.timesteps_before_training:
			allow_train = True

		if ep_finished:
			if allow_train and args.use_checkpoints:
				RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

			state, _ = env.reset()
			last_expl_reward = ep_total_reward
			ep_total_reward, ep_timesteps = 0, 0
			ep_num += 1

def train_offline(RL_agent, env, eval_env, args):
	RL_agent.replay_buffer.load_D4RL(d4rl.qlearning_dataset(env))

	evals = []
	start_time = time.time()

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args, d4rl=True)
		RL_agent.train()


def maybe_evaluate_and_print(RL_agent, eval_env, t, start_time, args, expl, value, tdd, f, d4rl=False):
	# if t % args.eval_freq == 0 and allow_train:

	total_reward = np.zeros(args.eval_eps)
	for ep in range(args.eval_eps):
		state, _ = eval_env.reset()
		done = False
		while not done:
			action = RL_agent.select_action(np.array(state), args.use_checkpoints, use_exploration=False)
			state, reward, terminated, truncated, _ = eval_env.step(action)
			done = terminated or truncated
			total_reward[ep] += reward
	eval = total_reward.mean()

	print(f"Step:{t} eval: {eval:.2f} expl: {expl:.2f} value: {value:.2f} be: {tdd:.2f}  time:{round((time.time()-start_time)/60.,2)} min(s)")
	if d4rl:
		total_reward = eval_env.get_normalized_score(total_reward) * 100
		print(f"D4RL score: {total_reward.mean():.3f}")



	log = [eval, expl, value, tdd]
	f.write(",".join(map(str, log)) + '\n')
	f.flush()
		# dd = np.load(f"{home_directory}/experiments/mrt/{args.file_name}.npy")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	torch.set_num_threads(1)
	# RL
	parser.add_argument("--env", default="Humanoid-v4", type=str)
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--mrt", action='store_true')
	parser.add_argument("--mrt_direct", action='store_true', help="Direct bias estimate (no EMA smoothing); bias = td_mean per step")
	parser.add_argument("--bias_ema_alpha", type=float, default=0.005, help="EMA decay for bias estimate")
	parser.add_argument("--mrt_interval", default=250, type=int)
	parser.add_argument("--mrt_tau", default=0.0, type=float, help="MRT soft-update rate (0=hard sync, e.g. 0.005=soft)")
	parser.add_argument("--reward_centering", action='store_true')
	parser.add_argument("--gpl", action='store_true')
	parser.add_argument("--offline", default=False, action='store_true')
	parser.add_argument('--use_checkpoints', default=False, action='store_true')
	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=1e3, type=int)
	parser.add_argument("--eval_eps", default=5, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)
	# File
	parser.add_argument('--file_name', default=None)
	parser.add_argument('--d4rl_path', default="./d4rl_datasets", type=str)
	parser.add_argument('--task', default='test')
	args = parser.parse_args()
	
	if args.offline:
		import d4rl
		d4rl.set_dataset_path(args.d4rl_path)
		args.use_checkpoints = False

	# Build experiment name: td7[_mrt_a<alpha>][_mrt_direct][_rc][_gpl][_<task>]
	algo_parts = ['td7']
	if args.mrt:
		algo_parts.append(f'mrt_a{args.bias_ema_alpha:g}')
		if args.mrt_tau > 0:
			algo_parts.append('soft')
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

	# Save experiment config
	config_path = os.path.join(data_dir, f"{file_stem}_config.json")
	with open(config_path, 'w') as cf:
		json.dump(vars(args), cf, indent=2)

	# Open metrics CSV
	f = open(os.path.join(data_dir, f"{file_stem}.csv"), 'w')
	f.write("eval_reward,expl_reward,value,be_error\n")

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	print("---------------------------------------")
	print(f"Algorithm: TD7, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	env.action_space.seed(args.seed)
	eval_env.action_space.seed(args.seed + 100)

	random.seed(args.seed)
	np.random.seed(args.seed)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)  # 如果使用 GPU
	# torch.cuda.manual_seed_all(args.seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False

	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	hp = TD7.Hyperparameters(mrt_tau=args.mrt_tau, bias_ema_alpha=args.bias_ema_alpha)
	RL_agent = TD7.Agent(state_dim, action_dim, max_action, args.offline,
	                     mrt=args.mrt, mrt_direct=args.mrt_direct,
	                     mrt_interval=args.mrt_interval,
	                     reward_centering=args.reward_centering,
	                     gpl=args.gpl,
	                     hp=hp)

	if args.offline:
		train_offline(RL_agent, env, eval_env, args)
	else:
		train_online(RL_agent, env, eval_env, args, f)
