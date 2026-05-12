import numpy as np


def eval_policy(policy, env, eval_episodes=3):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            done = terminated or truncated
    avg_reward /= eval_episodes
    return avg_reward
