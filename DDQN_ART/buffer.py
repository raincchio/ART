import numpy as np
import torch


class ReplayBuffer:
    """Uniform replay buffer storing Atari frames as uint8 to save memory."""

    def __init__(self, obs_shape: tuple, capacity: int = 1_000_000, device: str = "cuda"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.not_dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action: int, next_obs, reward: float, done: bool):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.not_dones[self.ptr] = 1.0 - float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.not_dones[idx], device=self.device),
        )

    def __len__(self):
        return self.size
