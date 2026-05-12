import copy

import numpy as np
import torch
import torch.nn.functional as F

from network import NatureDQN


class DDQNAgent:
    """
    Double DQN agent with optional ART (bias-correcting Q-network).

    ART mechanism (mirrors TD7_MRT.py):
      - bias_net: separate Q-network trained on bias-corrected Bellman targets
      - bias (EMA scalar): tracks E[Q_online(s,a) - y] across the replay buffer
      - bias_net target: y - bias  →  bias_net ≈ Q* corrected for overestimation
      - Action selection uses bias_net when ART is enabled
      - Hard target update: Q_online ← bias_net, Q_target ← bias_net (every target_update_freq steps)
    """

    def __init__(
        self,
        n_actions: int,
        obs_shape: tuple,
        device: str = "cuda",
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        art: bool = False,
        bias_ema_alpha: float = 0.005,
    ):
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.art = art
        self.bias_ema_alpha = bias_ema_alpha
        self.training_steps = 0
        self.pre_sync_callback = None  # set externally by diagnostics

        self.online_net = NatureDQN(n_actions).to(device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=lr, eps=1.5e-4
        )

        if art:
            self.bias_net = NatureDQN(n_actions).to(device)
            self.bias_optimizer = torch.optim.Adam(
                self.bias_net.parameters(), lr=lr, eps=1.5e-4
            )
            self.bias = 0.0  # EMA of E[Q_online(s,a) - y]: positive = overestimation

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        net = self.bias_net if self.art else self.online_net
        return int(net(obs_t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train(self, batch, batch_size: int):
        state, action, next_state, reward, not_done = batch

        # ---- Double DQN target ----------------------------------------
        with torch.no_grad():
            # online net selects the next action
            next_action = self.online_net(next_state).argmax(dim=1, keepdim=True)
            # target net evaluates it
            next_q = self.target_net(next_state).gather(1, next_action)
            y = reward + not_done * self.gamma * next_q   # (B,1)

        # ---- Main Q-network update ------------------------------------
        current_q = self.online_net(state).gather(1, action.unsqueeze(1))  # (B,1)

        # ART: update global bias EMA before touching gradients
        if self.art:
            with torch.no_grad():
                td_mean = (current_q - y).mean().item()
            self.bias += self.bias_ema_alpha * (td_mean - self.bias)

        loss = F.smooth_l1_loss(current_q, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # ---- ART: bias_net update -------------------------------------
        # bias_net is trained to satisfy: bias_net(s,a) ≈ y - bias
        # i.e., an unbiased Q-estimate corrected for systematic overestimation
        if self.art:
            bias_q = self.bias_net(state).gather(1, action.unsqueeze(1))
            bias_target = y - self.bias          # (B,1)
            bias_loss = F.smooth_l1_loss(bias_q, bias_target.detach())
            self.bias_optimizer.zero_grad()
            bias_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bias_net.parameters(), 10.0)
            self.bias_optimizer.step()

        # ---- Hard target update ---------------------------------------
        self.training_steps += 1
        self.did_sync = False
        if self.training_steps % self.target_update_freq == 0:
            self.did_sync = True
            # pre_sync_callback fires HERE, before weights are overwritten.
            # Diagnostics should hook into this to measure Q1 vs Q2 gap.
            if self.pre_sync_callback is not None:
                self.pre_sync_callback(self.training_steps)
            if self.art:
                # bias_net drives both networks (mirrors TD7_MRT target update logic)
                self.online_net.load_state_dict(self.bias_net.state_dict())
                self.target_net.load_state_dict(self.bias_net.state_dict())
            else:
                self.target_net.load_state_dict(self.online_net.state_dict())

        return current_q.mean().item(), loss.item()
