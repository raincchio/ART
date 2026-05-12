import copy
import torch
import torch.nn.functional as F
from network import Actor, QF

device = torch.device("cpu")


class Trainer(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            ptbs=None,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            mrt=False,
            mrt_direct=False,
            reward_centering=False,
            gpl=False,
            bias_ema_alpha=0.005,
            rc_alpha=0.001,
            beta_lr=1e-4,
    ):
        self.ptbs = torch.as_tensor(ptbs).float().to(device)
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.qf = QF(state_dim, action_dim, len(ptbs)).to(device)
        self.qf_target = copy.deepcopy(self.qf)
        self.qf_optimizer = torch.optim.Adam(self.qf.parameters(), lr=3e-4)

        # ART: parallel bias-correcting Q network
        if mrt or mrt_direct:
            self.bias_qf = QF(state_dim, action_dim, len(ptbs)).to(device)
            self.bias_qf_optimizer = torch.optim.Adam(self.bias_qf.parameters(), lr=3e-4)
            self.bias_ema = 0.0
            self.bias_ema_alpha = bias_ema_alpha

        # Reward Centering: EMA of mean reward
        self.reward_centering = reward_centering
        if reward_centering:
            self.reward_mean = 0.0
            self.rc_alpha = rc_alpha

        # GPL: adaptive pessimism coefficient β
        self.gpl = gpl
        if gpl:
            self.gpl_beta = 0.5   # matches TD3's clipped double-Q at init
            self.beta_lr = beta_lr

        self.mrt = mrt
        self.mrt_direct = mrt_direct

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.from_numpy(state[None]).float().to(device)
        return self.actor(state).cpu().detach().numpy()[0]

    def _bellman_target(self, next_state, action, reward, not_done):
        """Compute Q-target for a batch. Assumes reward already reward-centered if needed."""
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
        next_q1s, next_q2s = self.qf_target(next_state, next_action)
        min_next_qs = torch.min(next_q1s, next_q2s)
        return reward + not_done * self.discount * min_next_qs + self.ptbs

    def train(self, replay_buffer, batch_size=100, q_idx=None):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Reward centering: subtract EMA of mean reward to reduce correlated Bellman bias
        if self.reward_centering:
            self.reward_mean += self.rc_alpha * (reward.mean().item() - self.reward_mean)
            reward = reward - self.reward_mean

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            next_q1s, next_q2s = self.qf_target(next_state, next_action)

            if self.gpl:
                # GPL: pessimistic target = mean - β * uncertainty
                next_qmean = (next_q1s + next_q2s) / 2
                next_uncertainty = (next_q1s - next_q2s).abs()
                min_next_qs = next_qmean - self.gpl_beta * next_uncertainty
            else:
                min_next_qs = torch.min(next_q1s, next_q2s)

            target_qs = reward + not_done * self.discount * min_next_qs + self.ptbs

        current_q1s, current_q2s = self.qf(state, action)

        critic_loss = F.mse_loss(current_q1s, target_qs) + F.mse_loss(current_q2s, target_qs)
        self.qf_optimizer.zero_grad()
        critic_loss.backward()
        self.qf_optimizer.step()

        # GPL: dual gradient update for β
        # β increases when TD error is positive (Q overestimates target), becoming more pessimistic
        if self.gpl:
            with torch.no_grad():
                current_qmean = (current_q1s + current_q2s) / 2
                td_error_mean = (current_qmean - target_qs).mean().item()
            self.gpl_beta = max(0.0, self.gpl_beta + self.beta_lr * td_error_mean)

        # ART: estimate bias from an independent batch sampled after the critic update.
        # Using separate data decouples b̂ from the gradient step: the critic just minimised
        # its loss on batch-1, so TD errors on batch-1 are artificially small.  An independent
        # batch gives an unbiased estimate of E[Q - TQ] under the current Q function.
        if self.mrt or self.mrt_direct:
            b_s, b_a, b_ns, b_r, b_nd = replay_buffer.sample(batch_size)
            with torch.no_grad():
                if self.reward_centering:
                    b_r = b_r - self.reward_mean
                b_target = self._bellman_target(b_ns, b_a, b_r, b_nd)
                b_q1, b_q2 = self.qf(b_s, b_a)
                td_mean = ((b_q1 + b_q2) / 2 - b_target).mean().item()
                if self.mrt_direct:
                    self.bias_ema = td_mean          # direct: no smoothing
                else:
                    self.bias_ema += self.bias_ema_alpha * (td_mean - self.bias_ema)

            # Update bias_qf toward bias-corrected target: Q_target - b̂
            bias_q1s, bias_q2s = self.bias_qf(state, action)
            bias_target = target_qs - self.bias_ema
            bias_critic_loss = (F.mse_loss(bias_q1s, bias_target)
                                + F.mse_loss(bias_q2s, bias_target))
            self.bias_qf_optimizer.zero_grad()
            bias_critic_loss.backward()
            self.bias_qf_optimizer.step()

        # Actor update
        curr_action = self.actor(state)
        if self.mrt or self.mrt_direct:
            q_values = self.bias_qf.Q1(state, curr_action)
            actor_loss = -q_values[:, q_idx].mean()
        elif self.gpl:
            q1_a, q2_a = self.qf(state, curr_action)
            q_pess = (q1_a + q2_a) / 2 - self.gpl_beta * (q1_a - q2_a).abs()
            actor_loss = -q_pess[:, q_idx].mean()
        else:
            q_values = self.qf.Q1(state, curr_action)
            actor_loss = -q_values[:, q_idx].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.total_it += 1

        # Soft target updates
        if self.mrt or self.mrt_direct:
            # bias_qf drives both qf and qf_target (low-bias values guide the main network)
            for param, target_param in zip(self.bias_qf.parameters(), self.qf_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.bias_qf.parameters(), self.qf.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            for param, target_param in zip(self.qf.parameters(), self.qf_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return current_q1s.mean().item(), critic_loss.item() / 2

    def save(self, filename):
        torch.save(self.qf.state_dict(), filename + "_critic")
        torch.save(self.qf_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.qf.load_state_dict(torch.load(filename + "_critic"))
        self.qf_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.qf_target = copy.deepcopy(self.qf)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
