import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

import buffer

@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = 1e6
    discount: float = 0.99
    target_update_rate: int = 250
    exploration_noise: float = 0.1

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # LAP
    alpha: float = 0.4
    min_priority: float = 1

    # TD3+BC
    lmbda: float = 0.1

    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4
    reset_weight: float = 0.9

    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 3e-4

    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4

    # Reward Transformation
    rt_lr: float = 3e-4

    # Reward Centering
    reward_centering_alpha: float = 0.001

    # ART bias EMA
    bias_ema_alpha: float = 0.005

    # MRT sync mode: 0.0 = hard copy every target_update_rate steps; >0 = soft Polyak rate per step
    mrt_tau: float = 0.0

    # GPL pessimism coefficient learning rate
    gpl_beta_lr: float = 1e-4

def AvgL1Norm(x, eps=1e-8):
    return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)

def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu):
        super(Actor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)


    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        return torch.tanh(self.l3(a))


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)


    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs


    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Critic, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2*zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2*zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)


    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)



class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, offline=False, mrt=False, mrt_direct=False, mrt_interval=1000, reward_centering=False, gpl=False, hp=Hyperparameters()):
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.hp = hp

        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        # ART: parallel bias-correcting Q network (mrt = EMA-smoothed; mrt_direct = raw)
        self.mrt_direct = mrt_direct
        if mrt or mrt_direct:
            self.bias_critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
            self.bias_critic_optimizer = torch.optim.Adam(self.bias_critic.parameters(), lr=hp.critic_lr)
            self.bias = 0.0  # EMA (or raw, when mrt_direct) of mean TD error: approximates E[epsilon_Q] per Eq.22

        self.reward_centering = reward_centering
        if reward_centering:
            self.reward_mean = 0.0

        # GPL: adaptive pessimism — β * |Q1 - Q2| penalizes uncertain targets
        self.gpl = gpl
        if gpl:
            self.gpl_beta = 0.5   # matches clipped double-Q at init

        self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = buffer.LAP(state_dim, action_dim, self.device, hp.buffer_size, hp.batch_size,
                                        max_action, normalize_actions=True, prioritized=True)

        self.max_action = max_action
        self.offline = offline
        self.mrt = mrt
        self.mrt_interval = mrt_interval

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0


    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            state = torch.tensor(state.reshape(1,-1), dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                action = action + torch.randn_like(action) * self.hp.exploration_noise

            return action.clamp(-1,1).cpu().data.numpy().flatten() * self.max_action


    def train(self):

        state, action, next_state, reward, not_done = self.replay_buffer.sample()

        # Reward centering: subtract EMA of batch rewards to remove correlated Bellman bias
        if self.reward_centering:
            self.reward_mean += self.hp.reward_centering_alpha * (reward.mean().item() - self.reward_mean)
            reward = reward - self.reward_mean

        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        #########################
        # Update Critic
        #########################
        # next_action = None
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(-1,1)

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)

            Q_next = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs)
            if self.gpl:
                # GPL: pessimistic target = mean - β * |Q1 - Q2|
                Q_next_reduced = Q_next.mean(1, keepdim=True) - self.gpl_beta * (Q_next[:, 0:1] - Q_next[:, 1:2]).abs()
            else:
                Q_next_reduced = Q_next.min(1, keepdim=True)[0]
            Q_target = reward + not_done * self.hp.discount * Q_next_reduced.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        value_difference = Q - Q_target
        td_loss = value_difference.abs()

        critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ART: estimate bias from an independent batch sampled AFTER the critic update.
        # The critic just minimised its loss on the primary batch, so TD errors there are
        # artificially small.  An independent batch gives an unbiased estimate of E[Q - TQ]
        # under the current Q.  (Mirrors TD3_MRT/Trainer.py:124-139.)
        if self.mrt or self.mrt_direct:
            primary_ind = self.replay_buffer.ind  # save: update_priority() below uses these
            b_state, b_action, b_next_state, b_reward, b_not_done = self.replay_buffer.sample()
            self.replay_buffer.ind = primary_ind  # restore for update_priority()

            with torch.no_grad():
                if self.reward_centering:
                    b_reward = b_reward - self.reward_mean

                b_target_zs = self.fixed_encoder_target.zs(b_next_state)
                b_noise = (torch.randn_like(b_action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip, self.hp.noise_clip)
                b_next_action = (self.actor_target(b_next_state, b_target_zs) + b_noise).clamp(-1, 1)
                b_target_zsa = self.fixed_encoder_target.zsa(b_target_zs, b_next_action)

                b_Q_next = self.critic_target(b_next_state, b_next_action, b_target_zsa, b_target_zs)
                if self.gpl:
                    b_Q_next_red = b_Q_next.mean(1, keepdim=True) - self.gpl_beta * (b_Q_next[:, 0:1] - b_Q_next[:, 1:2]).abs()
                else:
                    b_Q_next_red = b_Q_next.min(1, keepdim=True)[0]
                b_Q_target = b_reward + b_not_done * self.hp.discount * b_Q_next_red.clamp(self.min_target, self.max_target)

                b_zs = self.fixed_encoder.zs(b_state)
                b_zsa = self.fixed_encoder.zsa(b_zs, b_action)
                b_Q = self.critic(b_state, b_action, b_zsa, b_zs)

                td_mean = (b_Q - b_Q_target).mean().item()
                if self.mrt_direct:
                    self.bias = td_mean                       # raw: no smoothing
                else:
                    self.bias += self.hp.bias_ema_alpha * (td_mean - self.bias)

        # GPL: dual gradient update — β tracks mean TD error; positive error → more pessimism
        if self.gpl:
            td_error_mean = value_difference.mean().item()
            self.gpl_beta = max(0.0, self.gpl_beta + self.hp.gpl_beta_lr * td_error_mean)

        # Update bias_critic toward bias-corrected target: y_Q2 = Q_target - hat_b
        if self.mrt or self.mrt_direct:
            bias_Q = self.bias_critic(state, action, fixed_zsa, fixed_zs)
            bias_td_loss = (bias_Q - Q_target + self.bias).abs()
            bias_critic_loss = LAP_huber(bias_td_loss)

            self.bias_critic_optimizer.zero_grad()
            bias_critic_loss.backward()
            self.bias_critic_optimizer.step()

        #########################
        # Update LAP
        #########################
        priority = td_loss.max(1)[0].clamp(min=self.hp.min_priority).pow(self.hp.alpha)
        self.replay_buffer.update_priority(priority)

        #########################
        # Update Actor
        #########################
        if self.training_steps % self.hp.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            if self.mrt or self.mrt_direct:
                Q = self.bias_critic(state, actor, fixed_zsa, fixed_zs)
                actor_loss = -Q.mean()
            elif self.gpl:
                Q = self.critic(state, actor, fixed_zsa, fixed_zs)
                Q_pess = Q.mean(1, keepdim=True) - self.gpl_beta * (Q[:, 0:1] - Q[:, 1:2]).abs()
                actor_loss = -Q_pess.mean()
            else:
                Q = self.critic(state, actor, fixed_zsa, fixed_zs)
                actor_loss = -Q.mean()

            if self.offline:
                actor_loss = actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        #########################
        self.training_steps += 1
        # Update Iteration
        #########################
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())

            if self.mrt or self.mrt_direct:
                self.critic.load_state_dict(self.bias_critic.state_dict())
                self.critic_target.load_state_dict(self.bias_critic.state_dict())
            else:
                self.critic_target.load_state_dict(self.critic.state_dict())

            self.replay_buffer.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min
        return Q[:,0].mean().item(), (td_loss**2).mean().item()


    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            self.train_and_reset()


    # Batch training
    def train_and_reset(self):
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.hp.steps_before_checkpointing:
                self.best_min_return *= self.hp.reset_weight
                self.max_eps_before_update = self.hp.max_eps_when_checkpointing

            self.train()

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8