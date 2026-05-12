import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class QF(nn.Module):
    def __init__(self, state_dim, action_dim, n_rew=1):
        super(QF, self).__init__()

        # Q1 architecture
        self.n_rew = n_rew
        self.l1 = nn.ModuleList()
        self.l2 = nn.ModuleList()
        self.l3 = nn.ModuleList()

        for _ in range(n_rew):
            self.l1.append(nn.Linear(state_dim + action_dim, 256))
            self.l2.append(nn.Linear(256, 256))
            self.l3.append(nn.Linear(256, 1))

        # Q2 architecture
        self.l4 = nn.ModuleList()
        self.l5 = nn.ModuleList()
        self.l6 = nn.ModuleList()

        for _ in range(n_rew):
            self.l4.append(nn.Linear(state_dim + action_dim, 256))
            self.l5.append(nn.Linear(256, 256))
            self.l6.append(nn.Linear(256, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1s = []
        q2s = []

        for i in range(self.n_rew):
            tmp = F.relu(self.l1[i](sa))
            tmp = F.relu(self.l2[i](tmp))
            tmp = self.l3[i](tmp)
            q1s.append(tmp)

            tmp2 = F.relu(self.l4[i](sa))
            tmp2 = F.relu(self.l5[i](tmp2))
            tmp2 = self.l6[i](tmp2)
            q2s.append(tmp2)

        return torch.cat(q1s, 1), torch.cat(q2s, 1)

    def Q1(self, state, action, index=None):
        sa = torch.cat([state, action], 1)
        q1s = []
        if index==None:
            for i in range(self.n_rew):
                tmp = F.relu(self.l1[i](sa))
                tmp = F.relu(self.l2[i](tmp))
                tmp = self.l3[i](tmp)
                q1s.append(tmp)

            return torch.cat(q1s, 1)
        else:
            tmp = F.relu(self.l1[index](sa))
            tmp = F.relu(self.l2[index](tmp))
            tmp = self.l3[index](tmp)
            return tmp


class VF(nn.Module):
    def __init__(self, state_dim):
        super(VF, self).__init__()

        # Q1 architecture
        self.l1 = nn.ModuleList()
        self.l2 = nn.ModuleList()
        self.l3 = nn.ModuleList()

        for _ in range(1):
            self.l1.append(nn.Linear(state_dim, 256))
            self.l2.append(nn.Linear(256, 256))
            self.l3.append(nn.Linear(256, 1))

        # Q2 architecture
        self.l4 = nn.ModuleList()
        self.l5 = nn.ModuleList()
        self.l6 = nn.ModuleList()

        for _ in range(1):
            self.l4.append(nn.Linear(state_dim, 256))
            self.l5.append(nn.Linear(256, 256))
            self.l6.append(nn.Linear(256, 1))

    def forward(self, state):
        sa = state

        tmp = F.relu(self.l1[i](sa))
        tmp = F.relu(self.l2[i](tmp))
        v1 = self.l3[i](tmp)

        tmp2 = F.relu(self.l4[i](sa))
        tmp2 = F.relu(self.l5[i](tmp2))
        v2 = self.l6[i](tmp2)

        return v1, v2

    def V(self, state):
        sa = state

        tmp = F.relu(self.l1(sa))
        tmp = F.relu(self.l2(tmp))
        v = self.l3(tmp)

        return v

class V(nn.Module):
    def __init__(self, state_dim):
        super(V, self).__init__()

        # v architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        sa = state
        tmp = F.relu(self.l1(sa))
        tmp = F.relu(self.l2(tmp))
        v_value = self.l3(tmp)
        return v_value


class Q(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(Q, self).__init__()

        # v architecture
        self.l1 = nn.Linear(state_dim+act_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        tmp = F.relu(self.l1(sa))
        tmp = F.relu(self.l2(tmp))
        q_value = self.l3(tmp)
        return q_value


class RewardTransformation(nn.Module):
    def __init__(self, mrt_rm_var):
        super(RewardTransformation, self).__init__()

        if 'a' in mrt_rm_var:
            self.a = 1
        else:
            self.a = nn.Parameter(torch.tensor(1.0))

        if 'b' in mrt_rm_var:
            self.b = 0
        else:
            self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, r):
        if torch.is_tensor(self.a):
            a = torch.clamp(self.a, min=0.1, max=5)
        else:
            a = self.a

        r_t = r*a + self.b

        return r_t