import numpy as np
import random
from collections import deque, namedtuple
import logging

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

import config as cfg


random.seed(cfg.AGENT.TORCH_SEED)
np.random.seed(cfg.AGENT.TORCH_SEED)
torch.manual_seed(cfg.AGENT.TORCH_SEED)

Transition = namedtuple("Transition",
                        ("state", "action", "state_dot", "reward"))


class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, xdim, udim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(xdim + udim, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        out = self.model(xu)
        return out


class Actor(nn.Module):
    def __init__(self, xdim, udim):
        super().__init__()
        self.model = nn.Linear(xdim, udim, bias=False)

    def forward(self, x):
        out = self.model(x)
        return out


class Agent():
    def __init__(self):
        self.memory = ReplayMemory(cfg.AGENT.REPLAY_CAPACITY)
        self.Q_function = DQN(2, 1)
        self.actor = Actor(2, 1)

        self.M = cfg.AGENT.M

        self.HJB_optimizer = optim.Adam(
            self.Q_function.parameters(), lr=cfg.AGENT.HJB_OPTIM_LR)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=cfg.AGENT.ACTOR_OPTIM_LR)

    def set_data(self, data):
        x, u, xdot, r = [torch.tensor(d).float().flatten()[None, :]
                         for d in data]
        self.memory.push(x, u, xdot, r)

    def optimize_model(self):
        if len(self.memory) < cfg.AGENT.BATCH_SIZE:
            return

        transitions = self.memory.sample(cfg.AGENT.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        state_dot_batch = torch.cat(batch.state_dot)
        reward_batch = torch.cat(batch.reward)

        # Q-function update
        state_batch.requires_grad_(True)
        action_batch.requires_grad_(True)

        Q = self.Q_function(state_batch, action_batch)

        dQdx = autograd.grad(Q.sum(), state_batch, create_graph=True)[0]
        dQdu = autograd.grad(Q.sum(), action_batch, create_graph=True)[0]

        HJB_error = (
            torch.sum(dQdx * state_dot_batch, dim=1)
            - self.M * torch.sqrt(torch.sum(dQdu ** 2, dim=1))
            + reward_batch.flatten()
        ).square().mean()

        # logging.debug(
        #     "HJB error = "
        #     f"{torch.sum(dQdx * state_dot_batch, dim=1).mean():+7.4f} "
        #     f"{self.M * torch.sqrt(torch.sum(dQdu ** 2, dim=1)).mean():+7.4f} "
        #     f"{reward_batch.flatten().mean():+7.4f} = "
        #     f"{HJB_error:+7.5f}"
        # )

        self.HJB_optimizer.zero_grad()
        HJB_error.backward()
        self.HJB_optimizer.step()

        # Actor update
        self.Q_function.eval()

        # state_batch.requires_grad_(False)
        action_batch = self.actor(state_batch)

        Q = self.Q_function(state_batch, action_batch)
        dQdu = autograd.grad(Q, action_batch,
                             grad_outputs=torch.ones_like(Q),
                             create_graph=True)[0]
        actor_error = torch.sum(dQdu.square(), dim=1).mean()

        actor_param = self.actor.model.weight.detach().numpy().T.copy()

        self.actor_optimizer.zero_grad()
        actor_error.backward(retain_graph=True)

        self.actor_optimizer.step()

        self.Q_function.train()

        info = dict(
            HJB_error=HJB_error.detach().numpy(),
            actor_error=actor_error.detach().numpy(),
            actor_param=actor_param,
        )
        return info


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(2, 1),
    )
    # model.weight = nn.Parameter(torch.ones(1, 2))
    # model.bias = nn.Parameter(torch.zeros(1))

    model2 = nn.Linear(3, 2)

    z = torch.rand(5, 3)
    x = model2(z)
    y = model(x)

    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")

    print(f"model2 param: {model2.weight.data}")

    grad = autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    print(f"grad: {grad}")

    error = (grad ** 2).mean()
    print(f"error: {error}")

    model.zero_grad()
    model2.zero_grad()
    error.backward()
    # print(f"model grad: {model.weight.grad}")
    print(f"model2 grad: {model2.weight.grad}")

#     exp = 4 * x.T * z
#     print(f"Expected model2 grad: {exp}")
