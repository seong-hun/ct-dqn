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


class ReplayMemory():
    def __init__(self, capacity, transition):
        self.data = deque(maxlen=capacity)
        self.transition = transition

    def push(self, *args):
        self.data.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def __len__(self):
        return len(self.data)

    def get_batch(self):
        return self.transition(*zip(*list(self.data)))


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
            nn.Tanh()
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


class BaseAgent:
    def set_data(self, obs, action, reward, next_obs, info):
        pass

    def optimize_model(self):
        pass

    def get_action(self, obs):
        pass

    def get_info(self):
        return 0


class CMRACAgent(BaseAgent):
    Transition = namedtuple("Transition", ("t", "eta", "chi"))
    memory_capacity = cfg.CMRAC.AGENT.MEMORY_CAPACITY

    def __init__(self):
        self.memory = ReplayMemory(
            capacity=self.memory_capacity + 1,
            transition=self.Transition
        )

        self.min_eigval = 0

    def set_data(self, obs, action, reward, next_obs, info):
        eta, chi = obs
        t = info["t"]
        self.memory.push(t, eta, chi)

    def optimize_model(self):
        memory_len = len(self.memory)

        if memory_len <= self.memory_capacity:
            return

        batch = self.memory.get_batch()

        min_eigvals = np.zeros(memory_len - 1)

        for i in range(memory_len - 1):
            Omega = np.sum(
                [eta.dot(eta.T) for j, eta in enumerate(batch.eta) if j != i],
                axis=0)
            min_eigvals[i] = np.linalg.eigvals(Omega).min()

        del self.memory.data[min_eigvals.argmin()]

        self.min_eigval = min_eigvals.min()

    def get_action(self, state):
        if len(self.memory) < self.memory_capacity:
            return 0, 0

        Omega, M = self.sum(self.memory)
        return Omega, M

    def sum(self, memory):
        batch = self.Transition(*zip(*list(memory.data)))
        Omega = np.sum([eta.dot(eta.T) for eta in batch.eta], axis=0)
        M = np.sum(
            [eta.dot(chi.T) for eta, chi in zip(batch.eta, batch.chi)],
            axis=0
        )
        return Omega, M

    def get_info(self):
        stacked_time = np.array([data.t for data in self.memory.data])
        stacked_time = fill_nan(stacked_time, self.memory_capacity)
        min_eigval = self.min_eigval
        return dict(stacked_time=stacked_time, min_eigval=min_eigval)


class QLCMRACAgent(BaseAgent):
    def __init__(self):
        self.memory = ReplayMemory(cfg.AGENT.REPLAY_CAPACITY)
        self.Q_function = DQN(2, 1)
        self.actor = Actor(2, 1)

        self.M = cfg.AGENT.M

        self.Q_optimizer = optim.Adam(
            self.Q_function.parameters(), lr=cfg.AGENT.HJB_OPTIM_LR)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=cfg.AGENT.ACTOR_OPTIM_LR,
                                          betas=cfg.AGENT.ACTOR_OPTIM_BETAS)

    def set_data(self, data):
        proc_data = [torch.tensor(d).float().flatten()[None, :] for d in data]
        self.memory.push(*proc_data)

    def optimize_model(self):
        if len(self.memory) < cfg.AGENT.BATCH_SIZE:
            return

        transitions = self.memory.sample(cfg.AGENT.BATCH_SIZE)
        batch = self.Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        state_dot_batch = torch.cat(batch.state_dot)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        rintdiff_batch = torch.cat(batch.rintdiff)

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

        with torch.no_grad():
            next_action_batch = self.actor(next_state_batch)
            next_Q = self.Q_function(next_state_batch, next_action_batch)

        opt_error = (next_Q + rintdiff_batch - Q).square().mean()

        # logging.debug(
        #     "HJB error = "
        #     f"{torch.sum(dQdx * state_dot_batch, dim=1).mean():+7.4f} "
        #     f"{self.M * torch.sqrt(torch.sum(dQdu ** 2, dim=1)).mean():+7.4f} "
        #     f"{reward_batch.flatten().mean():+7.4f} = "
        #     f"{HJB_error:+7.5f}"
        # )

        # logging.debug("HJB error: "
        #               f"{HJB_error:+7.4f} | "
        #               "OPT error: "
        #               f"{opt_error:+7.4}")

        Q_error = HJB_error + opt_error

        self.Q_optimizer.zero_grad()
        Q_error.backward()
        self.Q_optimizer.step()

        # Actor update
        # self.Q_function.eval()

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

        # self.Q_function.train()

        info = dict(
            HJB_error=Q_error.detach().numpy(),
            actor_error=actor_error.detach().numpy(),
            actor_param=actor_param,
        )
        return info


def fill_nan(arr, size, axis=0):
    if arr.shape[axis] < size:
        pad_width = [[0, 0] for _ in range(np.ndim(arr))]
        pad_width[axis][1] = size - arr.shape[axis]
        return np.pad(arr, pad_width, constant_values=np.nan)
    else:
        return arr


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
