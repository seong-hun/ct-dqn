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


random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)


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

    def flush(self):
        self.data.clear()


class DQN(nn.Module):
    def __init__(self, xdim, udim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(xdim + udim, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=1)
        out = self.model(xu)
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
    Gamma_c = cfg.CMRAC.GAMMA_C

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

    def get_action(self, obs):
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

    def composite_law(self, x, xr, What, action):
        Omega, M = action

        if np.ndim(Omega) != 0:
            norm = np.linalg.eigvals(Omega).max()
        else:
            norm = 1

        return - self.Gamma_c * (np.dot(Omega, What) - M) / norm

    def observation(self, env):
        xi = env.filter.xi.state
        eta = env.filter.eta.state
        x = env.system.x.state[:, None]
        xr = env.system.xr.state
        chi = env.filter.get_chi(xi, x, xr)
        return eta, chi


class QLCMRACAgent(BaseAgent):
    Transition = namedtuple("Transition",
                            ("z", "what", "R", "next_z", "next_what"))

    def __init__(self):
        self.memory = ReplayMemory(
            capacity=cfg.QLCMRAC.AGENT.REPLAY_CAPACITY,
            transition=self.Transition,
        )
        self.Q_function = DQN(2 * cfg.DIM.n, cfg.DIM.q * cfg.DIM.m)

        self.M = cfg.QLCMRAC.AGENT.M
        self.factor = np.exp(-cfg.GAMMA * cfg.QLCMRAC.H)

        self.Q_optimizer = optim.Adam(
            self.Q_function.parameters(), lr=cfg.QLCMRAC.AGENT.HJB_OPTIM_LR)
        self.loss = nn.MSELoss()

    def set_data(self, obs, action, reward, next_obs, info):
        z, What = obs
        R = reward
        next_z, next_What = next_obs

        proc_data = [torch.tensor(d).float().flatten()[None, :]
                     for d in [z, What, R, next_z, next_What]]
        self.memory.push(*proc_data)

    def composite_term(self, x, xr, What):
        self.Q_function.eval()

        z = np.vstack((x, xr)).T
        z = torch.FloatTensor(z)
        what = torch.FloatTensor(What).T
        z.requires_grad_(True)
        what.requires_grad_(True)

        Q = self.Q_function(z, what)
        dQdu = autograd.grad(Q.sum(), what, create_graph=True)[0]
        dQdu = dQdu.detach().numpy()

        comp = - self.M * dQdu / np.linalg.norm(dQdu)
        return comp.T

    def observation(self, env):
        x = env.system.x.state[:, None]
        xr = env.system.xr.state
        z = np.vstack((x, xr))
        What = env.controller.What.state
        return z, What

    def get_action(self, obs):
        pass

    def optimize_model(self):
        if len(self.memory) < cfg.QLCMRAC.AGENT.REPLAY_CAPACITY:
            return

        self.Q_function.train()

        for i in range(cfg.QLCMRAC.AGENT.ITER_LIMIT):
            transitions = self.memory.sample(cfg.QLCMRAC.AGENT.BATCH_SIZE)
            batch = self.Transition(*zip(*transitions))

            z_batch = torch.cat(batch.z)
            what_batch = torch.cat(batch.what)
            R_batch = torch.cat(batch.R)
            next_z_batch = torch.cat(batch.next_z)
            next_what_batch = torch.cat(batch.next_what)

            Q = self.Q_function(z_batch, what_batch)

            with torch.no_grad():
                next_Q = self.Q_function(next_z_batch, next_what_batch)

            loss = self.loss(self.factor * next_Q + R_batch, Q)

            self.Q_optimizer.zero_grad()
            loss.backward()
            self.Q_optimizer.step()

            logging.debug(f"[{i+1:02d}/{cfg.QLCMRAC.AGENT.ITER_LIMIT}] "
                          f"Loss: {loss:07.4f}")
            if loss < cfg.QLCMRAC.AGENT.ITER_THR:
                break

        self.memory.flush()

        info = dict(
            loss=loss.detach().numpy(),
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
