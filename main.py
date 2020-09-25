import numpy as np
import numpy.linalg as nla
import scipy
import sys
import logging
import os
import time
from pathlib import Path
import shutil

from fym.core import BaseEnv, BaseSystem
import fym.logging
import fym.agents.LQR as LQR

from agent import Agent
import config as cfg


class System(BaseSystem):
    A = np.array([
        [0, 1],
        [-0.01859521, 0.015162375]
    ])
    B = np.vstack((0, 1))

    def __init__(self):
        super().__init__(cfg.SYSTEM.INITIAL_STATE)

    def set_dot(self, u):
        self.dot = self.deriv(self.state, u)

    def deriv(self, x, u):
        return self.A.dot(x) + self.B.dot(u)

    def get_optimal_gain(self, Q, R):
        K, _, eigvals, _ = LQR.clqr(self.A, self.B, Q, R)
        return K, eigvals

    def get_random_stable_gain(self):
        Q = np.diag(np.random.rand(2)) * 0.01
        R = np.diag(np.random.rand(1)) * 0.1
        K, eigvals = self.get_optimal_gain(Q, R)
        return K, eigvals


def find_analytic_q_function(A, B, Q, R, Gamma):
    # Find P
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    # Find F
    F = scipy.linalg.solve_continuous_are(
        0.5 * nla.inv(R).dot(B.T).dot(P).dot(B),
        np.eye(1),
        R,
        nla.inv(Gamma)
    )
    # print(F)

    # Find E
    E = F.dot(nla.inv(R)).dot(B.T).dot(P)
    # print(E)

    # Find G
    # tmp = 0.25 * P.dot(B).dot(nla.inv(R)).dot(F).dot(nla.inv(R)).dot(B.T).dot(P)
    G = scipy.linalg.solve_lyapunov(
        0.5 * A.T,
        E.T.dot(Gamma).dot(E) - Q
    )
    # print(G)

    # Check HJB
    print(0.5 * (G.dot(A) + A.T.dot(G)) - E.T.dot(Gamma).dot(E) + Q)
    print(
        0.5 * (E.dot(B) + B.T.dot(E.T))
        # - F.T.dot(Gamma).dot(F)
        + R
    )
    print(G.dot(B) + A.T.dot(E.T) - 2 * E.T.dot(Gamma).dot(F))


class Env(BaseEnv):
    Q = cfg.AGENT.Q
    R = cfg.AGENT.R
    # Gamma = cfg.AGENT.GAMMA

    def __init__(self):
        super().__init__(dt=cfg.TIME_STEP, max_t=cfg.FINAL_TIME,
                         ode_step_len=cfg.ODE_STEP_LEN)
        self.system = System()
        self.rint = BaseSystem()

        optim_gain = self.system.get_optimal_gain(self.Q, self.R)
        self.optimal_gain, self.optimal_eigvals = optim_gain

        # A, B = self.system.A, self.system.B
        # Q, R, Gamma = self.Q, self.R, self.Gamma
        # find_analytic_q_function(A, B, Q, R, Gamma)

    def reset(self):
        super().reset()
        self.system.state = np.random.uniform(-1, 1,
                                              size=self.system.state.shape)
        self.system.state *= np.deg2rad(np.vstack((30, 30)))
        self.K, self.eigvals = self.system.get_random_stable_gain()
        self.random_param = np.random.uniform(low=[0, 1, 0],
                                              high=[2, 10, np.pi],
                                              size=(10, 3))

    def step(self):
        t = self.clock.get()
        x = self.system.state
        rint = self.rint.state
        u = self.get_input(t, x)
        xdot = self.system.deriv(x, u)
        r = self.get_reward(x, u)

        *_, done = self.update()

        next_x = self.system.state
        rintdiff = self.rint.state - rint
        return (x, u, xdot, r, next_x, rintdiff), done

    def set_dot(self, t):
        x = self.system.state
        u = self.get_input(t, x)
        self.system.set_dot(u)
        self.rint.dot = self.get_reward(x, u)

    def get_input(self, t, x):
        u = - self.K.dot(x)
        for a, w, p in self.random_param:
            u = u + np.deg2rad(1) * (a * np.sin(w * t + p))
        return u

    def get_reward(self, x, u):
        r = x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)
        return r

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        state_dict = self.observe_dict(y)
        x = state_dict["system"]
        rint = state_dict["rint"]
        u = self.get_input(t, x)
        xdot = self.system.deriv(x, u)
        return dict(
            t=t,
            x=x,
            u=u,
            xdot=xdot,
            rint=rint,
        )


def set_logger():
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        filename='main.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def run(agent, i):
    logging.info(f"Starting run {i:03d}/{cfg.EPISODE_LEN:03d}")

    logging.debug("Creating and resetting Env")
    env = Env()
    env.reset()

    logging.debug(f"Eigenvalues: {env.eigvals[0]:5.2f}, {env.eigvals[1]:5.2f}")

    rundir = os.path.join("data", f"run-{i:03d}")
    env.logger = fym.logging.Logger(path=os.path.join(rundir, "traj.h5"))
    env.logger.set_info(
        optimal_gain=env.optimal_gain,
        optimal_eigvals=env.optimal_eigvals,
    )
    agent_logger = fym.logging.Logger(path=os.path.join(rundir, "agent.h5"))

    t0 = time.time()

    while True:
        env.render()

        t = env.clock.get()

        data, done = env.step()

        agent.set_data(data)
        info = agent.optimize_model()

        if info is not None:
            agent_logger.record(t=t, info=info)

        if done:
            break

    env.close()
    agent_logger.close()

    dt = time.time() - t0

    logging.info(f"Data was saved in {rundir}")
    logging.info(f"Running was finised in {dt:5.3f} s")


def main():
    set_logger()

    # Clear the data directory
    for d in Path("data").iterdir():
        shutil.rmtree(d)
        logging.info(f"Data {d} is removed")

    # Set a Q-learning agent
    agent = Agent()

    # Get multiple traectory data
    for i in range(cfg.EPISODE_LEN):
        run(agent, i)

    # # Q-Learning
    # agent.set_data(os.path.join("data", "run"))
    # agent.learn()


if __name__ == "__main__":
    main()
