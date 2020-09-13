import numpy as np
import numpy.linalg as nla
import scipy
import sys
import logging
import os
import time

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

    def get_random_stable_gain(self):
        Q = np.diag(np.random.rand(2)) * 0.01
        R = np.diag(np.random.rand(1)) * 0.1
        K, _, eigvals, _ = LQR.clqr(self.A, self.B, Q, R)
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
        - F.T.dot(Gamma).dot(F)
        + R
    )
    print(G.dot(B) + A.T.dot(E.T) - 2 * E.T.dot(Gamma).dot(F))


class Env(BaseEnv):
    Q = cfg.AGENT.Q
    R = cfg.AGENT.R
    Gamma = cfg.AGENT.GAMMA

    def __init__(self):
        super().__init__(dt=cfg.TIME_STEP, max_t=cfg.FINAL_TIME)
        self.system = System()

        # A, B = self.system.A, self.system.B
        # Q, R, Gamma = self.Q, self.R, self.Gamma
        # find_analytic_q_function(A, B, Q, R, Gamma)

    def reset(self):
        super().reset()
        self.K, self.eigvals = self.system.get_random_stable_gain()

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.system.state
        u = self.get_input(t, x)
        self.system.set_dot(u)

    def get_input(self, t, x):
        u = - self.K.dot(x)
        return u

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        state_dict = self.observe_dict(y)
        x = state_dict["system"]
        u = self.get_input(t, x)
        xdot = self.system.deriv(x, u)
        return dict(
            t=t,
            x=x,
            u=u,
            xdot=xdot
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


def run(i):
    logging.info(f"Starting run {i:03d}")

    logging.debug("Creating and resetting Env")
    env = Env()
    env.reset()

    logging.debug(f"Eigenvalues: {env.eigvals[0]:5.2f}, {env.eigvals[1]:5.2f}")

    datapath = os.path.join("data", "run", f"{i:03d}.h5")
    env.logger = fym.logging.Logger(path=datapath)

    t = time.time()

    while True:
        # env.render()
        done = env.step()

        if done:
            break

    env.close()

    t = time.time() - t

    logging.info(f"Data was saved in {datapath}")
    logging.info(f"Running was finised in {t:5.3f} s")


def main():
    set_logger()

    # Get multiple traectory data
    for i in range(cfg.EPISODE_LEN):
        run(i)

    # Q-Learning
    agent = Agent()
    agent.set_data(os.path.join("data", "run"))
    agent.learn()


if __name__ == "__main__":
    main()