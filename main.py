import numpy as np
import numpy.linalg as nla
import scipy
import sys
import logging
import os
import time
from pathlib import Path
import shutil
# import control

from fym.core import BaseEnv, BaseSystem
import fym.logging
import fym.agents.LQR

from agent import BaseAgent, CMRACAgent, QLCMRACAgent
import config as cfg


def phip(xp):
    x1, x2 = xp.ravel()
    return np.vstack([x1, x2, np.abs(x1) * x2, np.abs(x2) * x2, x1 ** 3])


def phi(x):
    xp = x[:2]
    return phip(xp)


class AugmentedSystem(BaseEnv):
    Ap = cfg.PLANT.Ap
    Bp = cfg.PLANT.Bp
    Hp = cfg.PLANT.Hp
    Wast = cfg.PLANT.Wast
    Lambda = cfg.PLANT.LAMBDA

    def __init__(self):
        super().__init__()
        self.xp = BaseSystem(cfg.PLANT.INITIAL_STATE)
        self.eI = BaseSystem(shape=(cfg.DIM.m, 1))

        m = cfg.DIM.m
        self.A = np.block([
            [self.Ap, np.zeros((cfg.DIM.np, m))],
            [self.Hp, np.zeros((m, m))]])
        self.B = np.block([[self.Bp], [np.zeros((m, m))]])

    def set_dot(self, u, zcmd):
        xp = self.xp.state
        self.dot = self.deriv(xp, u, zcmd)

    def deriv(self, xp, u, zcmd):
        Delta = self.Wast.T.dot(phip(xp))
        xpdot = self.Ap.dot(xp) + self.Bp.dot(self.Lambda).dot(u + Delta)
        eIdot = self.Hp.dot(xp) - zcmd
        dot = np.vstack((xpdot, eIdot))
        return dot


class ReferenceModel(BaseSystem):
    def __init__(self, A, B):
        super().__init__(cfg.REFMODEL.INITIAL_STATE)

        # Calculate Ac and Br
        m = cfg.DIM.m
        Q = cfg.REFMODEL.Q
        R = cfg.REFMODEL.R
        self.K, *_ = fym.agents.LQR.clqr(A, B, Q, R)
        self.Ac = A - B.dot(self.K)
        self.Br = np.block([[np.zeros((cfg.DIM.np, m))], [-np.eye(m)]])

    def deriv(self, xr, zcmd):
        dot = self.Ac.dot(xr) + self.Br.dot(zcmd)
        return dot

    def set_dot(self, zcmd):
        xr = self.state
        self.dot = self.deriv(xr, zcmd)


class MainSystem(BaseEnv):
    def __init__(self):
        super().__init__()
        self.x = AugmentedSystem()
        self.xr = ReferenceModel(A=self.x.A, B=self.x.B)

    def set_dot(self, u, zcmd):
        self.x.set_dot(u, zcmd)
        self.xr.set_dot(zcmd)

    def get_optimal_gain(self):
        return self.xr.K

    def deriv(self, xp, xr, u, zcmd):
        xdot = self.x.deriv(xp, u, zcmd)
        xrdot = self.xr.deriv(xr, zcmd)
        dot = np.vstack((xdot, xrdot))
        return dot

    def over_limit(self):
        done = False
        if np.rad2deg(np.abs(self.x.xp.state[1])) > 90:
            done = True
            logging.info("OVER LIMIT")
        return done


class Filter(BaseEnv):
    tauf = cfg.FILTER.tauf
    n = cfg.DIM.n

    def __init__(self, A, B, K):
        super().__init__()
        self.xi = BaseSystem(shape=(cfg.DIM.m, 1))
        self.eta = BaseSystem(shape=(cfg.DIM.q, 1))

        self.A = A
        self.K = K
        self.Ar = A - B.dot(K)
        self.Bdagger = np.linalg.pinv(B)

    def set_dot(self, u, x, xr):
        tauf = self.tauf
        n = self.n

        xi = self.xi.state
        eta = self.eta.state

        e = x - xr

        uad = u + self.K.dot(x)
        self.xi.dot = 1 / tauf * (
            + uad
            + self.Bdagger.dot(1 / tauf * np.eye(n) + self.Ar).dot(e)
            - xi
        )

        self.eta.dot = 1 / tauf * (phi(x) - eta)

    def get_chi(self, xi, x, xr):
        e = x - xr
        chi = 1 / self.tauf * self.Bdagger.dot(e) - xi
        return chi


class Controller:
    def __init__(self, system):
        super().__init__()
        self.K = system.get_optimal_gain()

    def get_input(self, t, x):
        pass

    def set_dot(self, x, xr, u, aciton):
        pass

    def get_base(self, x):
        ubase = -self.K.dot(x)
        return ubase

    def get_info(self):
        return None

    def observation(self, env):
        return None


class LQR(Controller):
    def get_input(self, t, x):
        u = self.get_base(x)
        return u


class MRAC(Controller, BaseEnv):
    def __init__(self, system):
        super().__init__(system)
        q = cfg.DIM.q
        m = cfg.DIM.m
        self.What = BaseSystem(shape=(q, m))

        self.Gamma = cfg.MRAC.GAMMA
        Ac = system.xr.Ac
        Q = cfg.MRAC.Q
        self.P = scipy.linalg.solve_lyapunov(Ac.T, -Q)
        self.B = system.x.B

    def get_input(self, t, x):
        What = self.What.state
        ubase = self.get_base(x)
        uad = -What.T.dot(phi(x))
        u = ubase + uad
        return u

    def set_dot(self, x, xr, u, action):
        e = x - xr
        P = self.P
        B = self.B
        self.What.dot = np.dot(self.Gamma, phi(x)).dot(e.T).dot(P).dot(B)


class CMRAC(MRAC):
    def __init__(self, system):
        super().__init__(system)
        self.filter = Filter(A=system.x.A, B=system.x.B, K=system.xr.K)

        self.Gamma_c = cfg.CMRAC.GAMMA_C

    def set_dot(self, x, xr, u, action):
        e = x - xr
        P = self.P
        B = self.B

        Omega, M = action

        What = self.What.state

        if np.ndim(Omega) != 0:
            norm = np.linalg.eigvals(Omega).max()
        else:
            norm = 1

        self.What.dot = (
            self.Gamma * phi(x).dot(e.T).dot(P).dot(B)
            - self.Gamma_c * (np.dot(Omega, What) - M) / norm
        )
        self.filter.set_dot(u, x, xr)

    def get_info(self):
        return None

    def observation(self, env):
        xi = self.filter.xi.state
        eta = self.filter.eta.state
        x = env.system.x.state[:, None]
        xr = env.system.xr.state
        chi = self.filter.get_chi(xi, x, xr)
        return eta, chi


class Commander():
    def get_command(self, t):
        if t < 10:
            zcmd = 0
        elif t < 17:
            zcmd = 1
        elif t < 25:
            zcmd = 0
        elif t < 32:
            zcmd = -1
        else:
            zcmd = 0

        return zcmd


class Env(BaseEnv):
    Q = cfg.AGENT.Q
    R = cfg.AGENT.R
    gamma = cfg.AGENT.gamma

    def __init__(self, system, controller):
        super().__init__(dt=cfg.TIME_STEP, max_t=cfg.FINAL_TIME,
                         ode_step_len=cfg.ODE_STEP_LEN)
        self.system = system
        self.controller = controller
        self.commander = Commander()
        self.rint = BaseSystem()

    def reset(self):
        super().reset()
        obs = self.controller.observation(self)
        return obs

    def step(self, action):
        t = self.clock.get()

        *_, done = self.update(action=action)

        done = self.system.over_limit() or done

        next_obs = self.controller.observation(self)

        reward = 0

        info = dict(
            t=t,
            controller=self.controller.get_info(),
        )

        return next_obs, reward, done, info

    def set_dot(self, t, action):
        x = self.system.x.state[:, None]
        xr = self.system.xr.state

        u = self.controller.get_input(t, x)
        zcmd = self.commander.get_command(t)

        self.system.set_dot(u, zcmd)
        self.controller.set_dot(x, xr, u, action)
        self.rint.dot = np.exp(-self.gamma * t) * self.get_reward(x, u)

    def get_reward(self, x, u):
        r = x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)
        return r

    def logger_callback(self, i, t, y, t_hist, ode_hist):
        state_dict = self.observe_dict(y)
        x = np.vstack([val for val in state_dict["system"]["x"].values()])
        u = self.controller.get_input(t, x)
        zcmd = self.commander.get_command(t)
        return dict(
            t=t,
            state=state_dict,
            u=u,
            zcmd=zcmd,
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


def run(env, agent, name):
    logging.info(f"Starting run {name}")

    rundir = os.path.join("data", f"{name}")
    env.logger = fym.logging.Logger(path=os.path.join(rundir, "traj.h5"))
    env.logger.set_info(name=name)
    agent_logger = fym.logging.Logger(path=os.path.join(rundir, "agent.h5"))

    t0 = time.time()

    logging.debug("Resetting Env")
    obs = env.reset()

    while True:
        env.render()

        action = agent.get_action(obs)
        agent_info = agent.get_info()

        next_obs, reward, done, info = env.step(action)

        agent.set_data(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            info=info
        )
        agent.optimize_model()

        agent_logger.record(
            t=info["t"],
            info=agent_info,
        )

        if done:
            break

        obs = next_obs

    env.close()
    agent_logger.close()

    dt = time.time() - t0

    logging.info(f"Data was saved in {rundir}")
    logging.info(f"Running was finised in {dt:5.3f} s")


def main():
    set_logger()

    # agentlist = ["LQR", "MRAC", "CMRAC", "QLCMRAC"]
    agentlist = ["QLCMRAC"]

    # Clear the data directory
    datapath = Path("data")
    if not datapath.exists():
        datapath.mkdir()

    for d in Path("data").iterdir():
        if d.name in agentlist:
            shutil.rmtree(d)
            logging.info(f"Data {d} is removed")

    for name in agentlist:
        system = MainSystem()

        if name == "LQR":
            controller = LQR(system)
            agent = BaseAgent()
        elif name == "MRAC":
            controller = MRAC(system)
            agent = BaseAgent()
        elif name == "CMRAC":
            controller = CMRAC(system)
            agent = CMRACAgent()
        elif name == "QLCMRAC":
            controller = CMRAC(system)
            agent = QLCMRACAgent()

        env = Env(system, controller)
        run(env, agent, name)


if __name__ == "__main__":
    main()
