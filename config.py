from types import SimpleNamespace as SN
import numpy as np


PLANT = SN(
    INITIAL_STATE=np.vstack([0.3, 0]),
    Ap=np.array([[0, 1], [0, 0]]),
    Bp=np.array([[0], [1]]),
    Hp=np.array([[1, 0]]),
    Wast=np.vstack([-18.59521, 15.162375, -62.45153, 9.54708, 21.45291]),
    LAMBDA=np.diag([1.0])
)

DIM = SN(
    np=2,
    n=3,
    m=1,
    q=5,
)

REFMODEL = SN(
    INITIAL_STATE=np.vstack([0.3, 0, 0]),
    Q=np.diag([2800, 1, 15000]),
    R=np.diag([50]),
)

MRAC = SN(
    GAMMA=1e4,
    Q=np.eye(3),
)

CMRAC = SN(
    GAMMA_C=1e2,
    AGENT=SN(
        MEMORY_CAPACITY=100,
    ),
)

AGENT = SN(
    Q=REFMODEL.Q,
    R=REFMODEL.R,
    gamma=1e-2,
    # GAMMA=np.diag([0.01]),
    REPLAY_CAPACITY=50000,
    BATCH_SIZE=100,
    M=1e3,
    HJB_OPTIM_LR=1e-3,
    ACTOR_OPTIM_LR=1e-1,
    ACTOR_OPTIM_MOMENTUM=0.9,
    ACTOR_OPTIM_BETAS=(0.9, 0.999),
    TORCH_SEED=0,
)

FILTER = SN(
    tauf=1e-3,
)

TIME_STEP = 1e-2
FINAL_TIME = 40
EPISODE_LEN = 150
ODE_STEP_LEN = 10
