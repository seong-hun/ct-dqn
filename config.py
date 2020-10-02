from types import SimpleNamespace as SN
import numpy as np


SYSTEM = SN(
    INITIAL_STATE=np.zeros((2, 1))
)

AGENT = SN(
    Q=np.diag([10, 5]),
    R=np.diag([3]),
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

TIME_STEP = 0.1
FINAL_TIME = 20
EPISODE_LEN = 150
ODE_STEP_LEN = 10
