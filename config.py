from types import SimpleNamespace as SN
import numpy as np


SYSTEM = SN(
    INITIAL_STATE=np.zeros((2, 1))
)

AGENT = SN(
    Q=np.diag([10, 5]),
    R=np.diag([3]),
    GAMMA=np.diag([0.1])
)

TIME_STEP = 0.01
FINAL_TIME = 50
EPISODE_LEN = 10
