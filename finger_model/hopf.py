# Include Required Libraries
import matplotlib.pyplot as plt
import jax.numpy as np
from jax.experimental.ode import odeint as odeint_jax

ALPHA = 0
BETA = 1
W_SWING = 2
W_STANCE = 3
THETA = 4
SCALE1 = 5
SCALE2 = 6
MU = 7
K = 8
N_OF_PARAMS = 9


# Function to solve Hopf Equations
def hopf_solve(interval, *params):

    def model(y, t, _params):
        b = 100
        x1 = y[0]
        y1 = y[1]
        x2 = y[2]
        y2 = y[3]
        r1 = np.sqrt(x1 ** 2 + y1 ** 2)
        r2 = np.sqrt(x2 ** 2 + y2 ** 2)
        delta1 = y2 * np.cos(_params[THETA]) - x2 * np.sin(_params[THETA])
        delta2 = y1 * np.cos(-_params[THETA]) - x1 * np.sin(-_params[THETA])
        w1 = _params[W_STANCE] / (np.exp(-b * x1) + 1) + _params[W_SWING] / (np.exp(b * x1) + 1)
        w2 = _params[W_STANCE] / (np.exp(-b * x2) + 1) + _params[W_SWING] / (np.exp(b * x2) + 1)
        dx1dt = _params[ALPHA] * (_params[MU] - r1 ** 2) * x1 - w1 * y1
        dy1dt = _params[BETA] * (_params[MU] - r1 ** 2) * y1 + w1 * x1 + _params[K] * delta1
        dx2dt = _params[ALPHA] * (_params[MU] - r2 ** 2) * x2 - w2 * y2
        dy2dt = _params[BETA] * (_params[MU] - r2 ** 2) * y2 + w2 * x2 + _params[K] * delta2
        return np.array([dx1dt, dy1dt, dx2dt, dy2dt])

    init_x = [0.0, -1.0, 0.0, 1.0]
    sol = odeint_jax(model, init_x, interval, params)
    x1 = sol[0]
    y1 = sol[1]
    x2 = sol[2]
    y2 = sol[3]
    return [x1, y1, x2, y2]


def create_CPG(interval, *params):
    [x1, _, x2, _] = hopf_solve(interval, *params)
    CPG1 = (x1[::]) * params[SCALE1]
    CPG2 = (x2[::]) * params[SCALE2]
    return np.array(CPG1), np.array(CPG2)
