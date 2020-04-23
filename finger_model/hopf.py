# Include Required Libraries
from math import sin, cos, sqrt, exp, pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

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


class hopf():
    def __init__(self, *params):
        self.params = params
        # self.params[ALPHA] = params[0]
        # self.params[BETA] = params[1]
        # self.params[W_SWING] = params[2]  # frequency of the swing phase
        # self.params[W_STANCE] = params[3]    # frequency of the stance phase
        # self.params[THETA] = params[4]  # required phase difference between the 2 oscillators
        # self.params.current_ED = params[5]
        # self.params.current_FP = params[6]
        # self.params[MU] = params[13]
        # self.params[K] = params[14]  # coupling between phases of oscillators
        # self.params.X0 = X0    # List with initial condition
        self.init_x = [0.0, -1.0, 0.0, 1.0]
        self.b = 100

    # Function for set of Hopf Equations
    # Returns rate of change of x and y wrt time as array of floats
    def update(self, X, t):
        x1 = X[0]
        y1 = X[1]
        x2 = X[2]
        y2 = X[3]
        r1 = sqrt(x1 ** 2 + y1 ** 2)
        r2 = sqrt(x2 ** 2 + y2 ** 2)
        delta1 = y2 * cos(self.params[THETA]) - x2 * sin(self.params[THETA])
        delta2 = y1 * cos(-self.params[THETA]) - x1 * sin(-self.params[THETA])
        w1 = self.params[W_STANCE] / (exp(-self.b * x1) + 1) + self.params[W_SWING] / (exp(self.b * x1) + 1)
        w2 = self.params[W_STANCE] / (exp(-self.b * x2) + 1) + self.params[W_SWING] / (exp(self.b * x2) + 1)
        dx1dt = self.params[ALPHA] * (self.params[MU] - r1 ** 2) * x1 - w1 * y1
        dy1dt = self.params[BETA] * (self.params[MU] - r1 ** 2) * y1 + w1 * x1 + self.params[K] * delta1
        dx2dt = self.params[ALPHA] * (self.params[MU] - r2 ** 2) * x2 - w2 * y2
        dy2dt = self.params[BETA] * (self.params[MU] - r2 ** 2) * y2 + w2 * x2 + self.params[K] * delta2
        return np.array([dx1dt, dy1dt, dx2dt, dy2dt])

    # Function to solve Hopf Equations
    def solve(self, interval):
        sol = odeint(self.update, self.init_x, interval)
        x1 = sol[:, 0]
        y1 = sol[:, 1]
        x2 = sol[:, 2]
        y2 = sol[:, 3]
        return [x1, y1, x2, y2]


def create_CPG(interval, *params):
    oscillator = hopf(*params)
    [x1, _, x2, _] = oscillator.solve(interval)
    CPG1 = (x1[::]) * params[SCALE1]
    CPG2 = (x2[::]) * params[SCALE2]
    return np.array(CPG1), np.array(CPG2)