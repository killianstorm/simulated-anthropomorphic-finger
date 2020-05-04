from dynamic_model import *
from oscillator import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as num


def oscillator_loss_function(loss):
    # Create reference.
    params = [2, 2,  # a, b
              1.5 * num.pi, num.pi / 4,  # w_swing, w_stance
              num.pi,  # theta
              1, 0.5,  # scale1, scale2
              1, 1]  # mu, k
    reference = simulate_oscillator(interval, *params)
    # animation(reference, "reference")

    init_params = params
    init_params[SCALE1] = 0.1
    init_params[SCALE2] = 0.1

    tau2 = np.arange(0., 2., 0.1)
    tau3 = np.arange(0., 2., 0.1)

    B, D = np.meshgrid(tau2, tau3)
    nu = num.sqrt(1 + (2 * D * B) ** 2) / num.sqrt((1 - B ** 2) ** 2 + (2 * D * B) ** 2)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(B, D, nu)
    plt.xlabel('b')
    plt.ylabel('d')
    plt.show()

    # Learn parameters to reproduce reference.
    # CMA_ES_oscillator(reference)
    # grad_oscillator(reference, loss_end_effector, 50, 0.1, *params)

oscillator_loss_function(None)