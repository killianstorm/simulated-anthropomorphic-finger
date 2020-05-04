from finger_model.dynamic_model import *
from finger_model.oscillator import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as num

from datetime import datetime


def oscillator_loss_function(loss, interval, params, tau2, tau3):
    # Create reference.

    reference = simulate_oscillator(interval, *params)
    # animation(reference, "reference")

    init_params = params

    vals = num.zeros((tau2.shape[0], tau3.shape[0]))
    for i in range(tau2.shape[0]):
        for j in range(tau3.shape[0]):
            # print(i, j)
            init_params[SCALE1] = tau2[i]
            init_params[SCALE2] = tau3[j]
            temp = simulate_oscillator(interval, *init_params)
            vals[i, j] = loss(reference, temp)

    TAU2, TAU3 = np.meshgrid(tau2, tau3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(TAU2, TAU3, vals)
    plt.xlabel('tau2')
    plt.ylabel('tau3')
    plt.show()

    # Learn parameters to reproduce reference.
    # CMA_ES_oscillator(reference)
    # grad_oscillator(reference, loss_end_effector, 50, 0.1, *params)


def animation(history, dt, name=None, history2=None):
    key_points = history['positions']


    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    images = []
    di = 10
    N = key_points.shape[1]
    for i in tqdm(range(0, N, di)):
        plt.cla()

        plt.plot(key_points[:4, i], key_points[4:, i], marker='.')
        plt.scatter(history['end_effector'][0][:i], history['end_effector'][1][:i], s=0.1)


        if history2 is not None:
            plt.plot(history2['positions'][:4, i], history2['positions'][4:, i], marker='.')
            plt.scatter(history2['end_effector'][0][:i], history2['end_effector'][1][:i], s=0.1)


        plt.axhline(0)
        plt.axis('equal')
        plt.axis([-1, 4, -1, 2])
        images.append(fig2image(fig))

    filename = str(name) + datetime.now().strftime("_%d-%b-%Y_(%H:%M:%S.%f)") + ".mp4"

    if name is None:
        ImageSequenceClip(images, fps=int(1/dt/di)).ipython_display()
    else:
        ImageSequenceClip(images, fps=int(1/dt/di)).write_videofile(filename)
    return filename
