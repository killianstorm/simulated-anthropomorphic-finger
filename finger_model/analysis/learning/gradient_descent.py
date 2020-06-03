from finger_model.simulation.optimizer import *
from finger_model.simulation.loss_functions import *

import numpy as num
import matplotlib.pyplot as plt
from finger_model.tools import plots

from datetime import datetime


def plot_torques_or_forces(torques, interval, title, tendons=False):

    if tendons:
        plt.plot(interval, torques[:, 3], label="ED")
        plt.plot(interval, torques[:, 2], label="FP")
        plt.plot(interval, torques[:, 1], label="IO")
        plt.plot(interval, torques[:, 0], label="FS")
        plt.ylabel("Force [N]")
    else:
        plt.plot(interval, torques[:, 0], label="MCP")
        plt.plot(interval, torques[:, 1], label="PIP")
        plt.plot(interval, torques[:, 2], label="DIP")
        plt.title(title)
        plt.ylabel("Torque [Nm]")

    plt.title(title)
    plt.legend(loc="upper left")
    plt.xlabel("Time [s]")
    plt.savefig(title + ".png", dpi=244)
    plt.show()


def simulate_ctrnn_params_and_animate(params, name, tendons=False):
    """
    Simulate ctrnn parameters and create animated comparison
    between physical and simulated.

    arguments:
        params: dict containing reference, interval and params for ctrnn.
        name: name of the animation
    contains
    """
    reference = params['reference']
    dt = params['interval'][1] - params['interval'][0]

    print("Simulating the solution...", end="")
    approximation = simulate_rnn_oscillator(params)
    print("DONE")

    loss = loss_end_effector(reference, approximation)
    print("The loss is: ", loss)

    print("Plotting comparison between reference and approximated")

    plt.plot(reference['end_effector'][0], reference['end_effector'][1], label="reference")
    plt.plot(approximation['end_effector'][0], approximation['end_effector'][1], label="approximated")
    ti = "Comparison reference and approximated for " + name
    plt.title(ti)
    plt.savefig(ti + ".png", dpi=244)
    plt.show()

    print("Plotting the used forces/torques")

    title = "Forces" if tendons else "Torques"
    if 'torques' in reference:
        plot_torques_or_forces(reference['torques'], params['interval'], title + " for reference trajectory " + name, tendons)
    plot_torques_or_forces(approximation['torques'], params['interval'], title + " for approximated trajectory " + name, tendons)

    print("Animating the comparison between the reference and the approximation.")
    plots.animate(reference, dt, name, approximation, di=10)

    print("Process has finished.")


def learn_gradient_descent(reference, interval, iterations, learning_rate, name, loss_function=loss_angles, tendons=False):
    """
    Learn given reference trajectory by using gradient descent. An animation is created at the end.
    arguments:
        reference: the trajectory to be approximated
        interval: the interval
        iterations: the number of iterations
        learning_rate: the learning rate
        name: the file name of the animation
        loss_function: the loss function
    """

    name = str(name)

    # Plot reference trajectory.
    plt.cla()
    plt.plot(reference['end_effector'][0], reference['end_effector'][1])
    ti = "Reference trajectory " + name
    plt.title(ti)
    plt.savefig(ti + ".png", dpi=244)
    plt.show()

    size = RNN_SIZE_TENDONS if tendons else RNN_SIZE_TORQUES

    # Random initialisation.
    init_params = {
        'interval': interval,
        'reference': reference,
        RNN_TAUS: np.array(num.random.rand(size), dtype="float64"),
        RNN_BIASES: np.array(num.random.rand(size), dtype="float64"),
        RNN_GAINS: np.array(num.random.rand(size), dtype="float64"),
        RNN_STATES: np.array(num.random.rand(size), dtype="float64"),
        RNN_WEIGHTS: np.array(num.random.rand(size * size), dtype="float64")
    }

    # 0.5 initialisation
    # init_params = {
    #     'interval': interval,
    #     'reference': reference,
    #     RNN_TAUS: np.zeros(size) + 0.5,
    #     RNN_BIASES: np.zeros(size) + 0.5,
    #     RNN_GAINS: np.zeros(size) + 0.5,
    #     RNN_STATES: np.zeros(size) + 0.5,
    #     RNN_WEIGHTS: np.zeros(size * size) + 0.5
    # }


    # Params to take grad of.
    grad_params = [RNN_TAUS, RNN_BIASES, RNN_GAINS, RNN_STATES, RNN_WEIGHTS]

    print("Starting gradient descent...")
    gradbest = gradient_descent(loss_function, iterations, learning_rate, grad_params, init_params)
    print("Gradient descent has finished.")

    print("The best solution seems to be:")
    print(gradbest)

    simulate_ctrnn_params_and_animate(gradbest, name, tendons)

