from finger_model.simulation.optimizer import *
from finger_model.simulation.loss_functions import *

import numpy as num
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from finger_model.tools import plots

from datetime import datetime
import matplotlib as mpl


def plot_finger():
    plt.plot([0, lengths[0], lengths[0] + lengths[1], np.sum(lengths)], [0, 0, 0, 0], marker='.', linewidth=3, markersize=15, color='#1f77b4')
    plt.text(0 - 0.015, 0.025, 'MCP')
    plt.text(lengths[0] - 0.015, 0.025, 'PIP')
    plt.text(lengths[0] + lengths[1] - 0.015, 0.025, 'DIP')

    axes = plt.gca()
    axes.set_xlim([-0.175, 0.275])
    axes.set_ylim([-0.25, 0.15])
    axes.set_aspect('equal')

    if ENABLE_PIANO_KEY:
        rect = patches.Rectangle((pianokey_coordinates[0], pianokey_coordinates[1] - pianokey_height), 1,
                                 pianokey_height, linewidth=3, edgecolor='k', facecolor='none')
        axes.add_patch(rect)
        plt.text(pianokey_coordinates[0] + 0.05, pianokey_coordinates[1] - pianokey_height + 0.05, "PIANO KEY")

    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.legend()


def array_to_dict(a, size):
    d = {}
    d[RNN_TAUS] = a[:size]
    d[RNN_BIASES] = a[size:2 * size]
    d[RNN_GAINS] = a[2 * size:3 * size]
    d[RNN_STATES] = a[3 * size:4 * size]
    d[RNN_WEIGHTS] = a[4 * size:].reshape((size, size))
    return d


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
    print("The end effector loss is: ", loss)

    print("Plotting comparison between reference and approximated")


    plt.plot(reference['end_effector'][0], reference['end_effector'][1], label="reference")
    plt.plot(approximation['end_effector'][0], approximation['end_effector'][1], label="approximated")
    ti = "Comparison reference and approximated for " + name
    plt.title(ti)
    plot_finger()
    plt.savefig(ti + ".png", dpi=244)
    plt.show()

    print("Plotting the used forces/torques")

    title = "Forces" if tendons else "Torques"
    if 'torques' in reference:
        plot_torques_or_forces(reference['torques'], params['interval'], title + " for reference " + name, tendons)
    plot_torques_or_forces(approximation['torques'], params['interval'], title + " for approximated " + name, tendons)

    print("Animating the comparison between the reference and the approximation.")
    plots.animate(reference, dt, name, approximation)

    print("Process has finished.")


def learn_gradient_descent(reference, interval, iterations, name, loss_function=loss_angles, tendons=False):
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

    print("Performing gradient descent on model with " + "tendons" if ENABLE_TENDONS else "torques" + " for " + str(iterations) + " iterations.")
    print("Experiment name: " + name)
    print("Reference duration: " + str(interval[-1]))
    print("dt: " + str(interval[1] - interval[0]))

    starttime = datetime.now()
    print("Start time is " + str(starttime.strftime("_%d-%b-%Y_(%H:%M:%S.%f)")))


    # Plot reference trajectory.
    plt.cla()
    plt.plot(reference['end_effector'][0], reference['end_effector'][1], label="end effector trajectory")
    ti = "Reference trajectory " + name
    plt.title(ti)
    plot_finger()
    plt.savefig(ti + ".png", dpi=244)
    plt.show()

    size = RNN_SIZE_TENDONS if tendons else RNN_SIZE_TORQUES

    # Random initialisation.
    # init_params = {
    #     'interval': interval,
    #     'reference': reference,
    #     RNN_TAUS: np.array(num.random.rand(size), dtype="float64"),
    #     RNN_BIASES: np.array(num.random.rand(size), dtype="float64"),
    #     RNN_GAINS: np.array(num.random.rand(size), dtype="float64"),
    #     RNN_STATES: np.array(num.random.rand(size), dtype="float64"),
    #     RNN_WEIGHTS: np.array(num.random.rand(size * size), dtype="float64")
    # }

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

    init_params = {
        'interval': interval,
        'reference': reference,
        RNN_TAUS: np.ones(size),
        RNN_BIASES: np.ones(size),
        RNN_GAINS: np.ones(size),
        RNN_STATES: np.ones(size),
        RNN_WEIGHTS: np.ones(size * size)
    }

    losses = []


    current_iteration = [0]

    lines = []

    def callback(params):
        p = array_to_dict(params, RNN_SIZE_TENDONS)
        p['interval'] = interval
        sim = simulate_rnn_oscillator(p)
        loss = loss_function(sim, reference)
        losses.append(loss)
        l, = plt.plot(sim['end_effector'][0], sim['end_effector'][1])
        lines.append(l)

        print("### Current loss for iteration " + str(current_iteration[0]) + " is: " + str(loss))

        if current_iteration[0] % 5 == 0:
            print("Current optimal params are ")
            print(params)

        current_iteration[0] += 1

        temptime = datetime.now()
        print("Time passed since start: " + str(temptime - starttime))
        print("Average time per iteration: " + str((temptime - starttime) / current_iteration[0]))
        print()


    def after():
        plt.plot(reference['end_effector'][0], reference['end_effector'][1], label='reference')

        cmap0 = mpl.colors.LinearSegmentedColormap.from_list(
            'white2black', ['white', 'black'])
        # plot
        norm = mpl.colors.Normalize(vmin=0, vmax=len(losses))
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap0), fraction=.1)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Approximated trajectory after n number of iterations', rotation=270)
        title = "Convergence of the " + name + " after " + str(len(losses)) + " iterations"
        plt.title(title)
        plot_finger()

        grey = [0.9, 0.9, 0.9]
        offset = 0.9 / (len(lines) + 1)
        for l in lines:
            color = tuple(grey)
            l.set_color(color)
            grey[0] -= offset
            grey[1] -= offset
            grey[2] -= offset

        plt.savefig(title + ".png", dpi=244)
        plt.show()

        plt.plot(np.arange(0, len(losses), 1), losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss (RMSE)")
        ti = "Loss convergence for " + name
        plt.title(ti)
        plt.savefig(ti + ".png", dpi=244)
        plt.show()

        print("The losses convergence:")
        print(losses)

    # Params to take grad of.
    grad_params = [RNN_TAUS, RNN_BIASES, RNN_GAINS, RNN_STATES, RNN_WEIGHTS]

    gradbest = gradient_descent(loss_function, iterations, grad_params, init_params, callback=(callback, after))
    print("Gradient descent has finished.")

    print("The best solution seems to be:")
    print(gradbest)

    endtime = datetime.now()
    print("End time: " + str(starttime.strftime("_%d-%b-%Y_(%H:%M:%S.%f)")))
    print("Total time passed: " + str(endtime - starttime))
    print("Average time per iteration: " + str((endtime - starttime) / iterations))

    simulate_ctrnn_params_and_animate(gradbest, name, tendons)

