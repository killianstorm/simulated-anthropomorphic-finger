from hopf import *
from dynamic_model import *

from moviepy.editor import ImageSequenceClip
from datetime import datetime
from tqdm import tqdm

import cma
import numpy as num
import jax.numpy as np
import matplotlib.pyplot as plt

# Interval.
tmax, dt = 20., 0.01
refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)


# Lower and upper bounds.
lb = [1, 1, 2 * num.pi / 1.5, 2 * num.pi / 1.5, 1 * num.pi / 2, 0, 0, 1, 1]  # lower bounds of the tunable parameters
ub = [5, 5, 2 * num.pi / 0.8, 2 * num.pi / 0.8, 4 * num.pi / 2, 200, 200, 1, 1]

# TODO do grad descent on scale params
# create animations while optimising


def fig2image(fig):
    fig.canvas.draw()
    data = num.fromstring(fig.canvas.tostring_rgb(), dtype=num.uint8, sep='')
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def animation(history, name):
    key_points = history['positions']


    fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
    ax = fig.add_subplot(111)

    images = []
    di = 10
    N = key_points.shape[1]
    for i in tqdm(range(0, N, di)):
        plt.cla()
        plt.plot(key_points[:4, i], key_points[4:, i], marker='.')
        plt.scatter(history['end_effector'][0][:i], history['end_effector'][1][:i], s=0.5)
        plt.axhline(0)
        plt.axis('equal')
        plt.axis([-1, 4, -1, 2])
        images.append(fig2image(fig))
    ImageSequenceClip(images, fps=int(1/dt/di)).write_videofile(str(name) + datetime.now().strftime("_%d-%b-%Y_(%H:%M:%S.%f)") + ".mp4")


def denormalize(normalized):
    return num.multiply((num.array(ub) - num.array(lb)), normalized) + num.array(lb)


def grad_oscillator(reference, loss, iterations, learning_rate, *init):

    optimised_params = np.array(init)

    @jit
    def _loss_wrapper(_reference, _interval, *_params):
        _simulated = simulate_oscillator(_interval, *_params)
        return loss(_reference, _simulated)

    grad_params = [SCALE1]
    static_params = [i for i in PARAMS if i not in grad_params]


    # Create list of grad functions for each parameter of the loss wrapper.
    grad_functions = [jit(grad(_loss_wrapper, i + 2)) if i in grad_params else None for i in PARAMS]

    for i in range(iterations):

        # Calculate gradients
        grad_values = np.array([func(reference, interval, *optimised_params) if func is not None else 0. for func in grad_functions])

        # Perform gradient descent.
        optimised_params -= learning_rate * grad_values

        print("Iteration " + str(i) + " " + str(optimised_params))


def CMA_ES_oscillator(reference):

    all_losses = []
    N = N_OF_PARAMS - 2 # omit mu and k
    pop_size = 4 + num.floor(3 * log(N))
    best = cma.optimization_tools.BestSolution()

    es = cma.CMAEvolutionStrategy(N * [0.5], 0.5, {'bounds': [0, 1], 'popsize': pop_size, 'maxiter': 50, 'verb_append': best.evalsall})
    logger = cma.CMADataLogger().register(es, append = best.evalsall)

    while not es.stop():
        solutions = es.ask()
        loss_vect = []
        for i in range(len(solutions)):

            # Add constant values for mu and k.
            norm = num.concatenate([solutions[i], [0, 0]])

            # Denormalize solution from [0, 1] to [lb, ub].
            try_params = denormalize(norm)

            # Simulate the parameters.
            simulated = simulate_oscillator(interval, *try_params)

            # Calculate and store loss.
            loss = num.sqrt(((reference['end_effector'] - simulated['end_effector']) ** 2).mean())
            loss_vect.append(loss)

            # Plot reference and simulated.
            plt.plot(reference['end_position'][:4], reference['end_position'][4:])
            plt.scatter(reference['end_effector'][0], reference['end_effector'][1])
            plt.scatter(simulated['end_effector'][0], simulated['end_effector'][1])
            plt.title("Loss: " + str(loss))
            plt.show()

        # Feedback losses.
        es.tell(solutions, loss_vect)
        logger.add()
        es.disp()
        all_losses += loss_vect

    # Plot loss convergence.
    plt.plot(num.arange(len(all_losses)), all_losses)
    plt.title("Params ")
    plt.show()

    # Store best solution.
    best.update(es.best)
    cma.plot()
    print("Optimal solution:")
    print(best)
    print(best.x)


# Create reference.
params = [2, 2,  # a, b
          1.5 * num.pi, num.pi / 4,  # w_swing, w_stance
          num.pi,  # theta
          1, 0.5,  # scale1, scale2
          1, 1]                              # mu, k
reference = simulate_oscillator(interval, *params)
# animation(reference, "reference")

init_params = params
init_params[SCALE1] = 0.1
init_params[SCALE2] = 0.1

# Learn parameters to reproduce reference.
# CMA_ES_oscillator(reference)
grad_oscillator(reference, loss_end_effector, 50, 0.1, *params)