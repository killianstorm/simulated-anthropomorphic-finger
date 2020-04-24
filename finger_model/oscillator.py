from finger_model.hopf import *
from finger_model.dynamic_model import *

import cma
import numpy as np
import matplotlib.pyplot as plt


# Interval.
tmax, dt = 5., 0.01
interval = np.arange(0, tmax + dt, dt)


# Lower and upper bounds.
lb = [1, 1, 2 * np.pi / 1.5, 2 * np.pi / 1.5, 1 * np.pi / 2, 0, 0, 1, 1]  # lower bounds of the tunable parameters
ub = [5, 5, 2 * np.pi / 0.8, 2 * np.pi / 0.8, 4 * np.pi / 2, 200, 200, 1, 1]


def denormalize(normalized):
    return np.multiply((np.array(ub) - np.array(lb)), normalized) + np.array(lb)


def grad_oscillator(reference, loss, iterations, learning_rate, *init):

    optimised_params = np.array(init)

    @jit
    def _loss_wrapper(_reference, _interval, *_params):
        _simulated = simulate_oscillator(_interval, *_params)
        return loss(_reference, _simulated)

    # Create list of grad functions for each parameter of the loss wrapper.
    grad_functions = [jit(grad(_loss_wrapper, i + 2)) for i in range(N_OF_PARAMS - 2)]

    for i in range(iterations):

        # Calculate gradients
        grad_values = np.array([func(reference, interval, *optimised_params) for func in grad_functions])

        # Perform gradient descent.
        optimised_params[:7] -= learning_rate * grad_values

        print("Iteration " + str(i) + " " + str(optimised_params))


def CMA_ES_oscillator(reference):

    all_losses = []
    N = N_OF_PARAMS - 2 # omit mu and k
    pop_size = 4 + np.floor(3 * log(N))
    best = cma.optimization_tools.BestSolution()

    es = cma.CMAEvolutionStrategy(N * [0.5], 0.5, {'bounds': [0, 1], 'popsize': pop_size, 'maxiter': 50, 'verb_append': best.evalsall})
    logger = cma.CMADataLogger().register(es, append = best.evalsall)

    while not es.stop():
        solutions = es.ask()
        loss_vect = []
        for i in range(len(solutions)):

            # Add constant values for mu and k.
            norm = np.concatenate([solutions[i], [0, 0]])

            # Denormalize solution from [0, 1] to [lb, ub].
            try_params = denormalize(norm)

            # Simulate the parameters.
            simulated = simulate_oscillator(interval, *try_params)

            # Calculate and store loss.
            loss = np.sqrt(((reference['end_effector'] - simulated['end_effector']) ** 2).mean())
            loss_vect.append(loss)

            # Plot reference and simulated.
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
    plt.plot(np.arange(len(all_losses)), all_losses)
    plt.title("Params ")
    plt.show()

    # Store best solution.
    best.update(es.best)
    cma.plot()
    print("Optimal solution:")
    print(best)
    print(best.x)


# Create reference.
params = [2, 2,                              # a, b
          2 * np.pi / 1.2, 2 * np.pi / 1.2,  # w_swing, w_stance
          np.pi,                             # theta
          100, 100,                          # scale1, scale2
          1, 1]                              # mu, k
reference = simulate_oscillator(interval, *params)



# Learn parameters to reproduce reference.
CMA_ES_oscillator(reference)

# grad_oscillator(reference, loss_end_effector, 2000, 0.01, *(1.9, 1.9, 2*np.pi, 2*np.pi, np.pi, 12, 12, 1, 1))