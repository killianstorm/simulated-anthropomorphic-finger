from finger_model.hopf import *
from finger_model.dynamic_model import *

import cma
import numpy as np
import matplotlib.pyplot as plt


# Interval.
tmax, dt = 3., 0.01
interval = np.arange(0, tmax + dt, dt)


# a, b, w_swing, w_stance, theta, mu, k, scale1, scale2, X0

# Lower and upper bounds.
lb = [1, 1, 2 * np.pi / 1.5, 2 * np.pi / 1.5, 1 * np.pi / 2, 10, 10, 1, 1]  # lower bounds of the tunable parameters
ub = [5, 5, 2 * np.pi / 0.8, 2 * np.pi / 0.8, 4 * np.pi / 2, 20, 20, 1, 1]


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
        grad_values = [func(reference, interval, *optimised_params) for func in grad_functions]

        # Perform gradient descent.
        optimised_params -= learning_rate * grad_values

        print("Approximations: " + str(optimised_params))


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
    print(best)
    print(best.x)


# a = self.params[0]
# b = self.params[1]
# w_swing = self.params[2]
# w_stance = self.params[3]
# theta = self.params[4]
# mu = self.params[13]
# k = self.params[14]
# X0 = self.X0_init
# X0[0] = self.params[11]
# X0[2] = self.params[12]


# Create reference.
params = [2, 2,                              # a, b
          2 * np.pi / 1.2, 2 * np.pi / 1.2,  # w_swing, w_stance
          np.pi,                             # theta
          15, 15,                            # scale1, scale2
          1, 1]                              # mu, k


# params = [1, 50, 2 * pi / 1.7, 2 * pi / 1, 0.7 * pi, 150, 150, 600, 600, 600, 600, 0, 0, 1, 1]
# X0 = [0.0, -1.0, 0.0, 1.0]
# a, b, w_swing, w_stance, theta, mu, k, scale1, scale2, X0

# Generate reference trajectory.
reference = simulate_oscillator(interval, *params)



# Learn parameters to reproduce reference.
CMA_ES_oscillator(reference)

# grad_oscillator(reference, )