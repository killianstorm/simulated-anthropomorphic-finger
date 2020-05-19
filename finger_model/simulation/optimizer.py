from jax import value_and_grad, jit
from simulation.simulator import *

import cma
import numpy as num
import matplotlib.pyplot as plt


def grad_oscillator(loss, iterations, learning_rate, grad_params_names, init):

    print("START GRAD")

    @jit
    def _loss_wrapper(p):
        _reference = p['reference']
        _interval = p['interval']
        _simulated = simulate_rnn_oscillator(p)
        return loss(_reference, _simulated)

    # Create list of grad functions for each parameter of the loss wrapper.
    grad_functions = jit(value_and_grad(lambda gradient_params, static_params: _loss_wrapper({**gradient_params, **static_params})))

    grad_params = {}
    static_params = {}
    momentum = {}

    for key in init:
        if key in grad_params_names:
            grad_params[key] = init[key]
        else:
            static_params[key] = init[key]
        momentum[key] = 0.

    beta = 0.9

    for i in range(iterations):
        vals, grads = grad_functions(grad_params, static_params)

        print("ITERATION ", i, " LOSS: ", vals)

        if i % 10 == 0:
            print("GRADS: ", grads, " MOMENTUM: ", momentum)

            print("## PARAMS")
            print(grad_params)
            print("##")

        for key in grads:
            momentum[key] = beta * momentum[key] + (1 - beta) * grads[key]
            grad_params[key] -= learning_rate * momentum[key]

    return {**grad_params, **static_params}


def CMA_ES_oscillator(reference, interval, lb, ub, iterations):


    def denormalize(normalized):
        return num.multiply((num.array(ub) - num.array(lb)), normalized) + num.array(lb)

    all_losses = []
    N = N_OF_PARAMS - 2 # omit mu and k
    pop_size = 4 + num.floor(3 * log(N))
    best = cma.optimization_tools.BestSolution()

    es = cma.CMAEvolutionStrategy(N * [0.5], 0.5, {'bounds': [0, 1], 'popsize': pop_size, 'maxiter': iterations, 'verb_append': best.evalsall})
    logger = cma.CMADataLogger().register(es, append = best.evalsall)

    while not es.stop():
        solutions = es.ask()
        loss_vect = []
        for i in range(len(solutions)):

            # Add constant values for mu and k.
            norm = num.concatenate([solutions[i], [0, 0]])

            # Denormalize solution from [0, 1] to [lb, ub].
            try_params = denormalize(norm)
            print(try_params)

            # Simulate the parameters.
            simulated = simulate_hopf(interval, *try_params)

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
    return best