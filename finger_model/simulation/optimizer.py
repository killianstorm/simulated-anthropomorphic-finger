from jax import value_and_grad
from simulation.simulator import *


def gradient_descent(loss, iterations, learning_rate, grad_params_names, init):
    """
    Performs gradient descent.
    arguments:
        loss: the loss function
        iterations: the number of iterations
        learning_rate: the learning rate
        grad_params_names: the names of the params of which to optimise
        init: dict containing initial values for the parameters to optimise
    """

    print("START GRAD")

    @jit
    def _loss_wrapper(p):
        """
        Loss function wrapper which simulates the trajectory and calculates the loss.
        """
        _reference = p['reference']
        _interval = p['interval']
        _simulated = simulate_rnn_oscillator(p)
        return loss(_reference, _simulated)

    # Create grad function.
    grad_function = jit(value_and_grad(lambda gradient_params, static_params: _loss_wrapper({**gradient_params, **static_params})))

    # Dict with parameters to take grad of.
    grad_params = {}

    # Dict with static parameters.
    static_params = {}

    # Dict with momentum values.
    momentum = {}

    # Sort parameters into grad or static dicts.
    for key in init:
        if key in grad_params_names:
            grad_params[key] = init[key]
        else:
            static_params[key] = init[key]
        momentum[key] = 0.

    # Beta value for momentum
    beta = 0.9

    # Perform gradient descent.
    for i in range(iterations):

        # Take grad and get value.
        vals, grads = grad_function(grad_params, static_params)

        print("ITERATION ", i, " LOSS: ", vals)

        # Every 10th iteration, print current grads, momentum and optimised parameters.
        if i % 10 == 0:
            print("GRADS: ", grads, " MOMENTUM: ", momentum)

            print("## Current optimal parameters")
            print(grad_params)
            print("##")

        # Update parameters and momentum.
        for key in grads:
            momentum[key] = beta * momentum[key] + (1 - beta) * grads[key]
            grad_params[key] -= learning_rate * momentum[key]

    return {**grad_params, **static_params}
