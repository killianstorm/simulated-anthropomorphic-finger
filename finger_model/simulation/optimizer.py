from jax import value_and_grad
from simulation.simulator import *
from datetime import datetime

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

    starttime = datetime.now()
    print("Current time is " + str(starttime.strftime("_%d-%b-%Y_(%H:%M:%S.%f)")))

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

        print("########### ITERATION ", i, " LOSS: ", vals)

        # Every 5th iteration, print current grads, momentum and optimised parameters.
        if i % 5 == 0:
            print("GRADS: ", grads, " MOMENTUM: ", momentum)

            print("## Current optimal parameters")
            print(grad_params)
            print("##")

        # Update parameters and momentum.
        for key in grads:
            momentum[key] = beta * momentum[key] + (1 - beta) * grads[key]
            grad_params[key] -= learning_rate * momentum[key]

        nexttime = datetime.now()

        if i == 0:
            print("Compiling took (in seconds): " + str(nexttime.timestamp() - starttime.timestamp()))
            starttime = datetime.now()
        else:
            print("Current time: " + str(nexttime.strftime("_%d-%b-%Y_(%H:%M:%S.%f)")))
            print("Time passed: " + str(nexttime - starttime))
            print("Average seconds per iteration: " + str((nexttime.timestamp() - starttime.timestamp()) / i))

        print ("###########\n")

    return {**grad_params, **static_params}
