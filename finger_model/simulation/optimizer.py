from jax import value_and_grad
from simulation.simulator import *
from datetime import datetime

from scipy.optimize import minimize

import matplotlib.pyplot as plt

def minimise_with_gradient_descent(loss, iterations, grad_params_names, init, callback=None):
    """
    Performs gradient descent.
    arguments:
        loss: the loss function
        iterations: the number of iterations
        learning_rate: the learning rate
        grad_params_names: the names of the params of which to optimise
        init: dict containing initial values for the parameters to optimise
    """

    size = RNN_SIZE_TORQUES
    if ENABLE_TENDONS:
        size = RNN_SIZE_TENDONS

    @jit
    def _loss_wrapper(p):
        """
        Loss function wrapper which simulates the trajectory and calculates the loss.
        """
        _reference = p['reference']
        _interval = p['interval']
        _simulated = simulate_ctrnn(p)
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

    def dict_to_array(d):
        total = np.concatenate([
            d[RNN_TAUS],
            d[RNN_BIASES],
            d[RNN_GAINS],
            d[RNN_STATES],
            d[RNN_WEIGHTS].ravel()
        ])
        return total

    def array_to_dict(a):
        d = {}
        d[RNN_TAUS] = a[:size]
        d[RNN_BIASES] = a[size:2 * size]
        d[RNN_GAINS] = a[2 * size:3 * size]
        d[RNN_STATES] = a[3 * size:4 * size]
        d[RNN_WEIGHTS] = a[4 * size:].reshape((size, size))
        return d

    def default_callback(params):
        print("Iteration done")
        print("Params: ")
        print(params)

    if callback is None:
        iteration_callback = default_callback
    else:
        iteration_callback = callback[0]

    def objective(params):
        p = array_to_dict(params)
        return _loss_wrapper({**p, **static_params})

    objective_with_grad = jit(value_and_grad(objective))

    result = minimize(
        objective_with_grad,
        dict_to_array(grad_params),
        jac=True,
        method='CG',
        options={
            'maxiter': iterations,
            'disp': True
        },
        callback=iteration_callback)

    if callback is not None:
        after = callback[1]
        after()

    best_params = array_to_dict(result.x)

    return {**best_params, **static_params}