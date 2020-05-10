from simulation.optimizer import *
from simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

# Interval.
tmax, dt = 10., 0.1

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

init_params = {
    'interval': interval,
    RNN_TAU1: 1.,
    RNN_TAU2: 1.,
    RNN_TAU3: 1.,
    RNN_TAU4: 1.,
    RNN_BIAS1: 1.,
    RNN_BIAS2: 1.,
    RNN_BIAS3: 1.,
    RNN_BIAS4: 1.,

    RNN_WEIGHTS: np.array([1., 1., 1., 1.,  # Weights
                           1., 1., 1., 1.,
                           1., 1., 1., 1.,
                           1., 1., 1., 1.])
}

# cpg = ctrnn(interval, init_params, True)

# Create reference.
# reference = simulate_rnn_oscillator(init_params)
# reference = simulate_sin(interval, 10., 4., 15., 20.)
reference = simulate_constant(interval, 0., 0., 40., 40.)
plots.animation(reference, dt, "tendons", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

# Params to take grad.
grad_params = [RNN_TAU1, RNN_TAU2, RNN_TAU3, RNN_TAU4, RNN_BIAS1, RNN_BIAS2, RNN_BIAS3, RNN_BIAS4, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAU1: 1.,
    RNN_TAU2: 1.,
    RNN_TAU3: 1.,
    RNN_TAU4: 1.,
    RNN_BIAS1: 1.,
    RNN_BIAS2: 1.,
    RNN_BIAS3: 1.,
    RNN_BIAS4: 1.,

    RNN_WEIGHTS: np.array([1., 1., 1., 1.,  # Weights
                           1., 1., 1., 1.,
                           1., 1., 1., 1.,
                           1., 1., 1., 1.])
}


iterations = 10000
learning_rate = 0.1
print("GOGOGO")
gradbest = grad_oscillator(loss_end_effector, iterations, learning_rate, grad_params, init_params)
print(gradbest)

approximation = simulate_rnn_oscillator(gradbest)

loss = loss_end_effector(reference, approximation)
print("LOSS: ", loss)
# plots.animation(approximation, dt, "opt1_approx")
plots.animation(reference, dt, "opt1_approx_both", approximation)
