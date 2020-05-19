from simulation.optimizer import *
from simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 1., 0.0001

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

p_keystroke = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array([17. if i < interval[-1] / 2 else 0. for i in interval]),
    'F_ed': np.array([0. if i < interval[-1] / 2 else 30. for i in interval])
}
reference = simulate_predefined(p_keystroke)

# plots.animation(reference, dt, "keystroke", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

# Params to take grad.
grad_params = [RNN_TAUS, RNN_BIAS, RNN_GAINS, RNN_STATES, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAUS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_BIAS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_GAINS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_STATES: num.random.rand(RNN_SIZE_TENDONS),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE_TENDONS * RNN_SIZE_TENDONS)
}

iterations = 100000
learning_rate = 1.
print("GOGOGO")
gradbest = grad_oscillator(loss_angles, iterations, learning_rate, grad_params, init_params)
print(gradbest)

approximation = simulate_rnn_oscillator(gradbest)

loss = loss_end_effector(reference, approximation)
print("LOSS: ", loss)
# plots.animation(approximation, dt, "opt1_approx")
plots.animation(reference, dt, "opt1_approx_both", approximation)
