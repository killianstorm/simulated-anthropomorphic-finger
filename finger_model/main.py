from simulation.optimizer import *
from simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 1., 0.001

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

init_params = {
    'interval': interval,
    RNN_TAU: num.random.rand(RNN_SIZE),
    RNN_BIAS: num.random.rand(RNN_SIZE),
    RNN_STATES: num.random.rand(RNN_SIZE),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE * RNN_SIZE)
}

# cpg = ctrnn(interval, init_params, True)

# Create reference.
# reference = simulate_rnn_oscillator(init_params)
#
# plt.cla()
# plt.plot(interval, reference['torques'][:, 0])
# plt.plot(interval, reference['torques'][:, 1])
# plt.plot(interval, reference['torques'][:, 2])
# plt.plot(interval, reference['torques'][:, 3])
# plt.title("Reference torques")
# plt.show()

reference = simulate_sin(interval, 10., 4., 15., 20.)
# reference = simulate_constant(interval, 0., 0., 40., 40.)
# plots.animation(reference, dt, "tendons", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

# Params to take grad.
grad_params = [RNN_TAU, RNN_BIAS, RNN_STATES, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAU: num.random.rand(RNN_SIZE),
    RNN_BIAS: num.random.rand(RNN_SIZE),
    RNN_STATES: num.random.rand(RNN_SIZE),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE * RNN_SIZE)
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
