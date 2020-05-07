import numpy as num
from oscillator import *
import jax.numpy as np
import matplotlib.pyplot as plt
import plots

# Interval.
tmax, dt = 5., 0.001

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

init_params = {
    'interval': interval,
    RNN_TAU1: 1.,
    RNN_TAU2: 1.,
    RNN_TAU3: 1.,
    RNN_BIAS1: 1.,
    RNN_BIAS2: 1.,
    RNN_BIAS3: 1.,

    RNN_WEIGHTS: np.array([1., 1., 1.,  # Weights
                           1., 1., 1.,
                           1., 1., 1.])
}

# cpg = ctrnn(interval, *init_params)
# plt.plot(interval, cpg[:, 0])
# plt.plot(interval, cpg[:, 1])
# plt.plot(interval, cpg[:, 2])
# plt.title("CPG")
# plt.show()

# Create reference.
reference = simulate_rnn_oscillator(init_params)
# reference = simulate_sin(interval, 2., 1.5, 1.)
# plots.animation(reference, dt, "opt1")

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

# Params to take grad.
grad_params = [RNN_TAU1, RNN_TAU2, RNN_TAU3, RNN_BIAS1, RNN_BIAS2, RNN_BIAS3, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAU1: .5,
    RNN_TAU2: .5,
    RNN_TAU3: .5,
    RNN_BIAS1: .5,
    RNN_BIAS2: .5,
    RNN_BIAS3: .5,

    RNN_WEIGHTS: np.array([.5, .5, .5,  # Weights
                           .5, .5, .5,
                           .5, .5, .5])
}

# init_params = [.5, .5, .5,  # Taus
#                .5, .5, .5,  # Biases
#
#                .5, .5, .5,  # Weights
#                .5, .5, .5,
#                .5, .5, .5]

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
