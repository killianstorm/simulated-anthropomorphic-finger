import numpy as num
from oscillator import *
import jax.numpy as np
import matplotlib.pyplot as plt
import plots

# Interval.
tmax, dt = 5., 0.01
refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

init_params = {
    'interval': interval,
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

# cpg = ctrnn(interval, *init_params)
# plt.plot(interval, cpg[:, 0])
# plt.plot(interval, cpg[:, 1])
# plt.plot(interval, cpg[:, 2])
# plt.title("CPG")
# plt.show()

# Create reference.
# reference = simulate_rnn_oscillator(init_params)
reference = simulate_sin(interval, 2., 1.5, 1.)
# plots.animation(reference, dt, "opt1")

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

gradbest = {
    'interval': interval,
    'reference': reference,
    RNN_TAU1: 0.76543037,
    RNN_TAU2: 0.86420127,
    RNN_TAU3: 0.9167606,
    RNN_BIAS1: -1.3312947,
    RNN_BIAS2: -1.53163464,
    RNN_BIAS3: 0.38569354,

    RNN_WEIGHTS: np.array([0.37403231, 0.38423551, 0.35306435, 0.29270996, 0.30891777,
                             0.26351807, 0.2352118, 0.25596476, 0.18884815])
}

# {'rnn_tau1': DeviceArray(0.76543037, dtype=float64), 'rnn_tau2': DeviceArray(0.86420127, dtype=float64),
#  'rnn_tau3': DeviceArray(0.9167606, dtype=float64), 'rnn_bias1': DeviceArray(-1.3312947, dtype=float64),
#  'rnn_bias2': DeviceArray(-1.53163464, dtype=float64), 'rnn_bias3': DeviceArray(0.38569354, dtype=float64),
#  'rnn_weights': DeviceArray([0.37403231, 0.38423551, 0.35306435, 0.29270996, 0.30891777,
#                              0.26351807, 0.2352118, 0.25596476, 0.18884815], dtype=float64)}
# ITERATION
# 40000
# LOSS: 0.14016753553257194
approximation = simulate_rnn_oscillator(gradbest)

loss = loss_end_effector(reference, approximation)
print("LOSS: ", loss)
# plots.animation(approximation, dt, "opt1_approx")
plots.animation(reference, dt, "opt1_approx_both", approximation)
#
# {'rnn_tau1': DeviceArray(0.76780443, dtype=float64), 'rnn_tau2': DeviceArray(0.86653094, dtype=float64), 'rnn_tau3': DeviceArray(0.91994624, dtype=float64), 'rnn_bias1': DeviceArray(-1.35703084, dtype=float64), 'rnn_bias2': DeviceArray(-1.54789137, dtype=float64), 'rnn_bias3': DeviceArray(0.38924751, dtype=float64), 'rnn_weights': DeviceArray([0.37260763, 0.38287166, 0.35086563, 0.29066813, 0.30696377,
#              0.26035898, 0.23136951, 0.25228856, 0.18289266],            dtype=float64)}
# ITERATION  41000  LOSS:  0.14006244041969945
# ITERATION  41100  LOSS:  0.1400522060424578