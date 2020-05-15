from simulation.optimizer import *
from simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 2., 0.0001

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)


### CTRNN
# init_params = {
#     'interval': interval,
#     RNN_TAU: num.random.rand(RNN_SIZE_TORQUES),
#     RNN_BIAS: num.random.rand(RNN_SIZE_TORQUES),
#     RNN_STATES: num.random.rand(RNN_SIZE_TORQUES),
#     RNN_WEIGHTS: num.random.rand(RNN_SIZE_TORQUES * RNN_SIZE_TORQUES)
# }

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

### SINE
p_sine = {
    'amplitudes': np.array([1., -1., -.5]),
    'phases': np.array([2.5, 5., 5.]),
    'interval': interval
}
reference = simulate_sin(p_sine)

### CONSTANT
# p_constant = {
#     'tau1': 2.,
#     'tau2': -1.,
#     'tau3': -.5,
#     'interval': interval
# }
# reference = simulate_constant(p_constant)


# plots.animation(reference, dt, "sine")

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

# Params to take grad.
grad_params = [RNN_TAUS, RNN_BIAS, RNN_STATES, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAUS: num.random.rand(RNN_SIZE_TORQUES),
    RNN_BIAS: num.random.rand(RNN_SIZE_TORQUES),
    RNN_STATES: num.random.rand(RNN_SIZE_TORQUES),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE_TORQUES * RNN_SIZE_TORQUES)
}

# grad_params = ['tau1', 'tau2', 'tau3']
# init_params = {
#     'interval': interval,
#     'reference': reference,
#     'tau1': .5,
#     'tau2': .5,
#     'tau3': .5,
# }

iterations = 2000
learning_rate = 0.1
print("GOGOGO")
gradbest = grad_oscillator(loss_end_effector, iterations, learning_rate, grad_params, init_params)
print(gradbest)

approximation = simulate_rnn_oscillator(gradbest)

loss = loss_end_effector(reference, approximation)
print("LOSS: ", loss)
# plots.animation(approximation, dt, "opt1_approx")
plots.animation(reference, dt, "opt1_approx_both", approximation)
