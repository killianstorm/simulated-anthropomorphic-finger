from simulation.optimizer import *
from simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 2., 0.1

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

init_params = {
    'interval': interval,
    RNN_TAUS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_BIAS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_STATES: num.random.rand(RNN_SIZE_TENDONS),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE_TENDONS * RNN_SIZE_TENDONS)
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

p_sine = {
    'interval': interval,
    'amplitudes': np.array([0., 0., 40., 23.]),
    'phases': np.array([1., 1., .21, 1.])
}

reference = simulate_sin(p_sine)
# p_constant = {
#     'F_fs': 25.,
#     'F_io': 0.,
#     'F_fp': 0.,
#     'F_ed': 40.,
#     'interval': interval
# }
# reference = simulate_constant(p_constant)
# print(reference['torques'])

# plots.animation(reference, dt, "tendons", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

p_predefined = {
    'interval': interval,
    'F_fs': reference['torques'][:, 0],
    'F_io': reference['torques'][:, 1],
    'F_fp': reference['torques'][:, 2],
    'F_ed': reference['torques'][:, 3],
}

reference = simulate_predefined(p_predefined)
plots.animation(reference, dt, "predefined", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()




# Params to take grad.
grad_params = [RNN_TAUS, RNN_BIAS, RNN_STATES, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAUS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_BIAS: num.random.rand(RNN_SIZE_TENDONS),
    RNN_STATES: num.random.rand(RNN_SIZE_TENDONS),
    RNN_WEIGHTS: num.random.rand(RNN_SIZE_TENDONS * RNN_SIZE_TENDONS)
}

grad_params = ['F_fs']
init_params = {
    'interval': interval,
    'reference': reference,
    'F_fs': 1.,
    'F_io': 0.,
    'F_fp': 0.,
    'F_ed': 40.,
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
