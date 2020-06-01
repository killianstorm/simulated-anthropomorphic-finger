from finger_model.simulation.optimizer import *
from finger_model.simulation.loss_functions import *

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 1., 0.0001

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
reference = simulate_sine(p_sine)

plt.cla()
plt.plot(interval, reference['torques'][:, 0], label="MCP")
plt.plot(interval, reference['torques'][:, 1], label="PIP")
plt.plot(interval, reference['torques'][:, 2], label="DIP")
plt.title("Reference torques")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend(loc="upper left")
plt.show()

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
grad_params = [RNN_TAUS, RNN_BIASES, RNN_STATES, RNN_GAINS, RNN_WEIGHTS]
init_params = {
    'interval': interval,
    'reference': reference,
    RNN_TAUS: num.random.rand(RNN_SIZE_TORQUES),
    RNN_BIASES: num.random.rand(RNN_SIZE_TORQUES),
    RNN_STATES: num.random.rand(RNN_SIZE_TORQUES),
    RNN_GAINS: num.random.rand(RNN_SIZE_TORQUES),
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

iterations = 100000
learning_rate = 1.
print("GOGOGO")
# gradbest = grad_oscillator(loss_angles, iterations, learning_rate, grad_params, init_params)
gradbest = {
    'interval': interval,
    RNN_TAUS: np.array([0.09893328, 0.64548118, 1.93432572]),
    RNN_BIASES: np.array([-0.95394665, -0.35918627, -2.96284246]),
    RNN_STATES: np.array([-1.04861539, -0.23063669,  5.4683151 ]),
    RNN_GAINS: np.array([-1.76394312, -0.5693699 , -3.07278049]),
    RNN_WEIGHTS: np.array([ 1.20546856,  0.15172386, -1.95157362,  0.20661285, 0.29229113, -1.86179027, -3.37062968, -3.42195594, -2.24576479])
}
# reference = simulate_rnn_oscillator(test_params)

print(gradbest)

approximation = simulate_rnn_oscillator(gradbest)

plt.cla()
plt.plot(interval, approximation['torques'][:, 0], label="MCP")
plt.plot(interval, approximation['torques'][:, 1], label="PIP")
plt.plot(interval, approximation['torques'][:, 2], label="DIP")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.title("Predicted torques")
plt.legend(loc="upper left")
plt.show()

loss = loss_end_effector(reference, approximation)
print("LOSS: ", loss)
# plots.animation(approximation, dt, "opt1_approx")
# plots.animation(reference, dt, "predicted_both", approximation)
