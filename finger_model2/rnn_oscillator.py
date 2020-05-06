import jax.numpy as np
import jax.ops
import jax.lax
import matplotlib.pyplot as plt
# importing the CTRNN class
from CTRNN import CTRNN

# params
RNN_SIZE = 3
step_size = 0.01

RNN_TAU1 = 'rnn_tau1'
RNN_TAU2 = 'rnn_tau2'
RNN_TAU3 = 'rnn_tau3'
RNN_BIAS1 = 'rnn_bias1'
RNN_BIAS2 = 'rnn_bias2'
RNN_BIAS3 = 'rnn_bias3'
RNN_WEIGHTS = 'rnn_weights'
RNN_PARAMS = [RNN_TAU1, RNN_TAU2, RNN_TAU3, RNN_BIAS1, RNN_BIAS2, RNN_BIAS3]
RNN_ALL_PARAMS = [i for i in range(2 * RNN_SIZE + RNN_SIZE * RNN_SIZE)]


def ctrnn(interval, *rnn_params):
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    step_size = interval[1] - interval[0]

    states = np.array([0, 0, 0])
    outputs = sigmoid(states)
    gains = np.ones(RNN_SIZE)

    # Set up network.
    taus = np.array([rnn_params[RNN_TAU1], rnn_params[RNN_TAU2], rnn_params[RNN_TAU3]])
    biases = np.array([rnn_params[RNN_BIAS1], rnn_params[RNN_BIAS2], rnn_params[RNN_BIAS3]])
    weights = np.array(rnn_params[RNN_WEIGHTS:]).reshape(RNN_SIZE, RNN_SIZE)

    # def step(state, t):
    #     _states, _index, _outputs = state
    #
    #     external_inputs = np.zeros(RNN_SIZE)  # zero external_inputs
    #     total_inputs = external_inputs + weights.dot(_outputs)
    #     _states += step_size * (1 / taus) * (total_inputs - _states)
    #     _current = np.array(sigmoid(gains * (_states + biases)))
    #     jax.ops.index_update(_outputs, _index.astype(int), _outputs[0])
    #     jax.ops.index_update(_outputs, _index.astype(int), _outputs[0])
    #     jax.ops.index_update(_outputs, _index.astype(int), _outputs[0])
    #
    #     return _states, _index + 3, _outputs

    # initialize network
    #network.randomize_outputs(0.1, 0.2)

    out = []

    # simulate network
    for i in interval:
        external_inputs = np.zeros(RNN_SIZE)  # zero external_inputs
        total_inputs = external_inputs + weights.dot(outputs)
        states += step_size * (1 / taus) * (total_inputs - states)
        outputs = np.array(sigmoid(gains * (states + biases)))
        out.append(outputs)

    # out = np.zeros((interval.shape[0], 3))
    # states_final, _, outputs_final = jax.lax.scan(step, (states, 0., outputs), interval)
    return np.array(out)


# # plot oscillator output
# plt.plot(np.arange(0, run_duration, step_size), outputs[:, 0])
# plt.plot(np.arange(0, run_duration, step_size), outputs[:, 1])
# plt.xlabel('Time')
# plt.ylabel('Neuron outputs')
# plt.show()