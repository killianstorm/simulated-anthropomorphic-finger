import jax.numpy as np
import matplotlib.pyplot as plt

RNN_SIZE = 4
step_size = 0.01

RNN_TAU = 'rnn_tau'
RNN_BIAS = 'rnn_bias'
RNN_STATES = 'rnn_states'
RNN_WEIGHTS = 'rnn_weights'
# RNN_PARAMS = [RNN_TAU1, RNN_TAU2, RNN_TAU3, RNN_TAU4, RNN_BIAS1, RNN_BIAS2, RNN_BIAS3, RNN_BIAS4]


def ctrnn(interval, rnn_params, plot=False):
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

    # plot oscillator output
    if plot:
        plt.plot(np.arange(0, interval.shape[0], step_size), outputs[:, 0])
        plt.plot(np.arange(0, interval.shape[0], step_size), outputs[:, 1])
        plt.xlabel('Time')
        plt.ylabel('Neuron outputs')
        plt.show()

    # out = np.zeros((interval.shape[0], 3))
    # states_final, _, outputs_final = jax.lax.scan(step, (states, 0., outputs), interval)
    return np.array(out)