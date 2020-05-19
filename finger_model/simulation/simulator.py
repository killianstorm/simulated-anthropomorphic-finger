from cpg.hopf import *
from cpg.rnn_oscillator import *
from simulation.dynamic_model import *
from jax import jit
import jax
import jax.ops
import jax.lax

steps = 50

# MM, FM = finger_dynamic_model()

# tendon_force_range = np.array([0., 40.])

RADII = {
    J_DIP: {
        T_FP: 0.010 / 2.,
        T_ED: 0.010 / 2.
    },
    J_PIP: {
        T_FS: 0.035 / 2.,
        T_IO: 0.020 / 2.,
        T_FP: 0.042 / 2.,
        T_ED: 0.020 / 2.
    },
    J_MCP: {
        T_FS: 0.050 / 2.,
        T_IO: 0.025 / 2.,
        T_FP: 0.044 / 2.,
        T_ED: 0.044 / 2.
    }
}


@jit
def simulate_sin(p):
    amplitudes, phases, interval = p['amplitudes'], p['phases'], p['interval']

    @jit
    def ode(y, time, _amplitudes, _phases):
        _inputs = np.array([_a * np.sin(_p * time) for _a, _p in zip(_amplitudes, _phases)])

        if ENABLE_TENDONS:
            _inputs = np.abs(_inputs)

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_inputs,
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, amplitudes, phases)
    results = _calculate_positions(history, interval[1] - interval[2])

    results['torques'] = np.array([_a * np.sin(_p * interval) for _a, _p in zip(amplitudes, phases)]).transpose()

    if ENABLE_TENDONS:
        results['torques'] = np.abs(results['torques'])

    return results


@jit
def simulate_sin_RK4(p):
    amplitudes, phases, interval = p['amplitudes'], p['phases'], p['interval']

    @jit
    def ode(y, time, _amplitudes, _phases):
        _inputs = np.array([_a * np.sin(_p * time) for _a, _p in zip(_amplitudes, _phases)])

        if ENABLE_TENDONS:
            _inputs = np.abs(_inputs)

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_inputs,
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    dt = interval[1] - interval[0]

    def step(current_state, index):
        _amplitudes, _phases, _xi = current_state
        k1 = dt * ode(_xi, index, amplitudes, phases).flatten()
        k2 = dt * ode(_xi + .5 * k1, index + .5 * dt, amplitudes, phases).flatten()
        k3 = dt * ode(_xi + .5 * k2, index + .5 * dt, amplitudes, phases).flatten()
        k4 = dt * ode(_xi + k3, index + dt, amplitudes, phases).flatten()
        _out = _xi + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return (_amplitudes, _phases, _out), _out

    _, history = jax.lax.scan(step, (amplitudes, phases, initial_positions), p['interval'])

    # history = odeint_jax(ode, initial_positions, interval, amplitudes, phases)
    results = _calculate_positions(history, interval[1] - interval[2])

    results['torques'] = np.array([_a * np.sin(_p * interval) for _a, _p in zip(amplitudes, phases)]).transpose()

    if ENABLE_TENDONS:
        results['torques'] = np.abs(results['torques'])

    return results



@jit
def simulate_rnn_oscillator(p):
    if ENABLE_TENDONS:
        size = RNN_SIZE_TENDONS
        max_val = MAX_FORCE_TENDONS
    else:
        size = RNN_SIZE_TORQUES
        max_val = MAX_TORQUE

    dt = p['interval'][1] - p['interval'][0]

    # Set up network.
    taus = p[RNN_TAUS]
    biases = p[RNN_BIAS]
    weights = np.array(p[RNN_WEIGHTS]).reshape(size, size)
    states = p[RNN_STATES]
    gains = p[RNN_GAINS]
    out = jax.nn.sigmoid(states)

    def step(current_state, index):
        _output, _state = current_state
        external_inputs = np.zeros(size)  # zero external_inputs
        total_inputs = external_inputs + np.dot(weights, _output)
        _state += dt * (1 / taus) * (total_inputs - _state)
        _out = jax.nn.sigmoid(gains * (_state + biases))
        return (_out, _state), _out

    _, outputs = jax.lax.scan(step, (out, states), p['interval'])

    if ENABLE_TENDONS:
        outputs = np.multiply(max_val, outputs)
    else:
        outputs = np.multiply(max_val, outputs) - max_val / 2


    @jit
    def ode(y, time, _dt, _torques):

        _index = time / _dt
        _out = _torques[_index.astype(int)]

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_out,
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, p['interval'], dt, outputs)

    results = _calculate_positions(history, dt)
    results['torques'] = outputs

    return results


@jit
def simulate_constant(p):
    if ENABLE_TENDONS:
        inputs, interval = (p['F_fs'], p['F_io'], p['F_fp'], p['F_ed']), p['interval']
        size = RNN_SIZE_TENDONS
    else:
        inputs, interval = (p['tau1'], p['tau2'], p['tau3']), p['interval']
        size = RNN_SIZE_TORQUES

    @jit
    def ode(y, time, _inputs):
        params = [y,
                  lengths[0], masses[0], inertias[0],
                  lengths[1], masses[1], inertias[1],
                  lengths[2], masses[2], inertias[2],
                  *_inputs,
                  9.8, 0.5]
        return np.linalg.inv(MM(*params)) @ FM(*params)

    history = odeint_jax(ode, initial_positions, interval, inputs)

    results = _calculate_positions(history, interval[1] - interval[0])

    results['torques'] = np.array([interval.shape[0] * [i] for i in inputs]).transpose()

    return results


@jit
def simulate_predefined(p):
    if ENABLE_TENDONS:
        inputs, interval = (p['F_fs'], p['F_io'], p['F_fp'], p['F_ed']), p['interval']
    else:
        inputs, interval = (p['tau1'], p['tau2'], p['tau3']), p['interval']

    inputs = np.array([*inputs]).transpose()

    dt = interval[1] - interval[0]

    @jit
    def ode(y, time, _dt, _torques):

        _index = time / _dt
        _out = _torques[_index.astype(int)]

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_out,
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, dt, inputs)
    results = _calculate_positions(history, interval[1] - interval[0])
    results['torques'] = inputs

    return results


def _calculate_positions(history, dt):
    x_1 = lengths[0] * np.sin(history[:, 0])
    y_1 = - lengths[0] * np.cos(history[:, 0])

    x_2 = x_1 + lengths[1] * np.sin(history[:, 1])
    y_2 = y_1 - lengths[1] * np.cos(history[:, 1])

    x_3 = x_2 + lengths[2] * np.sin(history[:, 2])
    y_3 = y_2 - lengths[2] * np.cos(history[:, 2])

    end_effector = np.array([x_3, y_3])
    positions = np.array([np.zeros(len(x_1)), x_1, x_2, x_3, np.zeros(len(x_1)), y_1, y_2, y_3])
    velocities = np.array([history[:, 3], history[:, 4], history[:, 5]])
    accelerations = np.array(
        [np.gradient(velocities[0], dt), np.gradient(velocities[1], dt), np.gradient(velocities[2], dt)])
    # accelerations = np.array([np.array([np.roll(velocities[0], -1) - velocities[0]])[:-1] / dt,
    #                           np.array([np.roll(velocities[1], -1) - velocities[1]])[:-1] / dt,
    #                           np.array([np.roll(velocities[2], -1) - velocities[2]])[:-1] / dt,
    #                           np.array([np.roll(velocities[3], -1) - velocities[3]])[:-1] / dt
    #                           ])

    end_position = [0, x_1[-1], x_2[-1], x_3[-1], 0, y_1[-1], y_2[-1], y_3[-1]]

    results = {
        'end_effector': end_effector,
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'end_position': end_position,
        'angles': history
    }

    return results