from cpg.hopf import *
from cpg.rnn_oscillator import *
from simulation.dynamic_model import *
from jax import jit
import jax
import jax.ops
import jax.lax


@jit
def simulate_sine(p):
    """
    Simulates the finger with a series of sine inputs.
    arguments: dict containing amplitudes, phases and interval
    """

    amplitudes, phases, interval = p['amplitudes'], p['phases'], p['interval']

    @jit
    def ode(y, time, _amplitudes, _phases):

        # Calculate sines for current time instant.
        _inputs = np.array([_a * np.sin(_p * time) for _a, _p in zip(_amplitudes, _phases)])

        # If using tendons, take abs since no negative values are allowed.
        if ENABLE_TENDONS:
            _inputs = np.abs(_inputs)

        # Construct params.
        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_inputs,
                   9.8, 0.5]

        # Solve equations of motion for current time.
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    # Solve equations of motion for interval.
    history = odeint_jax(ode, initial_positions, interval, amplitudes, phases)

    # Prettify history.
    results = calculate_positions(history)

    # Add torque history.
    results['torques'] = np.array([_a * np.sin(_p * interval) for _a, _p in zip(amplitudes, phases)]).transpose()
    if ENABLE_TENDONS:
        results['torques'] = np.abs(results['torques'])

    return results


@jit
def simulate_sine_RK4(p):
    """
    WARNING: DEPRECATED SINCE IT DOES NOT HAVE THE REQUIRED ACCURACY

    Simulates the finger with a series of sine inputs by using Runge Kutta 4.
    arguments: p containing amplitudes, phases and interval
    """
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

    # Solve with Runge Kutta 4
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
    results = calculate_positions(history)

    results['torques'] = np.array([_a * np.sin(_p * interval) for _a, _p in zip(amplitudes, phases)]).transpose()

    if ENABLE_TENDONS:
        results['torques'] = np.abs(results['torques'])

    return results



@jit
def simulate_rnn_oscillator(p):
    """
    Simulates the finger with a continuous time recurrent neural network (CTRNN).
    arguments: dict containing taus, biases, states, gains and weights
    """

    # Determine size of CTRNN.
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

    # Step function for network.
    def step(current_state, index):
        _output, _state = current_state
        external_inputs = np.zeros(size)  # No external inputs.
        total_inputs = external_inputs + np.dot(weights, _output)
        _state += dt * (1 / taus) * (total_inputs - _state)
        _out = jax.nn.sigmoid(gains * (_state + biases))
        return (_out, _state), _out

    # Simulate network to get output.
    _, outputs = jax.lax.scan(step, (out, states), p['interval'])

    # Scale output according to max values of force or torque.
    if ENABLE_TENDONS:
        outputs = np.multiply(max_val, outputs)
    else:
        outputs = np.multiply(max_val, outputs) - max_val / 2

    @jit
    def ode(y, time, _dt, _torques):

        # Calculate which index for current time instant.
        _index = time / _dt
        _out = _torques[_index.astype(int)]

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_out,
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    # Solve with CTRNN outputs on interval.
    history = odeint_jax(ode, initial_positions, p['interval'], dt, outputs)

    results = calculate_positions(history)
    results['torques'] = outputs

    return results


@jit
def simulate_constant(p):
    """
    Simulates the finger with constant values.
    arguments: dict containing constant values for each tendon/joint and the interval
    """

    if ENABLE_TENDONS:
        inputs, interval = (p['F_fs'], p['F_io'], p['F_fp'], p['F_ed']), p['interval']
    else:
        inputs, interval = (p['tau1'], p['tau2'], p['tau3']), p['interval']

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

    results = calculate_positions(history)

    results['torques'] = np.array([interval.shape[0] * [i] for i in inputs]).transpose()

    return results


@jit
def simulate_predefined(p):
    """
    Simulates the finger with arrays of predefined forces/torques.
    arguments: dict containing arrays of forces/torques and the interval
    """

    if ENABLE_TENDONS:
        inputs, interval = (p['F_fs'], p['F_io'], p['F_fp'], p['F_ed']), p['interval']
    else:
        inputs, interval = (p['tau1'], p['tau2'], p['tau3']), p['interval']

    inputs = np.array([*inputs]).transpose()

    dt = interval[1] - interval[0]

    @jit
    def ode(y, time, _dt, _torques):

        # Calculate which index for current time instant.
        _index = time / _dt
        _out = _torques[_index.astype(int)]

        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   *_out,
                   9.8, 0.3]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, dt, inputs)
    results = calculate_positions(history)
    results['torques'] = inputs

    return results


def calculate_positions(history):
    """
    Converts results from odeint into a dict containing coordinates, velocities and angles.
    """

    x_1 = lengths[0] * np.sin(history[:, 0])
    y_1 = - lengths[0] * np.cos(history[:, 0])

    x_2 = x_1 + lengths[1] * np.sin(history[:, 1])
    y_2 = y_1 - lengths[1] * np.cos(history[:, 1])

    x_3 = x_2 + lengths[2] * np.sin(history[:, 2])
    y_3 = y_2 - lengths[2] * np.cos(history[:, 2])

    end_effector = np.array([x_3, y_3])
    positions = np.array([np.zeros(len(x_1)), x_1, x_2, x_3, np.zeros(len(x_1)), y_1, y_2, y_3])
    velocities = np.array([history[:, 3], history[:, 4], history[:, 5]])

    end_position = [0, x_1[-1], x_2[-1], x_3[-1], 0, y_1[-1], y_2[-1], y_3[-1]]

    results = {
        'end_effector': end_effector,
        'positions': positions,
        'velocities': velocities,
        'end_position': end_position,
        'angles': history
    }

    return results