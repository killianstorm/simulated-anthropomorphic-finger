from cpg.hopf import *
from cpg.rnn_oscillator import *
from simulation.dynamic_model import *
from jax import jit
import jax

steps = 50000

# MM, FM = finger_dynamic_model()

tendon_force_range = np.array([0., 40.])

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
def simulate_sin(interval, a1, a2, a3, a4):

    @jit
    def ode(y, time, _f1, _f2, _f3, _f4):
        _params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   _f1 * np.sin(time), _f2 * np.sin(0.5 * time), _f3 * np.sin(2  * time), _f4 * np.sin(0.1 * time),
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, a1, a2, a3, a4)

    return _calculate_positions(history)


@jit
def simulate_rnn_oscillator(p):

    dt = p['interval'][1] - p['interval'][0]

    states = np.zeros((4,))
    outputs = jax.nn.sigmoid(states)
    gains = np.ones(RNN_SIZE)

    # Set up network.
    taus = np.array([p[RNN_TAU1], p[RNN_TAU2], p[RNN_TAU3], p[RNN_TAU4]])
    biases = np.array([p[RNN_BIAS1], p[RNN_BIAS2], p[RNN_BIAS3], p[RNN_BIAS4]])
    weights = np.array(p[RNN_WEIGHTS]).reshape(RNN_SIZE, RNN_SIZE)

    @jit
    def ode(state, time, _dt, _gains, _taus, _biases, _weights):
        y, _outputs, _states = state
        external_inputs = np.zeros(RNN_SIZE)  # zero external_inputs
        total_inputs = external_inputs + np.dot(_weights, _outputs)
        _states += _dt * (1 / _taus) * (total_inputs - _states)
        _outputs = np.array(jax.nn.sigmoid(_gains * (_states + _biases)))

        # Denormalize
        _outputs = np.multiply((tendon_force_range[1] - tendon_force_range[0]), _outputs) + tendon_force_range[0]

        _params = [y,
                  lengths[0], masses[0], inertias[0],
                  lengths[1], masses[1], inertias[1],
                  lengths[2], masses[2], inertias[2],
                  _outputs[0], _outputs[1], _outputs[2], _outputs[3],
                  9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params), _outputs, _states

    history, _, _ = odeint_jax(ode, (initial_positions, outputs, states), p['interval'], dt, gains, taus, biases, weights)

    results = _calculate_positions(history)

    return results


@jit
def simulate_hopf(interval, *params):


    # CPG1 = np.sin(interval)

    #
    # CPG1 = (1., dict(zip(interval, CPG1)))
    # flat, tree = tree_flatten(CPG1)

    dt = interval[1] - interval[0]
    CPG1, CPG2 = create_CPG(interval, *params)
    # plt.plot(interval, CPG1)
    # plt.plot(interval, CPG2)
    # plt.show()

    @jit
    def ode(y, time, _dt, _cpg1, _cpg2):
        index = time / _dt
        # _cpg = hopf_model(hopf, time, params)
        _params = [y,
                  lengths[0], masses[0], inertias[0], 0.,
                  lengths[1], masses[1], inertias[1], _cpg1[index.astype(int)],
                  lengths[2], masses[2], inertias[2], _cpg2[index.astype(int)],
                  9.8, 0.5]
        # _oscillator_index.append(_oscillator_index[-1] + 1)
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, dt, CPG1, CPG2)

    return _calculate_positions(history)


def rescale_torques(y, F_fs, F_io, F_fp, F_ed):
    # DIP Moments caused by tendons FP and ED.
    M_FP_DIP = - F_fp * RADII[J_DIP][T_FP]
    M_ED_DIP = - F_ed * RADII[J_DIP][T_ED]

    # PIP Moments caused by tendons FS, IO, FP and ED.
    M_FS_PIP = - F_fs * RADII[J_PIP][T_FS]
    M_IO_PIP = - F_io * RADII[J_PIP][T_IO]
    M_FP_PIP = - F_fp * RADII[J_PIP][T_FP]
    M_ED_PIP = - F_ed * RADII[J_PIP][T_ED]

    # MCP Moments caused by tendons FS, IO, FP and ED
    M_FS_MCP = - F_fs * RADII[J_MCP][T_FS]
    M_IO_MCP = - F_io * RADII[J_MCP][T_IO]
    M_FP_MCP = - F_fp * RADII[J_MCP][T_FP]
    M_ED_MCP = - F_ed * RADII[J_MCP][T_ED]

    tau3 = M_FP_DIP + M_ED_DIP
    tau2 = M_FS_PIP - M_IO_PIP + M_FP_PIP - M_ED_PIP
    tau1 = M_FS_MCP + M_IO_MCP + M_FP_MCP - M_ED_MCP

    dip_angle_bounds = (np.pi / 2, np.pi + 0.1745)
    pip_angle_bounds = (np.pi / 4, np.pi)
    mcp_angle_bounds = (np.pi / 2 + 0.1745, np.pi + np.pi / 4)

    thetas = y[:2]
    thetasd = y[3:]

    break_point = 999
    alpha1 = np.pi / 2. + thetas[0]
    alpha2 = np.pi - (thetas[0] - thetas[1])
    alpha3 = np.pi - (thetas[1] - thetas[2])

    if not(mcp_angle_bounds[1] > alpha1 > mcp_angle_bounds[0]):
        tau1 += break_point * thetasd[0]

    if not(mcp_angle_bounds[1] > alpha1 > mcp_angle_bounds[0]):
        tau1 += break_point * thetasd[0]

    if not(mcp_angle_bounds[1] > alpha1 > mcp_angle_bounds[0]):
        tau1 += break_point * thetasd[0]

    tau1 += - np.heaviside(-tau1, 1) * np.heaviside(mcp_angle_bounds[0] - alpha1, 1) * break_point * thetasd[0] \
            - np.heaviside(tau1, 1) * np.heaviside(alpha1 - mcp_angle_bounds[1], 1) * break_point * thetasd[0]
    tau2 += - np.heaviside(-tau2, 1) * np.heaviside(pip_angle_bounds[0] - alpha2, 1) * break_point * thetasd[1]\
            - np.heaviside(tau2, 1) * np.heaviside(alpha2 - pip_angle_bounds[1], 1) * break_point * thetasd[1]
    tau3 += - np.heaviside(-tau3, 1) * np.heaviside(dip_angle_bounds[0] - alpha3, 1) * break_point * thetasd[2]\
            - np.heaviside(tau3, 1) * np.heaviside(alpha3 - dip_angle_bounds[1], 1) * break_point * thetasd[2]

    return tau1, tau2, tau3


@jit
def simulate_constant(interval, F_fs, F_io, F_fp, F_ed):
    @jit
    def ode(y, time, _F_fs, _F_io, _F_fp, _F_ed):

        # _tau1, _tau2, _tau3 = rescale_torques(y, _F_fs, _F_io, _F_fp, _F_ed)

        params = [y,
                   lengths[0], masses[0], inertias[0],
                   lengths[1], masses[1], inertias[1],
                   lengths[2], masses[2], inertias[2],
                   _F_fs, _F_io, _F_fp, _F_ed,
                   9.8, 0.5]
        return np.linalg.inv(MM(*params)) @ FM(*params)

    history = odeint_jax(ode, initial_positions, interval, F_fs, F_io, F_fp, F_ed)
    return _calculate_positions(history)


def _calculate_positions(history):

    x_1 = lengths[0] * np.sin(history[:, 0])
    y_1 = - lengths[0] * np.cos(history[:, 0])

    x_2 = x_1 + lengths[1] * np.sin(history[:, 1])
    y_2 = y_1 - lengths[1] * np.cos(history[:, 1])

    x_3 = x_2 + lengths[2] * np.sin(history[:, 2])
    y_3 = y_2 - lengths[2] * np.cos(history[:, 2])

    end_effector = np.array([x_3, y_3])
    positions = np.array([np.zeros(len(x_1)), x_1, x_2, x_3, np.zeros(len(x_1)), y_1, y_2, y_3])
    velocities = np.array([history[:, 3], history[:, 4], history[:, 5]])
    accelerations = np.array([np.gradient(velocities[0]), np.gradient(velocities[1]), np.gradient(velocities[2])])
    end_position = [0, x_1[-1], x_2[-1], x_3[-1], 0, y_1[-1], y_2[-1], y_3[-1]]

    results = {
        'end_effector': end_effector,
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'end_position': end_position
    }

    return results