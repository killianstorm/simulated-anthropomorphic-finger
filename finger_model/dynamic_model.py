
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as num
from sympy import *
from sympy.physics.mechanics import *

from hopf import *
from rnn_oscillator import *

from moviepy.editor import ImageSequenceClip

from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

from jax import jit, grad, value_and_grad, vmap
from jax.experimental.ode import odeint as odeint_jax

init_printing()

lengths = [.5, .25, .125]
masses = [1., .5, .25]
inertias = [masses[0] * (lengths[0] ** 2) * (1. / 12.),
            masses[1] * (lengths[1] ** 2) * (1. / 12.),
            masses[2] * (lengths[2] ** 2) * (1. / 12.)]


def fig2image(fig):
    fig.canvas.draw()
    data = num.fromstring(fig.canvas.tostring_rgb(), dtype=num.uint8, sep='')
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def finger_dynamic_model():

    thetas = [dynamicsymbols('theta1'),
              dynamicsymbols('theta2'),
              dynamicsymbols('theta3')]

    thetasd = [dynamicsymbols('theta1', 1),
               dynamicsymbols('theta2', 1),
               dynamicsymbols('theta3', 1)]

    thetasdd = [dynamicsymbols('theta1', 2),
                dynamicsymbols('theta2', 2),
                dynamicsymbols('theta3', 2)]

    t = symbols('t')

    l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr, tau1, tau2, tau3 = symbols(
        'l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr, tau1, tau2, tau3')
    x1 = l1 * sin(thetas[0])
    y1 = -l1 * cos(thetas[0])

    x2 = x1 + l2 * sin(thetas[1])
    y2 = y1 - l2 * cos(thetas[1])

    x3 = x2 + l3 * sin(thetas[2])
    y3 = y2 - l3 * cos(thetas[2])

    xc1 = (l1 / 2.) * sin(thetas[0])
    yc1 = - (l1 / 2.) * cos(thetas[0])

    xc2 = x1 + (l2 / 2.) * sin(thetas[1])
    yc2 = y1 - (l2 / 2.) * cos(thetas[1])

    xc3 = x2 + (l3 / 2.) * sin(thetas[2])
    yc3 = y2 - (l3 / 2.) * cos(thetas[2])

    x1d = diff(x1, t)
    y1d = diff(y1, t)

    x2d = diff(x2, t)
    y2d = diff(y2, t)

    x3d = diff(x3, t)
    y3d = diff(y3, t)

    xc1d = diff(xc1, t)
    yc1d = diff(yc1, t)

    xc2d = diff(xc2, t)
    yc2d = diff(yc2, t)

    xc3d = diff(xc3, t)
    yc3d = diff(yc3, t)

    alpha1 = pi - (thetas[1] - thetas[0])
    alpha1d = diff(alpha1, t)

    alpha2 = pi - (thetas[2] - thetas[1])
    alpha2d = diff(alpha2, t)

    V = m1 * g * yc1 + m2 * g * yc2 + m3 * g * yc3
    T1 = Rational(1, 2) * m1 * (xc1d ** 2 + yc1d ** 2) + Rational(1, 2) * (thetasd[0] ** 2) * I1
    T2 = Rational(1, 2) * m2 * (xc2d ** 2 + yc2d ** 2) + Rational(1, 2) * (thetasd[1] ** 2) * I2
    T3 = Rational(1, 2) * m3 * (xc3d ** 2 + yc3d ** 2) + Rational(1, 2) * (thetasd[2] ** 2) * I3
    L = (T1 + T2 + T3) - V

    N = ReferenceFrame('N')
    r_pp = ReferenceFrame('rigid_pp')
    r_pp.set_ang_vel(N, thetasd[0] * N.z)

    r_mp = ReferenceFrame('rigid_mp')
    r_mp.set_ang_vel(N, thetasd[1] * N.z)

    r_dp = ReferenceFrame('rigid_dp')
    r_dp.set_ang_vel(N, thetasd[2] * N.z)

    j_pp_mp = ReferenceFrame('joint_pp_mp')
    j_pp_mp.set_ang_vel(N, alpha1d * N.z)

    j_mp_dp = ReferenceFrame('joint_mp_dp')
    j_mp_dp.set_ang_vel(N, alpha2d * N.z)

    FL = [(r_pp, tau1 * N.z),  # Rigid bodies
          (r_mp, tau2 * N.z),
          (r_dp, tau3 * N.z),

          (r_pp, -c_fr * thetasd[0] * N.z),  # Joints
          (j_pp_mp, -c_fr * alpha1d * N.z),
          (j_mp_dp, -c_fr * alpha2d * N.z)
          ]

    LM = LagrangesMethod(L, [thetas[0], thetas[1], thetas[2]], forcelist=FL, frame=N)
    equations = LM.form_lagranges_equations()

    y = [thetas[0], thetas[1], thetas[2],
         thetasd[0], thetasd[1], thetasd[2]]

    parameters = [
                l1, m1, I1, tau1,
                l2, m2, I2, tau2,
                l3, m3, I3, tau3,
                g, c_fr]

    unknowns = [Dummy() for i in y]
    unknown_dict = dict(zip(y, unknowns))

    mm = LM.mass_matrix_full.subs(unknown_dict)
    fm = LM.forcing_full.subs(unknown_dict)
    # mm = LM.mass_matrix_full
    # fm = LM.forcing_full

    mapping = {'sin': np.sin, 'cos': np.cos, 'pi': np.pi, 'array': np.array, 'ImmutableDenseMatrix': np.array}

    mass_matrix = lambdify([unknowns] + parameters, mm, mapping)
    forcing_matrix = lambdify([unknowns] + parameters, fm, mapping)

    # equations_of_motion = np.linalg.inv(mass_matrix(*)) * forcing_matrix

    #equations_of_motion = LM.rhs()
    #mass_matrix = LM.mass_matrix
    #forcing = LM.forcing

    # equations_of_motion_lambda = lambdify((y,
    #                                        l1, m1, I1, tau1,
    #                                        l2, m2, I2, tau2,
    #                                        l3, m3, I3, tau3,
    #                                        g, c_fr), equations_of_motion, mapping)

    return mass_matrix, forcing_matrix


# Loss function: SSE of end effector positions.
@jit
def loss_end_effector(reference, simulated):
    return np.sqrt(np.mean((reference['end_effector'] - simulated['end_effector']) ** 2))


# Loss function: SSE of velocities.
def loss_velocities(reference, simulated):
    return np.sqrt(((reference['velocities'] - simulated['velocities']) ** 2).mean())


# Loss function: SSE of all positions.
def loss_positions(reference, simulated):
    return np.sqrt(((reference['positions'] - simulated['positions']) ** 2).mean())


# Loss function: SSE of accelerations.
def loss_accelerations(reference, simulated):
    return np.sqrt(((reference['accelerations'] - simulated['accelerations']) ** 2).mean())


# @jit
# def loss_mix(tau1, tau2, tau3, reference):
#     trajectory = _calculate_positions(tau1, tau2, tau3)
#     return np.square((np.sum((reference['positions'] - trajectory['positions']) ** 2)) /
#                      np.sum((reference['accelerations'] - trajectory['accelerations']) ** 2))


def plot(reference, loss, x0, xt, title):
    """
    Plots a given function for interval [x0, xt].imp
    """
    x = num.linspace(x0, xt, num=abs(xt - x0) * 5)
    y = num.zeros(len(x))

    index = 0
    for i in x:
        y[index] = loss(i, 0., 0., reference)
        index += 1

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Torque')
    plt.ylabel('Cost')
    plt.show()


def plot_losses(reference, x0, xt):
    """
    Plots all losses given in Solver for interval [x0, xt]
    """
    print("Plotting losses...", end='')
    plot(reference, loss_end_effector, x0, xt, 'End-effector loss function')
    plot(reference, loss_positions, x0, xt, 'Positions loss function')
    plot(reference, loss_velocities, x0, xt, 'Velocities loss function')
    plot(reference, loss_accelerations, x0, xt, 'Accelerations loss function')
    # plot(reference, loss_mix, x0, xt, 'Mix loss function')

    print("DONE")


def gradient_descent(reference, loss, iterations, learning_rate, init):

    # Construct grad function.
    gradient_loss = jit(grad(loss, 1))

    # Assign initial tau1.
    tau1_start = init
    print("Starting with tau1: " + str(tau1_start))

    # Number of iterations.
    for i in range(iterations):

        # Calculate gradient
        g = gradient_loss(reference, tau1_start, 0., 0.)
        print("Gradient: " + str(g))

        # Perform gradient descent.
        tau1_start -= learning_rate * g

        print("Tau1 approximation: " + str(tau1_start))


# Construct ode
initial_positions = np.array([0., 0., 0.,
                              0., 0., 0.])
MM, FM = finger_dynamic_model()


@jit
def simulate_sin(interval, a1, a2, a3):

    @jit
    def ode(y, time, _f1, _f2, _f3):
        _params = [y,
                   lengths[0], masses[0], inertias[0], np.sin(_f1 * time),
                   lengths[1], masses[1], inertias[1], np.sin(_f2 * time),
                   lengths[2], masses[2], inertias[2], np.sin(_f3 * time),
                   9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params)

    history = odeint_jax(ode, initial_positions, interval, a1, a2, a3, mxstep=500)

    return _calculate_positions(history)


@jit
def simulate_rnn_oscillator(p):

    dt = p['interval'][1] - p['interval'][0]

    states = np.array([0., 0., 0.])
    outputs = jax.nn.sigmoid(states)
    gains = np.ones(RNN_SIZE)

    # Set up network.
    taus = np.array([p[RNN_TAU1], p[RNN_TAU2], p[RNN_TAU3]])
    biases = np.array([p[RNN_BIAS1], p[RNN_BIAS2], p[RNN_BIAS3]])
    weights = np.array(p[RNN_WEIGHTS]).reshape(RNN_SIZE, RNN_SIZE)

    @jit
    def ode(state, time, _dt, _gains, _taus, _biases, _weights):
        y, _outputs, _states = state
        external_inputs = np.zeros(RNN_SIZE)  # zero external_inputs
        total_inputs = external_inputs + np.dot(_weights, _outputs)
        _states += _dt * (1 / _taus) * (total_inputs - _states)
        _outputs = np.array(jax.nn.sigmoid(_gains * (_states + _biases)))

        _params = [y,
                  lengths[0], masses[0], inertias[0], _outputs[0],
                  lengths[1], masses[1], inertias[1], _outputs[1],
                  lengths[2], masses[2], inertias[2], _outputs[2],
                  9.8, 0.5]
        return np.linalg.inv(MM(*_params)) @ FM(*_params), _outputs, _states

    history, _, _ = odeint_jax(ode, (initial_positions, outputs, states), p['interval'], dt, gains, taus, biases, weights)

    results = _calculate_positions(history)

    return results


@jit
def simulate_oscillator(interval, *params):


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

    history = odeint_jax(ode, initial_positions, interval, dt, CPG1, CPG2, mxstep=500)

    return _calculate_positions(history)


@jit
def simulate_constant(tau1, tau2, tau3, interval):
    @jit
    def ode(y, time, _tau1, _tau2, _tau3):
        params = [y,
                   lengths[0], masses[0], inertias[0], _tau1,
                   lengths[1], masses[1], inertias[1], _tau2,
                   lengths[2], masses[2], inertias[2], _tau3,
                   9.8, 0.5]
        return np.linalg.inv(MM(*params)) @ FM(*params)

    history = odeint_jax(ode, initial_positions, interval, tau1, tau2, tau3)
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


def plot_movements(reference):
    plt.plot(reference['positions'][0, 0], reference['positions'][0, 1])
    plt.plot(reference['positions'][1, 0], reference['positions'][1, 1])
    plt.plot(reference['positions'][2, 0], reference['positions'][2, 1])
    plt.legend(('Vingertip', 'Tussenstuk', 'Laatste'))

    plt.title("Individual trajectories")
    plt.show()

    plt.plot(reference['accelerations'][0], reference['accelerations'][1])
    plt.title("Accelerations")
    plt.show()

# Create reference
# reference = solve_for(5., 0., 0.)
# gradient_descent(reference, loss_end_effector, 1000, 0.0001, 1.)

