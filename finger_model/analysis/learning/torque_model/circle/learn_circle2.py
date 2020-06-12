from finger_model.analysis.learning.gradient_descent import *
import time

# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

r = 0.03
offset = np.sum(lengths) - r / 2

x = np.linspace(-r, r, int((tmax / dt) / 2) + 1)
x2 = x[::-1][:-1]

z = np.sqrt(-x ** 2 + r ** 2)
z2 = -np.sqrt(-x2 ** 2 + r ** 2)

x = np.concatenate([x + offset, x2 + offset])
z = np.concatenate([z, z2])
plt.plot(x, z)
plt.show()

xz = np.array([x, z])

reference = {
    'interval': interval,
    'end_effector': xz
}

p_opt = {
    'interval': interval,
    'reference': reference,
    RNN_TAUS: np.array([ 1.56465126, 0.94478561, 1.67513426]),
    RNN_BIASES: np.array([-0.27607084, 1.28427122, 0.24184255]),
    RNN_GAINS: np.array([2.37533802, -2.71218046, 0.53411891]),
    RNN_STATES: np.array([ -0.1075834, -1.95932735, -0.75545409]),
    RNN_WEIGHTS: np.array([2.82870448, 1.39636011, 1.41379944, -1.14895493, 0.70051312, 1.03068617, 1.81113158, 0.36068029, 1.38334333])
}

simulate_ctrnn_params_and_animate(p_opt, "circle_notendons_1")

# Learn to reproduce trajectory using gradient descent.
# learn_gradient_descent(reference, interval, 250, .1, loss_function=loss_end_effector, tendons=False, name="circle_torques")
