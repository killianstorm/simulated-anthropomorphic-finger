from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

# Predefined keystroke trajectory.
p_sine = {
    'amplitudes': np.array([.2, -.2, -.1]),
    'phases': np.array([2.5, 5., 5.]),
    'interval': interval
}
reference = simulate_sine(p_sine)
plots.animate(reference, dt, "keystroke", tendons=True, di=100)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 10, 1., loss_function=loss_angles, tendons=False, name="keystroke")
