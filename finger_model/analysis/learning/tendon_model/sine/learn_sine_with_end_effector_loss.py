from finger_model.analysis.learning.gradient_descent import *

# Interval.
tmax, dt = 1.8, 0.001
interval = num.arange(0, tmax + dt, dt)

p_sine = {
    'amplitudes': np.array([0., 2., 15., 25]),
    'phases': np.array([0., 2.5, 3., 1.75]),
    'interval': interval
}

name = "sine trajectory \n with angle loss function"
reference = simulate_sine(p_sine)
# plots.animate(reference, dt, name, tendons=True)

loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 2, loss_function=loss_function, tendons=True, name=name)
