from finger_model.analysis.learning.gradient_descent import *
import time

# Interval.
tmax, dt = 1., 0.001
interval = num.arange(0, tmax + dt, dt)

# Predefined keystroke trajectory.
p_sine = {
    'amplitudes': np.array([.2, -.2, -.1]),
    'phases': np.array([2.5, 5., 5.]),
    'interval': interval
}
t1 = time.time()
reference = simulate_sine(p_sine)
print("Time passed: ", time.time() - t1)
plots.animate(reference, dt, "keystroke", tendons=True)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 250, .1, loss_function=loss_angles, tendons=False, name="keystroke")
