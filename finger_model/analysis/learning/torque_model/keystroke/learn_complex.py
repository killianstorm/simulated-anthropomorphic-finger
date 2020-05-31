from finger_model.analysis.learning.gradient_descent import *
import time

# Interval.
tmax, dt = 4., 0.00001
interval = num.arange(0, tmax + dt, dt)

# Predefined keystroke trajectory.
p_sine = {
    'amplitudes': np.array([2., -1., -.5]),
    'phases': np.array([3., 3., 3.]),
    'interval': interval
}


t1 = time.time()
reference = simulate_sine(p_sine)
print("Time passed: ", time.time() - t1)
plots.animate(reference, dt, "keystroke", tendons=False, di=100)


# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 10000, .01, loss_function=loss_angles, tendons=False, name="complex01")
