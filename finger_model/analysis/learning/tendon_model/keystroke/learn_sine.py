from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1.8, 0.0001
interval = num.arange(0, tmax + dt, dt)


p_sine = {
    'amplitudes': np.array([0., 2., 15., 25]),
    'phases': np.array([0., 2.5, 3., 1.75]),
    'interval': interval
}

import time
t1 = time.time()
reference = simulate_sine(p_sine)
print("Time passed: ", time.time() - t1)
# plots.animate(reference, dt, "sine_ligaments", tendons=True, di=100)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 1000, .01, loss_function=loss_angles, tendons=True, name="sine")
