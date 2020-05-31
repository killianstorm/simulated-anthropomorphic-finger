from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1.5, 0.0001
interval = num.arange(0, tmax + dt, dt)


length = interval.shape[0]
F_fp = np.array([(7.0 if 0 < i < 0.5 else 12.5 if 0.5 < i < 1. else 9.5) for i in interval])
F_io = np.array([(7.5 if 0 < i < 0.4 else 2. if 0.4 < i < 1. else 5.) for i in interval])
F_ed = np.array([(12. if 0 < i < 0.3 else 15.0 if 0.3 < i < 0.8 else 30.) for i in interval])


p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': F_io,
    'F_fp': F_fp,
    'F_ed': F_ed,
}

import time
t1 = time.time()
reference = simulate_predefined(p_predefined)
print("Time passed: ", time.time() - t1)
plots.animate(reference, dt, "force_change", tendons=True, di=100)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 1000, .01, loss_function=loss_angles, tendons=True, name="force_change")
