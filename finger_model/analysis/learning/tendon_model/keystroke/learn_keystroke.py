from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

ed, fp = [], []
which = False
period = 5000 * dt
for i in interval:
    if i % period == 0:
        which = not which

    if which:
        fp.append(10.)
        ed.append(0.)
    else:
        ed.append(10.)
        fp.append(0.)



p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array(fp),
    'F_ed': np.array(ed),
}

import time
t1 = time.time()
reference = simulate_predefined(p_predefined)
print("Time passed: ", time.time() - t1)
# plots.animate(reference, dt, "keystroke", tendons=True, di=100)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 100, .01, loss_function=loss_angles, tendons=True, name="keystroke")
