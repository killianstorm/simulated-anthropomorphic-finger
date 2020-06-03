from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1.5, 0.001
interval = num.arange(0, tmax + dt, dt)

ed, fp = [], []
for i in interval:
    if i < tmax / 8.:
        fp.append(5.)
        ed.append(0.)
    else:
        fp.append(0.)
        ed.append(7.5)

p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array(fp),
    'F_ed': np.array(ed),
}

name = "keystroke trajectory \n with angle loss function"
reference = simulate_predefined(p_predefined)
# plots.animate(reference, dt, name, tendons=True)

loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 250, loss_function=loss_function, tendons=True, name=name)
