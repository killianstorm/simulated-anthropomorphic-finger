from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

ed, fp = [], []
which = False
period = 8 * dt
for i in interval:
    if i % period == 0:
        which = not which

    if which:
        fp.append(3.)
        ed.append(0.)
    else:
        ed.append(3.)
        fp.append(0.)

p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array(fp),
    'F_ed': np.array(ed),
}

reference = simulate_predefined(p_predefined)
plots.animate(reference, dt, "keystroke", tendons=True, di=100)

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 3, .1, loss_function=loss_angles, tendons=False, name="keystroke")
