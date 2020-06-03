from finger_model.analysis.learning.gradient_descent import *

# Interval.
tmax, dt = 1., 0.001
interval = num.arange(0, tmax + dt, dt)

fp = []
for i in interval:
    fp.append(10.)

p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array(fp),
    'F_ed': np.zeros(interval.shape[0]),
}

name = "full grasp trajectory \n with end-effector loss function"
reference = simulate_predefined(p_predefined)
# plots.animate(reference, dt, name, tendons=True)

loss_function = loss_end_effector

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 250, loss_function=loss_function, tendons=True, name=name)
