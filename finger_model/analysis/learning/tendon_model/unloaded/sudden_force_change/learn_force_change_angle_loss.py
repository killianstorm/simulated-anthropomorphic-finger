from finger_model.analysis.learning.gradient_descent import *


# Interval.
tmax, dt = 1.5, 0.001
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

name = "sudden force change trajectory \n with angle loss function"
reference = simulate_predefined(p_predefined)
# plots.animate(reference, dt, name, tendons=True)

loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 250, loss_function=loss_function, tendons=True, name=name)
