from finger_model.analysis.learning.gradient_descent import *

# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

r = 0.06
offset = np.sum(lengths) - r / 2

x = np.linspace(-r, r, int((tmax / dt) / 2) + 1)
x2 = x[::-1][:-1]

z = np.sqrt(-x ** 2 + r ** 2)
z2 = -np.sqrt(-x2 ** 2 + r ** 2)

x = np.concatenate([x + offset, x2 + offset])
z = np.concatenate([z, z2])
plt.plot(x, z)
plt.show()

xz = np.array([x, z])

reference = {
    'interval': interval,
    'end_effector': xz
}


# p = {
#     'interval': interval,
#     'F_fs': np.zeros(interval.shape[0]),
#     'F_io': np.zeros(interval.shape[0]),
#     'F_fp': np.zeros(interval.shape[0]),
#     'F_ed': np.zeros(interval.shape[0])
# }
# approximation = simulate_predefined(p)
#
# plots.animate(reference, dt, 'EFRKF', approximation, di=100, tendons=True)

learn_gradient_descent(reference, interval, 2, .01, loss_function=loss_end_effector, tendons=True, name="circle")
