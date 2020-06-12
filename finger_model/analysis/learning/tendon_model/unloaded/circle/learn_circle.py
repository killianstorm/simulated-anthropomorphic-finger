from analysis.learning.gradient_descent import *

# Precision loss after 14 iterations. Loss = 0.1495

# Interval.
tmax, dt = 1., 0.0001
interval = num.arange(0, tmax + dt, dt)

r = 0.03
offset = np.sum(lengths) - r

x = np.linspace(-r, r, int((tmax / dt) / 2) + 1)
x2 = x[::-1][:-1]

z = np.sqrt(-x ** 2 + r ** 2)
z2 = -np.sqrt(-x2 ** 2 + r ** 2)

x = np.concatenate([x2 + offset, x + offset])
z = np.concatenate([z2, z])
plt.plot(x, z)
plt.show()

xz = np.array([x, z])

reference = {
    'interval': interval,
    'end_effector': xz
}

name = "perfect circle trajectory"


learn_gradient_descent(reference, interval, 250, loss_function=loss_end_effector, tendons=True, name=name)
