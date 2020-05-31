from finger_model.analysis.learning.gradient_descent import *
import time

# Interval.
tmax, dt = 2., 0.00001
interval = num.arange(0, tmax + dt, dt)

# Predefined keystroke trajectory.
length = interval.shape[0]
MCP = [(.50  if 0 < i < 0.5 else 0.125 if 0.5 < i < 1. else -.250 if 0.5 < i < 1. else -.25) for i in interval]
PIP = [(.25  if 0 < i < 0.5 else -.375 if 0.5 < i < 1. else 0.100 if 0.5 < i < 1. else 0.05) for i in interval]
DIP = [(.015 if 0 < i < 0.5 else -.250 if 0.5 < i < 1. else 0.050 if 0.5 < i < 1. else -.01) for i in interval]

# plt.plot(interval, MCP)
# plt.plot(interval, DIP)
# plt.plot(interval, PIP)
# plt.show()

p_predefined = {
    'interval': interval,
    'tau1': MCP,
    'tau2': PIP,
    'tau3': DIP
}

t1 = time.time()
reference = simulate_predefined(p_predefined)
print("Time passed: ", time.time() - t1)
# plots.animate(reference, dt, "keystroke", tendons=False, di=100)


# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 10000, .01, loss_function=loss_angles, tendons=False, name="complex01")
