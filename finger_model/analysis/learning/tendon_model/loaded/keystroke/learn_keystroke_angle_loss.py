import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../..'))
from analysis.learning.gradient_descent import *
import pickle

# Precision errors after 144 iterations. Loss = 0.84501

# Interval.
tmax, dt = 2., 0.001
interval = num.arange(0, tmax + dt, dt)

infile = open("full_piano_keystroke.pickle",'rb')
reference = pickle.load(infile)
infile.close()

name = "loaded piano keystroke trajectory \n with angle loss function"
loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 2, loss_function=loss_function, tendons=True, name=name)