from finger_model.analysis.learning.gradient_descent import *

# Precision erros after 144 iterations. Loss = 0.84501

# Interval.
tmax, dt = 2., 0.001
interval = num.arange(0, tmax + dt, dt)

name = "loaded piano keystroke trajectory \n with angle loss function"

import pickle
# outfile = open("full_piano_keystroke.pickle",'wb')
# pickle.dump(reference, outfile)
# outfile.close()

infile = open("full_piano_keystroke.pickle",'rb')
reference = pickle.load(infile)
infile.close()
plots.animate(reference, dt, name, tendons=True)

loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
learn_gradient_descent(reference, interval, 25, loss_function=loss_function, tendons=True, name=name)
