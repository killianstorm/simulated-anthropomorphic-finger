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
# plots.animate(reference, dt, name, tendons=True)

loss_function = loss_angles

# Learn to reproduce trajectory using gradient descent.
# learn_gradient_descent(reference, interval, 250, loss_function=loss_function, tendons=True, name=name)

opt_params = {'rnn_tau': np.array([0.73714265, 1.62787514, 0.26231022, 0.94657235]), 'rnn_bias': np.array([0.28648788, 1.18492026, 0.94938257, 1.70822662]), 'rnn_gains': np.array([-4.94507942,  0.34342934, -1.81595887,  2.34423985]), 'rnn_states': np.array([-0.48489297,  1.67354513,  2.3559032 ,  1.73951868]), 'rnn_weights': np.array([[-5.00560947,  1.95213028,  1.16496038,  1.7793703 ],
             [ 0.4906549 ,  0.53195009,  0.99715998,  0.51077542],
             [ 0.17478629, -0.28697391,  0.9973187 , -0.39750044],
             [ 1.01390176,  0.96669816,  1.02389297,  0.96567561]]),
              'interval': interval}

approx = simulate_ctrnn(opt_params)
plots.animate(reference, dt, name, approx)

