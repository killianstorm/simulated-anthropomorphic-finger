from simulation.optimizer import *
from simulation.loss_functions import *

import csv

import jax.numpy as np
import matplotlib.pyplot as plt
from tools import plots

import numpy as num

# Interval.
tmax, dt = 3., 0.01

refresh_rate = tmax/dt
interval = num.arange(0, tmax + dt, dt)

ed, fp = [], []
which = False
period = 150 * dt
for i in interval:
    if i % period == 0:
        which = not which

    fp.append(7.5)
    ed.append(2.)

    # if which:
    #     fp.append(7.)
    #     ed.append(8.5)
    # else:
    #     fp.append(7.)
    #     fp.append(7.)


p_predefined = {
    'interval': interval,
    'F_fs': np.zeros(interval.shape[0]),
    'F_io': np.zeros(interval.shape[0]),
    'F_fp': np.array(fp),
    'F_ed': np.array(ed),
}

reference = simulate_predefined(p_predefined)
# plots.animation(reference, dt, "predefined", tendons=True)

plt.plot(reference['end_effector'][0], reference['end_effector'][1])
plt.title("Reference")
plt.show()

mcp_angles = reference['angles'][:, 0]
pip_angles = reference['angles'][:, 1]
dip_angles = reference['angles'][:, 2]


def r2d(rads):
    return rads * 180 / np.pi


rel_mcp = r2d((np.pi / 2. + mcp_angles))
rel_pip = r2d((np.pi - (mcp_angles - pip_angles)))
rel_dip = r2d((np.pi - (pip_angles - dip_angles)))


# Read csv.
def read_csv(file, keys):
    mydict = {}
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        mydict['time'] = []
        mydict['F'] = []
        for rows in reader:
            mydict['time'].append(num.float64(rows[0]))
            mydict['F'].append(num.float64(rows[1]))

        mydict[keys[0]] = np.array(mydict['time'], dtype="float64")
        mydict[keys[1]] = np.array(mydict['F'], dtype="float64")
    return mydict


measured = read_csv("ipj_coupling.csv", ('PIP', 'DIP'))


# plt.cla()

axes = plt.gca()
plt.plot(rel_pip, rel_dip, label="Simulated")
plt.plot(180 - measured['PIP'], 180 - measured['DIP'], label="Physical")
plt.title("IPJ coupling")
plt.xlabel("PIP relative angle [degrees]")
plt.ylabel("DIP relative angle [degrees]")
plt.legend()
plt.savefig("ipj_coupling.png", dpi=244)
plt.show()

