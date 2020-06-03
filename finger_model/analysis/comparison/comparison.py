import csv
import matplotlib.pyplot as plt

from finger_model.simulation.optimizer import *
from finger_model.simulation.loss_functions import *
from tools import plots

import numpy as num

# Offset to translate physical coordinates to simulated.
OFFSET = (-0.0625, -0.05)


def compare_sim_to_phys(method, title):
    """
    Compares simulated to physical trajectories.
    """

    method = str(method)
    title = str(title)

    # Read csv files.
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

    # Read in physical forces and trajectory.
    F_FP = read_csv(method + '/' + method + '_FP.csv', ('time', 'F'))
    F_IO = read_csv(method + '/' + method + '_IO.csv', ('time', 'F'))
    F_ED = read_csv(method + '/' + method + '_ED.csv', ('time', 'F'))
    physical_trajectory = read_csv(method + '/' + method + '_xz.csv', ('x', 'z'))
    physical_trajectory['x'] += OFFSET[0]
    physical_trajectory['z'] += OFFSET[1]

    # Plot physical forces.
    plt.plot(F_ED['time'], F_ED['F'], label="ED")
    plt.plot(F_FP['time'], F_FP['F'], label="FP")
    plt.plot(F_IO['time'], F_IO['F'], label="IO")
    plt.title("Forces for " + title)
    plt.xlabel("time [s]")
    plt.ylabel("F [N]")
    plt.legend()
    # plt.savefig("forces_" + method + ".png", dpi=244)
    plt.show()

    # Interval of 5 seconds.
    tmax, dt = 5., 5. / F_FP['time'].shape[0]
    interval = num.arange(0, tmax + dt, dt)[:-1]

    # Simulate physical forces on simulator.
    params = {
        'interval': interval,
        'F_fs': np.zeros(interval.shape[0], dtype="float64"),
        'F_io': F_IO['F'],
        'F_fp': F_FP['F'],
        'F_ed': F_ED['F']
    }
    simulated_trajectory = simulate_predefined(params)

    loss = np.sqrt(np.mean((np.array([physical_trajectory['x'], physical_trajectory['z']]) - simulated_trajectory['end_effector']) ** 2))
    print("The loss is: " + str(loss))

    # Plot simulated trajectory along physical trajectory.
    plt.plot([0, lengths[0], lengths[0] + lengths[1], np.sum(lengths)], [0, 0, 0, 0], marker='.', linewidth=3, markersize=15)
    plt.text(0 - 0.015, 0.025, 'MCP')
    plt.text(lengths[0] - 0.015, 0.025, 'PIP')
    plt.text(lengths[0] + lengths[1] - 0.015, 0.025, 'DIP')

    axes = plt.gca()
    axes.set_xlim([-0.125, 0.225])
    axes.set_ylim([-0.25, 0.15])

    plt.scatter(simulated_trajectory['end_effector'][0], simulated_trajectory['end_effector'][1], s=5., label="simulation")
    plt.scatter(physical_trajectory['x'], -physical_trajectory['z'], s=5., label="physical")
    plt.legend(loc='lower left')
    plt.title("Comparison for " + title)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    # plt.savefig("comparison_" + method + ".png", dpi=244)
    plt.show()

    # Create animation
    # plots.animate(simulated_trajectory, dt, "comparison_" + str(method), tendons=True)

    # # Read csv.
    # def read_csv(file, keys):
    #     mydict = {}
    #     with open(file, mode='r') as infile:
    #         reader = csv.reader(infile)
    #         mydict['time'] = []
    #         mydict['F'] = []
    #         for rows in reader:
    #             mydict['time'].append(num.float64(rows[0]))
    #             mydict['F'].append(num.float64(rows[1]))
    #
    #         mydict[keys[0]] = np.array(mydict['time'], dtype="float64")
    #         mydict[keys[1]] = np.array(mydict['F'], dtype="float64")
    #     return mydict
    #
    # measured = read_csv("ipj_coupling.csv", ('PIP', 'DIP'))
    #
    # # plt.cla()
    #
    # def r2d(rads):
    #     return rads * 180 / np.pi
    #
    # mcp_angles = simulated_trajectory['angles'][:, 3]
    # pip_angles = simulated_trajectory['angles'][:, 4]
    # dip_angles = simulated_trajectory['angles'][:, 5]
    #
    # rel_mcp = r2d((np.pi / 2. + mcp_angles))
    # rel_pip = r2d((np.pi - (mcp_angles - pip_angles)))
    # rel_dip = r2d((np.pi - (pip_angles - dip_angles)))
    #
    # plt.scatter(rel_pip, rel_dip, label="Simulated")
    # plt.scatter(180 - measured['PIP'], 180 - measured['DIP'], label="Physical")
    # plt.title("IPJ coupling simplified expression")
    # plt.xlabel("PIP relative angle [degrees]")
    # plt.ylabel("DIP relative angle [degrees]")
    # plt.legend()
    # plt.show()

# Full grasp.
compare_sim_to_phys("grasp", "full grasp")

# # Isometric PIP.
compare_sim_to_phys("0PIP", "extended PIP with higher friction")
#
# # Isometric MCP.
compare_sim_to_phys("0MCP", "extended MCP with higher friction")
#
# # Complex.
compare_sim_to_phys("complex", "complex trajectory")

