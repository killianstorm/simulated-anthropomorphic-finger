##################################
# Variables defining the finger. #
##################################

# Max force when using tendons.
MAX_FORCE_TENDONS = 40.

# Max torque when using torques.
MAX_TORQUE = 2.

# True if using tendons, False if using torques
ENABLE_TENDONS = True

# True if using ligaments, False if not.
ENABLE_LIGAMENTS = True

# Lengths of each phalanx.
lengths = np.array([0.09955, 0.06687, 0.04291])

# Mass of each phalanx.
masses = np.array([.150, 0.060, .025])

# Inertia of each phalanx
inertias = np.array([masses[0] * (lengths[0] ** 2) * (1. / 12.),
                     masses[1] * (lengths[1] ** 2) * (1. / 12.),
                     masses[2] * (lengths[2] ** 2) * (1. / 12.)])

# The intitial positions of the finger.
initial_positions = np.array([np.pi / 2, np.pi / 2, np.pi / 2,    # Angles
                                     0.,        0.,         0.])  # Angle velocities

# Keys for dictionaries.
T_FS = 'fs'
T_IO = 'io'
T_FP = 'fp'
T_ED = 'ed'

J_DIP = 'dip'
J_PIP = 'pip'
J_MCP = 'mcp'

# Moment arms of each joint for each tendon.
RADII = {
    J_DIP: {
        T_FP: 0.024 / 2.,
        T_ED: 0.010 / 2.
    },
    J_PIP: {
        T_FS: 0.034 / 2.,
        T_IO: 0.020 / 2.,
        T_FP: 0.042 / 2.,
        T_ED: 0.020 / 2.
    },
    J_MCP: {
        T_FS: 0.052 / 2.,
        T_IO: 0.024 / 2.,
        T_FP: 0.044 / 2.,
        T_ED: 0.036 / 2.
    }
}
