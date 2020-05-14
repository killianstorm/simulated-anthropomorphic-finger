import jax.numpy as np

from jax import jit


# Loss function: SSE of end effector positions.
@jit
def loss_end_effector(reference, simulated):
    return np.sqrt(np.mean((reference['end_effector'] - simulated['end_effector']) ** 2))


# Loss function: SSE of velocities.
@jit
def loss_velocities(reference, simulated):
    return np.sqrt(((reference['velocities'] - simulated['velocities']) ** 2).mean())


# Loss function: SSE of all positions.
@jit
def loss_positions(reference, simulated):
    return np.sqrt(((reference['positions'] - simulated['positions']) ** 2).mean())


# Loss function: SSE of accelerations.
@jit
def loss_accelerations(reference, simulated):
    return np.sqrt(np.mean((reference['accelerations'] - simulated['accelerations']) ** 2))


@jit
def loss_end_position(reference, simulated):
    return np.sqrt(np.mean((reference['end_position'] - simulated['end_position']) ** 2))


@jit
def loss_angles(reference, simulated):
    return np.sqrt(np.mean((reference['angles'] - simulated['angles']) ** 2))


def loss_whatever(reference, simulated):
    return simulated['end_position'][-1]