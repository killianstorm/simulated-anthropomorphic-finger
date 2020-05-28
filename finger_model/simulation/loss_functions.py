import jax.numpy as np

from jax import jit


@jit
def loss_end_effector(reference, simulated):
    """
    Loss function: SSE of end effector positions.
    """
    return np.sqrt(np.mean((reference['end_effector'] - simulated['end_effector']) ** 2))


@jit
def loss_velocities(reference, simulated):
    """
    Loss function: SSE of velocities.
    """
    return np.sqrt(((reference['velocities'] - simulated['velocities']) ** 2).mean())


@jit
def loss_positions(reference, simulated):
    """
    Loss function: SSE of all positions.
    """
    return np.sqrt(((reference['positions'] - simulated['positions']) ** 2).mean())


@jit
def loss_end_position(reference, simulated):
    """
    Loss function: SSE of end effector final position.
    """
    return np.sqrt(np.mean((reference['end_position'] - simulated['end_position']) ** 2))


@jit
def loss_angles(reference, simulated):
    """
    Loss function: SSE of angles and angle velocities.
    """
    return np.sqrt(np.mean((reference['angles'] - simulated['angles']) ** 2))
