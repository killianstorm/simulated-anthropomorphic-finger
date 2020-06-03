from finger_model.simulation.simulator import *
from moviepy.editor import ImageSequenceClip

import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as num
import tqdm
from datetime import datetime
import math

T_FS = 'fs'
T_IO = 'io'
T_FP = 'fp'
T_ED = 'ed'

J_DIP = 'dip'
J_PIP = 'pip'
J_MCP = 'mcp'

RADII = {
    J_DIP: {
        T_FP: 0.010 / 2.,
        T_ED: 0.010 / 2.
    },
    J_PIP: {
        T_FS: 0.035 / 2.,
        T_IO: 0.020 / 2.,
        T_FP: 0.042 / 2.,
        T_ED: 0.020 / 2.
    },
    J_MCP: {
        T_FS: 0.050 / 2.,
        T_IO: 0.025 / 2.,
        T_FP: 0.044 / 2.,
        T_ED: 0.044 / 2.
    }
}


FS_COLOR = 'g'
IO_COLOR = 'y'
FP_COLOR = 'r'
ED_COLOR = 'k'


def fig2image(fig):
    fig.canvas.draw()
    data = num.fromstring(fig.canvas.tostring_rgb(), dtype=num.uint8, sep='')
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def rotate(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin. The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def animate(reference, dt, name=None, approximation=None, tendons=True, di=1):
    """
    Creates an animation of a simulated trajectory. It also allows a second trajectory to be plotted for comparison.
    """

    # Only to be used if no data about other phalanges.
    draw_end_effector_trajectory = False

    di = int(0.01 / dt) # Set framerate of 100 Hz

    if approximation is None:
        key_points = reference['positions']
        torques = reference['torques']
        angles = reference['angles']
    else:
        key_points = approximation['positions']
        torques = approximation['torques']
        angles = approximation['angles']

    fig = plt.figure(figsize=(8.3333, 6.25), dpi=288)
    ax = fig.add_subplot(111)

    images = []
    N = key_points.shape[1]

    for i in tqdm.tqdm(range(0, N, di)):

        plt.cla()

        axes = plt.gca()
        axes.set_xlim([-0.175, 0.275])
        axes.set_ylim([-0.25, 0.15])
        axes.set_aspect('equal')

        # Plot reference or approximated finger movement.
        if approximation is None:
            plt.plot(key_points[:4, i], key_points[4:, i], marker='.', linewidth=5, markersize=5, label="ref. finger")
            plt.scatter(reference['end_effector'][0][:i], reference['end_effector'][1][:i], s=0.1, label="ref. trajectory")
        else:
            plt.plot(key_points[:4, i], key_points[4:, i], marker='.', linewidth=5, markersize=5, label="approx. finger")
            plt.scatter(approximation['end_effector'][0][:i], approximation['end_effector'][1][:i], s=0.1, label="approx. trajectory")

            if 'positions' in reference:
                plt.plot(reference['positions'][:4, i], reference['positions'][4:, i], marker='.', label="ref. finger")
            plt.scatter(reference['end_effector'][0][:i], reference['end_effector'][1][:i], s=0.1, label="ref. trajectory")

            plt.legend()

        if ENABLE_PIANO_KEY:

            p_c = pianokey_coordinates[1]
            if key_points[7, i] < pianokey_coordinates[1]:
                p_c = key_points[7, i]

            rect = patches.Rectangle((pianokey_coordinates[0], p_c - pianokey_height), 1,
                                     pianokey_height, linewidth=3, edgecolor='k', facecolor='none')
            axes.add_patch(rect)

            plt.text(pianokey_coordinates[0] + 0.05, p_c - pianokey_height + 0.05, "PIANO KEY")

        # Plot reference trajectory.

        if tendons:

            ### FS tendon ###
            # MCP
            x, y = key_points[0, i], key_points[4, i]
            mcp_x, mcp_y = rotate((x, y - RADII[J_MCP][T_FS]), (x, y), - (num.pi / 2 - angles[i, 0]))
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_MCP][T_FS], color=FS_COLOR, linewidth=0.1))

            # PIP
            x, y = key_points[1, i], key_points[5, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_PIP][T_FS], color=FS_COLOR, linewidth=0.1))
            pip_x, pip_y = rotate((x, y - RADII[J_PIP][T_FS]), (x, y), -(num.pi / 2 - angles[i, 1]))

            plt.plot([-0.2, -0.1, mcp_x, pip_x], [-0.1, -RADII[J_MCP][T_FS], mcp_y, pip_y], color=FS_COLOR)

            plt.text(-0.25, -0.125, 'FS: ' + str(round(torques[i, 0], 2)))

            ### FP tendon ###
            # MCP
            x, y = key_points[0, i], key_points[4, i]
            mcp_x, mcp_y = rotate((x, y - RADII[J_MCP][T_FP]), (x, y), - (num.pi / 2 - angles[i, 0]))
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_MCP][T_FP], color=FP_COLOR, linewidth=0.1))

            # PIP
            x, y = key_points[1, i], key_points[5, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_PIP][T_FP], color=FP_COLOR, linewidth=0.1))
            pip_x, pip_y = rotate((x, y - RADII[J_PIP][T_FP]), (x, y), -(num.pi / 2 - angles[i, 1]))

            # DIP
            x, y = key_points[2, i], key_points[6, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_DIP][T_FP], color=FP_COLOR, linewidth=0.1))
            dip_x, dip_y = rotate((x, y - RADII[J_DIP][T_FP]), (x, y), -(num.pi / 2 - angles[i, 1]))

            plt.plot([-0.1, -0.1, mcp_x, pip_x, dip_x], [-0.1, -0.05, mcp_y, pip_y, dip_y], color=FP_COLOR)
            plt.text(-0.125, -0.125, 'FP: ' + str(round(torques[i, 2], 2)))

            ### IO tendon ###
            # MCP
            x, y = key_points[0, i], key_points[4, i]
            mcp_x, mcp_y = rotate((x, y - RADII[J_MCP][T_IO]), (x, y), - (num.pi / 2 - angles[i, 0]))
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_MCP][T_IO], color=IO_COLOR, linewidth=0.1))

            # PIP
            x, y = key_points[1, i], key_points[5, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_PIP][T_IO], color=IO_COLOR, linewidth=0.1))
            pip_x, pip_y = rotate((x, y + RADII[J_PIP][T_IO]), (x, y), -(num.pi / 2 - angles[i, 1]))

            plt.plot([-0.2, -0.1, mcp_x, pip_x], [-RADII[J_MCP][T_IO], -RADII[J_MCP][T_IO], mcp_y, pip_y], color=IO_COLOR)

            plt.text(-0.25, -RADII[J_MCP][T_IO] - 0.025, 'IO: ' + str(round(torques[i, 1], 2)))


            ### ED tendon ###
            # MCP
            x, y = key_points[0, i], key_points[4, i]
            mcp_x, mcp_y = rotate((x, y + RADII[J_MCP][T_ED]), (x, y), - (num.pi / 2 - angles[i, 0]))
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_MCP][T_ED], color=ED_COLOR, linewidth=0.1, fill=False))

            # PIP
            x, y = key_points[1, i], key_points[5, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_PIP][T_ED], color=ED_COLOR, linewidth=0.1, fill=False))
            pip_x, pip_y = rotate((x, y + RADII[J_PIP][T_ED]), (x, y), -(num.pi / 2 - angles[i, 1]))

            # DIP
            x, y = key_points[2, i], key_points[6, i]
            ax.add_patch(patches.Circle((x, y), radius=RADII[J_DIP][T_ED], color=ED_COLOR, linewidth=0.1, fill=False))
            dip_x, dip_y = rotate((x, y + RADII[J_DIP][T_ED]), (x, y), -(num.pi / 2 - angles[i, 1]))

            plt.plot([-0.2, -0.1, mcp_x, pip_x, dip_x], [RADII[J_MCP][T_ED], RADII[J_MCP][T_ED], mcp_y, pip_y, dip_y], color=ED_COLOR)

            plt.text(-0.25, -RADII[J_MCP][T_ED] + 0.025, 'ED: ' + str(round(torques[i, 3], 2)))
        else:
            xpos = -0.25
            plt.text(xpos, 0.15, 'Approx. MCP: ' + str(round(torques[i, 0], 2)))
            plt.text(xpos, 0.10, 'Approx. PIP: ' + str(round(torques[i, 1], 2)))
            plt.text(xpos, 0.05, 'Approx. DIP: ' + str(round(torques[i, 2], 2)))

        plt.axhline(0)
        plt.axis('equal')
        plt.axis([-0.3, 0.3, -0.3, 0.3])
        images.append(fig2image(fig))

    filename = str(name) + datetime.now().strftime("_%d-%b-%Y_(%H:%M:%S.%f)") + ".mp4"

    if name is None:
        ImageSequenceClip(images, fps=int(1/dt/di)).ipython_display()
    else:
        ImageSequenceClip(images, fps=int(1/dt/di)).write_videofile(filename)
    return filename