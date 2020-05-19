from simulation.simulator import *
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


def oscillator_loss_function(loss, interval, params, tau2, tau3):
    # Create reference.

    reference = simulate_hopf(interval, *params)
    # animation(reference, "reference")

    init_params = params

    vals = num.zeros((tau2.shape[0], tau3.shape[0]))
    for i in range(tau2.shape[0]):
        for j in range(tau3.shape[0]):
            # print(i, j)
            init_params[SCALE1] = tau2[i]
            init_params[SCALE2] = tau3[j]
            temp = simulate_hopf(interval, *init_params)
            vals[i, j] = loss(reference, temp)

    TAU2, TAU3 = np.meshgrid(tau2, tau3)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(TAU2, TAU3, vals)
    plt.xlabel('tau2')
    plt.ylabel('tau3')
    plt.show()

    # Learn parameters to reproduce reference.
    # CMA_ES_oscillator(reference)
    # grad_oscillator(reference, loss_end_effector, 50, 0.1, *params)


def rotate(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def animation(history, dt, name=None, history2=None, tendons=False):
    key_points = history['positions']

    fig = plt.figure(figsize=(8.3333, 6.25), dpi=288)
    ax = fig.add_subplot(111)

    images = []
    di = 100
    N = key_points.shape[1]

    torques = history['torques']
    angles = history['angles']

    for i in tqdm.tqdm(range(0, N, di)):

        plt.cla()

        plt.plot(key_points[:4, i], key_points[4:, i], marker='.', linewidth=5, markersize=5)
        plt.scatter(history['end_effector'][0][:i], history['end_effector'][1][:i], s=0.1)

        if tendons:

            alpha1 = (num.pi / 2 + angles[i, 0])
            alpha2 = num.pi - (angles[i, 0] - angles[i, 1])
            alpha3 = num.pi - (angles[i, 1] - angles[i, 2])

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
            plt.text(xpos, 0.15, 'MCP: ' + str(round(torques[i, 0], 2)))
            plt.text(xpos, 0.10, 'PIP: ' + str(round(torques[i, 1], 2)))
            plt.text(xpos, 0.05, 'D IP: ' + str(round(torques[i, 2], 2)))



        if history2 is not None:
            plt.plot(history2['positions'][:4, i], history2['positions'][4:, i], marker='.')
            plt.scatter(history2['end_effector'][0][:i], history2['end_effector'][1][:i], s=0.1)

            xpos = -0.25
            plt.text(xpos, -0.05, 'Pred. MCP: ' + str(round(torques[i, 0], 2)))
            plt.text(xpos, -0.10, 'Pred. PIP: ' + str(round(torques[i, 1], 2)))
            plt.text(xpos, -0.15, 'Pred. DIP: ' + str(round(torques[i, 2], 2)))

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