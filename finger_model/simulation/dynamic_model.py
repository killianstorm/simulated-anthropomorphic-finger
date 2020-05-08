from sympy import *
from sympy.physics.mechanics import *
from jax.config import config

import jax.numpy as np

config.update("jax_enable_x64", True)

init_printing()

lengths = [0.100, 0.065, 0.035]
masses = [.2, .1, .05]
inertias = [masses[0] * (lengths[0] ** 2) * (1. / 12.),
            masses[1] * (lengths[1] ** 2) * (1. / 12.),
            masses[2] * (lengths[2] ** 2) * (1. / 12.)]
initial_positions = np.array([0., 0., 0.,
                              0., 0., 0.])


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


def finger_dynamic_model():

    thetas = [dynamicsymbols('theta1'),
              dynamicsymbols('theta2'),
              dynamicsymbols('theta3')]

    thetasd = [dynamicsymbols('theta1', 1),
               dynamicsymbols('theta2', 1),
               dynamicsymbols('theta3', 1)]

    thetasdd = [dynamicsymbols('theta1', 2),
                dynamicsymbols('theta2', 2),
                dynamicsymbols('theta3', 2)]

    t = symbols('t')

    l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr, F_fs, F_io, F_fp, F_ed = symbols(
        'l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr, F_fs, F_io, F_fp, F_ed')
    x1 = l1 * sin(thetas[0])
    y1 = -l1 * cos(thetas[0])

    x2 = x1 + l2 * sin(thetas[1])
    y2 = y1 - l2 * cos(thetas[1])

    x3 = x2 + l3 * sin(thetas[2])
    y3 = y2 - l3 * cos(thetas[2])

    xc1 = (l1 / 2.) * sin(thetas[0])
    yc1 = - (l1 / 2.) * cos(thetas[0])

    xc2 = x1 + (l2 / 2.) * sin(thetas[1])
    yc2 = y1 - (l2 / 2.) * cos(thetas[1])

    xc3 = x2 + (l3 / 2.) * sin(thetas[2])
    yc3 = y2 - (l3 / 2.) * cos(thetas[2])

    x1d = diff(x1, t)
    y1d = diff(y1, t)

    x2d = diff(x2, t)
    y2d = diff(y2, t)

    x3d = diff(x3, t)
    y3d = diff(y3, t)

    xc1d = diff(xc1, t)
    yc1d = diff(yc1, t)

    xc2d = diff(xc2, t)
    yc2d = diff(yc2, t)

    xc3d = diff(xc3, t)
    yc3d = diff(yc3, t)

    alpha1 = Rational(1, 2) * pi + thetas[0]
    alpha1d = diff(alpha1, t)

    alpha2 = pi - (thetas[0] - thetas[1])
    alpha2d = diff(alpha2, t)

    alpha3 = pi - (thetas[1] - thetas[2])
    alpha3d = diff(alpha3, t)

    V = m1 * g * yc1 + m2 * g * yc2 + m3 * g * yc3
    T1 = Rational(1, 2) * m1 * (xc1d ** 2 + yc1d ** 2) + Rational(1, 2) * (thetasd[0] ** 2) * I1
    T2 = Rational(1, 2) * m2 * (xc2d ** 2 + yc2d ** 2) + Rational(1, 2) * (thetasd[1] ** 2) * I2
    T3 = Rational(1, 2) * m3 * (xc3d ** 2 + yc3d ** 2) + Rational(1, 2) * (thetasd[2] ** 2) * I3
    L = (T1 + T2 + T3) - V

    N = ReferenceFrame('N')
    r_pp = ReferenceFrame('rigid_pp')
    r_pp.set_ang_vel(N, thetasd[0] * N.z)

    r_mp = ReferenceFrame('rigid_mp')
    r_mp.set_ang_vel(N, thetasd[1] * N.z)

    r_dp = ReferenceFrame('rigid_dp')
    r_dp.set_ang_vel(N, thetasd[2] * N.z)

    j_pp_mp = ReferenceFrame('joint_pp_mp')
    j_pp_mp.set_ang_vel(N, alpha2d * N.z)

    j_mp_dp = ReferenceFrame('joint_mp_dp')
    j_mp_dp.set_ang_vel(N, alpha3d * N.z)

    # DIP Moments caused by tendons FP and ED.
    M_FP_DIP = - F_fp * RADII[J_DIP][T_FP]
    M_ED_DIP = - F_ed * RADII[J_DIP][T_ED]

    # PIP Moments caused by tendons FS, IO, FP and ED.
    M_FS_PIP = - F_fs * RADII[J_PIP][T_FS]
    M_IO_PIP = - F_io * RADII[J_PIP][T_IO]
    M_FP_PIP = - F_fp * RADII[J_PIP][T_FP]
    M_ED_PIP = - F_ed * RADII[J_PIP][T_ED]

    # MCP Moments caused by tendons FS, IO, FP and ED
    M_FS_MCP = - F_fs * RADII[J_MCP][T_FS]
    M_IO_MCP = - F_io * RADII[J_MCP][T_IO]
    M_FP_MCP = - F_fp * RADII[J_MCP][T_FP]
    M_ED_MCP = - F_ed * RADII[J_MCP][T_ED]

    tau3 = M_FP_DIP - M_ED_DIP
    tau2 = M_FS_PIP - M_IO_PIP + M_FP_PIP - M_ED_PIP
    tau1 = M_FS_MCP + M_IO_MCP + M_FP_MCP - M_ED_MCP

    dip_angle_bounds = (pi / 2, pi + 0.1745)
    pip_angle_bounds = (pi / 4, pi)
    mcp_angle_bounds = (pi / 2 + 0.1745, pi + pi / 4)

    break_point = 25

    FL = [(r_pp, tau1 * N.z),  # Tendon torques
          (r_mp, tau2 * N.z),
          (r_dp, tau3 * N.z),

          (r_pp,    (-c_fr * alpha1d - Heaviside(-tau1) * Heaviside(mcp_angle_bounds[0] - alpha1) * break_point * alpha1d - Heaviside(tau1) * Heaviside(alpha1 - mcp_angle_bounds[1]) * break_point * alpha1d) * N.z),  # Joints
          (j_pp_mp, (-c_fr * alpha2d - Heaviside(-tau2) * Heaviside(pip_angle_bounds[0] - alpha2) * break_point * alpha2d - Heaviside(tau2) * Heaviside(alpha2 - pip_angle_bounds[1]) * break_point * alpha2d) * N.z),
          (j_mp_dp, (-c_fr * alpha3d - Heaviside(-tau3) * Heaviside(dip_angle_bounds[0] - alpha3) * break_point * alpha3d - Heaviside(tau3) * Heaviside(alpha3 - dip_angle_bounds[1]) * break_point * alpha3d) * N.z)
          ]

    LM = LagrangesMethod(L, [thetas[0], thetas[1], thetas[2]], forcelist=FL, frame=N)
    equations = LM.form_lagranges_equations()

    y = [thetas[0], thetas[1], thetas[2],
         thetasd[0], thetasd[1], thetasd[2]]

    parameters = [
                l1, m1, I1,
                l2, m2, I2,
                l3, m3, I3,
                F_fs, F_io, F_fp, F_ed,
                g, c_fr]

    unknowns = [Dummy() for i in y]
    unknown_dict = dict(zip(y, unknowns))

    mm = LM.mass_matrix_full.subs(unknown_dict)
    fm = LM.forcing_full.subs(unknown_dict)
    # mm = LM.mass_matrix_full
    # fm = LM.forcing_full

    mapping = {'sin': np.sin, 'cos': np.cos, 'pi': np.pi, 'array': np.array, 'ImmutableDenseMatrix': np.array, 'Heaviside': lambda x: np.heaviside(x, 1)}

    mass_matrix = lambdify([unknowns] + parameters, mm, mapping)
    forcing_matrix = lambdify([unknowns] + parameters, fm, mapping)

    # equations_of_motion = np.linalg.inv(mass_matrix(*)) * forcing_matrix

    #equations_of_motion = LM.rhs()
    #mass_matrix = LM.mass_matrix
    #forcing = LM.forcing

    # equations_of_motion_lambda = lambdify((y,
    #                                        l1, m1, I1, tau1,
    #                                        l2, m2, I2, tau2,
    #                                        l3, m3, I3, tau3,
    #                                        g, c_fr), equations_of_motion, mapping)

    return mass_matrix, forcing_matrix


MM, FM = finger_dynamic_model()

