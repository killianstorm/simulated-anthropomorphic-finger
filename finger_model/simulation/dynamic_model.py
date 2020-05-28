from sympy import *
from sympy.physics.mechanics import *
from jax.config import config

import jax.numpy as np

from finger_model.simulation.parameters import *

config.update("jax_enable_x64", True)

init_printing()


def equations_of_motion():
    """
    Constructs the equations of motion by using the variables declared above.
    returns: lambdified mass and force matrix
    """

    # Angles
    theta1 = dynamicsymbols('theta1')
    theta2 = dynamicsymbols('theta2')
    theta3 = dynamicsymbols('theta3')

    # Angle velocities.
    thetad1 = dynamicsymbols('theta1', 1)
    thetad2 = dynamicsymbols('theta2', 1)
    thetad3 = dynamicsymbols('theta3', 1)

    # Time.
    t = symbols('t')

    if ENABLE_TENDONS:
        # Tendons are activated, use forces.
        F_fs, F_io, F_fp, F_ed = dynamicsymbols('F_fs, F_io, F_fp, F_ed')
        l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr = symbols(
            'l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr')
    else:
        # If not, use torques.
        tau1, tau2, tau3 = dynamicsymbols('tau1, tau2, tau3')
        l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr,  = symbols(
            'l1, m1, I1, l2, m2, I2, l3, m3, I3, g, c_fr')

    # Proximal phalanx coordinates
    x1 = l1 * sin(theta1)
    z1 = -l1 * cos(theta1)

    # Middle phalanx coordinates
    x2 = x1 + l2 * sin(theta2)
    z2 = z1 - l2 * cos(theta2)

    # Distal phalanx coordinates
    x3 = x2 + l3 * sin(theta3)
    z3 = z2 - l3 * cos(theta3)

    # Proximal phalanx center of gravity coordinates
    xc1 = (l1 / 2.) * sin(theta1)
    zc1 = - (l1 / 2.) * cos(theta1)

    # Middle phalanx center of gravity coordinates
    xc2 = x1 + (l2 / 2.) * sin(theta2)
    zc2 = z1 - (l2 / 2.) * cos(theta2)

    # Distal phalanx center of gravity coordinates
    xc3 = x2 + (l3 / 2.) * sin(theta3)
    zc3 = z2 - (l3 / 2.) * cos(theta3)

    # Proximal phalanx center of gravity velocities
    xc1d = diff(xc1, t)
    zc1d = diff(zc1, t)

    # Middle phalanx center of gravity velocities
    xc2d = diff(xc2, t)
    zc2d = diff(zc2, t)

    # Distal phalanx center of gravity velocities
    xc3d = diff(xc3, t)
    zc3d = diff(zc3, t)

    # Relative angle MCP joint.
    alpha1 = Rational(1, 2) * pi + theta1
    alpha1d = diff(alpha1, t)

    # Relative angle PIP joint.
    alpha2 = pi - (theta1 - theta2)
    alpha2d = diff(alpha2, t)

    # Relative angle DIP joint.
    alpha3 = pi - (theta2 - theta3)
    alpha3d = diff(alpha3, t)

    # Potential energy.
    V = m1 * g * zc1 + m2 * g * zc2 + m3 * g * zc3

    # Kinectic energy.
    T1 = Rational(1, 2) * m1 * (xc1d ** 2 + zc1d ** 2) + Rational(1, 2) * (thetad1 ** 2) * I1
    T2 = Rational(1, 2) * m2 * (xc2d ** 2 + zc2d ** 2) + Rational(1, 2) * (thetad2 ** 2) * I2
    T3 = Rational(1, 2) * m3 * (xc3d ** 2 + zc3d ** 2) + Rational(1, 2) * (thetad3 ** 2) * I3

    # Lagrangian.
    L = (T1 + T2 + T3) - V

    # Declaration of angular movements.
    N = ReferenceFrame('N')
    r_pp = ReferenceFrame('rigid_pp')
    r_pp.set_ang_vel(N, thetad1 * N.z)

    r_mp = ReferenceFrame('rigid_mp')
    r_mp.set_ang_vel(N, thetad2 * N.z)

    r_dp = ReferenceFrame('rigid_dp')
    r_dp.set_ang_vel(N, thetad3 * N.z)

    j_pp_mp = ReferenceFrame('joint_pp_mp')
    j_pp_mp.set_ang_vel(N, alpha2d * N.z)

    j_mp_dp = ReferenceFrame('joint_mp_dp')
    j_mp_dp.set_ang_vel(N, alpha3d * N.z)

    def d2r(d):
        """
        Degrees to radians
        """
        return d * (pi / 180.)

    # Joint angle bounds.
    mcp_angle_bounds = (d2r(95), d2r(225))
    pip_angle_bounds = (d2r(60), d2r(220))
    dip_angle_bounds = (d2r(80), d2r(185))


    if tendons:
        # DIP moment caused by FP.
        M_FP_DIP = - F_fp * RADII[J_DIP][T_FP]

        # DIP moment caused by ED.
        M_ED_DIP = - F_ed * ((alpha2 - pip_angle_bounds[0]) / (pip_angle_bounds[1] - pip_angle_bounds[0])) * lengths[2] * 0.18 # Lengths[2] because there is no pulley, only several attachments to the length of the dip

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

        # Moments are added to become torques for each joint.
        tau3 = M_FP_DIP - M_ED_DIP
        tau2 = M_FS_PIP - M_IO_PIP + M_FP_PIP - M_ED_PIP
        tau1 = M_FS_MCP + M_IO_MCP + M_FP_MCP - M_ED_MCP

    # Friction coefficients.
    c_fr_mcp = 0.1
    c_fr_pip = 0.1
    c_fr_dip = 0.1

    if ENABLE_LIGAMENTS:
        # If using ligaments.
        c_ligament = 25.

        # Force list with activated ligaments.
        FL = [(r_pp, tau1 * N.z),  # Tendon torques
              (r_mp, tau2 * N.z),
              (r_dp, tau3 * N.z),

              (r_pp,    (-c_fr_mcp * alpha1d - Heaviside(-tau1) * Heaviside(mcp_angle_bounds[0] - alpha1) * c_ligament * alpha1d - Heaviside(tau1) * Heaviside(alpha1 - mcp_angle_bounds[1]) * c_ligament * alpha1d) * N.z),  # Joints
              (j_pp_mp, (-c_fr_pip * alpha2d - Heaviside(-tau2) * Heaviside(pip_angle_bounds[0] - alpha2) * c_ligament * alpha2d - Heaviside(tau2) * Heaviside(alpha2 - pip_angle_bounds[1]) * c_ligament * alpha2d) * N.z),
              (j_mp_dp, (-c_fr_dip * alpha3d - Heaviside(-tau3) * Heaviside(dip_angle_bounds[0] - alpha3) * c_ligament * alpha3d - Heaviside(tau3) * Heaviside(alpha3 - dip_angle_bounds[1]) * c_ligament * alpha3d) * N.z)
              ]
    else:
        # If not using ligaments.
        # Force list with disabled ligaments.
        FL = [(r_pp, tau1 * N.zt),  # Tendon torques
              (r_mp, tau2 * N.z),
              (r_dp, tau3 * N.z),

              (r_pp,    (-c_fr_mcp * alpha1d) * N.z),  # Joints
              (j_pp_mp, (-c_fr_pip * alpha2d) * N.z),
              (j_mp_dp, (-c_fr_dip * alpha3d) * N.z)]

    # Construction of Euler-Lagrange equations.
    LM = LagrangesMethod(L, [theta1, theta2, theta3], forcelist=FL, frame=N)

    # Calculate equations of motion.
    equations = LM.form_lagranges_equations()

    # Set unknowns.
    y = [theta1, theta2, theta3,
         thetad1, thetad2, thetad3]

    if ENABLE_TENDONS:
        # If using tendons, request forces.
        parameters = [
                    l1, m1, I1,
                    l2, m2, I2,
                    l3, m3, I3,
                    F_fs, F_io, F_fp, F_ed,
                    g, c_fr]
    else:
        # If using tendons, request torques.
        parameters = [
                    l1, m1, I1,
                    l2, m2, I2,
                    l3, m3, I3,
                    tau1, tau2, tau3,
                    g, c_fr]

    # Construct mass and force matrix by substituting unknowns with Dummy.
    unknowns = [Dummy() for _ in y]
    unknown_dict = dict(zip(y, unknowns))

    mm = LM.mass_matrix_full.subs(unknown_dict)
    fm = LM.forcing_full.subs(unknown_dict)

    # Map sympy functions to jax numpy functions when lambdifying
    mapping = {'sin': np.sin, 'cos': np.cos, 'pi': np.pi, 'array': np.array, 'ImmutableDenseMatrix': np.array, 'Heaviside': lambda x: np.heaviside(x, 1)}

    # Lambdify matrices.
    mass_matrix = lambdify([unknowns] + parameters, mm, mapping)
    forcing_matrix = lambdify([unknowns] + parameters, fm, mapping)

    return mass_matrix, forcing_matrix


# Construct equations of motion.
MM, FM = equations_of_motion()
