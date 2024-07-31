# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import drjit as dr
from mitsuba import Vector2f

from .constants import PI
from .utils import rotation, theta_phi_from_unit_vec,\
    spherical_vectors_from_theta_phi, component_transform


def iso_antenna_pattern(theta, phi, slant_angle):
    r"""
    Isotropic antenna pattern in the local spherical coordinate system of
    the antenna.

    Input
    ------
    theta : [num_samples], float
        Elevation angle [rad]

    phi : [num_samples], float
        Azimuth angle [rad]

    slant_angle : float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    -------
    : [num_samples, 2], float
        Isotropic antenna pattern in the local spherical
        coordinate system (phi_prime_hat, theta_prime_hat).
    """

    # Number of samples
    n = dr.shape(theta)[0]

    f = dr.zeros(Vector2f, n)
    f.x = dr.sin(slant_angle) # Horizontal component
    f.y = dr.cos(slant_angle) # Vertical component

    return f

def dipole_pattern(theta, phi, slant_angle):
    r"""
    Short dipole pattern with linear polarization

    Input
    ------
    theta : [num_samples], float
        Elevation angle [rad]

    phi : [num_samples], float
        Azimuth angle [rad]

    slant_angle : float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    -------
    : [num_samples, 2], float
        Isotropic antenna pattern in the local spherical
        coordinate system (phi_prime_hat, theta_prime_hat).
    """
    # Number of samples
    n = dr.shape(theta)[0]

    k = dr.sqrt(1.5)
    c = k*dr.sin(theta)

    f = dr.zeros(Vector2f, n)
    f.x = c*dr.sin(slant_angle) # Horizontal component
    f.y = c*dr.cos(slant_angle) # Vertical component

    return f

def hw_dipole_pattern(theta, phi, slant_angle):
    r"""
    Half-wavelength dipole pattern with linear polarization

    Input
    ------
    theta : [num_samples], float
        Elevation angle [rad]

    phi : [num_samples], float
        Azimuth angle [rad]

    slant_angle : float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    -------
    : [num_samples, 2], float
        Isotropic antenna pattern in the local spherical
        coordinate system (phi_prime_hat, theta_prime_hat).
    """
    # Number of samples
    n = dr.shape(theta)[0]

    k = dr.sqrt(1.643)
    EPSILON = 1e-9
    c = k*dr.cos(PI/2.*dr.cos(theta))/dr.sin(theta+EPSILON)

    f = dr.zeros(Vector2f, n)
    f.x = c*dr.sin(slant_angle) # Horizontal component
    f.y = c*dr.cos(slant_angle) # Vertical component

    return f

def tr38901_antenna_pattern(theta, phi, slant_angle):
    r"""
    Antenna pattern from 3GPP TR 38.901 (Table 7.3-1) in the local spherical
    coordinate system of the antenna.

    Input
    ------
    theta : [num_samples], float
        Elevation angle [rad]

    phi : [num_samples], float
        Azimuth angle [rad]

    slant_angle : float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    Output
    -------
    : [num_samples, 2], float
        Isotropic antenna pattern in the local spherical
        coordinate system (phi_prime_hat, theta_prime_hat).
    """

    # Number of samples
    n = dr.shape(theta)[0]

    # Wrap phi to [-PI,PI]
    phi = phi+PI
    phi -= dr.floor(phi/(2.*PI))*2.*PI
    phi -= PI

    theta_3db = phi_3db = 65./180.*PI
    a_max = sla_v = 30.
    g_e_max = 8.
    a_v = -dr.min([12.*((theta-PI/2.)/theta_3db)**2, sla_v])
    a_h = -dr.min([12.*(phi/phi_3db)**2, a_max])
    a_db = -dr.min([-(a_v + a_h), a_max]) + g_e_max
    a = dr.power(10., a_db/10.)
    c = dr.sqrt(a)

    f = dr.zeros(Vector2f, n)
    f.x = c*dr.sin(slant_angle) # Horizontal component
    f.y = c*dr.cos(slant_angle) # Vertical component

    return f

def jones_antenna_field(pattern, orientation, slant_angle, d_world):
    r"""
    Computes the Jones vector of the an antenna with pattern `pattern`
    and orientation `orientation` with respect to the world frame, as
    well as the corresponding unit vector basis.

    The input `pattern` is expected to be a callable that takes as input
    the elevation and azimuth angles in the local antenna frame and returns
    the antenna field in the spherical antenna local coordinate system.

    Note: Only real-valued antenna patterns are supported, and therefore
    phases are assumed to be 0 and the imaginary component ignored.

    Input
    ------
    pattern : callable
        Callable that provides the antenna pattern in the spherical local
        frame and takes as input the elevation angle, azimuth angle, and
        slant angle, in the local frame:

        pattern(theta, phi, slant_angle) -> Vector2f

    orientation : [3], float
        Orientation of the antenna with respect to the world frame [rad].
        Format: (z, y, x).

    slant_angle : float
        Slant angle of the linear polarization [rad].
        A slant angle of zero means vertical polarization.

    d_world : [num_samples, 3], float
        Direction of propagation of the field in the world frame

    Output
    -------
    f_world : [num_samples, 2], float
        Antenna pattern in the world frame

    phi_hat_world, theta_hat_world : [num_samples, 3], float
        Spherical basis vector component (x,y) in the world frame
    """

    # Transformation from local to world frame
    to_world = rotation(orientation)

    # Spherical basis vector in the world frame
    theta_world, phi_world = theta_phi_from_unit_vec(d_world)
    theta_hat_world, phi_hat_world =\
        spherical_vectors_from_theta_phi(theta_world, phi_world)

    # Initial rays directions in the local frame
    d_local = to_world.inverse().transform_affine(d_world)

    # Antenna field in the local frame
    theta_local, phi_local = theta_phi_from_unit_vec(d_local)
    f_local = pattern(theta_local, phi_local, slant_angle)

    # Antenna field in the world frame
    theta_hat_local, phi_hat_local =\
        spherical_vectors_from_theta_phi(theta_local, phi_local)
    cpn_transform = component_transform(theta_hat_local, phi_hat_local,
                                                theta_hat_world, phi_hat_world,
                                                transform=to_world)
    f_world = cpn_transform.transform_affine(f_local)

    return f_world, phi_hat_world, theta_hat_world
