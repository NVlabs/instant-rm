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
import mitsuba as mi
import numpy as np
from mitsuba import Vector3f, Point2f, Transform4f, Vector4f, Matrix3f,\
    Transform3f, ScalarTransform4f, Int, Float64, Point2i, Vector3f, Point2i,\
    Matrix4d, Vector2f

from .constants import PI


def fibonacci_lattice(num_points):
    """
    Generates a Fibonacci lattice for the unit square

    Input
    -----
    num_points : int
        Number of points

    Output
    -------
    points : [num_points, 2]
        Generated rectangular coordinates of the lattice points
    """

    golden_ratio = (1.+dr.sqrt(Float64(5.)))/2.
    ns = dr.arange(Float64, 0, num_points)

    x = ns/golden_ratio
    x = x - dr.floor(x)
    y = ns/(num_points-1)

    points = Point2f()
    points.x = x
    points.y = y

    return points

def complex_sqrt(x):
    r"""
    Computes the square root of a complex scalar `x`

    Input
    ------
    x : complex
        Complex number for which to compute the square root

    Output
    -------
    x : complex
        The square root of `x`
    """

    x_r = dr.real(x)
    x_i = dr.imag(x)

    x_i_sqr = dr.sqr(x_i)

    epsilon = 1e-10
    r = dr.safe_sqrt(dr.sqr(x_r) + x_i_sqr+epsilon)
    r_sqrt = dr.safe_sqrt(r+epsilon)

    f = dr.rcp(dr.safe_sqrt(dr.sqr(x_r + r) + x_i_sqr+epsilon))*r_sqrt
    x_sqrt = mi.Complex2f(f*(x_r + r), x_i*f)

    return x_sqrt

def rotation(orientation):
    r"""
    Computes the transform corresponding to a rotation by angles `orientation`

    Input
    ------
    orientation : [3], float
        Rotation angles with respect to axes [z, y, x] [rad].

    Output
    -------
    : mi.ScalarTransform4f
        Transform
    """

    r2d = 180./PI

    mat =     Transform4f.rotate(axis=[0, 0, 1], angle=orientation.x*r2d)\
            @ Transform4f.rotate(axis=[0, 1, 0], angle=orientation.y*r2d)\
            @ Transform4f.rotate(axis=[1, 0, 0], angle=orientation.z*r2d)
    return mat

def theta_phi_from_unit_vec(v):
    r"""
    Computes zenith and azimuth angles corresponding to unit-norm vectors `v`

    Input
    ------
    v : [num_samples,3], float
        Unit vectors

    Output
    -------
    theta : [num_samples], float
        Zenith angles

    phi : [num_samples], float
        Azimuth angles
    """

    if dr.shape(v)[1] != 3:
		raise ValueError("Input vectors must be 3-dimensional.")
    # Clip to ensure numerical stability
    theta = dr.acos(v.z)
    phi = dr.atan2(v.y, v.x)
    return theta, phi

def spherical_vectors_from_theta_phi(theta, phi):
    r"""
    Computes the spherical unit vectors `theta_hat` and `phi_hat` from the
    zenith and azimuth angles

    Input
    -------
    theta : [num_samples], float
        Zenith angle [rad]

    phi : [num_samples], float
        Azimuth angle [rad]

    Output
    --------
    theta_hat,phi_hat : [num_samples, 3], float
        Vectors of the spherical basis
    """

    # Number of samples
    n = dr.shape(theta)[0]

    # Initialize vectors
    theta_hat = dr.zeros(Vector3f, n)
    phi_hat = dr.zeros(Vector3f, n)

    cos_theta = dr.cos(theta)
    sin_theta = dr.sin(theta)
    cos_phi = dr.cos(phi)
    sin_phi = dr.sin(phi)

    theta_hat.x = cos_theta*cos_phi
    theta_hat.y = cos_theta*sin_phi
    theta_hat.z = -sin_theta

    phi_hat.x = -sin_phi
    phi_hat.y = cos_phi

    return theta_hat, phi_hat

def component_transform(theta_prime_hat, phi_prime_hat, theta_hat, phi_hat,
                        transform=None):
    """
    Computes basis change matrix for going from spherical coordinate
    system (`theta_prime_hat`, `phi_prime_hat`) to (`theta_hat`, `phi_hat`).

    An optional rotation matrix `transform` can be applied to
    (`theta_prime_hat`, `phi_prime_hat`) if these vectors are not provided
    in the same coordinate system as (`theta_hat`, `phi_hat`).

    Input
    -----
    theta_prime_hat : [..., 3], float
        Source unit vector for theta_hat component

    phi_prime_hat : [..., 3], float
        Source unit vector for phi_hat component

    theta_hat : [..., 3], float
        Target unit vector for theta_hat

    phi_hat : [..., 3], float
        Target unit vector for phi_hat component

    transform : [..., 2, 2], float
        Optional transform to apply to (theta_prime_hat, phi_prime_hat).
        This is useful if (theta_prime_hat, phi_prime_hat) is provided in a
        different coordinate system than (theta_hat, phi_hat).

    Output
    -------
    c : [..., 2, 2], float
        Change of basis matrix
    """

    if transform:
        # Represent the (theta_prime, phi_prime) in the same coordinate
        # system as (theta, phi)
        theta_prime_hat = transform.transform_affine(theta_prime_hat)
        phi_prime_hat = transform.transform_affine(phi_prime_hat)

    # Compute the matrix components
    c_11 = dr.dot(theta_hat, theta_prime_hat)
    c_12 = dr.dot(theta_hat, phi_prime_hat)
    c_21 = dr.dot(phi_hat, theta_prime_hat)
    c_22 = dr.dot(phi_hat, phi_prime_hat)

    c = Matrix3f(c_11, c_12, 0.0,
                c_21, c_22, 0.0,
                0.0,   0.0, 1.0)

    return Transform3f(c)

def jones_to_stokes(v_jones):
    r"""
    Converts a Jones vector to is Stokes representation

    Input
    ------
    v_jones : [num_samples, 2], float
        Jones vectors

    Output
    -------
    : [num_samples, 4], float
        Stokes representation of `v_jones`
    """

    # Number of samples
    n = dr.shape(v_jones)[0]

    # Jones components
    ex = v_jones.x # Horizontal component
    ey = v_jones.y # Vertical component
    ex_abs_sq = dr.abs(ex)**2
    ey_abs_sq = dr.abs(ey)**2

    v_stokes = dr.zeros(Vector4f, n)
    v_stokes.x = ex_abs_sq + ey_abs_sq
    v_stokes.y = ex_abs_sq - ey_abs_sq
    v_stokes.z = 2.*dr.real(ex*dr.conj(ey))
    v_stokes.w = -2.*dr.imag(ex*dr.conj(ey))

    return v_stokes

def specular_mueller_matrix(wi_local, c_eta):
    r"""
    Computes the Mueller matrix for specular reflection and the direction
    of the specularly reflected ray in the local coordinate system.

    Input
    ------
    wi_local : [num_samples, 3], float
        Incident direction in the local coordinate system

    c_eta : [num_samples], complex
        Complex relative permittivity

    Output
    -------
    m_local : Spectrum
        Mueller matrix in the local coordinate system

    ks_local : [num_samples, 3]
        Directions of specular reflection in the local coordinate system
    """

    # Cosine of incident angle
    cos_theta_i = dr.abs(wi_local.z)

    # Mueller matrix for specular reflection
    # Complex refractive index
    c_n = complex_sqrt(c_eta)
    # Mueller matrix for specular reflection
    m_local = mi.mueller.specular_reflection(cos_theta_i, c_n)

    # Direction of specularly reflected ray
    ks_local = mi.reflect(wi_local)

    return m_local, ks_local

def mueller_to_implicit(m_local, wi_local, ko_local):
    r"""
    Transforms the Mueller matrix `m_local` to operate in the Mitsuba implicit
    local coordinate system.

    The Mueller matrix is initially assumed to operate in the local coordinate
    system.

    See:
    https://mitsuba.readthedocs.io/en/latest/src/key_topics/polarization.html

    Input
    ------
    m_local : Spectrum
        Mueller matrix in the local coordinate system

    wi_local : [num_samples, 3], float
        Incident direction in the local coordinate system

    ko_local : [num_samples, 3], float
        Direction of the reflected ray in the local coordinate system

    Input
    ------
    m_local : Spectrum
        Mueller matrix in the Mitsuba implicit local coordinate system
    """

    ki_local = -wi_local

    # Normal in local coordinate system
    s_i_current = Vector3f(-ki_local.y, ki_local.x, 0.0)
    s_o_current = Vector3f(-ko_local.y, ko_local.x, 0.0)

    # S axis for the incident and specularly reflected ray in the implicit
    # Mitsuba coordinate system
    s_i_target = mi.mueller.stokes_basis(ki_local)
    s_o_target = mi.mueller.stokes_basis(ko_local)

    # Avoid singularity when the incident direction, and the normal are
    # collinear
    collinear = dr.eq(s_i_current, 0.)
    s_i_current = dr.select(collinear, Vector3f(1.,0.,0.),
                            dr.normalize(s_i_current))
    s_o_current = dr.select(collinear, Vector3f(1.,0.,0.),
                            dr.normalize(s_o_current))

    m_local = mi.mueller.rotate_mueller_basis(m_local,
                                        ki_local, s_i_current, s_i_target,
                                        ko_local, s_o_current, s_o_target)

    return m_local

def mp_to_world(center, orientation, size):
    """
    Build the `to_world` transformation that maps a default Mitsuba rectangle
    to the rectangle that defines the measurement plane.

    Input
    ------
    center : [3], float
        Center of the rectangle

    orientation : [3], float
        Orientation of the rectangle [rad]

    size : [2], float
        Scale of the rectangle.
        The width of the rectangle (in the local X direction) is scale[0]
        and its height (in the local Y direction) scale[1].

    Output
    -------
    : ScalarTransform4
        Rectangle to world transformation
    """

    orientation = 180. * orientation / PI
    return (
        ScalarTransform4f.translate(center)
        @ ScalarTransform4f.rotate(axis=[0, 0, 1], angle=orientation[0])
        @ ScalarTransform4f.rotate(axis=[0, 1, 0], angle=orientation[1])
        @ ScalarTransform4f.rotate(axis=[1, 0, 0], angle=orientation[2])
        @ ScalarTransform4f.scale([0.5 * size[0], 0.5 * size[1], 1])
    )

def mp_uv_2_cell_ind(uv, mp_num_cells):
    r"""
    Computes the indices of the hitted cells of the map from the UV coordinates

    Input
    ------
    uv : [num_samples, 2], float
        Coordinates of the intersected points in UV space

    mp_num_cells : [2], int
        Number of cells forming the measurement plane

    Output
    -------
    : [num_samples], int
        Cell indices in the flattened measurement plane
    """

    # Size of a cell in UV space
    cell_size_uv = Vector2f(mp_num_cells)

    # Cell indices in the 2D measurement plane
    # [num_samples, 2]
    cell_ind = Point2i(dr.floor(uv*cell_size_uv))

    # Cell indices for the flattened measurement plane
    # [num_samples]
    cell_ind = cell_ind[0]*mp_num_cells[1]+cell_ind[1]

    return cell_ind

def mat_inverse_double(m):
    """
    Invert a matrix using double-precision arithmetic

    Input
    -------
    m : mi.Spectrum
        Matrix to inverse as a Mitsuba Spectrum object

    Output
    -------
    : mi.Spectrum
        Matrix inverse as a Mitsuba Spectrum object
    """

    # Convert to a simple 4x4 f64-valued matrix
    a_f64 = Matrix4d(*[m[col][row].x for row in range(4) for col in range(4)])

    # Perform inverse
    inv_f64 = dr.inverse(a_f64)

    # Convert back to `mi.Spectrum`
    inv = mi.Spectrum(inv_f64)

    return inv
