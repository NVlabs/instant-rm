# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import mitsuba as mi
import drjit as dr

from .constants import PI
from .utils import fibonacci_lattice, jones_to_stokes, theta_phi_from_unit_vec
from .antenna import jones_antenna_field


class RayTube:
    r"""
    Class implementing a ray tube with the following features:

    * An axial ray, defining the origin and direction of the last segment of
    the ray tube
    * An electric field represented as a Stokes vector
    * Principal radii of curvature
    * A solid angle
    * An origin for the ray tube
    * An flag indicating if the ray is active
    * A length

    Initially, ray tubes are spawn on a lattice centered on the transmitter
    position.
    Radii of curvatures are initialized to 0, and angles 4PI/num_samples.
    The initial reference point is the transmitter position.
    The field of each ray is initialized using the transmit antenna pattern.

    The ray and field vector are represented in the world coordinate system.
    The Stokes field vector is represented in the Mitsuba implicit Stokes basis
    (https://mitsuba.readthedocs.io/en/latest/src/key_topics/polarization.html).

    Parameters
    -----------
    num_samples : int
        Number of samples used to compute the maps

    tx_position : [3], float
        Position of the transmitter

    tx_orientation : [3], float
        Orientation of the transmitter

    tx_slant_angle : float
        Antenna slant angle

    tx_pattern : callable
        Antenna pattern

    Properties
    -----------
    active : bool
        Flag set to `True` if the ray is active

    f_world: mi.Spectrum
        Electric field as a Stokes vector

    ray : mi.Ray
        Axial ray

    rho_1, rho_2 : float
        Radii of curvature [m]

    angle : float
        Solid angle [sr]

    origin : mi.Point3f
        Origin of the ray tube.
        Note: In general, it is different from the origin of the last segment
        of the ray tube

    length : float
        Total length of the ray [m]
    """

    # Makes this class a Dr.JiT PyTree
    DRJIT_STRUCT = {'active'   : dr.mask_t(mi.Float),
                    'f_world'  : mi.Spectrum,
                    'ray'      : mi.Ray3f,
                    'rho_1'    : mi.Float,
                    'rho_2'    : mi.Float,
                    'angle'    : mi.Float,
                    'origin'   : mi.Point3f,
                    'length'   : mi.Float
                    }

    def __init__(self, num_samples, tx_position, tx_orientation, tx_slant_angle,
                 tx_pattern):

        # Initialize the axial rays
        samples_on_square = fibonacci_lattice(num_samples)
        k_world = mi.warp.square_to_uniform_sphere(samples_on_square)
        self.ray = mi.Ray3f(o=tx_position, d=k_world)

        # Mask indicating which rays are active
        self.active = dr.full(dr.mask_t(mi.Float), True, num_samples)

        # Initialize field vector
        # Field vector radiated by the transmit antenna in the world frame and
        # using Jones representation
        ft_jones_world, phi_hat_world, _ = jones_antenna_field(tx_pattern,
                                        tx_orientation, tx_slant_angle,
                                        k_world)
        # Stokes representation of the field vector
        self.f_world = jones_to_stokes(ft_jones_world)
        # Rotate the Stokes vector to represent it in the implicit frame used by
        # Mitsuba.
        # The current reference frame vector `phi_hat_tx_world`
        to_implicit = mi.mueller.rotate_stokes_basis(k_world,
                        phi_hat_world, mi.mueller.stokes_basis(k_world))
        self.f_world = mi.Spectrum(to_implicit)@self.f_world

        # Initialize principal radii of curvature
        self.rho_1 = dr.zeros(mi.Float, num_samples)
        self.rho_2 = dr.zeros(mi.Float, num_samples)

        # Initialize solid angle of the ray tube
        self.angle = dr.full(mi.Float, 4.*PI/num_samples, num_samples)

        # Initialize the origin
        self.origin = dr.tile(tx_position, num_samples)

        # Initialize the length
        self.length = dr.zeros(mi.Float, num_samples)

        # Keep track of direction of departure to compute ASD and ESD
        self.kd_world = k_world

    def sqr_spreading_factor(self, s):
        r"""
        Returns the squared spreading factor evaluated at `s`

        Input
        ------
        s : float
            Position along the axial ray at which to evaluate the squared
            spreading factor

        Output
        -------
        : float
            Squared spreading factor
        """

        spherical = dr.eq(self.rho_1, 0.) & dr.eq(self.rho_2, 0.)

        a2_1 = dr.sqr(dr.rcp(s))
        a2_2 = self.rho_1*self.rho_2*dr.rcp((self.rho_1+s)*(self.rho_2+s))
        a2 = dr.select(spherical, a2_1, a2_2)
        return a2
