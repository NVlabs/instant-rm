# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import mitsuba as mi
import drjit as dr

from .constants import PI, SPEED_OF_LIGHT
from .utils import mueller_to_implicit, specular_mueller_matrix


class LambertianMaterial(mi.BSDF):
    r"""
    Mitsuba BSDF implementing the Lambertian scattering model from
    [Degli-Esposti07], but without cross-polarization leakage controlled by a
    settable parameter (XPD). Cross-polarization leakage are modeled as for
    specular reflection.

    [Degli-Esposi07] V. Degli-Esposti, F. Fuschini, E. M. Vitucci and
    G. Falciasecca, "Measurement and Modelling of Scattering From Buildings,"
    in IEEE Transactions on Antennas and Propagation.
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        # Read BSDF properties
        # Scattering coefficient
        s = 0.0
        if props.has_property('s'):
            s = props['s']
            if s < 0.0 or s > 1.0:
                msg = "Scattering coefficient must be in range (0,1)"
                raise ValueError(msg)
        self.s = mi.Float(s)
        # Real part of the relative permittivity
        eta_r = 1.0
        if props.has_property('eta_r'):
            eta_r = props['eta_r']
            if eta_r < 1.0:
                msg = "Real part of the relative permittivity must greater "\
                    "or equal to 1"
                raise ValueError(msg)
        self.eta_r = mi.Float(eta_r)
        # Imaginary part of the relative permittivity
        eta_i = 0.0
        if props.has_property('eta_i'):
            eta_i = props['eta_i']
            if eta_i < 0.0:
                msg = "Imaginary part of the relative permittivity must be "\
                    "greater or equal to 0"
                raise ValueError(msg)
        self.eta_i = mi.Float(eta_i)

    def sample(self, ctx, si, sample1, sample2, active=True):
        r"""
        Samples the radio material.

        Input
        ------
        ctx : mi.BSDFContext
         Context describing which lobes to sample, and whether radiance or
         importance are being transported.
         Not relevant for this material. Only importance sampling is handled.

        si : mi.SurfaceInteraction3f
            Surface interaction data structure describing the underlying
            surface position.

        sample1 : Float
            A uniformly distributed sample on (0,1). It is used to randomly
            sample reflection event, i.e., specular or diffuse, depending on the
            scattering coefficient.

        sample2 : Float
            A uniformly distributed sample on (0,1)^2. It is used to sample the
            direction for diffuse reflection.

        active : Bool
            Mask to specify active rays

        Output
        -------
        bs : mi.BSDFSample3f
            Object indicating the sampled direction and other information.

        value : mi.Spectrum
            Mueller matrix in the local coordinate system
        """

        wi_local = si.wi
        c_eta = mi.Complex2f(self.eta_r, -self.eta_i)

        # Mueller matrix for specular reflection and the direction of specular
        # reflection
        m_local, ks_local = specular_mueller_matrix(wi_local, c_eta)

        # Direction for diffusely reflected rays
        kd_local = self.diffuse_direction(sample2)
        # Weight for diffuse reflection
        w_diffuse = self.diffuse_weight(kd_local)

        # Scale Mueller matrices according to the scattering coefficient to
        # enable gradient for this parameter
        # Diffuse
        p = dr.sqr(self.s)
        EPSILON = 1e-9
        w_diffuse *= p*dr.detach(dr.rcp(p+EPSILON))
        # Specular
        r = 1. - p
        w_specular = r*dr.detach(dr.rcp(r+EPSILON))

        # Sample reflection type
        specular = self.reflection_type(sample1)

        # Weighting to apply to the Mueller matrix
        w = dr.select(specular, w_specular, w_diffuse)

        # Mueller matrix for diffuse reflection
        m_local = w*m_local
        # Direction of reflection
        ko_local = dr.select(specular, ks_local, kd_local)

        # Move to implicit Mitsuba local coordinate system
        m_local = mueller_to_implicit(m_local, wi_local, ko_local)

        # Instantiate and set the BSDFSample object
        bs = mi.BSDFSample3f()
        bs.pdf = 0.0 # Not used
        bs.sampled_component = dr.select(specular,
                                         mi.UInt32(0), # Specular
                                         mi.UInt32(1)) # Diffuse
        bs.sampled_type = 0 # Not used
        bs.wo = ko_local
        bs.eta = 1.0 # Not used

        return bs, m_local

    def eval(self, ctx, si, wo, active):
        raise NotImplementedError()

    def pdf(self, ctx, si, wo, active):
        raise NotImplementedError()

    def traverse(self, callback):
        callback.put_parameter('eta_r', self.eta_r,
                               mi.ParamFlags.Differentiable)
        callback.put_parameter('eta_i', self.eta_i,
                               mi.ParamFlags.Differentiable)
        callback.put_parameter('s', self.s,
                               mi.ParamFlags.Differentiable)

    def to_string(self):
        s = f"Lambertian[\n"\
            f"\teta_r={self.eta_r}"\
            f"\teta_i={self.eta_i}"\
            f"\ts={self.s}"\
            f"]"
        return s

    ###########################################
    # Utilities
    ###########################################

    def diffuse_weight(self, kd_local):
        """
        Weight for the Lambertian diffuse reflection

        Input
        ------
        kd_local : [num_samples, 3], float
            Reflection direction in local frame

        Output
        ------
        : [num_samples], float
            Weight
        """

        cos_theta_d = dr.abs(kd_local.z)
        w = cos_theta_d/PI

        return w

    def diffuse_direction(self, sample2):
        """
        Direction on the hemisphere of diffuse reflection

        Input
        ------
        sample2 : [num_samples, 2], float
            A uniformly distributed sample on (0,1)^2.

        Output
        -------
        : [num_samples, 3], float
            Diffuse reflection direction in the local coordinate system
        """

        kd_local = mi.warp.square_to_uniform_hemisphere(sample2)
        # Due to numerical error, it could be that kd_local.z is slightly
        # negative
        kd_local.z = dr.abs(kd_local.z)
        return kd_local

    def reflection_type(self, sample1):
        """
        Reflection type, i.e., specular or diffuse, depending on the scattering
        coefficient `s`.

        Input
        ------
        sample1 : [num_samples], float
            A uniformly distributed sample on (0,1)

        Output
        -------
        : [num_samples], bool
            True if specular, False if diffuse
        """

        p = dr.sqr(self.s)
        specular = sample1 > p
        return specular
