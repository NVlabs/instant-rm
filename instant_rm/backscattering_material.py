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
import scipy
import mitsuba as mi
import drjit as dr

from .constants import PI, SPEED_OF_LIGHT
from .utils import mueller_to_implicit, specular_mueller_matrix


class BackscatteringMaterial(mi.BSDF):
    r"""
    Mitsuba BSDF implementing the backscattering lobe model from
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
        # alpha_r
        alpha_r = 1
        if props.has_property('alpha_r'):
            alpha_r = props['alpha_r']
            if alpha_r < 1:
                raise ValueError("alpha_r must be >= 1")
        self.alpha_r = alpha_r
        # alpha_i
        alpha_i = 1
        if props.has_property('alpha_i'):
            alpha_i = props['alpha_i']
            if alpha_i < 1:
                raise ValueError("alpha_i must be >= 1")
        self.alpha_i = alpha_i
        # lambda_
        lambda_ = 1.0
        if props.has_property('lambda'):
            lambda_ = props['lambda']
            if lambda_ < 0 or lambda_ > 1:
                raise ValueError("lambda must be in (0,1)")
        self.lambda_ = mi.Float(lambda_)

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
        ks_local = mi.reflect(wi_local)
        c_eta = mi.Complex2f(self.eta_r, -self.eta_i)

        # Mueller matrix for specular reflection and the direction of specular
        # reflection
        m_local, ks_local = specular_mueller_matrix(wi_local, c_eta)

        # Direction for diffusely reflected rays
        kd_local = self.diffuse_direction(sample2)
        # Weight for diffuse reflection
        w_diffuse = self.diffuse_weight(wi_local, ks_local, kd_local)

        # Scale Mueller matrices according to the scattering coefficient to
        # enable gradient for this parameter
        # Diffuse
        p = dr.sqr(self.s)
        w_diffuse *= p*dr.detach(dr.rcp(p))
        # Specular
        r = 1. - p
        w_specular = r*dr.detach(dr.rcp(r))

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
        callback.put_parameter('alpha_i', self.alpha_i,
                               mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('alpha_r', self.alpha_r,
                               mi.ParamFlags.NonDifferentiable)
        callback.put_parameter('lambda', self.lambda_,
                               mi.ParamFlags.Differentiable)

    def to_string(self):
        s = f"Lambertian[\n"\
            f"\teta_r={self.eta_r}"\
            f"\teta_i={self.eta_i}"\
            f"\ts={self.s}"\
            f"\talpha_r={self.alpha_r}"\
            f"\talpha_i={self.alpha_i}"\
            f"\tlambda={self.lambda_}"\
            f"]"
        return s

    ###########################################
    # Utilities
    ###########################################

    def diffuse_normalization_factor(self, wi_local):
        r"""
        Computes the normalization factor for diffuse reflection,
        denoted by F_{alpha_r,alpha_i} in [Degli-Esposi07].

        Input
        ------
        wi_local : [num_samples, 3], float
            Incident direction in the local coordinate system

        Output
        -------
        : [num_samples], float
            Normalization factor
        """

        cos_theta_i = dr.abs(wi_local.z)
        sin_theta_i = dr.sqrt(1. - dr.sqr(cos_theta_i))

        # F_alpha_i and F_alpha_r
        f_alpha_i = dr.zeros(mi.Float)
        f_alpha_r = dr.zeros(mi.Float)

        # K_n
        # n ranges from 0 to n_max, and will be computed
        # sequentially thereafter
        k_n = dr.zeros(mi.Float)

        # As parallelization is done over samples, i.e.,
        # each thread computes a sample, the compute of
        # the normalization factor is done sequentially for
        # each sample.

        # Compute I_j for odd values of j
        alpha_max = np.maximum(self.alpha_i, self.alpha_r)
        for j in range(alpha_max+1):

            # Even j
            if (j % 2) == 0:
                # For even j, I_j is independant of the incidence
                # direction and therefore the same for all samples
                # ()
                i_j = 2.*PI/(j+1)

            # Odd j
            else:
                # ()
                n = (j-1)//2

                # Compute k_n
                # [num_samples]
                v = dr.power(sin_theta_i, 2*n)
                v *= scipy.special.binom(2*n, n)
                v /= np.power(2., 2.*n)
                k_n = k_n + v

                # Compute I_j
                # [num_samples]
                i_j = cos_theta_i*k_n*2.*PI/float(j+1)

            # Update f_alpha_i
            mask_i = 1.0 if j <= self.alpha_i else 0.0
            f_alpha_i += i_j*scipy.special.binom(self.alpha_i, j)*mask_i
            # Update f_alpha_r
            mask_r = 1.0 if j <= self.alpha_r else 0.0
            f_alpha_r += i_j*scipy.special.binom(self.alpha_r, j)*mask_r

        f_alpha_i /= np.power(2., self.alpha_i)
        f_alpha_r /= np.power(2., self.alpha_r)

        # Compute normalization factor
        f = self.lambda_*f_alpha_r + (1.-self.lambda_)*f_alpha_i

        return f

    def diffuse_weight(self, wi_local, ks_local, kd_local):
        r"""
        Computes the weight of the scattering field in equation
        (12) of Degli-Esposi07.

        Input
        ------
        wi_local : [num_samples, 3], float
            Incident direction in the local coordinate system

        ks_local : [num_samples, 3], float
            Direction of specular reflection in local frame

        kd_local : [num_samples, 3], float
            Reflection direction in local frame

        Output
        -------
        : [num_samples], float
            Weight for the scattered field
        """

        cos_psi_r = dr.dot(ks_local, kd_local)
        cos_psi_i = dr.dot(wi_local, kd_local)

        v_r = dr.power(0.5*(1.+cos_psi_r), self.alpha_r)
        v_i = dr.power(0.5*(1.+cos_psi_i), self.alpha_i)

        w = self.lambda_*v_r + (1.-self.lambda_)*v_i

        # Normalization
        f = self.diffuse_normalization_factor(wi_local)
        w_normalized = w*dr.rcp(f)

        return w_normalized

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
