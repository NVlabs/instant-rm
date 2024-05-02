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


class SmoothMaterial(mi.BSDF):
    r"""
    Mitsuba BSDF implementing a smooth material, i.e., a material that reflects
    specularly.
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)

        # Read material properties
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

        As this is a perfectly smooth material, there is not randomness and the
        direction of reflection is always the specular one.

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
            Not used.

        sample2 : Float
            Not used.

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

        # Move to implicit Mitsuba local coordinate system
        m_local = mueller_to_implicit(m_local, wi_local, ks_local)

        # Instantiate and set the BSDFSample object
        bs = mi.BSDFSample3f()
        bs.pdf = 0.0 # Specular reflection
        bs.sampled_component = mi.UInt32(0)
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.wo = ks_local
        bs.eta = 1.0 # Complex numbers not handled

        return bs, m_local

    def eval(self, ctx, si, wo, active):
        # Should never be called
        return 0.0

    def pdf(self, ctx, si, wo, active):
        # Should never be called
        return 0.0

    def traverse(self, callback):
        callback.put_parameter('eta_r', self.eta_r,
                               mi.ParamFlags.Differentiable)
        callback.put_parameter('eta_i', self.eta_i,
                               mi.ParamFlags.Differentiable)

    def to_string(self):
        s = f"Smooth[\n"\
            f"\teta_r={self.eta_r}"\
            f"\teta_i={self.eta_i}"\
            f"]"
        return s
