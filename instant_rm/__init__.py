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
mi.set_variant('cuda_ad_mono_polarized')


from .tracer import MapTracer
from .tracer_prb import PathlossMapRBPTracer

# Register the defined materials
# Smooth material
from .smooth_material import SmoothMaterial
mi.register_bsdf("smooth", lambda props: SmoothMaterial(props))
# Lambertian
from .lambertian_material import LambertianMaterial
mi.register_bsdf("lambertian", lambda props: LambertianMaterial(props))
# Backscattering
from .backscattering_material import BackscatteringMaterial
mi.register_bsdf("backscattering", lambda props: BackscatteringMaterial(props))
