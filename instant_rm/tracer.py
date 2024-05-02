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
from .antenna import iso_antenna_pattern, dipole_pattern, hw_dipole_pattern,\
    tr38901_antenna_pattern
from .ray_tube import RayTube
from .utils import mp_to_world, mp_uv_2_cell_ind, theta_phi_from_unit_vec

from mitsuba import Transform4f, Point2i, TensorXf, Spectrum,\
    ScalarTransform4f, UInt, Loop, Int


class MapTracer:
    r"""
    Radio map tracer.

    Parameters
    -----------
    scene : mi.Scene
        Mitsuba scene

    fc : float
        Carrier frequency [Hz]

    tx_pattern : str
        Transmitter antenna pattern.
        One of 'iso' 'dipole', 'hw_dipole', or 'tr38901'.

    tx_slant_angle : float
        Transmitter slant angle [rad].
        A slant angle of zero means vertical polarization.

    mp_center : [3], float
        Center of the map

    mp_orientation : [3], float
        Orientation of the map $(\alpha, \beta, \gamma)$ [rad].
        (0,0,0) means parallel to XY.

    mp_size : [2], float
        Size of the map [m]

    mp_cell_size : [2], float
        Size of the map cells [m]

    num_samples : int
        Number of samples

    max_depth : int
        Maximum path depths. 0 means line-of-sight only.

    seed : int
        Seed for the sampler
        Defaults to 1.

    rr_depth : int
        Path depth at which Russian roulette starts to be using, i.e., paths
        are randomly terminated with probably `rr_prob``.
        Defaults to -1, i.e., no random termination.

    rr_prob : float
        If Russian roulette is used, maximum probability with which to continue
        a path.
        Defaults to 0.95.

    Input
    ------
    tx_position : [3], float
        Position of the transmitter

    tx_orientation : [3], float
        Orientation of the transmitter (z, y, x) [rad]

    los : bool
        Enable/Disable LoS.
        Defaults to `True`, i.e., LoS is enabled.

    rms_ds : bool
        If set to `True`, computes and returns the RMS-DS map.
        Defaults to `True`.

    mda : bool
        If set to `True`, computes and returns the mean direction of arrival
        map. Defaults to `True`.

    mdd : bool
        If set to `True`, computes and returns the mean direction of departure
        map. Defaults to `True`.

    loop_record : bool
        Enable/Disable drjit loop recording.
        Loop recording is significantly faster. However, with loop recording,
        backpropagation *across* loop iterations is not possible.
        Default to `True`.

    Output
    ------
    pm : [num_cells_x, num_cells_y], float
        Path loss map

    rdsm : [num_cells_x, num_cells_y], TensorXf
        RMS-delay spread map in ns.
        Only returned in `rms_ds` is set to `True`.

    mdam = [num_cells_x, num_cells_y], TensorXf
        Mean direction of arrival map.
        Direction of arrival is in the world frame.
        Only returned in `mda` is set to `True`.

    mddm : [num_cells_x, num_cells_y], TensorXf
        Mean direction of departure map.
        Direction of departure is in the world frame.
        Only returned in `mdd` is set to `True`.
    """

    def __init__(self, scene, fc, tx_pattern, tx_slant_angle,
                 mp_center, mp_orientation, mp_size, mp_cell_size,
                 num_samples, max_depth, seed=1,
                 rr_depth=-1, rr_prob=0.95):

        if rr_depth == -1:
            rr_depth = max_depth + 1

        self.scene = scene
        self.tx_slant_angle = tx_slant_angle
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.mp_cell_size = mp_cell_size
        self.rr_depth = rr_depth
        self.rr_prob = rr_prob
        self.seed = seed

        # Wavelength [m]
        self.wavelength = SPEED_OF_LIGHT/fc
        # Number of cells forming the map
        self.mp_num_cells = np.ceil(mp_size/mp_cell_size).astype(int)
        # Number of cells forming the direction of arrival/departure maps
        self.mp_dir_num_cells = np.concatenate([self.mp_num_cells, [3]], axis=0)

        # Build the measurement plane
        self.meas_plane = mi.load_dict(
            {'type': 'rectangle',
             'to_world': mp_to_world(mp_center, mp_orientation, mp_size)
            })

        # Select the antenna pattern of the transmitter
        if tx_pattern == 'iso':
            self.tx_pattern_call = iso_antenna_pattern
        elif tx_pattern == 'dipole':
            self.tx_pattern_call = dipole_pattern
        elif tx_pattern == 'hw_dipole':
            self.tx_pattern_call = hw_dipole_pattern
        elif tx_pattern == 'tr38901':
            self.tx_pattern_call = tr38901_antenna_pattern

        # Sampler
        self.sampler = mi.load_dict({'type': 'independent'})

    def __call__(self, tx_position, tx_orientation, los=True, rms_ds=True,
                 mda=True, mdd=True, loop_record=True):

        # Transform CPU-side literals to device-side variables to avoid
        # recompilation of the kernel
        tx_position = mi.Point3f([dr.opaque(mi.Float, v)
                                  for v in tx_position])
        tx_orientation = mi.Point3f([dr.opaque(mi.Float, v)
                                     for v in tx_orientation])

        # Initialize ray tubes
        rt = RayTube(self.num_samples, tx_position, tx_orientation,
                      self.tx_slant_angle, self.tx_pattern_call)

        # Initialize the path loss map to zero
        pm = dr.zeros(TensorXf, self.mp_num_cells)

        # Initialize the weighted delay and squared delay maps to zero.
        # These are required to compute the RMS delay spread maps.
        dm = dr.zeros(TensorXf, self.mp_num_cells)
        d2m = dr.zeros(TensorXf, self.mp_num_cells)

        # Initialize the mean direction of departure and arrival maps
        # Size of the mean direction maps
        mdam = dr.zeros(TensorXf, self.mp_dir_num_cells)
        mddm = dr.zeros(TensorXf, self.mp_dir_num_cells)

        # Set the seed of the sampler
        self.sampler.seed(self.seed, self.num_samples)

        depth = UInt(0)
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)
        dr.set_flag(dr.JitFlag.LoopOptimize, False) # Helps with latency
        loop = Loop("main", state=lambda : (depth, rt))
        loop.set_max_iterations(self.max_depth+1)
        while loop(rt.active):

            # Test intersection with the scene
            si_scene = self.scene.ray_intersect(rt.ray, active=rt.active)

            # Test intersection with the measurement plane
            si_mp = self.meas_plane.ray_intersect(rt.ray, active=rt.active)
            # An intersection with the measurement plane is valid only if
            # (i) it was not obstructed by the scene, and (ii) the intersection
            # is valid.
            val_mp_int = rt.active & (si_mp.t < si_scene.t) & si_mp.is_valid()
            # Disable LoS if requested
            val_mp_int &= (depth > 0) | los
            # Update the maps
            self.add_to_maps(rms_ds, mda, mdd, pm, dm, d2m, mdam, mddm, si_mp,
                             rt, val_mp_int)

            # Update the active state of rays
            # Active rays are those that hit the scene
            rt.active &= si_scene.is_valid()

            # Computes Mueller matrices and scattered ray direction resulting
            # from intersecting the scene
            m_world, k_world = self.eval_interaction(si_scene, rt)

            # Update the field
            rt.f_world = m_world@rt.f_world
            # Due to numerical error, it could be that rt.f_world.x is slightly
            # negative
            rt.f_world &= (rt.f_world.x >= 0)

            # Spawn rays for next iteration
            rt.ray = si_scene.spawn_ray(d=k_world)
            depth += 1
            rt.active &= (depth <= self.max_depth)

            # Russian roulette
            rr_inactive = depth < self.rr_depth
            # Use the current path loss as probabilty of continuing the path
            rr_continue_prob = rt.f_world.x.x
            # User specify a maximum probability of continuing tracing
            rr_continue_prob = dr.minimum(rr_continue_prob, self.rr_prob)
            # Randomly stop tracing of rays
            rr_continue = self.sampler.next_1d() < rr_continue_prob
            rt.active &= rr_inactive | rr_continue
            # Scale the remaining rays accordingly to ensure an unbiased result
            rt.f_world[~rr_inactive] *= dr.rcp(rr_continue_prob)

        # Finalizes the computation of the radio maps
        pm, rdsm, mdam, mddm = self.finalize_maps(rms_ds, mda, mdd, pm, dm, d2m,
                                                  mdam, mddm)

        output = [pm]
        if rms_ds:
            output += [rdsm]
        if mda:
            output += [mdam]
        if mdd:
            output += [mddm]

        return output

    def add_to_maps(self, rms_ds, mda, mdd, pm, dm, d2m, mdam, mddm, si, rt,
                    hit):
        r"""
        Adds the contribution of the rays that hit the measurement plane to the
        radio maps.
        The maps are updated in place.

        Input
        ------
        rms_ds : bool
            If set to `True`, updates the RMS-DS map

        mda : bool
            If set to `True`, updates the mean direction of arrival map

        mdd : bool
            If set to `True`, updates the mean direction of departure map

        pm : [num_cells_x, num_cells_y], TensorXf
            Path loss map

        dm : [num_cells_x, num_cells_y], TensorXf
            Delay map

        d2m : [num_cells_x, num_cells_y], TensorXf
            Squared delay map

        mdam : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of arrival map

        mddm : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of departure map

        si : mi.SurfaceInteraction
            Information about the interaction of the rays with the measurement
            plane

        rt : RayTube
            Ray tubes

        hit : [num_samples], bool
            Array of booleans indicating for each ray if it hits the measurement
            plane
        """

        # Indices of the hit cells
        # [num_samples], int
        cell_ind = mp_uv_2_cell_ind(si.uv, self.mp_num_cells)

        # Spreading factor to account for the propagation loss from the last
        # intersection point with the scene
        # Propagation distance
        d = dr.norm(si.p - rt.origin)
        # Squared spreading factor
        a2 = rt.sqr_spreading_factor(d)
        # Ray tube radii of curvature evaluated at the intersection point with
        # the measurement plane
        rho_1 = rt.rho_1 + d
        rho_2 = rt.rho_2 + d

        # Ray power at the intersection point with the measurement plane
        p = rt.f_world.x*a2

        # Ray weight
        # Cosine of incident angle
        # The normal is always (0,0,1) in the local frame
        cos_theta = dr.abs(si.wi.z)
        w = rho_1*rho_2*rt.angle*dr.rcp(cos_theta)

        # Contribution to the path loss map
        a = p*w

        # Update the path loss map
        dr.scatter_reduce(dr.ReduceOp.Add, pm.array, value=a, index=cell_ind,
                                active=hit)

        # Update the weighted delay map
        if rms_ds:
            # Speed of light in meters per ns
            SPEED_OF_LIGHT_MNS = SPEED_OF_LIGHT / 1e9
            # Delay in ns and squared delay in ns^2
            tau = (rt.length + d) / SPEED_OF_LIGHT_MNS
            tau2 = dr.sqr(tau)
            # Weighted delay and squared delay
            wtau = a*tau
            wtau2 = a*tau2

            # Update the delay and squared delay maps
            dr.scatter_reduce(dr.ReduceOp.Add, dm.array, value=wtau,
                              index=cell_ind, active=hit)
            dr.scatter_reduce(dr.ReduceOp.Add, d2m.array, value=wtau2,
                            index=cell_ind, active=hit)

        # Update the direction of arrival map
        if mda:
            wka = -rt.ray.d
            wka *= a.x
            # Broadcast to the (x,y,z) inner dimension
            cell_ind_x = cell_ind * 3
            cell_ind_y = cell_ind_x + 1
            cell_ind_z = cell_ind_x + 2

            # x component
            dr.scatter_reduce(dr.ReduceOp.Add, mdam.array, value=wka.x,
                            index=cell_ind_x, active=hit)
            # y component
            dr.scatter_reduce(dr.ReduceOp.Add, mdam.array, value=wka.y,
                            index=cell_ind_y, active=hit)
            # z component
            dr.scatter_reduce(dr.ReduceOp.Add, mdam.array, value=wka.z,
                            index=cell_ind_z, active=hit)

        # Update the direction of departure map
        if mdd:
            wkd = rt.kd_world
            wkd *= a.x

            # x component
            dr.scatter_reduce(dr.ReduceOp.Add, mddm.array, value=wkd.x,
                            index=cell_ind_x, active=hit)
            # y component
            dr.scatter_reduce(dr.ReduceOp.Add, mddm.array, value=wkd.y,
                            index=cell_ind_y, active=hit)
            # z component
            dr.scatter_reduce(dr.ReduceOp.Add, mddm.array, value=wkd.z,
                            index=cell_ind_z, active=hit)

    def finalize_maps(self, rms_ds, mda, mdd, pm, dm, d2m, mdam, mddm):
        r"""
        Finalizes the computation of the maps

        Input
        ------
        rms_ds : bool
            If set to `True`, updates the RMS-DS map

        mda : bool
            If set to `True`, updates the mean direction of arrival map

        mdd : bool
            If set to `True`, updates the mean direction of departure map

        pm : [num_cells_x, num_cells_y], TensorXf
            Path loss map

        dm : [num_cells_x, num_cells_y], TensorXf
            Delay map.

        d2m : [num_cells_x, num_cells_y], TensorXf
            Squared delay map.

        mdam : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of arrival map

        mddm : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of departure map

        Output
        -------
        pm : [num_cells_x, num_cells_y], TensorXf
            Finalized path loss map

        rdsm : [num_cells_x, num_cells_y], TensorXf
            RMS delay spread map in ns

        mdam : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of arrival maps

        mddm : [num_cells_x, num_cells_y, 3], TensorXf
            Mean direction of departure maps
        """

        rcp_pm = dr.rcp(pm)
        rcp_pm_3d = dr.repeat(rcp_pm.array, 3)

        # Computing the RMS delay spread map
        if rms_ds:
            rdsm = dr.safe_sqrt(d2m*rcp_pm - dr.sqr(dm*rcp_pm))
            rdsm = dr.select(dr.eq(pm, 0.), np.nan, rdsm)
        else:
            rdsm = dm # Discarded later

        # Normalize the mean direction of arrival map
        if mda:
            mdam_array = mdam.array*rcp_pm_3d
            mdam = mi.TensorXf(mdam_array, shape=self.mp_dir_num_cells)

        # Normalize the mean direction of departure map
        if mdd:
            mddm_array = mddm.array*rcp_pm_3d
            mddm = mi.TensorXf(mddm_array, shape=self.mp_dir_num_cells)

        # Scaling the path loss map
        # Scaling factor
        cell_area = self.mp_cell_size[0]*self.mp_cell_size[1]
        scaling = dr.sqr(self.wavelength*dr.rcp(4.*PI))*dr.rcp(cell_area)
        pm *= scaling

        return pm, rdsm, mdam, mddm

    def eval_interaction(self, si_scene, rt):
        r"""
        Evaluates the interactions of rays with the scene.
        Returns the Mueller matrix and direction of the scattered ray.

        Input
        ------
        si_scene : mi.SurfaceInteraction
            Information about the interaction of the rays with the measurement
            plane

        rt : RayTube
            Ray tubes

        Output
        -------
        m_world : Matrix4
            Mueller matrix in the world coordinate system

        wo_world : Vector3
            Directions of spawn rays
        """

        wi_world = -rt.ray.d

        # Ensure the normal is oriented in the opposite incident wave
        # propagation direction
        normal_world = si_scene.n*dr.sign(dr.dot(si_scene.n, wi_world))

        # Sets the normal in `si_scene` and initializes the local frame
        si_scene.sh_frame.n = normal_world
        si_scene.initialize_sh_frame()
        si_scene.n = normal_world

        wi_local = si_scene.to_local(wi_world)
        si_scene.wi = wi_local

        # 'Importance' mode is used as we trace from source to target
        mode = mi.TransportMode.Importance
        ctx = mi.BSDFContext(mode=mode)

        # Samples the material
        sample1 = self.sampler.next_1d(rt.active)
        sample2 = self.sampler.next_2d(rt.active)
        s, m_local = si_scene.bsdf().sample(ctx, si_scene, sample1, sample2,
                                            rt.active)

        wo_local = s.wo
        wo_world = si_scene.to_world(wo_local)
        m_world = si_scene.to_world_mueller(m_local, -wi_local, wo_local)

        # The origin of the ray tube is moved to the interaction point.
        # To that aim, the field at that new origin, which must correspond to
        # s = 0, needs to be updated, which involves applying `m_world` as well
        # as the spreading factor computed using the former origin.
        specular = dr.eq(s.sampled_component, mi.UInt32(0))
        # Propagation distance
        d = dr.norm(si_scene.p - rt.origin)
        # Squared spreading factor
        w = rt.sqr_spreading_factor(d)
        # In the case of diffuse reflection, an additional weighting needs to
        # be applied
        w *= dr.select(specular, 1., rt.angle*(rt.rho_1+d)*(rt.rho_2+d))
        # Applies propagation loss
        m_world = mi.Matrix4f(*[m_world[col][row].x for row in range(4) for col in range(4)])
        m_world @= w
        m_world = Spectrum(m_world)

        # Updates the ray tube radii of curvature
        rt.rho_1 += (d & specular)
        rt.rho_2 += (d & specular)
        # Updates ray tube origin
        rt.origin = si_scene.p
        # Updates the angle
        rt.angle = dr.select(specular, rt.angle, 2.*PI)
        # Updates the length
        rt.length += d

        return m_world, wo_world
