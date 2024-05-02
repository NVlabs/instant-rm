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
from .utils import fibonacci_lattice, jones_to_stokes
from .antenna import iso_antenna_pattern, dipole_pattern, hw_dipole_pattern,\
    tr38901_antenna_pattern
from .ray_tube import RayTube
from .utils import mp_to_world, mp_uv_2_cell_ind, mat_inverse_double

from mitsuba import Vector3f, Transform4f, Point2i, TensorXf, Spectrum,\
    ScalarTransform4f, UInt, Loop, Point3f, Vector3f, Int


class PathlossMapRBPTracer:
    r"""
    Ray tracing of path loss maps with paths replay for fast gradient
    backpropagation.

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
        Center of the measurement plane

    mp_orientation : [3], float
        Orientation of the map $(\alpha, \beta, \gamma)$ [rad].
        (0,0,0) means parallel to XY.

    mp_size : [2], float
        Size of the measurement plane [m]

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

    func_loss : callable
        Loss function.
        Must be a callable that takes as input the path loss map and
        outputs a scalar.

    loop_record : bool
        If set to `True`, then enables Dr.JiT loop recording.
        Default to `True`.

    Output
    ------
    pm : [num_cells_x, num_cells_y], float
        Path loss map

    loss : float
        Loss value
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

        # Build the measurement plane
        self.meas_plane = mi.load_dict(
            {'type': 'rectangle',
            'to_world': mp_to_world(mp_center, mp_orientation, mp_size),
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

    def __call__(self, tx_position, tx_orientation, func_loss,
                 loop_record=True):

        # Transform CPU-side literals to device-side variables to avoid
        # recompilation of the kernel
        tx_position = mi.Point3f([dr.opaque(mi.Float, v)
                                  for v in tx_position])
        tx_orientation = mi.Point3f([dr.opaque(mi.Float, v)
                                     for v in tx_orientation])

        with dr.suspend_grad():
            # Radiative backpropagation for wireless maps requires
            # three stages, each involving the tracing of the maps.

            ## First stage: Fowrward pass
            # Computes the path loss map
            pm = self.forward(tx_position, tx_orientation, loop_record)

            # Evaluate the loss and the gradient of the loss with respect to
            # its inputs
            loss, d_loss = self.loss_gradient(pm, func_loss)

            ## Second stage: Computes combined end-to-end transfer matrix
            # A ray originating from the transmitter can intersect the
            # measurement plane multiple times, and to every intersection with
            # the measurement plane corresponds a Mueller matrix that represents
            # the transfer function of the channel from the transmitter to this
            # intersection point.
            # The following function computes, for every path, the sum of the
            # end-to-end transfer matrices (one for each intersection with the
            # measurement plane), weighted according to its incident angle and
            # radii of curvature at the intersection point, and gradient of the
            # loss function with respect to the intersected cell.
            comb_mat = self.compute_combined_mat(tx_position, tx_orientation,
                                                  loop_record, d_loss)

            ## Third stage
            # Computes the gradients.
            # The gradients are stored internally by Dr.JiT
            self.compute_grads(tx_position, tx_orientation, loop_record,
                               d_loss, comb_mat)

        return pm, loss

    ##############################################################
    # Utilities
    ##############################################################

    def forward(self, tx_position, tx_orientation, loop_record):
        """
        Computes the path loss map

        Input
        ------
        tx_position : [3], float
            Position of the transmitter

        tx_orientation : [3], float
            Orientation of the transmitter (z, y, x) [rad]

        loop_record : bool
            Enable/Disable drjit loop recording.

        Output
        ------
        pm : [num_cells_x, num_cells_y], float
            Path loss map
        """

        # Initialize ray tubes
        rt = RayTube(self.num_samples, tx_position, tx_orientation,
                      self.tx_slant_angle, self.tx_pattern_call)

        # Initialize the path loss map to zero
        pm = dr.zeros(TensorXf, self.mp_num_cells)

        # Set the seed of the Russian roulette sampler
        self.sampler.seed(self.seed, self.num_samples)

        depth = UInt(0)
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)
        loop = Loop("main", state=lambda : (depth, rt))
        loop.set_max_iterations(self.max_depth+1)
        dr.set_flag(dr.JitFlag.LoopOptimize, False) # Helps with latency
        while loop(rt.active):

            # Test intersection with the scene
            si_scene = self.scene.ray_intersect(rt.ray, active=rt.active)

            # Test intersection with the measurement plane
            si_mp = self.meas_plane.ray_intersect(rt.ray, active=rt.active)
            # An intersection with the measurement plane is valid only if
            # (i) it was not obstructed by the scene, and (ii) the intersection
            # is valid.
            val_mp_int = rt.active & (si_mp.t < si_scene.t) & si_mp.is_valid()

            # Indices of the intersected cells
            cell_ind = mp_uv_2_cell_ind(si_mp.uv, self.mp_num_cells)

            # Weights to apply to the amplitudes of the rays that hit the
            # measurement plane
            rw = self.ray_weights(si_mp, rt)

            # Update the path loss map
            self.add_to_pathloss_map(pm, si_mp, rt, rw, cell_ind, val_mp_int)

            # Update rays active state
            # Active rays are those that hit the scene, i.e., didn't bounce out
            # of the scene
            rt.active &= si_scene.is_valid()

            # Compute Mueller matrices and scattered ray direction resulting
            # from intersecting the scene
            m_world, k_world = self.eval_interaction(si_scene, rt)

            # Update the field being propagated
            rt.f_world = m_world@rt.f_world
            # Due to numerical error, it could be that rt.f_world.x is slightly
            # negative
            rt.f_world &= (rt.f_world.x >= 0)

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

            # Spawn rays for next iteration
            rt.ray = si_scene.spawn_ray(d=k_world)
            depth += 1
            rt.active &= (depth <= self.max_depth)

        return pm

    def compute_combined_mat(self, tx_position, tx_orientation, loop_record,
                              d_loss):
        """
        Computes combined end-to-end transfer matrices

        A ray originating from the transmitter can intersect the
        measurement plane multiple times, and to every intersection with
        the measurement plane corresponds a Mueller matrix that represents
        the transfer function of the channel from the transmitter to this
        intersection point.

        This function computes, for every path, the sum of the
        end-to-end transfer matrices (one for each intersection with the
        measurement plane), weighted according to its incident angle and
        radii of curvature at the intersection point, and gradient of the
        loss function with respect to the intersected cell.

        Input
        ------
        tx_position : [3], float
            Position of the transmitter

        tx_orientation : [3], float
            Orientation of the transmitter (z, y, x) [rad]

        loop_record : bool
            Enable/Disable drjit loop recording

        d_loss : [num_cells_x, num_cells_y], float
            Derivatives of the loss function with respect to the map cells

        Output
        ------
        combined_mat : [num_samples, 4, 4], float
            Combined end-to-end transfer matrices
        """

        # Initialize ray tubes
        rt = RayTube(self.num_samples, tx_position, tx_orientation,
                      self.tx_slant_angle, self.tx_pattern_call)

        # Combined end-to-end transfer matrices
        # Initialized to zero
        comb_mat = dr.zeros(Spectrum, self.num_samples)

        # End-to-end transfer matrix of the paths (Not combined)
        # This is required to compute `comb_mat`
        e2e_mat = dr.identity(Spectrum, self.num_samples)

        # Set the seed of the Russian roulette sampler
        self.sampler.seed(self.seed, self.num_samples)

        depth = UInt(0)
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)
        loop = Loop("main", state=lambda : (depth, rt, comb_mat, e2e_mat))
        loop.set_max_iterations(self.max_depth+1)
        dr.set_flag(dr.JitFlag.LoopOptimize, False) # Helps with latency
        while loop(rt.active):

            # Test intersection with the scene
            si_scene = self.scene.ray_intersect(rt.ray, active=rt.active)

            # Test intersection with the measurement plane
            si_mp = self.meas_plane.ray_intersect(rt.ray, active=rt.active)
            # An intersection with the measurement plane is valid only if
            # (i) it was not obstructed by the scene, and (ii) the intersection
            # is valid.
            val_mp_int = rt.active & (si_mp.t < si_scene.t) & si_mp.is_valid()

            # Indices of the intersected cells
            cell_ind = mp_uv_2_cell_ind(si_mp.uv, self.mp_num_cells)

            # Weights to apply to the amplitudes of the rays that hit the
            # measurement plane
            rw = self.ray_weights(si_mp, rt)

            # Scaling to apply to the Mueller matrices of the path suffixes
            cw = self.combined_weight(rw, cell_ind, val_mp_int, d_loss)

            # Update the combined end-to-end transfer matrix with the Mueller
            # matrices of the rays that hit the measurement plane
            comb_mat = dr.select(val_mp_int,
                                    comb_mat + mi.Spectrum(cw)@e2e_mat,
                                    comb_mat)
            comb_mat = Spectrum(comb_mat)

            # Updates rays active state
            # Active rays are those that hit the scene
            rt.active &= si_scene.is_valid()

            # Computes Mueller matrices and scattered ray direction resulting
            # from intersecting the scene
            m_world, k_world = self.eval_interaction(si_scene, rt)

            # Update the end-to-end transfer matrix
            e2e_mat = m_world@e2e_mat

            # Update the field being propagated
            rt.f_world = m_world@rt.f_world
            # Due to numerical error, it could be that rt.f_world.x is slightly
            # negative
            rt.f_world &= (rt.f_world.x >= 0)

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

            # Spawn rays for next iteration
            rt.ray = si_scene.spawn_ray(d=k_world)
            depth += 1
            rt.active &= (depth <= self.max_depth)

        return comb_mat

    def compute_grads(self, tx_position, tx_orientation, loop_record, d_loss,
                      comb_mat):
        """
        Computes gradients

        Input
        ------
        tx_position : [3], float
            Position of the transmitter

        tx_orientation : [3], float
            Orientation of the transmitter (z, y, x) [rad]

        loop_record : bool
            Enable/Disable drjit loop recording.

        d_loss : [num_cells_x, num_cells_y], float
            Derivatives of the loss function with respect to the map cells.
            Only required if `compute_w_mueller_mat` or  `compute_gradients` is
            set to `True`.

        comb_mat : [num_samples, 4, 4], float
            Combined Mueller matrices
        """

        # Initialize ray tubes
        rt = RayTube(self.num_samples, tx_position, tx_orientation,
                      self.tx_slant_angle, self.tx_pattern_call)

        # Set the seed of the Russian roulette sampler
        self.sampler.seed(self.seed, self.num_samples)

        depth = UInt(0)
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record)
        loop = Loop("main", state=lambda : (depth, rt, comb_mat))
        loop.set_max_iterations(self.max_depth+1)
        dr.set_flag(dr.JitFlag.LoopOptimize, False) # Helps with latency
        while loop(rt.active):

            # Test intersection with the scene
            si_scene = self.scene.ray_intersect(rt.ray, active=rt.active)

            # Test intersection with the measurement plane
            si_mp = self.meas_plane.ray_intersect(rt.ray, active=rt.active)
            # An intersection with the measurement plane is valid only if
            # (i) it was not obstructed by the scene, and (ii) the intersection
            # is valid.
            val_mp_int = rt.active & (si_mp.t < si_scene.t) & si_mp.is_valid()

            # Indices of the intersected cells
            cell_ind = mp_uv_2_cell_ind(si_mp.uv, self.mp_num_cells)

            # Weights to apply to the amplitudes of the rays that hit the
            # measurement plane
            rw = self.ray_weights(si_mp, rt)

            # Scaling to apply to the Mueller matrices of the path suffixes
            # Scaling to apply to the Mueller matrices of the path suffixes
            cw = self.combined_weight(rw, cell_ind, val_mp_int, d_loss)

            # The combined end-to-end transfer matrix is updated by subtracting
            # the term corresponding to the intersection with the measurement
            # plane. As the intersection is behind, further intersections
            # with the scene will not contribute to the gradient of related to
            # this intersection.
            comb_mat = dr.select(val_mp_int,
                                 comb_mat - mi.Spectrum(cw),
                                 comb_mat)
            comb_mat = Spectrum(comb_mat)

            # Updates rays active state
            # Active rays are those that hit the scene
            rt.active &= si_scene.is_valid()

            # Computes Mueller matrices and scattered ray direction resulting
            # from intersecting the scene
            # This is done with gradient tracking enabled.
            with dr.resume_grad():
                m_world, k_world = self.eval_interaction(si_scene, rt)

            # No need to propagate gradients through the direction of
            # propagation
            k_world = dr.detach(k_world)

            # Update the combined end-to-end transfer matrix by
            # canceling the current interaction with the scene
            m_world_inv = mat_inverse_double(m_world)
            comb_mat = comb_mat@m_world_inv

            # Only the Mueller matrix of the current interaction `m_world`
            # has derivative tracking enabled.
            # The combined end-to-end transfer matrix (path suffix) and the
            # incident field (path prefix) re detached.
            with dr.resume_grad():
                # Compute gradients for the intensity (path loss) component
                # only.
                # Gradients are internally stored by Dr.JiT
                f_total_world = comb_mat@m_world@rt.f_world
                dr.backward_from(f_total_world.x)

            # Update the field being propagated
            rt.f_world = m_world@rt.f_world
            # Due to numerical error, it could be that rt.f_world.x is slightly
            # negative
            rt.f_world &= (rt.f_world.x >= 0)

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

            # Spawn rays for next iteration
            rt.ray = si_scene.spawn_ray(d=k_world)
            depth += 1
            rt.active &= (depth <= self.max_depth)

    def loss_gradient(self, pm, func_loss):
        """
        Computes the gradient of `func_loss` with respect to its input
        `pm`.

        Input
        ------
        pm : [num_cells_x, num_cells_y], float
            Path loss map

        func_loss : callable
            Loss function.
            Must be a callable that takes as input the path loss map and
            outputs a scalar.

        Output
        -------
        loss : float
            Loss value

        d_loss : [num_cells_x, num_cells_y], float
            Derivatives of the loss function with respect to the path loss map
            cells evaluated for `pm`.
        """

        with dr.resume_grad():
            dr.enable_grad(pm)
            loss = func_loss(pm)
            dr.backward(loss)
            d_loss = dr.grad(pm)
            dr.disable_grad(pm)

        return loss, d_loss

    def ray_weights(self, si, rt):
        r"""
        Computes the weights that must be applied to the path loss of rays
        that hit the measurement plane.

        Input
        ------
        si : mi.SurfaceInteraction
            Information about the interaction of the rays with the measurement
            plane

        rt : RayTube
            Ray tubes

        Output
        -------
        : [num_samples], float
            Scaling to apply to the amplitudes of the rays intersecting the
            measurement plane
        """

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

        # Cosine of incident angle
        # The normal is always (0,0,1) in the local frame
        cos_theta = dr.abs(si.wi.z)

        # Ray weight
        w = rho_1*rho_2*rt.angle*dr.rcp(cos_theta)*a2

        # Antenna effective area normalization by the cell area
        cell_area = self.mp_cell_size[0]*self.mp_cell_size[1]
        w *= dr.sqr(self.wavelength*dr.rcp(4.*PI))*dr.rcp(cell_area)

        return w

    def add_to_pathloss_map(self, pm, si, rt, w, cell_ind, hit):
        r"""
        Adds the contribution of the rays that hit the measurement plane to the
        path loss map.
        The path loss map `pm` is updated in place.

        Input
        ------
        pm : [num_cells], float
            Path loss map

        si : mi.SurfaceInteraction
            Information about the interaction of the rays with the measurement
            plane

        rt : RayTube
            Ray tubes

        w : [num_samples], float
            Weightings for the ray amplitudes

        cell_ind : [num_samples], int
            Indices of the cells intersected by the rays in the flattened map

        hit : [num_samples], bool
            Array of booleans indicating for each ray if it hits the measurement
            plane
        """

        # Weighted ray amplitudes
        a = rt.f_world.x*w

        # Update the path loss map
        dr.scatter_reduce(dr.ReduceOp.Add, pm.array, value=a, index=cell_ind,
                                active=hit)

    def combined_weight(self, w, cell_ind, hit, d_loss):
        r"""
        Computes the weights applied to combine transfer matrices to compute the
        combined end-to-end transfer matrices.

        Input
        ------
        w : [num_samples], float
            Weightings for the ray path losses

        cell_ind : [num_samples], int
            Indices of the cells intersected by the rays in the flattened map

        hit : [num_samples], bool
            Array of booleans indicating for each ray if it hit the measurement
            plane

        d_loss : [num_cells_x, num_cells_y], float
            Derivatives of the loss function with respect to the map cells

        Output
        -------
        wm_init : Matrix4, float
            Scaled identity Mueller matrix.
        """

        # Gradient loss of the hit cells
        g = dr.gather(mi.Float, d_loss.array, cell_ind, hit)

        # Scale by gradient
        w *= g

        return w

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

        # Ensures that the normal is oriented in the opposite incident wave
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
        w = dr.select(specular, w, w*rt.angle*(rt.rho_1+d)*(rt.rho_2+d))
        # Applies propagation loss
        m_world = mi.Matrix4f(*[m_world[col][row].x for row in range(4) for col in range(4)])
        m_world @= w
        m_world = Spectrum(m_world)

        # Updates the ray tube radii of curvature
        rt.rho_1 = dr.select(specular, rt.rho_1+d, 0.0)
        rt.rho_2 = dr.select(specular, rt.rho_2+d, 0.0)
        # Updates ray tube origin
        rt.origin = si_scene.p
        # Updates the angle
        rt.angle = dr.select(specular, rt.angle, 2.*PI)

        return m_world, wo_world
