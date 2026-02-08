# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Smoothed Particle Hydrodynamics
#
# Shows how to implement a SPH fluid simulation.
#
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
# Reference Publication
# Matthias Müller, David Charypar, and Markus H. Gross.
# "Particle-based fluid simulation for interactive applications."
# Symposium on Computer animation. Vol. 2. 2003.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.render


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

    # add external potential
    particle_rho[i] = density_normalization * rho


@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]

    # clamp x left
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), smoothing_length: float, width: float, height: float, length: float
):
    tid = wp.tid()

    # grid size
    nr_x = wp.int32(width / 4.0 / smoothing_length)
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / 4.0 / smoothing_length)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos


# One-way coupling

# Helper functions for primitives
@wp.func
def sdf_box(p: wp.vec3, box_size: wp.vec3):
    q = wp.abs(p) - box_size
    return wp.length(wp.max(q, wp.vec3(0.0))) + wp.min(wp.max(q), 0.0)

@wp.func
def sdf_sphere(p: wp.vec3, r: float):
    return wp.length(p) - r

@wp.func
def sdf_capsule(p: wp.vec3, radius: float, half_height: float):
    t = wp.clamp(p[1], -half_height, half_height)
    closest = wp.vec3(0.0, t, 0.0)
    return wp.length(p - closest) - radius

@wp.kernel
def collide_particles_with_robot(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    # Robot Data
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo: wp.sim.ModelShapeGeometry,
    num_shapes: int,
    restitution: float,
    particle_radius: float
):
    tid = wp.tid()
    p_world = particle_x[tid]
    v_world = particle_v[tid]
    

    for i in range(num_shapes):
        body_idx = shape_body[i]
        X_wb = body_q[body_idx]       # World -> Body
        X_bs = shape_transform[i]     # Body -> Shape
        X_ws = wp.transform_multiply(X_wb, X_bs)
        X_sw = wp.transform_inverse(X_ws)
        
        p_local = wp.transform_point(X_sw, p_world)
        
        # 2. Compute SDF
        geo_type = shape_geo.type[i]
        scale = shape_geo.scale[i]
        d = 1000.0
        
        # scale reused for different primitives
        if geo_type == wp.sim.GEO_SPHERE:
            d = sdf_sphere(p_local, scale[0]) # r
        elif geo_type == wp.sim.GEO_BOX:
            d = sdf_box(p_local, scale) # l, w, h
        elif geo_type == wp.sim.GEO_CAPSULE:
            d = sdf_capsule(p_local, scale[0], scale[1]) # r, half-height
            
        # non-negigible particle radius
        if d < particle_radius:
            
            # get normal
            n_local = wp.vec3(0.0, 1.0, 0.0)
            if geo_type == wp.sim.GEO_CAPSULE:
                t = wp.clamp(p_local[1], -scale[1], scale[1])
                closest = wp.vec3(0.0, t, 0.0)
                n_local = wp.normalize(p_local - closest)
            elif geo_type == wp.sim.GEO_SPHERE:
                n_local = wp.normalize(p_local)
            elif geo_type == wp.sim.GEO_BOX:
                # rough approx
                ax = wp.abs(p_local[0])
                ay = wp.abs(p_local[1])
                az = wp.abs(p_local[2])
                if ax > ay and ax > az:
                    n_local = wp.vec3(wp.sign(p_local[0]), 0.0, 0.0)
                elif ay > az:
                    n_local = wp.vec3(0.0, wp.sign(p_local[1]), 0.0)
                else:
                    n_local = wp.vec3(0.0, 0.0, wp.sign(p_local[2]))

            n_world = wp.transform_vector(X_ws, n_local)
            overlap = particle_radius - d
            p_world = p_world + n_world * overlap 

            vn = wp.dot(v_world, n_world)
            if vn < 0.0:
                v_tangent = v_world - n_world * vn
                v_world = v_tangent * 0.9 - n_world * vn * restitution
    
    particle_x[tid] = p_world
    particle_v[tid] = v_world


class Example:
    def __init__(self, stage_path="example_sph.usd", verbose=False):
        self.verbose = verbose

        # ANYmal
        builder = wp.sim.ModelBuilder()
        urdf_path = "envs/warp_sim_envs/assets/ant.xml"
        wp.sim.parse_mjcf(urdf_path, 
        builder,
            armature=0.05,
            contact_ke=4.e4,
            contact_kd=1.e2,
            contact_kf=3.e3,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e2,
            enable_self_collisions=False,
            up_axis="y",
            collapse_fixed_joints=True,
        )

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -warp.pi * 0.5)
        self.inv_start_rot = wp.quat_inverse(self.start_rot)

        self.robot_model = builder.finalize()
        self.scale_robot_model(20.0)
        self.robot_state = self.robot_model.state()

        # render params
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.smoothing_length = 0.8  # NOTE change this to adjust number of particles
        self.width = 10.0  # x
        self.height = 10.0  # y
        self.length = 10.0  # z
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = 0.01 * self.smoothing_length**3  # reduce according to smoothing length
        self.dt = 0.01 * self.smoothing_length  # decrease sim dt by smoothing length
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.gravity = -9.81
        self.n = int(
            self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)

        # constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )  # integrate density kernel
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )

        # allocate arrays
        self.x = wp.empty(self.n, dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def step(self):
        with wp.ScopedTimer("step"):
            for _ in range(self.sim_step_to_frame_ratio):
                q_np = self.robot_state.joint_q.numpy()
                # Move forward 0.5m/s
                q_np[0] = self.sim_time * 0.5  
            
                # Bob up and down to splash
                q_np[1] = 0.55 + np.sin(self.sim_time * 5.0) * 0.1

                # 1. Set Position (Floating Base px, py, pz)
                # World is Y-up, so we lift it in Y
                q_np[0] = 0.0
                q_np[1] = 1.0 # Lift it up 1 meter
                q_np[2] = 0.0

                # 2. Set Orientation (Floating Base qx, qy, qz, qw)
                # We perform the EXACT rotation from the NERD code:
                # -90 degrees around X-axis
                rot_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -np.pi * 0.5)
                
                q_np[3] = rot_quat[0]
                q_np[4] = rot_quat[1]
                q_np[5] = rot_quat[2]
                q_np[6] = rot_quat[3]
                
                # Apply to Warp state
                self.robot_state.joint_q = wp.from_numpy(q_np, dtype=float)

                with wp.ScopedTimer("grid build", active=self.verbose):
                    # build grid
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("forces", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.a,
                            self.isotropic_exp,
                            self.base_density,
                            self.gravity,
                            self.pressure_normalization,
                            self.viscous_normalization,
                            self.smoothing_length,
                        ],
                    )

                    wp.launch(
                        kernel=collide_particles_with_robot,
                        dim=self.n,
                        inputs=[
                            self.x, self.v,
                            self.robot_state.body_q,
                            self.robot_model.shape_transform,
                            self.robot_model.shape_body,
                            self.robot_model.shape_geo,
                            self.robot_model.shape_count,
                            0.5,
                            self.smoothing_length * 0.5
                        ]
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

            self.sim_time += self.frame_dt

    
    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            
            # Render Fluid
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
            )
            
            # --- 1. Get Data from GPU to CPU (NumPy) ---
            body_q_np = self.robot_state.body_q.numpy()
            shape_xform_np = self.robot_model.shape_transform.numpy()
            shape_geo_type_np = self.robot_model.shape_geo.type.numpy()
            shape_geo_scale_np = self.robot_model.shape_geo.scale.numpy()
            shape_body_np = self.robot_model.shape_body.numpy()

            # --- 2. Iterate Over Robot Shapes ---
            for i in range(self.robot_model.shape_count):
                body_idx = shape_body_np[i]

                # [FIX] Manually unpack NumPy arrays into Warp types
                # Body Transform (World -> Body)
                bq = body_q_np[body_idx] # This is a 7-float array
                X_wb = wp.transform(
                    wp.vec3(bq[0], bq[1], bq[2]),
                    wp.quat(bq[3], bq[4], bq[5], bq[6])
                )

                # Shape Transform (Body -> Shape)
                sq = shape_xform_np[i] # This is a 7-float array
                X_bs = wp.transform(
                    wp.vec3(sq[0], sq[1], sq[2]),
                    wp.quat(sq[3], sq[4], sq[5], sq[6])
                )
                
                # Compute Global Transform (World -> Shape)
                X_ws = wp.transform_multiply(X_wb, X_bs)
                
                # Extract pos/rot for the renderer
                pos = wp.transform_get_translation(X_ws)
                rot = wp.transform_get_rotation(X_ws)
                
                # Get geometry info
                geo_type = shape_geo_type_np[i]
                scale = shape_geo_scale_np[i] # This is a 3-float array
                name = f"robot_shape_{i}"

                # Render based on type
                if geo_type == wp.sim.GEO_SPHERE:
                    # scale[0] is radius
                    self.renderer.render_sphere(name, pos, rot, radius=float(scale[0]))
                    
                elif geo_type == wp.sim.GEO_BOX:
                    # scale is (x, y, z) extents
                    self.renderer.render_box(name, pos, rot, extents=tuple(scale))
                    
                elif geo_type == wp.sim.GEO_CAPSULE:
                    # scale[0] = radius, scale[1] = half_height
                    self.renderer.render_capsule(name, pos, rot, radius=float(scale[0]), half_height=float(scale[1]))
                
                elif geo_type == wp.sim.GEO_MESH:
                    # Fallback for meshes -> Draw a small proxy sphere so we can see where it is
                    self.renderer.render_sphere(name, pos, rot, radius=0.05)
            
            self.renderer.end_frame()

    
    def scale_robot_model(self, scale_factor):
        joint_X_np = self.robot_model.joint_X_p.numpy()
        joint_X_np[:, :3] *= scale_factor
        self.robot_model.joint_X_p = wp.from_numpy(joint_X_np, dtype=wp.transform)
        shape_scale_np = self.robot_model.shape_geo.scale.numpy()
        shape_scale_np *= scale_factor
        self.robot_model.shape_geo.scale = wp.from_numpy(shape_scale_np, dtype=wp.vec3)
        shape_X_np = self.robot_model.shape_transform.numpy()
        shape_X_np[:, :3] *= scale_factor
        self.robot_model.shape_transform = wp.from_numpy(shape_X_np, dtype=wp.transform)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sph.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=480, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        for _ in range(args.num_frames):
            example.render()
            example.step()

        if example.renderer:
            example.renderer.save()
