# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time the individual mesh ray traversals on the replicated ASV bunny scene."""

import warp as wp

from . import spatial_query as spatial
from .benchmarks_utils import setup_once

wp.set_module_options({"enable_backward": False})


@wp.func_native(
    snippet="""
    float t, u, v, sign;
    wp::vec3 normal;
    int face;
    return wp::mesh_query_ray_ordered(mesh, start, direction, max_t, t, u, v, sign, normal, face);
    """
)
def ray_ordered(mesh: wp.uint64, start: wp.vec3, direction: wp.vec3, max_t: float) -> bool: ...


@wp.func_native(
    snippet="""
    float sign = 0.0f;
    wp::mesh_query_ray_closest_sign(wp::mesh_get(mesh), start, direction, sign);
    return sign;
    """
)
def ray_closest_sign(mesh: wp.uint64, start: wp.vec3, direction: wp.vec3) -> float: ...


def get_mesh_ray_kernel(variant):
    @wp.kernel
    def mesh_ray_variant(
        mesh: wp.uint64,
        camera: spatial.Camera,
        mesh_pos: wp.array[wp.vec3],
        mesh_rot: wp.array[wp.quat],
        width: int,
        height: int,
        results: wp.array[wp.vec3],
    ):
        tid = wp.tid()
        x = tid % width
        y = height - 1 - tid // width
        sx = 2.0 * float(x) / float(width) - 1.0
        sy = 2.0 * float(y) / float(height) - 1.0
        direction = wp.normalize(
            wp.quat_rotate(camera.rot, wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0))
        )
        inv = wp.transform_inverse(wp.transform(mesh_pos[0], mesh_rot[0]))
        start = wp.transform_point(inv, camera.pos)
        direction = wp.transform_vector(inv, direction)
        value = float(0.0)
        if wp.static(variant == "closest"):
            value = float(wp.mesh_query_ray(mesh, start, direction, 1.0e6).result)
        elif wp.static(variant == "any"):
            value = float(wp.mesh_query_ray_anyhit(mesh, start, direction, 1.0e6))
        elif wp.static(variant == "ordered"):
            value = float(ray_ordered(mesh, start, direction, 1.0e6))
        elif wp.static(variant == "count"):
            value = float(wp.mesh_query_ray_count_intersections(mesh, start, direction))
        elif wp.static(variant == "sign"):
            value = ray_closest_sign(mesh, start, direction)
        results[tid] = wp.vec3(value, value, value)

    return mesh_ray_variant


class MeshRayVariants:
    params = [["closest", "any", "ordered", "count", "sign"], [1080], [0, 8], ["lbvh", "cubql"]]
    param_names = ["variant", "resolution", "leaf_size", "constructor"]
    number = 5
    timeout = 120

    @setup_once
    def setup(self, variant, resolution, leaf_size, constructor):
        if not spatial.USD_AVAILABLE:
            raise NotImplementedError("USD is required for the mesh ray benchmark")
        # Each parameter cell gets the same geometry regardless of ASV setup order.
        spatial.seed = 42
        self.scene = spatial.BvhRayQuery()
        self.scene.setup(resolution, leaf_size, "cuda", constructor)
        scene = self.scene
        self.kernel = get_mesh_ray_kernel(variant)
        self.inputs = [scene.mesh.id, scene.camera, scene.mesh_pos, scene.mesh_rot, resolution, resolution]
        wp.launch(self.kernel, scene.num_rays, inputs=self.inputs, outputs=[scene.rays], device=scene.device)
        with wp.ScopedCapture(device=scene.device) as capture:
            for _ in range(spatial.NUM_TRIES):
                wp.launch(self.kernel, scene.num_rays, inputs=self.inputs, outputs=[scene.rays], device=scene.device)
        self.graph = capture.graph
        wp.capture_launch(self.graph)
        wp.synchronize_device(scene.device)

    def time_mesh_ray(self, variant, resolution, leaf_size, constructor):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.scene.device)
