# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark coherent AABB and exact mesh closest-point BVH queries.

This complements ``benchmark_bvh.py`` with the coherent, scattered workloads
where bottom-up Exclusive BVH traversal can avoid a meaningful root prefix.
The closest-point workload evaluates triangles rather than using leaf AABBs as
a distance proxy.
"""

import argparse
import json
import time
from statistics import median

import numpy as np

import warp as wp

_VERSION = "ebvh_benchmark_v2"
print(f"[Warp] BVH query benchmark version: {_VERSION}")

AABB_VALIDATION_CAPACITY = 16
WP_AABB_VALIDATION_CAPACITY = wp.constant(AABB_VALIDATION_CAPACITY)


@wp.kernel
def bvh_query_aabb_root_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[tid], query_uppers[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_root_sets_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    hit_counts: wp.array(dtype=wp.int32),
    hit_ids: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[tid], query_uppers[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        if count < WP_AABB_VALIDATION_CAPACITY:
            hit_ids[tid * WP_AABB_VALIDATION_CAPACITY + count] = primitive
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_exclusive_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    seed_primitives: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive(bvh_id, query_lowers[tid], query_uppers[tid], seed_primitives[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_exclusive_sets_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    seed_primitives: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
    hit_ids: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive(bvh_id, query_lowers[tid], query_uppers[tid], seed_primitives[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        if count < WP_AABB_VALIDATION_CAPACITY:
            hit_ids[tid * WP_AABB_VALIDATION_CAPACITY + count] = primitive
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_exclusive_nodes_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    seed_primitives: wp.array(dtype=wp.int32),
    cached_nodes: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    cached_nodes[tid] = wp.bvh_query_aabb_exclusive_node(
        bvh_id, query_lowers[tid], query_uppers[tid], seed_primitives[tid]
    )


@wp.kernel
def bvh_query_aabb_exclusive_cached_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    cached_nodes: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive_cached(bvh_id, query_lowers[tid], query_uppers[tid], cached_nodes[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_exclusive_cached_sets_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    cached_nodes: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
    hit_ids: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive_cached(bvh_id, query_lowers[tid], query_uppers[tid], cached_nodes[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        if count < WP_AABB_VALIDATION_CAPACITY:
            hit_ids[tid * WP_AABB_VALIDATION_CAPACITY + count] = primitive
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_cached_node_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    cached_nodes: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[tid], query_uppers[tid], cached_nodes[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        count += 1
    hit_counts[tid] = count


@wp.kernel
def bvh_query_aabb_cached_node_sets_kernel(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    cached_nodes: wp.array(dtype=wp.int32),
    hit_counts: wp.array(dtype=wp.int32),
    hit_ids: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[tid], query_uppers[tid], cached_nodes[tid])
    primitive = int(0)
    count = int(0)
    while wp.bvh_query_next(query, primitive):
        if count < WP_AABB_VALIDATION_CAPACITY:
            hit_ids[tid * WP_AABB_VALIDATION_CAPACITY + count] = primitive
        count += 1
    hit_counts[tid] = count


@wp.kernel
def mesh_query_point_root_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    result_faces: wp.array(dtype=wp.int32),
    result_barycentrics: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh_id, query_points[tid], 1.0e7)
    if query.result:
        result_faces[tid] = query.face
        result_barycentrics[tid] = wp.vec2(query.u, query.v)
    else:
        result_faces[tid] = -1
        result_barycentrics[tid] = wp.vec2(0.0, 0.0)


@wp.kernel
def mesh_query_point_seeded_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    seed_faces: wp.array(dtype=wp.int32),
    result_faces: wp.array(dtype=wp.int32),
    result_barycentrics: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_seeded(mesh_id, query_points[tid], 1.0e7, seed_faces[tid])
    if query.result:
        result_faces[tid] = query.face
        result_barycentrics[tid] = wp.vec2(query.u, query.v)
    else:
        result_faces[tid] = -1
        result_barycentrics[tid] = wp.vec2(0.0, 0.0)


@wp.kernel
def mesh_query_point_exclusive_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    seed_faces: wp.array(dtype=wp.int32),
    result_faces: wp.array(dtype=wp.int32),
    result_barycentrics: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_exclusive(mesh_id, query_points[tid], 1.0e7, seed_faces[tid])
    if query.result:
        result_faces[tid] = query.face
        result_barycentrics[tid] = wp.vec2(query.u, query.v)
    else:
        result_faces[tid] = -1
        result_barycentrics[tid] = wp.vec2(0.0, 0.0)


@wp.kernel
def mesh_query_point_exclusive_nodes_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    seed_faces: wp.array(dtype=wp.int32),
    cached_nodes: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    cached_nodes[tid] = wp.mesh_query_point_no_sign_exclusive_node(mesh_id, query_points[tid], 1.0e7, seed_faces[tid])


@wp.kernel
def mesh_query_point_exclusive_cached_kernel(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    seed_faces: wp.array(dtype=wp.int32),
    cached_nodes: wp.array(dtype=wp.int32),
    result_faces: wp.array(dtype=wp.int32),
    result_barycentrics: wp.array(dtype=wp.vec2),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_exclusive_cached(
        mesh_id, query_points[tid], 1.0e7, seed_faces[tid], cached_nodes[tid]
    )
    if query.result:
        result_faces[tid] = query.face
        result_barycentrics[tid] = wp.vec2(query.u, query.v)
    else:
        result_faces[tid] = -1
        result_barycentrics[tid] = wp.vec2(0.0, 0.0)


def _part_1_by_2(values):
    values = np.asarray(values, dtype=np.uint32) & np.uint32(0x3FF)
    values = (values | (values << np.uint32(16))) & np.uint32(0x030000FF)
    values = (values | (values << np.uint32(8))) & np.uint32(0x0300F00F)
    values = (values | (values << np.uint32(4))) & np.uint32(0x030C30C3)
    values = (values | (values << np.uint32(2))) & np.uint32(0x09249249)
    return values


def _morton_order(points, scene_lower=-10.0, scene_upper=10.0):
    scale = 1024.0 / (scene_upper - scene_lower)
    quantized = np.clip((points - scene_lower) * scale, 0.0, 1023.0).astype(np.uint32)
    codes = (
        _part_1_by_2(quantized[:, 0])
        | (_part_1_by_2(quantized[:, 1]) << np.uint32(1))
        | (_part_1_by_2(quantized[:, 2]) << np.uint32(2))
    )
    return np.argsort(codes, kind="stable")


def make_aabb_data(num_primitives, num_queries, rng):
    """Create the validated P08-style scattered coherent AABB workload."""
    scene_radius = 10.0
    scene_diameter = 2.0 * scene_radius
    item_size = scene_diameter * 0.005
    query_half_size = scene_diameter * 0.002

    centers = rng.uniform(-scene_radius, scene_radius, size=(num_primitives, 3)).astype(np.float32)
    half_sizes = rng.uniform(0.5 * item_size, item_size, size=(num_primitives, 1)).astype(np.float32)
    lowers = centers - half_sizes
    uppers = centers + half_sizes

    seeds = rng.integers(0, num_primitives, size=num_queries, dtype=np.int32)
    query_centers = centers[seeds] + rng.uniform(
        -0.3 * query_half_size, 0.3 * query_half_size, size=(num_queries, 3)
    ).astype(np.float32)
    query_lowers = query_centers - query_half_size
    query_uppers = query_centers + query_half_size

    order = _morton_order(query_centers)
    return lowers, uppers, query_lowers[order], query_uppers[order], seeds[order]


def make_mesh_data(num_triangles, num_queries, rng):
    """Create separated exact triangles and coherent closest-point queries."""
    scene_radius = 10.0
    scene_diameter = 2.0 * scene_radius
    triangle_size = scene_diameter * 0.0025

    centers = rng.uniform(-scene_radius, scene_radius, size=(num_triangles, 3)).astype(np.float32)
    sizes = rng.uniform(0.5 * triangle_size, triangle_size, size=(num_triangles, 1)).astype(np.float32)

    points = np.empty((num_triangles, 3, 3), dtype=np.float32)
    points[:, 0, :] = centers + np.concatenate((-sizes, -0.5 * sizes, -0.2 * sizes), axis=1)
    points[:, 1, :] = centers + np.concatenate((sizes, -0.5 * sizes, 0.1 * sizes), axis=1)
    points[:, 2, :] = centers + np.concatenate((np.zeros_like(sizes), sizes, 0.2 * sizes), axis=1)
    points = points.reshape((-1, 3))
    indices = np.arange(num_triangles * 3, dtype=np.int32)

    seeds = rng.integers(0, num_triangles, size=num_queries, dtype=np.int32)
    perturbation = 2.0 * sizes[seeds]
    query_points = centers[seeds] + rng.uniform(-1.0, 1.0, size=(num_queries, 3)).astype(np.float32) * perturbation

    order = _morton_order(query_points)
    return points, indices, query_points[order], seeds[order]


def _recorded_command(kernel, dim, inputs, device):
    return wp.launch(kernel, dim=dim, inputs=inputs, device=device, record_cmd=True)


def time_commands(commands, device, *, samples, batch_size, burn_in):
    """Return interleaved per-kernel wall times after a sustained burn-in."""
    launchers = {}
    if device.is_cuda:
        for name, command in commands.items():
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                for _ in range(batch_size):
                    command.launch()
            graph = capture.graph
            launchers[name] = lambda graph=graph: wp.capture_launch(graph)
    else:
        for name, command in commands.items():

            def launch_batch(command=command):
                for _ in range(batch_size):
                    command.launch()

            launchers[name] = launch_batch

    names = list(commands)
    for iteration in range(burn_in):
        order = names if iteration % 2 == 0 else reversed(names)
        for name in order:
            launchers[name]()
    wp.synchronize_device(device)

    timings = {name: [] for name in names}
    for sample in range(samples):
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        if sample % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            launchers[name]()
            wp.synchronize_device(device)
            timings[name].append((time.perf_counter() - start) * 1000.0 / batch_size)
    return timings


def validate_aabb_hit_sets(
    device,
    num_queries,
    kernels_and_inputs,
):
    """Run untimed query arms and require identical primitive-ID sets."""
    result_sets = {}
    for name, kernel, inputs in kernels_and_inputs:
        counts = wp.empty(num_queries, dtype=wp.int32, device=device)
        hit_ids = wp.full(
            num_queries * AABB_VALIDATION_CAPACITY,
            -1,
            dtype=wp.int32,
            device=device,
        )
        wp.launch(
            kernel,
            dim=num_queries,
            inputs=[*inputs, counts, hit_ids],
            device=device,
        )
        counts_np = counts.numpy()
        max_hits = int(counts_np.max(initial=0))
        if max_hits > AABB_VALIDATION_CAPACITY:
            raise RuntimeError(
                f"AABB validation capacity {AABB_VALIDATION_CAPACITY} is too small; "
                f"{name} returned {max_hits} hits for one query"
            )
        hit_ids_np = hit_ids.numpy().reshape(num_queries, AABB_VALIDATION_CAPACITY)
        result_sets[name] = (counts_np, np.sort(hit_ids_np, axis=1))

    reference_counts, reference_ids = result_sets["root"]
    for name, (counts, hit_ids) in result_sets.items():
        np.testing.assert_array_equal(counts, reference_counts, err_msg=f"AABB hit counts differ for {name}")
        np.testing.assert_array_equal(hit_ids, reference_ids, err_msg=f"AABB primitive-ID sets differ for {name}")
    return reference_counts, reference_ids


def benchmark_device(device_name, args, aabb_data, mesh_data):
    device = wp.get_device(device_name)
    batch_size = args.cuda_batch_size if device.is_cuda else args.cpu_batch_size

    with wp.ScopedDevice(device):
        lowers, uppers, query_lowers, query_uppers, aabb_seeds = aabb_data
        lowers_wp = wp.array(lowers, dtype=wp.vec3, device=device)
        uppers_wp = wp.array(uppers, dtype=wp.vec3, device=device)
        bvh = wp.Bvh(
            lowers_wp,
            uppers_wp,
            constructor=args.constructor,
            leaf_size=args.leaf_size,
        )
        bvh_exclusive = wp.Bvh(
            lowers_wp,
            uppers_wp,
            constructor=args.constructor,
            leaf_size=args.leaf_size,
            enable_exclusive=True,
        )
        query_lowers_wp = wp.array(query_lowers, dtype=wp.vec3, device=device)
        query_uppers_wp = wp.array(query_uppers, dtype=wp.vec3, device=device)
        aabb_seeds_wp = wp.array(aabb_seeds, dtype=wp.int32, device=device)
        aabb_cached_nodes = wp.empty(args.queries, dtype=wp.int32, device=device)
        root_hit_counts = wp.empty(args.queries, dtype=wp.int32, device=device)
        exclusive_hit_counts = wp.empty(args.queries, dtype=wp.int32, device=device)
        cached_hit_counts = wp.empty(args.queries, dtype=wp.int32, device=device)
        cached_node_hit_counts = wp.empty(args.queries, dtype=wp.int32, device=device)

        wp.launch(
            bvh_query_aabb_exclusive_nodes_kernel,
            dim=args.queries,
            inputs=[bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_seeds_wp, aabb_cached_nodes],
            device=device,
        )

        aabb_root_command = _recorded_command(
            bvh_query_aabb_root_kernel,
            args.queries,
            [bvh.id, query_lowers_wp, query_uppers_wp, root_hit_counts],
            device,
        )
        aabb_exclusive_command = _recorded_command(
            bvh_query_aabb_exclusive_kernel,
            args.queries,
            [
                bvh_exclusive.id,
                query_lowers_wp,
                query_uppers_wp,
                aabb_seeds_wp,
                exclusive_hit_counts,
            ],
            device,
        )
        aabb_cached_command = _recorded_command(
            bvh_query_aabb_exclusive_cached_kernel,
            args.queries,
            [bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_cached_nodes, cached_hit_counts],
            device,
        )
        aabb_cached_node_command = _recorded_command(
            bvh_query_aabb_cached_node_kernel,
            args.queries,
            [bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_cached_nodes, cached_node_hit_counts],
            device,
        )
        aabb_times = time_commands(
            {
                "root": aabb_root_command,
                "exclusive": aabb_exclusive_command,
                "exclusive_cached": aabb_cached_command,
                "cached_node": aabb_cached_node_command,
            },
            device,
            samples=args.samples,
            batch_size=batch_size,
            burn_in=args.burn_in,
        )
        aabb_root_command.launch()
        aabb_exclusive_command.launch()
        aabb_cached_command.launch()
        aabb_cached_node_command.launch()
        aabb_root_hits = root_hit_counts.numpy()
        aabb_exclusive_hits = exclusive_hit_counts.numpy()
        aabb_cached_hits = cached_hit_counts.numpy()
        aabb_cached_node_hits = cached_node_hit_counts.numpy()
        np.testing.assert_array_equal(aabb_exclusive_hits, aabb_root_hits)
        np.testing.assert_array_equal(aabb_cached_hits, aabb_root_hits)
        np.testing.assert_array_equal(aabb_cached_node_hits, aabb_root_hits)

        aabb_root_hits, aabb_root_ids = validate_aabb_hit_sets(
            device,
            args.queries,
            (
                (
                    "root",
                    bvh_query_aabb_root_sets_kernel,
                    [bvh.id, query_lowers_wp, query_uppers_wp],
                ),
                (
                    "exclusive",
                    bvh_query_aabb_exclusive_sets_kernel,
                    [bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_seeds_wp],
                ),
                (
                    "exclusive_cached",
                    bvh_query_aabb_exclusive_cached_sets_kernel,
                    [bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_cached_nodes],
                ),
                (
                    "cached_node",
                    bvh_query_aabb_cached_node_sets_kernel,
                    [bvh_exclusive.id, query_lowers_wp, query_uppers_wp, aabb_cached_nodes],
                ),
            ),
        )

        points, indices, query_points, mesh_seeds = mesh_data
        points_wp = wp.array(points, dtype=wp.vec3, device=device)
        indices_wp = wp.array(indices, dtype=wp.int32, device=device)
        mesh = wp.Mesh(
            points_wp,
            indices_wp,
            bvh_constructor=args.constructor,
            bvh_leaf_size=args.leaf_size,
        )
        mesh_exclusive = wp.Mesh(
            points_wp,
            indices_wp,
            bvh_constructor=args.constructor,
            bvh_leaf_size=args.leaf_size,
            enable_exclusive=True,
        )
        query_points_wp = wp.array(query_points, dtype=wp.vec3, device=device)
        mesh_seeds_wp = wp.array(mesh_seeds, dtype=wp.int32, device=device)
        mesh_cached_nodes = wp.empty(args.queries, dtype=wp.int32, device=device)
        root_faces = wp.empty(args.queries, dtype=wp.int32, device=device)
        root_barycentrics = wp.empty(args.queries, dtype=wp.vec2, device=device)
        seeded_faces = wp.empty(args.queries, dtype=wp.int32, device=device)
        seeded_barycentrics = wp.empty(args.queries, dtype=wp.vec2, device=device)
        exclusive_faces = wp.empty(args.queries, dtype=wp.int32, device=device)
        exclusive_barycentrics = wp.empty(args.queries, dtype=wp.vec2, device=device)
        cached_faces = wp.empty(args.queries, dtype=wp.int32, device=device)
        cached_barycentrics = wp.empty(args.queries, dtype=wp.vec2, device=device)

        wp.launch(
            mesh_query_point_exclusive_nodes_kernel,
            dim=args.queries,
            inputs=[mesh_exclusive.id, query_points_wp, mesh_seeds_wp, mesh_cached_nodes],
            device=device,
        )

        closest_root_command = _recorded_command(
            mesh_query_point_root_kernel,
            args.queries,
            [mesh.id, query_points_wp, root_faces, root_barycentrics],
            device,
        )
        closest_seeded_command = _recorded_command(
            mesh_query_point_seeded_kernel,
            args.queries,
            [mesh_exclusive.id, query_points_wp, mesh_seeds_wp, seeded_faces, seeded_barycentrics],
            device,
        )
        closest_exclusive_command = _recorded_command(
            mesh_query_point_exclusive_kernel,
            args.queries,
            [mesh_exclusive.id, query_points_wp, mesh_seeds_wp, exclusive_faces, exclusive_barycentrics],
            device,
        )
        closest_cached_command = _recorded_command(
            mesh_query_point_exclusive_cached_kernel,
            args.queries,
            [
                mesh_exclusive.id,
                query_points_wp,
                mesh_seeds_wp,
                mesh_cached_nodes,
                cached_faces,
                cached_barycentrics,
            ],
            device,
        )
        closest_times = time_commands(
            {
                "root": closest_root_command,
                "seeded_root": closest_seeded_command,
                "exclusive": closest_exclusive_command,
                "exclusive_cached": closest_cached_command,
            },
            device,
            samples=args.samples,
            batch_size=batch_size,
            burn_in=args.burn_in,
        )
        closest_root_command.launch()
        closest_seeded_command.launch()
        closest_exclusive_command.launch()
        closest_cached_command.launch()
        closest_root_faces = root_faces.numpy()
        closest_root_barycentrics = root_barycentrics.numpy()
        closest_seeded_faces = seeded_faces.numpy()
        closest_seeded_barycentrics = seeded_barycentrics.numpy()
        closest_exclusive_faces = exclusive_faces.numpy()
        closest_exclusive_barycentrics = exclusive_barycentrics.numpy()
        closest_cached_faces = cached_faces.numpy()
        closest_cached_barycentrics = cached_barycentrics.numpy()
        np.testing.assert_array_equal(closest_seeded_faces, closest_root_faces)
        np.testing.assert_array_equal(closest_exclusive_faces, closest_root_faces)
        np.testing.assert_array_equal(closest_cached_faces, closest_root_faces)
        np.testing.assert_allclose(closest_seeded_barycentrics, closest_root_barycentrics, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(closest_exclusive_barycentrics, closest_root_barycentrics, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(closest_cached_barycentrics, closest_root_barycentrics, rtol=1.0e-6, atol=1.0e-6)

    aabb_root_median = median(aabb_times["root"])
    aabb_exclusive_median = median(aabb_times["exclusive"])
    aabb_cached_median = median(aabb_times["exclusive_cached"])
    aabb_cached_node_median = median(aabb_times["cached_node"])
    closest_root_median = median(closest_times["root"])
    closest_seeded_median = median(closest_times["seeded_root"])
    closest_exclusive_median = median(closest_times["exclusive"])
    closest_cached_median = median(closest_times["exclusive_cached"])

    return {
        "device": str(device),
        "constructor": args.constructor,
        "leaf_size": args.leaf_size,
        "num_queries": args.queries,
        "aabb": {
            "num_primitives": args.aabb_primitives,
            "root": {"median_ms": aabb_root_median, "samples_ms": aabb_times["root"]},
            "exclusive": {
                "median_ms": aabb_exclusive_median,
                "samples_ms": aabb_times["exclusive"],
                "speedup": aabb_root_median / aabb_exclusive_median,
            },
            "exclusive_cached": {
                "median_ms": aabb_cached_median,
                "samples_ms": aabb_times["exclusive_cached"],
                "speedup": aabb_root_median / aabb_cached_median,
            },
            "cached_node": {
                "median_ms": aabb_cached_node_median,
                "samples_ms": aabb_times["cached_node"],
                "speedup": aabb_root_median / aabb_cached_node_median,
            },
            "hit_sum": int(aabb_root_hits.sum()),
            "hit_count_checksum": int(
                np.dot(aabb_root_hits.astype(np.int64), np.arange(1, args.queries + 1, dtype=np.int64))
            ),
            "hit_index_checksum": int(
                np.dot(
                    aabb_root_ids.astype(np.int64).ravel(),
                    np.arange(1, aabb_root_ids.size + 1, dtype=np.int64),
                )
            ),
            "full_sets_match": True,
        },
        "closest_point": {
            "num_triangles": args.mesh_triangles,
            "root": {"median_ms": closest_root_median, "samples_ms": closest_times["root"]},
            "seeded_root": {
                "median_ms": closest_seeded_median,
                "samples_ms": closest_times["seeded_root"],
                "speedup": closest_root_median / closest_seeded_median,
            },
            "exclusive": {
                "median_ms": closest_exclusive_median,
                "samples_ms": closest_times["exclusive"],
                "speedup_vs_root": closest_root_median / closest_exclusive_median,
                "speedup_vs_seeded_root": closest_seeded_median / closest_exclusive_median,
            },
            "exclusive_cached": {
                "median_ms": closest_cached_median,
                "samples_ms": closest_times["exclusive_cached"],
                "speedup_vs_root": closest_root_median / closest_cached_median,
                "speedup_vs_seeded_root": closest_seeded_median / closest_cached_median,
            },
            "result_count": int(np.count_nonzero(closest_root_faces >= 0)),
            "face_checksum": int(
                np.dot(closest_root_faces.astype(np.int64), np.arange(1, args.queries + 1, dtype=np.int64))
            ),
            "barycentric_checksum": float(np.sum(closest_root_barycentrics, dtype=np.float64)),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--constructor", choices=("sah", "median", "lbvh", "cubql"), default="sah")
    parser.add_argument("--leaf-size", type=int, default=1)
    parser.add_argument("--aabb-primitives", type=int, default=10_000)
    parser.add_argument("--mesh-triangles", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=200_000)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--cuda-batch-size", type=int, default=20)
    parser.add_argument("--cpu-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wp.init()
    wp.set_module_options({"enable_backward": False, "fast_math": True})

    rng = np.random.default_rng(args.seed)
    aabb_data = make_aabb_data(args.aabb_primitives, args.queries, rng)
    mesh_data = make_mesh_data(args.mesh_triangles, args.queries, rng)

    devices = ("cpu", "cuda") if args.device == "all" else (args.device,)
    results = [benchmark_device(device, args, aabb_data, mesh_data) for device in devices]
    print(json.dumps({"version": _VERSION, "results": results}, indent=2))
