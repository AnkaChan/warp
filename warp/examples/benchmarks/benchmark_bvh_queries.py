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

_VERSION = "ebvh_benchmark_v1"
print(f"[Warp] BVH query benchmark version: {_VERSION}")


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


def time_command(command, device, *, samples, batch_size, burn_in):
    """Return per-kernel wall times after a sustained burn-in."""
    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            for _ in range(batch_size):
                command.launch()
        graph = capture.graph

        for _ in range(burn_in):
            wp.capture_launch(graph)
        wp.synchronize_device(device)

        timings = []
        for _ in range(samples):
            start = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize_device(device)
            timings.append((time.perf_counter() - start) * 1000.0 / batch_size)
        return timings

    for _ in range(burn_in):
        command.launch()
    wp.synchronize_device(device)

    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(batch_size):
            command.launch()
        wp.synchronize_device(device)
        timings.append((time.perf_counter() - start) * 1000.0 / batch_size)
    return timings


def benchmark_device(device_name, args, aabb_data, mesh_data):
    device = wp.get_device(device_name)
    batch_size = args.cuda_batch_size if device.is_cuda else args.cpu_batch_size

    with wp.ScopedDevice(device):
        lowers, uppers, query_lowers, query_uppers, aabb_seeds = aabb_data
        bvh = wp.Bvh(
            wp.array(lowers, dtype=wp.vec3, device=device),
            wp.array(uppers, dtype=wp.vec3, device=device),
            constructor=args.constructor,
            leaf_size=args.leaf_size,
        )
        query_lowers_wp = wp.array(query_lowers, dtype=wp.vec3, device=device)
        query_uppers_wp = wp.array(query_uppers, dtype=wp.vec3, device=device)
        aabb_seeds_wp = wp.array(aabb_seeds, dtype=wp.int32, device=device)
        hit_counts = wp.empty(args.queries, dtype=wp.int32, device=device)

        aabb_command = _recorded_command(
            bvh_query_aabb_root_kernel,
            args.queries,
            [bvh.id, query_lowers_wp, query_uppers_wp, hit_counts],
            device,
        )
        aabb_times = time_command(
            aabb_command,
            device,
            samples=args.samples,
            batch_size=batch_size,
            burn_in=args.burn_in,
        )
        aabb_command.launch()
        aabb_hits = hit_counts.numpy()

        points, indices, query_points, mesh_seeds = mesh_data
        mesh = wp.Mesh(
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(indices, dtype=wp.int32, device=device),
            bvh_constructor=args.constructor,
            bvh_leaf_size=args.leaf_size,
        )
        query_points_wp = wp.array(query_points, dtype=wp.vec3, device=device)
        mesh_seeds_wp = wp.array(mesh_seeds, dtype=wp.int32, device=device)
        result_faces = wp.empty(args.queries, dtype=wp.int32, device=device)
        result_barycentrics = wp.empty(args.queries, dtype=wp.vec2, device=device)

        closest_command = _recorded_command(
            mesh_query_point_root_kernel,
            args.queries,
            [mesh.id, query_points_wp, result_faces, result_barycentrics],
            device,
        )
        closest_times = time_command(
            closest_command,
            device,
            samples=args.samples,
            batch_size=batch_size,
            burn_in=args.burn_in,
        )
        closest_command.launch()
        closest_faces = result_faces.numpy()
        closest_barycentrics = result_barycentrics.numpy()

    # Keep seed arrays alive and visible to future benchmark arms.
    _ = aabb_seeds_wp, mesh_seeds_wp

    return {
        "device": str(device),
        "constructor": args.constructor,
        "leaf_size": args.leaf_size,
        "num_queries": args.queries,
        "aabb": {
            "num_primitives": args.aabb_primitives,
            "median_ms": median(aabb_times),
            "samples_ms": aabb_times,
            "hit_sum": int(aabb_hits.sum()),
            "hit_checksum": int(np.dot(aabb_hits.astype(np.int64), np.arange(1, args.queries + 1, dtype=np.int64))),
        },
        "closest_point": {
            "num_triangles": args.mesh_triangles,
            "median_ms": median(closest_times),
            "samples_ms": closest_times,
            "result_count": int(np.count_nonzero(closest_faces >= 0)),
            "face_checksum": int(
                np.dot(closest_faces.astype(np.int64), np.arange(1, args.queries + 1, dtype=np.int64))
            ),
            "barycentric_checksum": float(np.sum(closest_barycentrics, dtype=np.float64)),
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
