# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


@wp.kernel
def query_aabb(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    num_bounds: int,
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, query_lowers[tid], query_uppers[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        hits[tid * num_bounds + index] = 1


@wp.kernel
def query_aabb_exclusive(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    num_bounds: int,
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive(bvh_id, query_lowers[tid], query_uppers[tid], seeds[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        hits[tid * num_bounds + index] = 1


@wp.kernel
def find_aabb_exclusive_nodes(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    tid = wp.tid()
    nodes[tid] = wp.bvh_query_aabb_exclusive_node(
        bvh_id,
        query_lowers[tid],
        query_uppers[tid],
        seeds[tid],
    )


@wp.kernel
def query_aabb_exclusive_cached(
    bvh_id: wp.uint64,
    query_lowers: wp.array(dtype=wp.vec3),
    query_uppers: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    num_bounds: int,
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive_cached(bvh_id, query_lowers[tid], query_uppers[tid], nodes[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        hits[tid * num_bounds + index] = 1


@wp.kernel
def query_mesh_points(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=float),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh_id, query_points[tid], max_distances[tid])

    results[tid] = 0
    faces[tid] = -1
    bary_u[tid] = 0.0
    bary_v[tid] = 0.0
    distances[tid] = max_distances[tid]
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        results[tid] = 1
        faces[tid] = query.face
        bary_u[tid] = query.u
        bary_v[tid] = query.v
        distances[tid] = wp.length(closest - query_points[tid])


@wp.kernel
def query_mesh_points_seeded(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_seeded(mesh_id, query_points[tid], max_distances[tid], seeds[tid])

    results[tid] = 0
    faces[tid] = -1
    bary_u[tid] = 0.0
    bary_v[tid] = 0.0
    distances[tid] = max_distances[tid]
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        results[tid] = 1
        faces[tid] = query.face
        bary_u[tid] = query.u
        bary_v[tid] = query.v
        distances[tid] = wp.length(closest - query_points[tid])


@wp.kernel
def query_mesh_points_exclusive(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_exclusive(mesh_id, query_points[tid], max_distances[tid], seeds[tid])

    results[tid] = 0
    faces[tid] = -1
    bary_u[tid] = 0.0
    bary_v[tid] = 0.0
    distances[tid] = max_distances[tid]
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        results[tid] = 1
        faces[tid] = query.face
        bary_u[tid] = query.u
        bary_v[tid] = query.v
        distances[tid] = wp.length(closest - query_points[tid])


@wp.kernel
def find_mesh_exclusive_nodes(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    tid = wp.tid()
    nodes[tid] = wp.mesh_query_point_no_sign_exclusive_node(
        mesh_id,
        query_points[tid],
        max_distances[tid],
        seeds[tid],
    )


@wp.kernel
def query_mesh_points_exclusive_cached(
    mesh_id: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    max_distances: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    cached_nodes: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    distances: wp.array(dtype=float),
):
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign_exclusive_cached(
        mesh_id,
        query_points[tid],
        max_distances[tid],
        seeds[tid],
        cached_nodes[tid],
    )

    results[tid] = 0
    faces[tid] = -1
    bary_u[tid] = 0.0
    bary_v[tid] = 0.0
    distances[tid] = max_distances[tid]
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        results[tid] = 1
        faces[tid] = query.face
        bary_u[tid] = query.u
        bary_v[tid] = query.v
        distances[tid] = wp.length(closest - query_points[tid])


def get_constructors(device):
    constructors = ["sah", "median"]
    if device.is_cuda:
        constructors.append("lbvh")
    if wp.is_cubql_available():
        constructors.append("cubql")
    return constructors


def make_bounds(stage=0):
    lowers = np.empty((24, 3), dtype=np.float32)
    uppers = np.empty((24, 3), dtype=np.float32)

    # The first two bounds share the x=1 plane. A query contained non-strictly
    # by bound 0's exclusive box must still visit bound 1 because touching
    # AABBs intersect in Warp.
    lowers[0] = (0.0, 0.0, 0.0)
    uppers[0] = (1.0, 1.0, 1.0)
    lowers[1] = (1.0, 0.2, 0.2)
    uppers[1] = (2.0, 0.8, 0.8)

    for i in range(2, len(lowers)):
        col = (i - 2) % 6
        row = (i - 2) // 6
        lower = np.array((4.0 + 2.5 * col, 2.5 * row, 1.7 * ((i * 5) % 4)), dtype=np.float32)
        extent = np.array((0.4 + 0.05 * (i % 3), 0.5 + 0.03 * (i % 4), 0.35 + 0.04 * (i % 5)), dtype=np.float32)
        lowers[i] = lower
        uppers[i] = lower + extent

    if stage == 1:
        shift = np.array((8.0, -3.0, 2.0), dtype=np.float32)
        lowers += shift
        uppers += shift
    elif stage == 2:
        old_lowers = lowers.copy()
        lowers[:, 0] = 22.0 - uppers[:, 0]
        uppers[:, 0] = 22.0 - old_lowers[:, 0]
        lowers[:, 1] -= 5.0
        uppers[:, 1] -= 5.0

    return lowers, uppers


def make_aabb_queries(lowers, uppers):
    box_0_mid_x = 0.5 * (lowers[0, 0] + uppers[0, 0])
    if np.isclose(uppers[0, 0], lowers[1, 0]):
        shared_x = uppers[0, 0]
    else:
        shared_x = lowers[0, 0]

    query_lowers = [
        (min(box_0_mid_x, shared_x), 0.3 + lowers[0, 1], 0.3 + lowers[0, 2]),
        lowers[4] - 0.05,
        np.minimum(lowers[8], lowers[9]),
        np.min(lowers, axis=0) - 0.1,
        np.max(uppers, axis=0) + 4.0,
        (shared_x, 0.4 + lowers[0, 1], 0.4 + lowers[0, 2]),
        lowers[15] - 0.02,
        lowers[20] - 0.02,
    ]
    query_uppers = [
        (max(box_0_mid_x, shared_x), 0.7 + lowers[0, 1], 0.7 + lowers[0, 2]),
        uppers[4] + 0.05,
        np.maximum(uppers[8], uppers[9]),
        np.max(uppers, axis=0) + 0.1,
        np.max(uppers, axis=0) + 5.0,
        (shared_x, 0.6 + lowers[0, 1], 0.6 + lowers[0, 2]),
        uppers[15] + 0.02,
        uppers[20] + 0.02,
    ]
    seeds = np.array((0, 4, 3, 12, 0, 0, -1, len(lowers) + 7), dtype=np.int32)
    return np.asarray(query_lowers, dtype=np.float32), np.asarray(query_uppers, dtype=np.float32), seeds


def expected_aabb_hits(lowers, uppers, query_lowers, query_uppers):
    return np.logical_and(
        np.all(query_lowers[:, None, :] <= uppers[None, :, :], axis=2),
        np.all(lowers[None, :, :] <= query_uppers[:, None, :], axis=2),
    ).astype(np.int32)


def run_aabb_queries(bvh, query_lowers, query_uppers, seeds, device, exclusive):
    query_lowers_device = wp.array(query_lowers, dtype=wp.vec3, device=device)
    query_uppers_device = wp.array(query_uppers, dtype=wp.vec3, device=device)
    hits = wp.zeros(len(query_lowers) * len(bvh.lowers), dtype=int, device=device)
    if exclusive:
        seeds_device = wp.array(seeds, dtype=int, device=device)
        wp.launch(
            query_aabb_exclusive,
            dim=len(query_lowers),
            inputs=[bvh.id, query_lowers_device, query_uppers_device, seeds_device, len(bvh.lowers)],
            outputs=[hits],
            device=device,
        )
    else:
        wp.launch(
            query_aabb,
            dim=len(query_lowers),
            inputs=[bvh.id, query_lowers_device, query_uppers_device, len(bvh.lowers)],
            outputs=[hits],
            device=device,
        )
    return hits.numpy().reshape((len(query_lowers), len(bvh.lowers)))


def find_aabb_nodes(bvh, query_lowers, query_uppers, seeds, device):
    query_lowers_device = wp.array(query_lowers, dtype=wp.vec3, device=device)
    query_uppers_device = wp.array(query_uppers, dtype=wp.vec3, device=device)
    seeds_device = wp.array(seeds, dtype=int, device=device)
    nodes = wp.empty(len(query_lowers), dtype=int, device=device)
    wp.launch(
        find_aabb_exclusive_nodes,
        dim=len(query_lowers),
        inputs=[bvh.id, query_lowers_device, query_uppers_device, seeds_device],
        outputs=[nodes],
        device=device,
    )
    return nodes.numpy()


def run_cached_aabb_queries(bvh, query_lowers, query_uppers, cached_nodes, device):
    query_lowers_device = wp.array(query_lowers, dtype=wp.vec3, device=device)
    query_uppers_device = wp.array(query_uppers, dtype=wp.vec3, device=device)
    cached_nodes_device = wp.array(cached_nodes, dtype=int, device=device)
    hits = wp.zeros(len(query_lowers) * len(bvh.lowers), dtype=int, device=device)
    wp.launch(
        query_aabb_exclusive_cached,
        dim=len(query_lowers),
        inputs=[bvh.id, query_lowers_device, query_uppers_device, cached_nodes_device, len(bvh.lowers)],
        outputs=[hits],
        device=device,
    )
    return hits.numpy().reshape((len(query_lowers), len(bvh.lowers)))


def check_aabb_queries(test, bvh, lowers, uppers, device, cached_nodes=None):
    query_lowers, query_uppers, seeds = make_aabb_queries(lowers, uppers)
    expected = expected_aabb_hits(lowers, uppers, query_lowers, query_uppers)
    baseline = run_aabb_queries(bvh, query_lowers, query_uppers, seeds, device, exclusive=False)
    exclusive = run_aabb_queries(bvh, query_lowers, query_uppers, seeds, device, exclusive=True)
    if cached_nodes is None:
        cached_nodes = find_aabb_nodes(bvh, query_lowers, query_uppers, seeds, device)
    cached = run_cached_aabb_queries(bvh, query_lowers, query_uppers, cached_nodes, device)

    np.testing.assert_array_equal(baseline, expected)
    np.testing.assert_array_equal(exclusive, expected)
    np.testing.assert_array_equal(cached, expected)
    # Both boxes that share the clipping plane must be reported.
    test.assertEqual(int(exclusive[0, 0]), 1)
    test.assertEqual(int(exclusive[0, 1]), 1)
    test.assertEqual(int(exclusive[5, 0]), 1)
    test.assertEqual(int(exclusive[5, 1]), 1)
    return cached_nodes


def test_bvh_exclusive_aabb(test, device):
    for constructor in get_constructors(device):
        for leaf_size in (1, 8):
            with test.subTest(constructor=constructor, leaf_size=leaf_size):
                lowers_np, uppers_np = make_bounds(stage=0)
                lowers = wp.array(lowers_np, dtype=wp.vec3, device=device)
                uppers = wp.array(uppers_np, dtype=wp.vec3, device=device)
                bvh = wp.Bvh(
                    lowers,
                    uppers,
                    constructor=constructor,
                    leaf_size=leaf_size,
                    enable_exclusive=True,
                )
                cached_nodes = check_aabb_queries(test, bvh, lowers_np, uppers_np, device)
                if constructor == "cubql":
                    query_lowers, query_uppers, _ = make_aabb_queries(lowers_np, uppers_np)
                    sentinel_hits = run_cached_aabb_queries(
                        bvh,
                        query_lowers,
                        query_uppers,
                        np.ones(len(query_lowers), dtype=np.int32),
                        device,
                    )
                    np.testing.assert_array_equal(
                        sentinel_hits,
                        expected_aabb_hits(lowers_np, uppers_np, query_lowers, query_uppers),
                    )

                lowers_np, uppers_np = make_bounds(stage=1)
                lowers.assign(lowers_np)
                uppers.assign(uppers_np)
                bvh.refit()
                check_aabb_queries(test, bvh, lowers_np, uppers_np, device, cached_nodes=cached_nodes)

                lowers_np, uppers_np = make_bounds(stage=2)
                lowers.assign(lowers_np)
                uppers.assign(uppers_np)
                bvh.rebuild()
                check_aabb_queries(test, bvh, lowers_np, uppers_np, device, cached_nodes=cached_nodes)

                # A CUDA rebuild replaces a compact host-built SAH/median
                # topology with a full LBVH. A following refit must use the
                # rebuilt leaf count rather than the old compact count.
                if device.is_cuda and constructor in ("sah", "median") and leaf_size > 1:
                    lowers_np, uppers_np = make_bounds(stage=1)
                    lowers.assign(lowers_np)
                    uppers.assign(uppers_np)
                    bvh.refit()
                    check_aabb_queries(test, bvh, lowers_np, uppers_np, device, cached_nodes=cached_nodes)

    # Missing Exclusive metadata must take the ordinary root-traversal path.
    lowers_np, uppers_np = make_bounds()
    lowers = wp.array(lowers_np, dtype=wp.vec3, device=device)
    uppers = wp.array(uppers_np, dtype=wp.vec3, device=device)
    bvh = wp.Bvh(lowers, uppers, constructor="sah", leaf_size=8)
    check_aabb_queries(test, bvh, lowers_np, uppers_np, device)

    groups = wp.zeros(len(lowers), dtype=int, device=device)
    with test.assertRaisesRegex(RuntimeError, "not supported with grouped BVHs"):
        wp.Bvh(lowers, uppers, groups=groups, enable_exclusive=True)


def test_bvh_exclusive_cached_stack_boundary(test, device):
    for num_bounds in (1 << 16, (1 << 16) + 1):
        with test.subTest(num_bounds=num_bounds):
            x = np.arange(num_bounds, dtype=np.float32)
            lowers_np = np.zeros((num_bounds, 3), dtype=np.float32)
            uppers_np = np.ones((num_bounds, 3), dtype=np.float32)
            lowers_np[:, 0] = x
            uppers_np[:, 0] = x + np.float32(0.5)
            lowers = wp.array(lowers_np, dtype=wp.vec3, device=device)
            uppers = wp.array(uppers_np, dtype=wp.vec3, device=device)
            bvh = wp.Bvh(
                lowers,
                uppers,
                constructor="median",
                leaf_size=1,
                enable_exclusive=True,
            )

            query_lowers = np.array(((-1.0, -1.0, -1.0),), dtype=np.float32)
            query_uppers = np.array(((float(num_bounds), 2.0, 2.0),), dtype=np.float32)
            seeds = np.zeros(1, dtype=np.int32)
            cached_nodes = find_aabb_nodes(bvh, query_lowers, query_uppers, seeds, device)
            # The top-down builder stores leaves first and places the root at
            # ``num_bounds``. The enclosing query must therefore select that
            # exact current root, not exercise the invalid-cache fallback.
            test.assertEqual(int(cached_nodes[0]), num_bounds)

            baseline = run_aabb_queries(bvh, query_lowers, query_uppers, seeds, device, exclusive=False)
            cached = run_cached_aabb_queries(bvh, query_lowers, query_uppers, cached_nodes, device)
            np.testing.assert_array_equal(baseline, np.ones_like(baseline))
            np.testing.assert_array_equal(cached, baseline)


def make_mesh_points(stage=0):
    points = np.empty((24 * 3, 3), dtype=np.float32)
    for face in range(24):
        col = face % 6
        row = face // 6
        if stage == 1:
            col = 5 - col
            row = (3 * row) % 4

        x = 3.0 * col + 0.04 * face
        y = 3.0 * row + 0.03 * (face % 6)
        z = 0.13 * ((face * 7) % 5)
        width = 0.8 + 0.03 * (face % 3)
        height = 0.7 + 0.02 * (face % 4)
        points[face * 3 + 0] = (x, y, z)
        points[face * 3 + 1] = (x + width, y, z)
        points[face * 3 + 2] = (x, y + height, z)
    return points


def make_mesh_queries(points):
    target_faces = np.array((0, 3, 7, 12, 18, 23, 5, 10, 15), dtype=np.int32)
    seeds = np.array((0, 4, 1, 23, 12, 0, -1, 29, 15), dtype=np.int32)
    offsets = np.array((0.11, 0.17, 0.09, 0.21, 0.14, 0.19, 0.12, 0.16, 0.15), dtype=np.float32)
    query_points = np.empty((len(target_faces), 3), dtype=np.float32)
    for query_index, face in enumerate(target_faces):
        triangle = points[face * 3 : face * 3 + 3]
        query_points[query_index] = 0.5 * triangle[0] + 0.2 * triangle[1] + 0.3 * triangle[2]
        query_points[query_index, 2] += offsets[query_index]

    max_distances = np.full(len(target_faces), 100.0, dtype=np.float32)
    max_distances[-1] = offsets[-1] * 0.5
    expected_results = np.ones(len(target_faces), dtype=np.int32)
    expected_results[-1] = 0
    return query_points, max_distances, seeds, target_faces, offsets, expected_results


def find_mesh_nodes(mesh, query_points, max_distances, seeds, device):
    query_points_device = wp.array(query_points, dtype=wp.vec3, device=device)
    max_distances_device = wp.array(max_distances, dtype=float, device=device)
    seeds_device = wp.array(seeds, dtype=int, device=device)
    nodes = wp.empty(len(query_points), dtype=int, device=device)
    wp.launch(
        find_mesh_exclusive_nodes,
        dim=len(query_points),
        inputs=[mesh.id, query_points_device, max_distances_device, seeds_device],
        outputs=[nodes],
        device=device,
    )
    return nodes.numpy()


def run_mesh_query(kernel, mesh, query_points, max_distances, seeds, device, cached_nodes=None):
    query_points_device = wp.array(query_points, dtype=wp.vec3, device=device)
    max_distances_device = wp.array(max_distances, dtype=float, device=device)
    seeds_device = wp.array(seeds, dtype=int, device=device)
    count = len(query_points)
    results = wp.empty(count, dtype=int, device=device)
    faces = wp.empty(count, dtype=int, device=device)
    bary_u = wp.empty(count, dtype=float, device=device)
    bary_v = wp.empty(count, dtype=float, device=device)
    distances = wp.empty(count, dtype=float, device=device)

    inputs = [mesh.id, query_points_device, max_distances_device]
    if kernel is query_mesh_points_exclusive_cached:
        cached_nodes_device = wp.array(cached_nodes, dtype=int, device=device)
        inputs.extend((seeds_device, cached_nodes_device))
    elif kernel is not query_mesh_points:
        inputs.append(seeds_device)
    wp.launch(
        kernel,
        dim=count,
        inputs=inputs,
        outputs=[results, faces, bary_u, bary_v, distances],
        device=device,
    )
    return {
        "results": results.numpy(),
        "faces": faces.numpy(),
        "bary_u": bary_u.numpy(),
        "bary_v": bary_v.numpy(),
        "distances": distances.numpy(),
    }


def check_mesh_queries(mesh, points, device, cached_nodes=None, check_cubql_sentinel=False):
    query_points, max_distances, seeds, target_faces, offsets, expected_results = make_mesh_queries(points)
    baseline = run_mesh_query(query_mesh_points, mesh, query_points, max_distances, seeds, device)
    seeded = run_mesh_query(query_mesh_points_seeded, mesh, query_points, max_distances, seeds, device)
    exclusive = run_mesh_query(query_mesh_points_exclusive, mesh, query_points, max_distances, seeds, device)
    if cached_nodes is None:
        cached_nodes = find_mesh_nodes(mesh, query_points, max_distances, seeds, device)
    cached = run_mesh_query(
        query_mesh_points_exclusive_cached,
        mesh,
        query_points,
        max_distances,
        seeds,
        device,
        cached_nodes=cached_nodes,
    )

    invalid_cached_nodes = cached_nodes.copy()
    invalid_cached_nodes[0] = -1
    invalid_cached_nodes[1] = np.iinfo(np.int32).max
    invalid_cached_nodes[2] = 1
    invalid_cached = run_mesh_query(
        query_mesh_points_exclusive_cached,
        mesh,
        query_points,
        max_distances,
        seeds,
        device,
        cached_nodes=invalid_cached_nodes,
    )

    np.testing.assert_array_equal(baseline["results"], expected_results)
    hit_mask = expected_results == 1
    np.testing.assert_array_equal(baseline["faces"][hit_mask], target_faces[hit_mask])
    np.testing.assert_allclose(baseline["distances"][hit_mask], offsets[hit_mask], rtol=2.0e-5, atol=2.0e-6)

    for result in (seeded, exclusive, cached, invalid_cached):
        np.testing.assert_array_equal(result["results"], baseline["results"])
        np.testing.assert_array_equal(result["faces"][hit_mask], baseline["faces"][hit_mask])
        np.testing.assert_allclose(result["bary_u"][hit_mask], baseline["bary_u"][hit_mask], atol=2.0e-6)
        np.testing.assert_allclose(result["bary_v"][hit_mask], baseline["bary_v"][hit_mask], atol=2.0e-6)
        np.testing.assert_allclose(result["distances"][hit_mask], baseline["distances"][hit_mask], atol=2.0e-6)

    if check_cubql_sentinel:
        sentinel_cached = run_mesh_query(
            query_mesh_points_exclusive_cached,
            mesh,
            query_points,
            max_distances,
            seeds,
            device,
            cached_nodes=np.ones(len(query_points), dtype=np.int32),
        )
        np.testing.assert_array_equal(sentinel_cached["results"], baseline["results"])
        np.testing.assert_array_equal(sentinel_cached["faces"][hit_mask], baseline["faces"][hit_mask])
        np.testing.assert_allclose(sentinel_cached["distances"][hit_mask], baseline["distances"][hit_mask], atol=2.0e-6)

    return cached_nodes


def test_mesh_exclusive_closest_point(test, device):
    indices_np = np.arange(24 * 3, dtype=np.int32)
    for constructor in get_constructors(device):
        for leaf_size in (1, 8):
            with test.subTest(constructor=constructor, leaf_size=leaf_size):
                points_np = make_mesh_points(stage=0)
                points = wp.array(points_np, dtype=wp.vec3, device=device)
                indices = wp.array(indices_np, dtype=int, device=device)
                mesh = wp.Mesh(
                    points,
                    indices,
                    bvh_constructor=constructor,
                    bvh_leaf_size=leaf_size,
                    enable_exclusive=True,
                )
                cached_nodes = check_mesh_queries(
                    mesh,
                    points_np,
                    device,
                    check_cubql_sentinel=constructor == "cubql",
                )

                points_np = make_mesh_points(stage=1)
                points.assign(points_np)
                mesh.refit()
                check_mesh_queries(
                    mesh,
                    points_np,
                    device,
                    cached_nodes=cached_nodes,
                    check_cubql_sentinel=constructor == "cubql",
                )

    # Both seeded query variants fall back safely when the mesh has no
    # primitive-to-leaf or Exclusive-node metadata.
    points_np = make_mesh_points()
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=int, device=device)
    mesh = wp.Mesh(points, indices, bvh_constructor="sah", bvh_leaf_size=8)
    check_mesh_queries(mesh, points_np, device)

    groups = wp.zeros(len(indices_np) // 3, dtype=int, device=device)
    with test.assertRaisesRegex(RuntimeError, "not supported with grouped meshes"):
        wp.Mesh(points, indices, groups=groups, enable_exclusive=True)


def assert_mesh_query_variants_match(test, mesh, query_points, max_distances, seeds, device, expected_face):
    baseline = run_mesh_query(query_mesh_points, mesh, query_points, max_distances, seeds, device)
    seeded = run_mesh_query(query_mesh_points_seeded, mesh, query_points, max_distances, seeds, device)
    exclusive = run_mesh_query(query_mesh_points_exclusive, mesh, query_points, max_distances, seeds, device)
    cached_nodes = find_mesh_nodes(mesh, query_points, max_distances, seeds, device)
    cached = run_mesh_query(
        query_mesh_points_exclusive_cached,
        mesh,
        query_points,
        max_distances,
        seeds,
        device,
        cached_nodes=cached_nodes,
    )

    test.assertEqual(int(baseline["results"][0]), 1)
    test.assertEqual(int(baseline["faces"][0]), expected_face)
    for result in (seeded, exclusive, cached):
        np.testing.assert_array_equal(result["results"], baseline["results"])
        np.testing.assert_array_equal(result["faces"], baseline["faces"])
        np.testing.assert_allclose(result["bary_u"], baseline["bary_u"], atol=2.0e-6)
        np.testing.assert_allclose(result["bary_v"], baseline["bary_v"], atol=2.0e-6)
        np.testing.assert_allclose(result["distances"], baseline["distances"], atol=2.0e-6)

    return cached_nodes


def test_mesh_exclusive_edge_cases(test, device):
    # The stock query's scale-invariant sliver test must be preserved. Squaring
    # its terms overflows for this valid large triangle and would incorrectly
    # discard face 0 before the seeded traversal skips its leaf.
    points_np = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0e10, 0.0, 0.0),
            (1.0e10, 4.0e4, 0.0),
            (0.0, 1.0e8, 0.0),
            (32.0, 1.0e8, 0.0),
            (0.0, 1.0e8 + 32.0, 0.0),
        ),
        dtype=np.float32,
    )
    indices_np = np.arange(6, dtype=np.int32)
    mesh = wp.Mesh(
        wp.array(points_np, dtype=wp.vec3, device=device),
        wp.array(indices_np, dtype=int, device=device),
        bvh_constructor="sah",
        bvh_leaf_size=1,
        enable_exclusive=True,
    )
    assert_mesh_query_variants_match(
        test,
        mesh,
        np.array(((0.0, 0.0, 1.0),), dtype=np.float32),
        np.array((10.0,), dtype=np.float32),
        np.array((0,), dtype=np.int32),
        device,
        expected_face=0,
    )

    if not device.is_cpu or not wp.is_cubql_available():
        return

    # A power-of-two centroid distribution makes cuBQL's CPU spatial-median
    # builder peel off the high faces until native conversion reaches its stack
    # limit. Faces 0..15 are then packed into one effective leaf even though
    # bvh_leaf_size=1. The warm seed is face 15, while face 0 is the true closest
    # face; skipping the leaf after testing only the nominal seed would fail.
    chain_points = np.empty((70 * 3, 3), dtype=np.float32)
    for face in range(70):
        x = np.float32(2.0**face)
        extent = np.float32(max(float(x) * (2.0**-10), 0.25))
        chain_points[face * 3 + 0] = (x, 0.0, 0.0)
        chain_points[face * 3 + 1] = (x + extent, 0.0, 0.0)
        chain_points[face * 3 + 2] = (x, extent, 0.0)

    chain_mesh = wp.Mesh(
        wp.array(chain_points, dtype=wp.vec3, device=device),
        wp.array(np.arange(70 * 3, dtype=np.int32), dtype=int, device=device),
        bvh_constructor="cubql",
        bvh_leaf_size=1,
        enable_exclusive=True,
    )
    query_points = np.array(((1.0, 0.0, 0.1),), dtype=np.float32)
    max_distances = np.array((1.0e7,), dtype=np.float32)
    cached_nodes = assert_mesh_query_variants_match(
        test,
        chain_mesh,
        query_points,
        max_distances,
        np.array((15,), dtype=np.int32),
        device,
        expected_face=0,
    )
    face_zero_node = find_mesh_nodes(
        chain_mesh,
        query_points,
        max_distances,
        np.array((0,), dtype=np.int32),
        device,
    )
    np.testing.assert_array_equal(cached_nodes, face_zero_node)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestBvhExclusive(unittest.TestCase):
    pass


add_function_test(TestBvhExclusive, "test_bvh_exclusive_aabb", test_bvh_exclusive_aabb, devices=devices)
add_function_test(
    TestBvhExclusive,
    "test_bvh_exclusive_cached_stack_boundary",
    test_bvh_exclusive_cached_stack_boundary,
    devices=cuda_devices,
)
add_function_test(
    TestBvhExclusive,
    "test_mesh_exclusive_closest_point",
    test_mesh_exclusive_closest_point,
    devices=devices,
)
add_function_test(
    TestBvhExclusive,
    "test_mesh_exclusive_edge_cases",
    test_mesh_exclusive_edge_cases,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
