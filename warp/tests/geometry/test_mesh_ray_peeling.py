# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.func
def store_ray_result(
    tid: int,
    hit: bool,
    face: int,
    t: float,
    u: float,
    v: float,
    sign: float,
    normal: wp.vec3,
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    results[tid] = int(hit)
    faces[tid] = -1
    ts[tid] = -1.0
    barycentrics[tid] = wp.vec2(-1.0, -1.0)
    signs[tid] = 0.0
    normals[tid] = wp.vec3(0.0)
    if hit:
        faces[tid] = face
        ts[tid] = t
        barycentrics[tid] = wp.vec2(u, v)
        signs[tid] = sign
        normals[tid] = normal


@wp.kernel
def find_ray_exclusive_nodes(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    tid = wp.tid()
    nodes[tid] = wp.mesh_query_ray_exclusive_node(mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid])


@wp.kernel
def find_ray_exclusive_endpoint_nodes(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    tid = wp.tid()
    nodes[tid] = wp.mesh_query_ray_exclusive_endpoint_node(
        mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid]
    )


@wp.kernel
def query_rays_root(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh_id, starts[tid], directions[tid], max_ts[tid])
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_seeded(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray_seeded(mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid])
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_exclusive(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray_exclusive(mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid])
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_exclusive_cached(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray_exclusive_cached(
        mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid], nodes[tid]
    )
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_exclusive_cached_bottom_up(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray_exclusive_cached_bottom_up(
        mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid], nodes[tid]
    )
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_exclusive_cached_peeling(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    results: wp.array(dtype=int),
    faces: wp.array(dtype=int),
    ts: wp.array(dtype=float),
    barycentrics: wp.array(dtype=wp.vec2),
    signs: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    query = wp.mesh_query_ray_exclusive_cached_peeling(
        mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid], nodes[tid]
    )
    store_ray_result(
        tid,
        query.result,
        query.face,
        query.t,
        query.u,
        query.v,
        query.sign,
        query.normal,
        results,
        faces,
        ts,
        barycentrics,
        signs,
        normals,
    )


@wp.kernel
def query_rays_exclusive_cached_peeling_status(
    mesh_id: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_ts: wp.array(dtype=float),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    statuses: wp.array(dtype=int),
):
    tid = wp.tid()
    statuses[tid] = wp.mesh_query_ray_exclusive_cached_peeling_status(
        mesh_id, starts[tid], directions[tid], max_ts[tid], seeds[tid], nodes[tid]
    )


@wp.kernel
def query_ray_exclusive_node_depths(
    mesh_id: wp.uint64,
    nodes: wp.array(dtype=int),
    depths: wp.array(dtype=int),
):
    tid = wp.tid()
    depths[tid] = wp.mesh_query_ray_exclusive_node_depth(mesh_id, nodes[tid])


def run_query(kernel, mesh, starts, directions, max_ts, seeds, nodes, device):
    count = len(starts)
    starts_wp = wp.array(starts, dtype=wp.vec3, device=device)
    directions_wp = wp.array(directions, dtype=wp.vec3, device=device)
    max_ts_wp = wp.array(max_ts, dtype=float, device=device)
    seeds_wp = wp.array(seeds, dtype=int, device=device)
    nodes_wp = wp.array(nodes, dtype=int, device=device)
    outputs = (
        wp.empty(count, dtype=int, device=device),
        wp.empty(count, dtype=int, device=device),
        wp.empty(count, dtype=float, device=device),
        wp.empty(count, dtype=wp.vec2, device=device),
        wp.empty(count, dtype=float, device=device),
        wp.empty(count, dtype=wp.vec3, device=device),
    )
    inputs = [mesh.id, starts_wp, directions_wp, max_ts_wp]
    if kernel is not query_rays_root:
        inputs.append(seeds_wp)
    if kernel in (
        query_rays_exclusive_cached,
        query_rays_exclusive_cached_bottom_up,
        query_rays_exclusive_cached_peeling,
    ):
        inputs.append(nodes_wp)
    wp.launch(kernel, dim=count, inputs=inputs, outputs=list(outputs), device=device)
    return tuple(output.numpy() for output in outputs)


def find_nodes(mesh, starts, directions, max_ts, seeds, device):
    count = len(starts)
    nodes = wp.empty(count, dtype=int, device=device)
    wp.launch(
        find_ray_exclusive_nodes,
        dim=count,
        inputs=[
            mesh.id,
            wp.array(starts, dtype=wp.vec3, device=device),
            wp.array(directions, dtype=wp.vec3, device=device),
            wp.array(max_ts, dtype=float, device=device),
            wp.array(seeds, dtype=int, device=device),
        ],
        outputs=[nodes],
        device=device,
    )
    return nodes.numpy()


def find_endpoint_nodes(mesh, starts, directions, max_ts, seeds, device):
    count = len(starts)
    nodes = wp.empty(count, dtype=int, device=device)
    wp.launch(
        find_ray_exclusive_endpoint_nodes,
        dim=count,
        inputs=[
            mesh.id,
            wp.array(starts, dtype=wp.vec3, device=device),
            wp.array(directions, dtype=wp.vec3, device=device),
            wp.array(max_ts, dtype=float, device=device),
            wp.array(seeds, dtype=int, device=device),
        ],
        outputs=[nodes],
        device=device,
    )
    return nodes.numpy()


def get_peeling_statuses(mesh, starts, directions, max_ts, seeds, nodes, device):
    count = len(starts)
    statuses = wp.empty(count, dtype=int, device=device)
    wp.launch(
        query_rays_exclusive_cached_peeling_status,
        dim=count,
        inputs=[
            mesh.id,
            wp.array(starts, dtype=wp.vec3, device=device),
            wp.array(directions, dtype=wp.vec3, device=device),
            wp.array(max_ts, dtype=float, device=device),
            wp.array(seeds, dtype=int, device=device),
            wp.array(nodes, dtype=int, device=device),
        ],
        outputs=[statuses],
        device=device,
    )
    return statuses.numpy()


def get_node_depths(mesh, nodes, device):
    depths = wp.empty(len(nodes), dtype=int, device=device)
    wp.launch(
        query_ray_exclusive_node_depths,
        dim=len(nodes),
        inputs=[mesh.id, wp.array(nodes, dtype=int, device=device)],
        outputs=[depths],
        device=device,
    )
    return depths.numpy()


def assert_query_equal(expected, actual):
    for expected_value, actual_value in zip(expected[:-1], actual[:-1], strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)
    np.testing.assert_allclose(actual[-1], expected[-1], rtol=2.0e-6, atol=2.0e-6)


def make_scattered_queries():
    rng = np.random.default_rng(20260818)
    triangle_count = 192
    centers = rng.uniform(-10.0, 10.0, size=(triangle_count, 3)).astype(np.float32)
    sizes = rng.uniform(0.025, 0.05, size=triangle_count).astype(np.float32)
    points = np.empty((triangle_count, 3, 3), dtype=np.float32)
    points[:, 0] = centers + sizes[:, None] * np.array((-1.0, -0.5, -0.2), dtype=np.float32)
    points[:, 1] = centers + sizes[:, None] * np.array((1.0, -0.5, 0.1), dtype=np.float32)
    points[:, 2] = centers + sizes[:, None] * np.array((0.0, 1.0, 0.2), dtype=np.float32)

    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    targets = 0.30 * points[:, 0] + 0.35 * points[:, 1] + 0.35 * points[:, 2]
    distances = 0.45 * sizes
    hit_starts = targets + normals * distances[:, None]
    hit_directions = -normals
    hit_max_ts = 1.05 * distances

    # Include coherent seeds, unrelated valid seeds, invalid seeds, and misses.
    starts = np.concatenate((hit_starts, hit_starts, hit_starts, centers + np.float32(100.0)))
    directions = np.concatenate(
        (
            hit_directions,
            hit_directions,
            hit_directions,
            np.tile(np.array((0.0, 0.0, 1.0), dtype=np.float32), (triangle_count, 1)),
        )
    )
    max_ts = np.concatenate((hit_max_ts, hit_max_ts, hit_max_ts, np.full(triangle_count, 1.0, dtype=np.float32)))
    face_ids = np.arange(triangle_count, dtype=np.int32)
    seeds = np.concatenate((face_ids, np.roll(face_ids, 17), np.full(triangle_count, -1, dtype=np.int32), face_ids))
    return points.reshape(-1, 3), starts, directions, max_ts, seeds


def assert_all_ray_variants(test, points, starts, directions, max_ts, seeds, leaf_size, device, require_peel=False):
    indices = np.arange(len(points), dtype=np.int32)
    mesh = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=int, device=device),
        bvh_constructor="sah",
        bvh_leaf_size=leaf_size,
        enable_exclusive=True,
    )
    nodes = find_nodes(mesh, starts, directions, max_ts, seeds, device)
    endpoint_nodes = find_endpoint_nodes(mesh, starts, directions, max_ts, seeds, device)
    root = run_query(query_rays_root, mesh, starts, directions, max_ts, seeds, nodes, device)
    seeded = run_query(query_rays_seeded, mesh, starts, directions, max_ts, seeds, nodes, device)
    exclusive = run_query(query_rays_exclusive, mesh, starts, directions, max_ts, seeds, nodes, device)
    cached = run_query(query_rays_exclusive_cached, mesh, starts, directions, max_ts, seeds, nodes, device)
    bottom_up = run_query(
        query_rays_exclusive_cached_bottom_up,
        mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    peeling = run_query(
        query_rays_exclusive_cached_peeling,
        mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    peeling_statuses = get_peeling_statuses(mesh, starts, directions, max_ts, seeds, endpoint_nodes, device)
    assert_query_equal(root, seeded)
    assert_query_equal(root, exclusive)
    assert_query_equal(root, cached)
    assert_query_equal(root, bottom_up)
    assert_query_equal(root, peeling)
    test.assertTrue(np.all((peeling_statuses & ~31) == 0))
    if require_peel:
        test.assertGreater(np.count_nonzero(peeling_statuses & (1 | 2 | 4)), 0)

    # Cached nodes are hints, not trusted roots. Exercise invalid and unrelated
    # current node IDs without changing any expected result.
    stale_nodes = np.roll(nodes, 31)
    stale_nodes[::3] = -1
    stale_nodes[1::3] = np.iinfo(np.int32).max
    stale_cached = run_query(query_rays_exclusive_cached, mesh, starts, directions, max_ts, seeds, stale_nodes, device)
    assert_query_equal(root, stale_cached)
    stale_bottom_up = run_query(
        query_rays_exclusive_cached_bottom_up, mesh, starts, directions, max_ts, seeds, stale_nodes, device
    )
    stale_peeling = run_query(
        query_rays_exclusive_cached_peeling, mesh, starts, directions, max_ts, seeds, stale_nodes, device
    )
    assert_query_equal(root, stale_bottom_up)
    assert_query_equal(root, stale_peeling)

    # Refit changes both inclusive and exclusive clipping planes. Previously
    # cached node IDs must be revalidated against the rebuilt E boxes.
    triangle_ids = np.arange(len(points) // 3, dtype=np.float32)
    offsets = np.stack(
        (
            2.0e-4 * np.sin(triangle_ids),
            2.0e-4 * np.cos(triangle_ids),
            1.0e-4 * np.sin(0.5 * triangle_ids),
        ),
        axis=1,
    ).astype(np.float32)
    refit_points = points.reshape(-1, 3, 3) + offsets[:, None, :]
    mesh.points.assign(refit_points.reshape(-1, 3))
    mesh.refit()
    refit_root = run_query(query_rays_root, mesh, starts, directions, max_ts, seeds, nodes, device)
    refit_cached = run_query(query_rays_exclusive_cached, mesh, starts, directions, max_ts, seeds, nodes, device)
    assert_query_equal(refit_root, refit_cached)
    refit_bottom_up = run_query(
        query_rays_exclusive_cached_bottom_up,
        mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    refit_peeling = run_query(
        query_rays_exclusive_cached_peeling,
        mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    assert_query_equal(refit_root, refit_bottom_up)
    assert_query_equal(refit_root, refit_peeling)

    # A newly constructed topology models a rebuild where an old node index
    # can be valid but describe a different subtree.
    rebuilt_points = refit_points[::-1].copy().reshape(-1, 3)
    rebuilt_mesh = wp.Mesh(
        wp.array(rebuilt_points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=int, device=device),
        bvh_constructor="sah",
        bvh_leaf_size=leaf_size,
        enable_exclusive=True,
    )
    rebuilt_root = run_query(query_rays_root, rebuilt_mesh, starts, directions, max_ts, seeds, nodes, device)
    rebuilt_cached = run_query(
        query_rays_exclusive_cached, rebuilt_mesh, starts, directions, max_ts, seeds, nodes, device
    )
    assert_query_equal(rebuilt_root, rebuilt_cached)
    rebuilt_bottom_up = run_query(
        query_rays_exclusive_cached_bottom_up,
        rebuilt_mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    rebuilt_peeling = run_query(
        query_rays_exclusive_cached_peeling,
        rebuilt_mesh,
        starts,
        directions,
        max_ts,
        seeds,
        endpoint_nodes,
        device,
    )
    assert_query_equal(rebuilt_root, rebuilt_bottom_up)
    assert_query_equal(rebuilt_root, rebuilt_peeling)

    test.assertGreater(np.count_nonzero(root[0]), 0)
    test.assertGreater(np.count_nonzero(root[0] == 0), 0)
    return root


def test_mesh_ray_exclusive_random(test, device):
    points, starts, directions, max_ts, seeds = make_scattered_queries()
    for leaf_size in (1, 8):
        assert_all_ray_variants(test, points, starts, directions, max_ts, seeds, leaf_size, device, require_peel=True)


def test_mesh_ray_exclusive_boundaries(test, device):
    # Two coplanar triangles share x=0. Rays on that Exclusive BVH clipping
    # plane exercise strict-boundary fallback and exact-t face tie handling.
    points = np.array(
        (
            (-1.0, -1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, -1.0, 0.0),
            (3.0, -1.0, 0.0),
            (2.5, 1.0, 0.0),
            (1.0e8 - 32.0, -1.0e8 - 32.0, 0.0),
            (1.0e8 + 32.0, -1.0e8 - 32.0, 0.0),
            (1.0e8, -1.0e8 + 32.0, 0.0),
        ),
        dtype=np.float32,
    )
    tiny = np.nextafter(np.float32(0.0), np.float32(1.0))
    starts = np.array(
        (
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 1.0),
            (2.5, 0.0, 1.0),
            (2.5, 0.0, 0.0),
            (8.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0e8, -1.0e8, 64.0),
        ),
        dtype=np.float32,
    )
    directions = np.array(
        (
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (tiny, tiny, -1.0),
            (-tiny, tiny, -1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=np.float32,
    )
    max_ts = np.array((2.0, 1.0, 1.0, 4.0, 2.0, 2.0, 2.0, 128.0), dtype=np.float32)
    seeds = np.array((1, 0, 2, 2, 2, 1, 0, 3), dtype=np.int32)
    for leaf_size in (1, 8):
        root = assert_all_ray_variants(test, points, starts, directions, max_ts, seeds, leaf_size, device)
        # Stock traversal's reciprocal overflows for the two nonzero minimum
        # subnormal components. An internal root therefore misses, while a
        # packed root leaf tests the triangles directly. Experimental paths
        # must preserve either stock result for the active topology.
        tiny_results = (0, 0, 1) if leaf_size == 1 else (1, 1, 1)
        np.testing.assert_array_equal(root[0][-3:], np.array(tiny_results, dtype=np.int32))
        test.assertEqual(root[1][-1], 3)


def test_mesh_ray_peeling_directed_status(test, device):
    # Each packed leaf's E-box is a half-space in x, with infinite bounds on
    # its other five faces. The low leaf's upper plane and the high leaf's
    # lower plane retain the sibling triangles exactly on the peeled boundary.
    points = np.array(
        (
            (-10.0, 9.0, -1.0),
            (-10.0, 11.0, -1.0),
            (-10.0, 10.0, 1.0),
            (10.0, -1.0, -1.0),
            (10.0, 1.0, -1.0),
            (10.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    indices = np.arange(len(points), dtype=np.int32)
    mesh = wp.Mesh(
        wp.array(points, dtype=wp.vec3, device=device),
        wp.array(indices, dtype=int, device=device),
        bvh_constructor="sah",
        bvh_leaf_size=1,
        enable_exclusive=True,
    )

    prep_starts = np.array(((-20.0, 10.0, 0.0), (-20.0, 0.0, 0.0)), dtype=np.float32)
    prep_directions = np.array(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32)
    prep_max_ts = np.array((20.0, 40.0), dtype=np.float32)
    prep_seeds = np.array((0, 1), dtype=np.int32)
    endpoint_nodes = find_endpoint_nodes(mesh, prep_starts, prep_directions, prep_max_ts, prep_seeds, device)
    np.testing.assert_array_equal(get_node_depths(mesh, endpoint_nodes, device), np.array((1, 1), dtype=np.int32))

    starts = np.array(((-20.0, 0.0, 0.0), (-20.0, 0.0, 0.0), (-20.0, 10.0, 0.0)), dtype=np.float32)
    directions = np.tile(np.array((1.0, 0.0, 0.0), dtype=np.float32), (3, 1))
    max_ts = np.array((40.0, 40.0, 20.0), dtype=np.float32)
    seeds = np.array((0, 1, 0), dtype=np.int32)
    nodes = np.array((endpoint_nodes[0], endpoint_nodes[1], endpoint_nodes[0]), dtype=np.int32)

    root = run_query(query_rays_root, mesh, starts, directions, max_ts, seeds, nodes, device)
    bottom_up = run_query(query_rays_exclusive_cached_bottom_up, mesh, starts, directions, max_ts, seeds, nodes, device)
    peeling = run_query(query_rays_exclusive_cached_peeling, mesh, starts, directions, max_ts, seeds, nodes, device)
    assert_query_equal(root, bottom_up)
    assert_query_equal(root, peeling)
    np.testing.assert_array_equal(root[0], np.ones(3, dtype=np.int32))
    np.testing.assert_array_equal(root[1], np.array((1, 1, 0), dtype=np.int32))
    np.testing.assert_array_equal(
        get_peeling_statuses(mesh, starts, directions, max_ts, seeds, nodes, device),
        np.array((2, 4, 1), dtype=np.int32),
    )


def test_mesh_ray_peeling_constructors(test, device):
    constructors = []
    if device.is_cuda:
        constructors.append("lbvh")
    if wp.is_cubql_available():
        constructors.append("cubql")

    points, starts, directions, max_ts, seeds = make_scattered_queries()
    indices = np.arange(len(points), dtype=np.int32)
    for constructor in constructors:
        mesh = wp.Mesh(
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(indices, dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=1,
            enable_exclusive=True,
        )
        nodes = find_endpoint_nodes(mesh, starts, directions, max_ts, seeds, device)
        root = run_query(query_rays_root, mesh, starts, directions, max_ts, seeds, nodes, device)
        bottom_up = run_query(
            query_rays_exclusive_cached_bottom_up, mesh, starts, directions, max_ts, seeds, nodes, device
        )
        peeling = run_query(query_rays_exclusive_cached_peeling, mesh, starts, directions, max_ts, seeds, nodes, device)
        assert_query_equal(root, bottom_up)
        assert_query_equal(root, peeling)
        test.assertGreater(np.count_nonzero(root[0]), 0)
        test.assertGreater(np.count_nonzero(root[0] == 0), 0)


devices = get_test_devices()


class TestMeshRayPeeling(unittest.TestCase):
    pass


add_function_test(TestMeshRayPeeling, "test_mesh_ray_exclusive_random", test_mesh_ray_exclusive_random, devices=devices)
add_function_test(
    TestMeshRayPeeling, "test_mesh_ray_exclusive_boundaries", test_mesh_ray_exclusive_boundaries, devices=devices
)
add_function_test(
    TestMeshRayPeeling, "test_mesh_ray_peeling_directed_status", test_mesh_ray_peeling_directed_status, devices=devices
)
add_function_test(
    TestMeshRayPeeling, "test_mesh_ray_peeling_constructors", test_mesh_ray_peeling_constructors, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
