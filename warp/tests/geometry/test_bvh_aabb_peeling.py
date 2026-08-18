# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices

_VERSION = "bvh_aabb_peeling_v1"
print(f"[Warp] {_VERSION}")


@wp.kernel
def find_cached_nodes(
    bvh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    tid = wp.tid()
    nodes[tid] = wp.bvh_query_aabb_exclusive_node(bvh_id, lowers[tid], uppers[tid], seeds[tid])


@wp.kernel
def collect_root_hits(
    bvh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    capacity: int,
    counts: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb(bvh_id, lowers[tid], uppers[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        slot = wp.atomic_add(counts, tid, 1)
        if slot < capacity:
            hits[tid * capacity + slot] = index


@wp.kernel
def collect_bottom_up_hits(
    bvh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    capacity: int,
    counts: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive_cached_bottom_up(bvh_id, lowers[tid], uppers[tid], nodes[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        slot = wp.atomic_add(counts, tid, 1)
        if slot < capacity:
            hits[tid * capacity + slot] = index


@wp.kernel
def collect_peeling_hits(
    bvh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    capacity: int,
    counts: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    query = wp.bvh_query_aabb_exclusive_cached_peeling(bvh_id, lowers[tid], uppers[tid], nodes[tid])
    index = int(-1)
    while wp.bvh_query_next(query, index):
        slot = wp.atomic_add(counts, tid, 1)
        if slot < capacity:
            hits[tid * capacity + slot] = index


@wp.kernel
def query_peeling_diagnostics(
    bvh_id: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    statuses: wp.array(dtype=int),
    peel_counts: wp.array(dtype=int),
    depths: wp.array(dtype=int),
):
    tid = wp.tid()
    statuses[tid] = wp.bvh_query_aabb_exclusive_cached_peeling_status(
        bvh_id,
        lowers[tid],
        uppers[tid],
        nodes[tid],
    )
    peel_counts[tid] = wp.bvh_query_aabb_exclusive_cached_peeling_count(
        bvh_id,
        lowers[tid],
        uppers[tid],
        nodes[tid],
    )
    depths[tid] = wp.bvh_query_aabb_exclusive_node_depth(bvh_id, nodes[tid])


def get_constructors(device):
    constructors = ["sah", "median"]
    if device.is_cuda:
        constructors.append("lbvh")
    if wp.is_cubql_available():
        constructors.append("cubql")
    return constructors


def make_bounds(stage=0, count=64):
    lowers = np.zeros((count, 3), dtype=np.float32)
    uppers = np.ones((count, 3), dtype=np.float32)
    positions = np.arange(count, dtype=np.float32)
    if stage == 1:
        positions += np.float32(0.375)
        lowers[:, 1] = np.float32(-0.25)
        uppers[:, 1] = np.float32(0.75)
    elif stage == 2:
        positions = positions[::-1].copy()
        lowers[:, 2] = np.float32(0.125)
        uppers[:, 2] = np.float32(1.125)
    lowers[:, 0] = positions
    uppers[:, 0] = positions + np.float32(1.0)
    return lowers, uppers


def make_queries(lowers, uppers):
    count = len(lowers)
    seeds = [0, count - 1, count // 4, count // 2, 0, count // 3]
    low_x = float(np.min(lowers[:, 0]))
    high_x = float(np.max(uppers[:, 0]))

    query_lowers = [
        (float(lowers[seeds[0], 0] + 0.25), -0.1, 0.25),
        (low_x - 2.0, -0.1, 0.25),
        tuple(lowers[seeds[2]] + np.float32(0.25)),
        (low_x - 1.0, -0.1, 0.25),
        (low_x - 2.0, -1.0, -1.0),
        (high_x + 1.0, 0.25, 0.25),
    ]
    query_uppers = [
        (high_x + 2.0, 0.6, 0.75),
        (float(uppers[seeds[1], 0] - 0.25), 0.6, 0.75),
        tuple(uppers[seeds[2]] - np.float32(0.25)),
        (high_x + 1.0, 0.6, 0.75),
        (high_x + 2.0, 2.0, 2.0),
        (high_x + 2.0, 0.75, 0.75),
    ]

    # Every item endpoint is also an item boundary. These degenerate queries
    # exercise inclusive hits on the clipping plane retained by a face peel.
    for primitive in (1, 8, 16, 31, 32, 48, count - 1):
        boundary_x = float(lowers[primitive, 0])
        query_lowers.append((boundary_x, 0.5, 0.5))
        query_uppers.append((boundary_x, 0.5, 0.5))
        seeds.append(max(0, primitive - 1))

    rng = np.random.default_rng(20260818)
    for _ in range(12):
        x0, x1 = np.sort(rng.uniform(low_x - 1.0, high_x + 1.0, size=2)).astype(np.float32)
        query_lowers.append((float(x0), -0.05, 0.2))
        query_uppers.append((float(x1), 0.8, 0.9))
        seeds.append(int(rng.integers(0, count)))

    return (
        np.asarray(query_lowers, dtype=np.float32),
        np.asarray(query_uppers, dtype=np.float32),
        np.asarray(seeds, dtype=np.int32),
    )


def make_seed_queries(lowers, uppers, seeds):
    seed_lowers = lowers[seeds] + np.float32(0.25)
    seed_uppers = uppers[seeds] - np.float32(0.25)
    return seed_lowers, seed_uppers


def expected_multisets(lowers, uppers, query_lowers, query_uppers):
    overlap = np.logical_and(
        np.all(query_lowers[:, None, :] <= uppers[None, :, :], axis=2),
        np.all(lowers[None, :, :] <= query_uppers[:, None, :], axis=2),
    )
    return [np.flatnonzero(row).astype(np.int32).tolist() for row in overlap]


def find_nodes(bvh, lowers, uppers, seeds, device):
    count = len(lowers)
    nodes = wp.empty(count, dtype=int, device=device)
    wp.launch(
        find_cached_nodes,
        dim=count,
        inputs=[
            bvh.id,
            wp.array(lowers, dtype=wp.vec3, device=device),
            wp.array(uppers, dtype=wp.vec3, device=device),
            wp.array(seeds, dtype=int, device=device),
        ],
        outputs=[nodes],
        device=device,
    )
    return nodes.numpy()


def collect_hits(kernel, bvh, query_lowers, query_uppers, device, nodes=None):
    query_count = len(query_lowers)
    capacity = len(bvh.lowers)
    counts = wp.zeros(query_count, dtype=int, device=device)
    hits = wp.full(query_count * capacity, value=-1, dtype=int, device=device)
    inputs = [
        bvh.id,
        wp.array(query_lowers, dtype=wp.vec3, device=device),
        wp.array(query_uppers, dtype=wp.vec3, device=device),
    ]
    if kernel is not collect_root_hits:
        inputs.append(wp.array(nodes, dtype=int, device=device))
    inputs.append(capacity)
    wp.launch(kernel, dim=query_count, inputs=inputs, outputs=[counts, hits], device=device)

    counts_np = counts.numpy()
    hits_np = hits.numpy().reshape(query_count, capacity)
    result = []
    for query_index, count in enumerate(counts_np):
        if count > capacity:
            raise AssertionError(f"query {query_index} emitted {count} hits into capacity {capacity}")
        result.append(sorted(hits_np[query_index, :count].tolist()))
    return result


def assert_variants_exact(test, bvh, lowers, uppers, query_lowers, query_uppers, nodes, device):
    expected = expected_multisets(lowers, uppers, query_lowers, query_uppers)
    root = collect_hits(collect_root_hits, bvh, query_lowers, query_uppers, device)
    bottom_up = collect_hits(collect_bottom_up_hits, bvh, query_lowers, query_uppers, device, nodes)
    peeling = collect_hits(collect_peeling_hits, bvh, query_lowers, query_uppers, device, nodes)
    test.assertEqual(root, expected)
    test.assertEqual(bottom_up, expected)
    test.assertEqual(peeling, expected)
    for query_hits in peeling:
        test.assertEqual(len(query_hits), len(set(query_hits)))


def get_diagnostics(bvh, query_lowers, query_uppers, nodes, device):
    query_count = len(query_lowers)
    statuses = wp.empty(query_count, dtype=int, device=device)
    peel_counts = wp.empty(query_count, dtype=int, device=device)
    depths = wp.empty(query_count, dtype=int, device=device)
    wp.launch(
        query_peeling_diagnostics,
        dim=query_count,
        inputs=[
            bvh.id,
            wp.array(query_lowers, dtype=wp.vec3, device=device),
            wp.array(query_uppers, dtype=wp.vec3, device=device),
            wp.array(nodes, dtype=int, device=device),
        ],
        outputs=[statuses, peel_counts, depths],
        device=device,
    )
    return statuses.numpy(), peel_counts.numpy(), depths.numpy()


def test_bvh_aabb_peeling(test, device):
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
                query_lowers, query_uppers, seeds = make_queries(lowers_np, uppers_np)
                seed_lowers, seed_uppers = make_seed_queries(lowers_np, uppers_np, seeds)
                cached_nodes = find_nodes(bvh, seed_lowers, seed_uppers, seeds, device)
                assert_variants_exact(
                    test,
                    bvh,
                    lowers_np,
                    uppers_np,
                    query_lowers,
                    query_uppers,
                    cached_nodes,
                    device,
                )

                statuses, peel_counts, depths = get_diagnostics(
                    bvh,
                    query_lowers,
                    query_uppers,
                    cached_nodes,
                    device,
                )
                test.assertTrue(np.all(depths >= 0))
                if constructor == "sah":
                    test.assertNotEqual(int(statuses[0]) & 2, 0)
                    test.assertNotEqual(int(statuses[1]) & 4, 0)
                    test.assertNotEqual(int(statuses[2]) & 1, 0)
                    test.assertNotEqual(int(statuses[3]) & 8, 0)
                    # The two directed stride queries peel successively at
                    # multiple ancestors, rather than only at cached A.
                    test.assertGreaterEqual(int(peel_counts[0]), 2)
                    test.assertGreaterEqual(int(peel_counts[1]), 2)
                    test.assertGreaterEqual(int(peel_counts[2]), 1)

                invalid_nodes = cached_nodes.copy()
                invalid_nodes[0::2] = -1
                invalid_nodes[1::2] = np.iinfo(np.int32).max
                assert_variants_exact(
                    test,
                    bvh,
                    lowers_np,
                    uppers_np,
                    query_lowers,
                    query_uppers,
                    invalid_nodes,
                    device,
                )

                # Refit keeps topology but moves all clipping planes. Rebuild
                # reverses primitive geometry so old integer caches may now
                # name unrelated reachable nodes; both remain exact.
                for stage, rebuild in ((1, False), (2, True)):
                    lowers_np, uppers_np = make_bounds(stage=stage)
                    lowers.assign(lowers_np)
                    uppers.assign(uppers_np)
                    if rebuild:
                        bvh.rebuild()
                    else:
                        bvh.refit()
                    query_lowers, query_uppers, _ = make_queries(lowers_np, uppers_np)
                    assert_variants_exact(
                        test,
                        bvh,
                        lowers_np,
                        uppers_np,
                        query_lowers,
                        query_uppers,
                        cached_nodes,
                        device,
                    )

    # Missing Exclusive BVH metadata takes the exact root fallback.
    lowers_np, uppers_np = make_bounds(stage=0)
    bvh = wp.Bvh(
        wp.array(lowers_np, dtype=wp.vec3, device=device),
        wp.array(uppers_np, dtype=wp.vec3, device=device),
        constructor="sah",
        leaf_size=8,
    )
    query_lowers, query_uppers, seeds = make_queries(lowers_np, uppers_np)
    assert_variants_exact(
        test,
        bvh,
        lowers_np,
        uppers_np,
        query_lowers,
        query_uppers,
        seeds,
        device,
    )

    # A two-axis crossing is an edge/corner residual and must be declined.
    grid_lowers = np.zeros((64, 3), dtype=np.float32)
    grid_uppers = np.ones((64, 3), dtype=np.float32)
    for primitive in range(64):
        grid_lowers[primitive, 0] = np.float32(2.0 * (primitive % 8))
        grid_lowers[primitive, 1] = np.float32(2.0 * (primitive // 8))
        grid_uppers[primitive, 0] = grid_lowers[primitive, 0] + np.float32(0.8)
        grid_uppers[primitive, 1] = grid_lowers[primitive, 1] + np.float32(0.8)
    grid_bvh = wp.Bvh(
        wp.array(grid_lowers, dtype=wp.vec3, device=device),
        wp.array(grid_uppers, dtype=wp.vec3, device=device),
        constructor="sah",
        leaf_size=1,
        enable_exclusive=True,
    )
    grid_seeds = np.array((27,), dtype=np.int32)
    seed_lowers, seed_uppers = make_seed_queries(grid_lowers, grid_uppers, grid_seeds)
    grid_nodes = find_nodes(grid_bvh, seed_lowers, seed_uppers, grid_seeds, device)
    grid_query_lowers = np.array(((2.0, 2.0, 0.25),), dtype=np.float32)
    grid_query_uppers = np.array(((13.0, 13.0, 0.75),), dtype=np.float32)
    assert_variants_exact(
        test,
        grid_bvh,
        grid_lowers,
        grid_uppers,
        grid_query_lowers,
        grid_query_uppers,
        grid_nodes,
        device,
    )
    statuses, peel_counts, _ = get_diagnostics(
        grid_bvh,
        grid_query_lowers,
        grid_query_uppers,
        grid_nodes,
        device,
    )
    test.assertNotEqual(int(statuses[0]) & 8, 0)

    nan_lowers = grid_query_lowers.copy()
    nan_lowers[0, 0] = np.nan
    statuses, peel_counts, _ = get_diagnostics(
        grid_bvh,
        nan_lowers,
        grid_query_uppers,
        grid_nodes,
        device,
    )
    test.assertNotEqual(int(statuses[0]) & 16, 0)
    test.assertEqual(int(peel_counts[0]), 0)


devices = get_test_devices()


class TestBvhAabbPeeling(unittest.TestCase):
    pass


add_function_test(TestBvhAabbPeeling, "test_bvh_aabb_peeling", test_bvh_aabb_peeling, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
