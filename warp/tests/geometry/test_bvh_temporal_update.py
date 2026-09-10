# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.geometry.test_bvh_exclusive import (
    expected_aabb_hits,
    find_aabb_nodes,
    get_constructors,
    make_aabb_queries,
    make_bounds,
)
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query_update(
    bvh: wp.uint64,
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    refine: bool,
    num_bounds: int,
    next_nodes: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    i = wp.tid()
    cached_node = nodes[i]
    query = wp.bvh_query_aabb_exclusive_update(bvh, lowers[i], uppers[i], cached_node, refine)
    next_nodes[i] = cached_node
    hit = int(-1)
    while wp.bvh_query_next(query, hit):
        hits[i * num_bounds + hit] += 1


def run_update(bvh, lows, highs, nodes, refine, device):
    next_nodes = wp.empty(len(nodes), dtype=int, device=device)
    hits = wp.zeros(len(nodes) * len(bvh.lowers), dtype=int, device=device)
    wp.launch(
        query_update,
        len(nodes),
        [
            bvh.id,
            wp.array(lows, dtype=wp.vec3, device=device),
            wp.array(highs, dtype=wp.vec3, device=device),
            wp.array(nodes, dtype=int, device=device),
            refine,
            len(bvh.lowers),
            next_nodes,
            hits,
        ],
        device=device,
    )
    return next_nodes.numpy(), hits.numpy().reshape(len(nodes), -1)


def test_temporal_update(test, device):
    for constructor in get_constructors(device):
        for leaf_size in (1, 8):
            for refine in (False, True):
                with test.subTest(constructor=constructor, leaf_size=leaf_size, refine=refine):
                    lows, highs = make_bounds()
                    bvh = wp.Bvh(
                        wp.array(lows, dtype=wp.vec3, device=device),
                        wp.array(highs, dtype=wp.vec3, device=device),
                        constructor=constructor,
                        leaf_size=leaf_size,
                        enable_exclusive=True,
                    )
                    qlo, qhi, seeds = make_aabb_queries(lows, highs)
                    nodes = find_aabb_nodes(bvh, qlo, qhi, seeds, device)
                    nodes[:3] = (-1, np.iinfo(np.int32).max, 1)
                    for stage in (0, 1, 2, 1):
                        lows, highs = make_bounds(stage)
                        bvh.lowers.assign(lows)
                        bvh.uppers.assign(highs)
                        if stage == 2:
                            bvh.rebuild()
                        else:
                            bvh.refit()
                        qlo, qhi, _ = make_aabb_queries(lows, highs)
                        nodes, hits = run_update(bvh, qlo, qhi, nodes, refine, device)
                        np.testing.assert_array_equal(hits, expected_aabb_hits(lows, highs, qlo, qhi))
                        test.assertTrue(np.all(nodes >= 0))

    # Ordinary BVHs require neither metadata nor a valid initial cache.
    lows, highs = make_bounds()
    bvh = wp.Bvh(wp.array(lows, dtype=wp.vec3, device=device), wp.array(highs, dtype=wp.vec3, device=device))
    qlo, qhi, _ = make_aabb_queries(lows, highs)
    for refine in (False, True):
        _, hits = run_update(bvh, qlo, qhi, np.full(len(qlo), -1, dtype=np.int32), refine, device)
        np.testing.assert_array_equal(hits, expected_aabb_hits(lows, highs, qlo, qhi))


def test_temporal_refine(test, device):
    lows = np.zeros((64, 3), dtype=np.float32)
    lows[:, 0] = 2 * np.arange(64)
    highs = lows + 0.5
    bvh = wp.Bvh(
        wp.array(lows, dtype=wp.vec3, device=device),
        wp.array(highs, dtype=wp.vec3, device=device),
        constructor="sah",
        leaf_size=1,
        enable_exclusive=True,
    )
    qlo, qhi = lows + 0.1, highs - 0.1
    seeds = np.arange(64, dtype=np.int32)
    leaf_nodes = find_aabb_nodes(bvh, qlo, qhi, seeds, device)
    root_nodes = find_aabb_nodes(bvh, qlo, qhi, np.full(64, -1, dtype=np.int32), device)
    unchanged, _ = run_update(bvh, qlo, qhi, root_nodes, False, device)
    np.testing.assert_array_equal(unchanged, root_nodes)
    refined, hits = run_update(bvh, qlo, qhi, root_nodes, True, device)
    np.testing.assert_array_equal(refined, leaf_nodes)
    np.testing.assert_array_equal(hits, np.eye(64, dtype=np.int32))
    test.assertTrue(np.all(refined != root_nodes))


class TestBvhTemporalUpdate(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestBvhTemporalUpdate, "test_temporal_update", test_temporal_update, devices=devices)
add_function_test(TestBvhTemporalUpdate, "test_temporal_refine", test_temporal_refine, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
