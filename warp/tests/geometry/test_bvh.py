# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def bvh_query_aabb(bvh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3, bounds_intersected: wp.array[int]):
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_query_ray(bvh_id: wp.uint64, start: wp.vec3, dir: wp.vec3, bounds_intersected: wp.array[int]):
    query = wp.bvh_query_ray(bvh_id, start, dir)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_query_sphere(bvh_id: wp.uint64, center: wp.vec3, radius: float, bounds_intersected: wp.array[int]):
    query = wp.bvh_query_sphere(bvh_id, center, radius)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_capsule_query(bvh_id: wp.uint64, p0: wp.vec3, p1: wp.vec3, radius: float, bounds_intersected: wp.array[int]):
    # capsule = ray (p0 -> p1) inflated by radius, bounded to the segment by max_dist = 1.0
    query = wp.bvh_query_capsule(bvh_id, p0, p1 - p0, radius)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr, 1.0):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_capsule_query_max_dist(
    bvh_id: wp.uint64,
    start: wp.vec3,
    direction: wp.vec3,
    radius: float,
    max_dist: float,
    bounds_intersected: wp.array[int],
):
    query = wp.bvh_query_capsule(bvh_id, start, direction, radius)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr, max_dist):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_capsule_query_decreasing_max_dist(
    bvh_id: wp.uint64,
    start: wp.vec3,
    direction: wp.vec3,
    initial_max_dist: float,
    reduced_max_dist: float,
    first_hit_count: wp.array[int],
    later_hits: wp.array[int],
):
    query = wp.bvh_query_capsule(bvh_id, start, direction, 0.0)
    bounds_nr = int(0)

    if wp.bvh_query_next(query, bounds_nr, initial_max_dist):
        first_hit_count[0] = 1

    while wp.bvh_query_next(query, bounds_nr, reduced_max_dist):
        later_hits[bounds_nr] = 1


@wp.kernel
def bvh_capsule_query_guarded(
    bvh_id: wp.uint64,
    start: wp.vec3,
    direction: wp.vec3,
    radius: float,
    max_dist: float,
    num_bounds: int,
    guarded_hits: wp.array[int],
):
    query = wp.bvh_query_capsule(bvh_id, start, direction, radius)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr, max_dist):
        if bounds_nr < 0 or bounds_nr >= num_bounds:
            guarded_hits[num_bounds + 1] = 0
        else:
            wp.atomic_add(guarded_hits, bounds_nr + 1, 1)


def aabb_overlap(a_lower, a_upper, b_lower, b_upper):
    if (
        a_lower[0] > b_upper[0]
        or a_lower[1] > b_upper[1]
        or a_lower[2] > b_upper[2]
        or a_upper[0] < b_lower[0]
        or a_upper[1] < b_lower[1]
        or a_upper[2] < b_lower[2]
    ):
        return 0
    else:
        return 1


def intersect_ray_aabb(start, rcp_dir, lower, upper):
    l1 = (lower[0] - start[0]) * rcp_dir[0]
    l2 = (upper[0] - start[0]) * rcp_dir[0]
    lmin = min(l1, l2)
    lmax = max(l1, l2)

    l1 = (lower[1] - start[1]) * rcp_dir[1]
    l2 = (upper[1] - start[1]) * rcp_dir[1]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    l1 = (lower[2] - start[2]) * rcp_dir[2]
    l2 = (upper[2] - start[2]) * rcp_dir[2]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    if lmax >= 0.0 and lmax >= lmin:
        return 1
    else:
        return 0


def sphere_aabb_overlap(center, radius, lower, upper):
    # squared distance from center to the AABB <= radius^2
    sq = 0.0
    for i in range(3):
        if center[i] < lower[i]:
            sq += (lower[i] - center[i]) ** 2
        elif center[i] > upper[i]:
            sq += (center[i] - upper[i]) ** 2
    return 1 if sq <= radius * radius else 0


def segment_aabb_overlap(p0, p1, radius, lower, upper):
    # segment [p0, p1] vs the AABB inflated by radius (slab test clamped to [0, 1])
    lo = [lower[i] - radius for i in range(3)]
    hi = [upper[i] + radius for i in range(3)]
    d = [p1[i] - p0[i] for i in range(3)]
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-8:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return 0
        else:
            ood = 1.0 / d[i]
            t1 = (lo[i] - p0[i]) * ood
            t2 = (hi[i] - p0[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return 0
    return 1


@wp.kernel
def count_sphere_hits(bvh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    q = wp.bvh_query_sphere(bvh_id, center, radius)
    idx = int(0)
    while wp.bvh_query_next(q, idx):
        wp.atomic_add(hits, 0, 1)


def test_bvh(test, type, device, leaf_size, constructor=None):
    rng = np.random.default_rng(123)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers, constructor=constructor, leaf_size=leaf_size)

    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=int, device=device)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    query_center = wp.vec3(5.0, 5.0, 5.0)
    query_radius = 3.0

    query_p0 = wp.vec3(0.0, 0.0, 0.0)
    query_p1 = wp.vec3(10.0, 10.0, 10.0)
    capsule_radius = 1.0

    for test_case in range(3):
        if type == "AABB":
            wp.launch(
                bvh_query_aabb,
                dim=1,
                inputs=[bvh.id, query_lower, query_upper, bounds_intersected],
                device=device,
            )
        elif type == "ray":
            wp.launch(bvh_query_ray, dim=1, inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device)
        elif type == "sphere":
            wp.launch(
                bvh_query_sphere, dim=1, inputs=[bvh.id, query_center, query_radius, bounds_intersected], device=device
            )
        else:  # capsule
            wp.launch(
                bvh_capsule_query,
                dim=1,
                inputs=[bvh.id, query_p0, query_p1, capsule_radius, bounds_intersected],
                device=device,
            )

        device_intersected = bounds_intersected.numpy()

        for i in range(num_bounds):
            lower = lowers[i]
            upper = uppers[i]
            if type == "AABB":
                host_intersected = aabb_overlap(lower, upper, query_lower, query_upper)
            elif type == "ray":
                host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
            elif type == "sphere":
                host_intersected = sphere_aabb_overlap(query_center, query_radius, lower, upper)
            else:  # capsule
                host_intersected = segment_aabb_overlap(query_p0, query_p1, capsule_radius, lower, upper)

            test.assertEqual(host_intersected, device_intersected[i])

        if test_case == 0 or test_case == 1:
            lowers = rng.random(size=(num_bounds, 3)) * 5.0
            uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0
            wp.copy(device_lowers, wp.array(lowers, dtype=wp.vec3, device=device))
            wp.copy(device_uppers, wp.array(uppers, dtype=wp.vec3, device=device))
            bounds_intersected.zero_()

            if test_case == 0:
                bvh.refit()
            else:
                bvh.rebuild()


def test_bvh_query_aabb(test, device):
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "AABB", device, leaf_size)


def test_bvh_query_ray(test, device):
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "ray", device, leaf_size)


def test_bvh_query_sphere(test, device):
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "sphere", device, leaf_size)

    # Zero-radius sphere should behave like a point query (only bounds that contain the point).
    lowers = wp.array([(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)], dtype=wp.vec3, device=device)
    uppers = wp.array([(1.0, 1.0, 1.0), (3.0, 3.0, 3.0)], dtype=wp.vec3, device=device)
    bvh = wp.Bvh(lowers, uppers)

    hit_count = wp.zeros(1, dtype=int, device=device)

    wp.launch(count_sphere_hits, dim=1, inputs=[bvh.id, wp.vec3(0.5, 0.5, 0.5), 0.0, hit_count], device=device)
    test.assertEqual(hit_count.numpy()[0], 1)


def test_bvh_query_capsule(test, device):
    """Validate conservative capsule queries and distance boundary semantics."""
    # The broad-phase inflates node AABBs by radius as an axis-aligned box, not a true sphere,
    # so it is conservative: it never misses a primitive within radius of the segment but may
    # return extra candidates near box corners. Tests validate this conservative semantics.
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "capsule", device, leaf_size)

    def query_hits(lowers, uppers, start, direction, radius, max_dist, leaf_size=1):
        device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
        device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
        bvh = wp.Bvh(device_lowers, device_uppers, leaf_size=leaf_size)
        hits = wp.zeros(len(lowers), dtype=int, device=device)
        wp.launch(
            bvh_capsule_query_max_dist,
            dim=1,
            inputs=[bvh.id, wp.vec3(*start), wp.vec3(*direction), radius, max_dist, hits],
            device=device,
        )
        return hits.numpy()

    float32_one = np.float32(1.0)
    after_one = np.nextafter(float32_one, np.float32(np.inf))
    endpoint_hits = query_hits(
        [(float32_one, -0.1, -0.1), (after_one, -0.1, -0.1)],
        [(1.1, 0.1, 0.1), (1.1, 0.1, 0.1)],
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.0,
        1.0,
    )
    np.testing.assert_array_equal(endpoint_hits, [1, 0])

    tangent_y = np.float32(0.1)
    outside_y = np.nextafter(tangent_y, np.float32(np.inf))
    parallel_hits = query_hits(
        [(0.25, tangent_y, -0.1), (0.25, outside_y, -0.1)],
        [(0.5, 0.2, 0.1), (0.5, 0.2, 0.1)],
        (0.0, 0.0, 0.0),
        (1.0, -0.0, 0.0),
        0.1,
        1.0,
    )
    np.testing.assert_array_equal(parallel_hits, [1, 0])

    before_zero = np.nextafter(np.float32(0.0), np.float32(-np.inf))
    start_hits = query_hits(
        [(-0.1, -0.1, -0.1), (-0.1, -0.1, -0.1)],
        [(0.0, 0.1, 0.1), (before_zero, 0.1, 0.1)],
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.0,
        1.0,
    )
    np.testing.assert_array_equal(start_hits, [1, 0])

    after_zero = np.nextafter(np.float32(0.0), np.float32(np.inf))
    max_dist_zero_hits = query_hits(
        [(-0.1, -0.1, -0.1), (after_zero, -0.1, -0.1)],
        [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1)],
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        0.0,
        0.0,
    )
    np.testing.assert_array_equal(max_dist_zero_hits, [1, 0])
    negative_radius_hits = query_hits(
        [(-0.1, -0.1, -0.1), (after_zero, -0.1, -0.1)],
        [(0.1, 0.1, 0.1), (0.1, 0.1, 0.1)],
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        -0.5,
        0.0,
    )
    np.testing.assert_array_equal(negative_radius_hits, max_dist_zero_hits)

    decreasing_lowers = wp.array([(0.25, -0.1, -0.1), (0.75, -0.1, -0.1)], dtype=wp.vec3, device=device)
    decreasing_uppers = wp.array([(0.3, 0.1, 0.1), (0.8, 0.1, 0.1)], dtype=wp.vec3, device=device)
    for leaf_size in (1, 4):
        bvh = wp.Bvh(decreasing_lowers, decreasing_uppers, leaf_size=leaf_size)
        first_hit_count = wp.zeros(1, dtype=int, device=device)
        later_hits = wp.zeros(2, dtype=int, device=device)
        wp.launch(
            bvh_capsule_query_decreasing_max_dist,
            dim=1,
            inputs=[
                bvh.id,
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.0, 0.0),
                1.0,
                0.0,
                first_hit_count,
                later_hits,
            ],
            device=device,
        )
        test.assertEqual(first_hit_count.numpy()[0], 1)
        test.assertEqual(later_hits.numpy().sum(), 0)


def assert_bvh_capsule_query_guarded(
    device,
    lowers,
    uppers,
    start,
    direction,
    max_dist,
    *,
    constructor,
    groups=None,
    leaf_size=1,
):
    """Assert that a capsule query returns every bound once without touching guards."""
    num_bounds = len(lowers)
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    device_groups = None if groups is None else wp.array(groups, dtype=int, device=device)
    bvh = wp.Bvh(
        device_lowers,
        device_uppers,
        constructor=constructor,
        groups=device_groups,
        leaf_size=leaf_size,
    )

    left_canary = np.int32(0x13579BDF)
    right_canary = np.int32(0x2468ACE)
    expected = np.ones(num_bounds + 2, dtype=np.int32)
    expected[0] = left_canary
    expected[-1] = right_canary
    initial = np.zeros(num_bounds + 2, dtype=np.int32)
    initial[0] = left_canary
    initial[-1] = right_canary
    guarded_hits = wp.array(initial, dtype=int, device=device)

    wp.launch(
        bvh_capsule_query_guarded,
        dim=1,
        inputs=[
            bvh.id,
            wp.vec3(*start),
            wp.vec3(*direction),
            0.0,
            max_dist,
            num_bounds,
            guarded_hits,
        ],
        device=device,
    )

    np.testing.assert_array_equal(guarded_hits.numpy(), expected)


def test_bvh_query_capsule_max_depth(test, device):
    """Preserve capsule results at the maximum ungrouped BVH depth."""
    num_bounds = 33
    exponents = 4 * np.arange(num_bounds - 1, -1, -1, dtype=np.float32) - 32
    x = -np.exp2(exponents)
    lowers = np.column_stack((x, np.full(num_bounds, -0.1), np.full(num_bounds, -0.1))).astype(np.float32)
    uppers = np.column_stack((x, np.full(num_bounds, 0.1), np.full(num_bounds, 0.1))).astype(np.float32)

    # Power-of-two spacing makes SAH repeatedly split off the lowest item. The
    # right-first capsule traversal therefore reaches the fixed-stack limit
    # while all deferred siblings remain live.
    assert_bvh_capsule_query_guarded(
        device,
        lowers,
        uppers,
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        float(2**97),
        constructor="sah",
    )


def test_bvh_query_capsule_grouped_max_depth(test, device):
    """Preserve full-root capsule results for maximum-depth grouped BVHs."""
    num_deep_bounds = 33
    exponents = 4 * np.arange(num_deep_bounds - 1, -1, -1, dtype=np.float32) - 32
    x = np.concatenate((np.array([-float(2**100)], dtype=np.float32), -np.exp2(exponents)))
    num_bounds = len(x)
    lowers = np.column_stack((x, np.full(num_bounds, -0.1), np.full(num_bounds, -0.1))).astype(np.float32)
    uppers = np.column_stack((x, np.full(num_bounds, 0.1), np.full(num_bounds, 0.1))).astype(np.float32)
    groups = np.concatenate((np.zeros(1, dtype=np.int32), np.ones(num_deep_bounds, dtype=np.int32)))

    # The right group adds an ancestor to the same SAH ladder used above.
    assert_bvh_capsule_query_guarded(
        device,
        lowers,
        uppers,
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        float(2**101),
        constructor="sah",
        groups=groups,
    )

    if device.is_cuda:
        group_bits = np.empty(33, dtype=np.uint32)
        group_bits[0] = 0
        for bit_count in range(1, len(group_bits)):
            group_bits[bit_count] = ((1 << bit_count) - 1) << (32 - bit_count)

        for packed_leaf in (False, True):
            if packed_leaf:
                lbvh_group_bits = np.concatenate((group_bits[:-1], np.repeat(group_bits[-1:], 4)))
                leaf_size = 4
            else:
                lbvh_group_bits = group_bits
                leaf_size = 1

            lbvh_num_bounds = len(lbvh_group_bits)
            lbvh_lowers = np.tile(np.array([[-0.5, -0.1, -0.1]], dtype=np.float32), (lbvh_num_bounds, 1))
            lbvh_uppers = np.tile(np.array([[0.5, 0.1, 0.1]], dtype=np.float32), (lbvh_num_bounds, 1))
            assert_bvh_capsule_query_guarded(
                device,
                lbvh_lowers,
                lbvh_uppers,
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                2.0,
                constructor="lbvh",
                groups=lbvh_group_bits.view(np.int32),
                leaf_size=leaf_size,
            )


def test_bvh_cubql_constructor(test, device):
    if not wp.is_cubql_available():
        test.skipTest("cuBQL is not available")

    for leaf_size in [1, 2, 4]:
        test_bvh(test, "AABB", device, leaf_size, constructor="cubql")
        test_bvh(test, "ray", device, leaf_size, constructor="cubql")


def test_bvh_ray_query_inside_and_outside_bounds(test, device):
    """Regression test for issue #288: BVH ray queries should detect intersections
    regardless of whether the ray origin is inside or outside the bounding volumes.

    Previously, rays starting outside the bounds would fail to detect intersections.
    """
    # Create a single AABB spanning x=[0.5, 1.0], extending across y and z axes
    lowers = ((0.5, -1.0, -1.0),)
    uppers = ((1.0, 1.0, 1.0),)

    device_lowers = wp.array(lowers, dtype=wp.vec3f, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3f, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    bounds_intersected = wp.zeros(shape=1, dtype=int, device=device)

    # Test both ray origins: outside (x=0.0) and inside (x=0.75) the AABB
    for x in (0.0, 0.75):
        query_start = wp.vec3(x, 0.0, 0.0)
        query_dir = wp.vec3(1.0, 0.0, 0.0)  # Ray pointing in +x direction

        wp.launch(bvh_query_ray, dim=1, inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device)

        device_intersected = bounds_intersected.numpy()
        # Both cases should detect the single intersection
        test.assertEqual(device_intersected.sum(), 1)


def test_bvh_refit_root_leaves(test, device):
    """Refit CUDA BVHs whose root is represented as a leaf.

    Single-node trees store the root leaf without a parent. LBVH can also pack
    the root into a leaf when ``leaf_size`` covers all primitives. The old and
    new AABBs occupy disjoint x-ranges, so the old query should miss after
    refit and the new query should hit the updated bounds.
    """
    old_single_lower = wp.vec3(0.0, 0.0, 0.0)
    old_single_upper = wp.vec3(1.0, 1.0, 1.0)
    new_single_lower = wp.vec3(2.0, 0.0, 0.0)
    new_single_upper = wp.vec3(3.0, 1.0, 1.0)

    cases = [
        (
            f"single_leaf_{constructor}",
            constructor,
            1,
            [old_single_lower],
            [old_single_upper],
            [new_single_lower],
            [new_single_upper],
            old_single_lower,
            old_single_upper,
            new_single_lower,
            new_single_upper,
            1,
        )
        for constructor in ("sah", "median", "lbvh")
    ]
    cases.append(
        (
            "packed_root_lbvh",
            "lbvh",
            2,
            [wp.vec3(0.0, 0.0, 0.0), wp.vec3(2.0, 0.0, 0.0)],
            [wp.vec3(1.0, 1.0, 1.0), wp.vec3(3.0, 1.0, 1.0)],
            [wp.vec3(4.0, 0.0, 0.0), wp.vec3(6.0, 0.0, 0.0)],
            [wp.vec3(5.0, 1.0, 1.0), wp.vec3(7.0, 1.0, 1.0)],
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(3.0, 1.0, 1.0),
            wp.vec3(4.0, 0.0, 0.0),
            wp.vec3(7.0, 1.0, 1.0),
            2,
        )
    )

    for (
        name,
        constructor,
        leaf_size,
        old_lowers,
        old_uppers,
        new_lowers,
        new_uppers,
        old_query_lower,
        old_query_upper,
        new_query_lower,
        new_query_upper,
        expected_new_hits,
    ) in cases:
        with test.subTest(name=name):
            lowers = wp.array(old_lowers, dtype=wp.vec3, device=device)
            uppers = wp.array(old_uppers, dtype=wp.vec3, device=device)
            bvh = wp.Bvh(lowers, uppers, constructor=constructor, leaf_size=leaf_size)

            wp.copy(lowers, wp.array(new_lowers, dtype=wp.vec3, device=device))
            wp.copy(uppers, wp.array(new_uppers, dtype=wp.vec3, device=device))
            bvh.refit()

            bounds_intersected = wp.zeros(shape=len(old_lowers), dtype=int, device=device)
            wp.launch(
                bvh_query_aabb,
                dim=1,
                inputs=[bvh.id, old_query_lower, old_query_upper, bounds_intersected],
                device=device,
            )
            test.assertEqual(bounds_intersected.numpy().sum(), 0, f"Expected miss at old bounds ({name})")

            bounds_intersected.zero_()
            wp.launch(
                bvh_query_aabb,
                dim=1,
                inputs=[bvh.id, new_query_lower, new_query_upper, bounds_intersected],
                device=device,
            )
            test.assertEqual(
                bounds_intersected.numpy().sum(), expected_new_hits, f"Expected hits at refit bounds ({name})"
            )


def get_random_aabbs(n, center, relative_shift, relative_size, rng):
    centers = rng.uniform(-0.5, 0.5, size=n * 3).reshape(n, 3) * relative_shift + center
    diffs = 0.5 * rng.random(n * 3).reshape(n, 3) * relative_size

    lowers = centers - diffs
    uppers = centers + diffs

    return lowers, uppers


@wp.kernel
def compute_num_contact_with_checksums(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    bvh_id: wp.uint64,
    counts: wp.array[int],
    check_sums: wp.array[int],
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    count = int(0)

    check_sum = int(0)
    index = int(0)
    while wp.bvh_query_next(query, index):
        check_sum = check_sum ^ index
        count += 1

    counts[tid] = count
    check_sums[tid] = check_sum


def test_capture_bvh_rebuild(test, device):
    with wp.ScopedDevice(device):
        rng = np.random.default_rng(123)

        num_item_bounds = 100000
        item_bound_size = 0.01

        relative_shift = 2

        num_test_bounds = 10000
        test_bound_relative_size = 0.05

        center = np.array([0.0, 0.0, 0.0])

        item_lowers_np, item_uppers_np = get_random_aabbs(num_item_bounds, center, relative_shift, item_bound_size, rng)
        item_lowers = wp.array(item_lowers_np, dtype=wp.vec3)
        item_uppers = wp.array(item_uppers_np, dtype=wp.vec3)
        bvh_1 = wp.Bvh(item_lowers, item_uppers)
        item_lowers_2 = wp.zeros_like(item_lowers)
        item_uppers_2 = wp.zeros_like(item_lowers)

        test_lowers_np, test_uppers_np = get_random_aabbs(
            num_test_bounds, center, relative_shift, test_bound_relative_size, rng
        )
        test_lowers = wp.array(test_lowers_np, dtype=wp.vec3)
        test_uppers = wp.array(test_uppers_np, dtype=wp.vec3)

        item_lowers_2_np, item_uppers_2_np = get_random_aabbs(
            num_item_bounds,
            center,
            relative_shift,
            item_bound_size,
            rng,
        )
        item_lowers_2.assign(item_lowers_2_np)
        item_uppers_2.assign(item_uppers_2_np)

        counts_1 = wp.empty(n=num_test_bounds, dtype=int)
        checksums_1 = wp.empty(n=num_test_bounds, dtype=int)
        counts_2 = wp.empty(n=num_test_bounds, dtype=int)
        checksums_2 = wp.empty(n=num_test_bounds, dtype=int)

        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.copy(item_lowers, item_lowers_2)
            wp.copy(item_uppers, item_uppers_2)
            bvh_1.rebuild()
            wp.launch(
                compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, bvh_1.id],
                outputs=[counts_1, checksums_1],
            )

        cuda_graph = capture.graph

        for _ in range(10):
            item_lowers_2_np, item_uppers_2_np = get_random_aabbs(
                num_item_bounds,
                center,
                relative_shift,
                item_bound_size,
                rng,
            )
            item_lowers_2.assign(item_lowers_2_np)
            item_uppers_2.assign(item_uppers_2_np)

            wp.capture_launch(cuda_graph)

            bvh_2 = wp.Bvh(item_lowers_2, item_uppers_2)
            wp.launch(
                compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, bvh_2.id],
                outputs=[counts_2, checksums_2],
                device=device,
            )

            assert_array_equal(counts_1, counts_2)
            assert_array_equal(checksums_1, checksums_2)


@wp.kernel
def tile_bvh_query_aabb_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.tile_bvh_query_aabb(bvh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_bvh_query_next(query)
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each bound is only reported once
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


@wp.kernel
def tile_bvh_query_ray_kernel(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.tile_bvh_query_ray(bvh_id, start, dir)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_bvh_query_next(query)
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each bound is only reported once
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


def test_tile_bvh_query(test, device):
    """Test tile-based BVH query and compare with single-threaded version."""
    rng = np.random.default_rng(456)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_aabb,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_single],
        device=device,
    )

    # Test with tile-based version
    block_dim = 64
    bounds_intersected_tile = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_aabb_kernel,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_tile],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tile_result = bounds_intersected_tile.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tile_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tile={tile_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        if (
            lower[0] < query_upper[0]
            and upper[0] > query_lower[0]
            and lower[1] < query_upper[1]
            and upper[1] > query_lower[1]
            and lower[2] < query_upper[2]
            and upper[2] > query_lower[2]
        ):
            test.assertEqual(tile_result[i], 1, f"Expected bound {i} to be intersected")
        else:
            test.assertEqual(tile_result[i], 0, f"Expected bound {i} to not be intersected")

    # Verify that no bound was reported more than once
    # (all values should be 0 or 1, never > 1)
    for i in range(num_bounds):
        test.assertIn(
            tile_result[i],
            [0, 1],
            f"Bound {i} was reported {tile_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )

    # Also test tile_query_valid-based loop
    bounds_intersected_count = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_valid_aabb_kernel,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_count],
        device=device,
        block_dim=block_dim,
    )
    count_result = bounds_intersected_count.numpy()
    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            count_result[i],
            f"tile_query_valid mismatch at bound {i}: single={single_result[i]}, count={count_result[i]}",
        )


def test_tile_bvh_query_ray(test, device):
    """Test tile-based BVH ray query and compare with single-threaded version."""
    rng = np.random.default_rng(789)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_ray,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_single],
        device=device,
    )

    # Test with tile-based version
    block_dim = 64
    bounds_intersected_tile = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_ray_kernel,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_tile],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tile_result = bounds_intersected_tile.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tile_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tile={tile_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
        test.assertEqual(tile_result[i], host_intersected, f"Expected bound {i} intersection to be {host_intersected}")

    # Verify that no bound was reported more than once
    # (all values should be 0 or 1, never > 1)
    for i in range(num_bounds):
        test.assertIn(
            tile_result[i],
            [0, 1],
            f"Bound {i} was reported {tile_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )

    # Also test tile_query_valid-based loop
    bounds_intersected_count = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_valid_ray_kernel,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_count],
        device=device,
        block_dim=block_dim,
    )
    count_result = bounds_intersected_count.numpy()
    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            count_result[i],
            f"tile_query_valid mismatch at bound {i}: single={single_result[i]}, count={count_result[i]}",
        )


# Tests for new bvh_query_*_tiled() API (primary naming convention)
@wp.kernel
def bvh_query_aabb_tiled_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.bvh_query_aabb_tiled(bvh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.bvh_query_next_tiled(query)
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


@wp.kernel
def bvh_query_ray_tiled_kernel(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.bvh_query_ray_tiled(bvh_id, start, dir)

    while wp.tile_query_valid(query):
        result_tile = wp.bvh_query_next_tiled(query)
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


def test_bvh_query_aabb_tiled(test, device):
    """Test bvh_query_aabb_tiled() API (new primary naming convention)."""
    rng = np.random.default_rng(456)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_aabb,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_single],
        device=device,
    )

    # Test with new tiled API
    block_dim = 64
    bounds_intersected_tiled = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=bvh_query_aabb_tiled_kernel,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_tiled],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tiled_result = bounds_intersected_tiled.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tiled_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tiled={tiled_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        if (
            lower[0] < query_upper[0]
            and upper[0] > query_lower[0]
            and lower[1] < query_upper[1]
            and upper[1] > query_lower[1]
            and lower[2] < query_upper[2]
            and upper[2] > query_lower[2]
        ):
            test.assertEqual(tiled_result[i], 1, f"Expected bound {i} to be intersected")
        else:
            test.assertEqual(tiled_result[i], 0, f"Expected bound {i} to not be intersected")

    # Verify that no bound was reported more than once
    for i in range(num_bounds):
        test.assertIn(
            tiled_result[i],
            [0, 1],
            f"Bound {i} was reported {tiled_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


def test_bvh_query_ray_tiled(test, device):
    """Test bvh_query_ray_tiled() API (new primary naming convention)."""
    rng = np.random.default_rng(789)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_ray,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_single],
        device=device,
    )

    # Test with new tiled API
    block_dim = 64
    bounds_intersected_tiled = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=bvh_query_ray_tiled_kernel,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_tiled],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tiled_result = bounds_intersected_tiled.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tiled_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tiled={tiled_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
        test.assertEqual(tiled_result[i], host_intersected, f"Expected bound {i} intersection to be {host_intersected}")

    # Verify that no bound was reported more than once
    for i in range(num_bounds):
        test.assertIn(
            tiled_result[i],
            [0, 1],
            f"Bound {i} was reported {tiled_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


@wp.kernel
def tile_bvh_query_valid_aabb_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.tile_bvh_query_aabb(bvh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_bvh_query_next(query)
        result_idx = wp.untile(result_tile)

        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


@wp.kernel
def tile_bvh_query_valid_ray_kernel(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    bounds_intersected: wp.array[int],
):
    query = wp.tile_bvh_query_ray(bvh_id, start, dir)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_bvh_query_next(query)
        result_idx = wp.untile(result_tile)

        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)


@wp.kernel
def bvh_query_aabb_mark_all(bvh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3, hit: wp.array(dtype=int)):
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        hit[bounds_nr] = 1


def test_bvh_degenerate_deep_tree(test, device):
    # Diagonally exponentially spaced boxes give the morton codes a long
    # shared-prefix chain, and the duplicate cluster at the tail pushes the
    # tree to the construction depth bound, maximizing traversal stack
    # pressure. A query covering everything must still enumerate every box.
    num_bounds = 256
    vals = np.power(0.5, np.arange(num_bounds) * 29.0 / num_bounds).astype(np.float32)
    lowers = np.stack([vals, vals, vals], axis=1)
    lowers[num_bounds - 64 :] = lowers[num_bounds - 64]
    uppers = lowers + 1e-7

    for constructor in ["sah", "median", "lbvh"]:
        for leaf_size in [1, 4]:
            bvh = wp.Bvh(
                wp.array(lowers, dtype=wp.vec3, device=device),
                wp.array(uppers, dtype=wp.vec3, device=device),
                constructor=constructor,
                leaf_size=leaf_size,
            )
            hit = wp.zeros(num_bounds, dtype=int, device=device)
            wp.launch(
                bvh_query_aabb_mark_all,
                dim=1,
                inputs=[bvh.id, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(2.0, 2.0, 2.0), hit],
                device=device,
            )
            test.assertEqual(
                int(hit.numpy().sum()),
                num_bounds,
                f"missing query results for constructor={constructor} leaf_size={leaf_size}",
            )


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()
cuda_devices_with_mempool = get_cuda_test_devices_with_mempool()


class TestBvh(unittest.TestCase):
    def test_bvh_codegen_adjoints_with_select(self):
        def kernel_fn(bvh: wp.uint64):
            v = wp.vec3(0.0, 0.0, 0.0)
            bounds_nr = int(0)

            if True:
                query_1 = wp.bvh_query_aabb(bvh, v, v)
                query_2 = wp.bvh_query_ray(bvh, v, v)

                wp.bvh_query_next(query_1, bounds_nr)
                wp.bvh_query_next(query_2, bounds_nr)
            else:
                query_1 = wp.bvh_query_aabb(bvh, v, v)
                query_2 = wp.bvh_query_ray(bvh, v, v)

                wp.bvh_query_next(query_1, bounds_nr)
                wp.bvh_query_next(query_2, bounds_nr)

        wp.Kernel(func=kernel_fn)

    def test_bvh_new_del(self):
        # test the scenario in which a bvh is created but not initialized before gc
        instance = wp.Bvh.__new__(wp.Bvh)
        instance.__del__()

    def test_bvh_cubql_groups_error(self):
        lowers = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        uppers = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device="cpu")
        groups = wp.array([0], dtype=int, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "Grouped BVHs"):
            wp.Bvh(lowers, uppers, constructor="cubql", groups=groups)


add_function_test(TestBvh, "test_bvh_aabb", test_bvh_query_aabb, devices=devices)
add_function_test(TestBvh, "test_bvh_degenerate_deep_tree", test_bvh_degenerate_deep_tree, devices=devices)
add_function_test(TestBvh, "test_bvh_ray", test_bvh_query_ray, devices=devices)
add_function_test(TestBvh, "test_bvh_sphere", test_bvh_query_sphere, devices=devices)
add_function_test(TestBvh, "test_bvh_capsule", test_bvh_query_capsule, devices=devices)
add_function_test(TestBvh, "test_bvh_capsule_max_depth", test_bvh_query_capsule_max_depth, devices=devices)
add_function_test(
    TestBvh,
    "test_bvh_capsule_grouped_max_depth",
    test_bvh_query_capsule_grouped_max_depth,
    devices=devices,
)
add_function_test(TestBvh, "test_bvh_cubql_constructor", test_bvh_cubql_constructor, devices=devices)
add_function_test(
    TestBvh,
    "test_bvh_ray_query_inside_and_outside_bounds",
    test_bvh_ray_query_inside_and_outside_bounds,
    devices=devices,
)
add_function_test(TestBvh, "test_bvh_refit_root_leaves", test_bvh_refit_root_leaves, devices=cuda_devices)
add_function_test(TestBvh, "test_tile_bvh_query_aabb", test_tile_bvh_query, devices=cuda_devices)
add_function_test(TestBvh, "test_tile_bvh_query_ray", test_tile_bvh_query_ray, devices=cuda_devices)

# Tests for new bvh_query_*_tiled() API
add_function_test(TestBvh, "test_bvh_query_aabb_tiled", test_bvh_query_aabb_tiled, devices=cuda_devices)
add_function_test(TestBvh, "test_bvh_query_ray_tiled", test_bvh_query_ray_tiled, devices=cuda_devices)

add_function_test(TestBvh, "test_capture_bvh_rebuild", test_capture_bvh_rebuild, devices=cuda_devices_with_mempool)

if __name__ == "__main__":
    unittest.main(verbosity=2)
