# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices_with_mempool,
    get_test_devices,
)


@wp.kernel
def query_aabb_hits(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    hits: wp.array(dtype=int),
):
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    primitive = int(-1)
    while wp.bvh_query_next(query, primitive):
        wp.atomic_add(hits, primitive, 1)


@wp.kernel
def record_frontier(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    tokens: wp.array(dtype=int),
    token_offset: int,
    token_capacity: int,
    root: int,
    count: wp.array(dtype=int),
    epoch: wp.array(dtype=int),
    resolved_root: wp.array(dtype=int),
):
    count[0] = wp.bvh_query_aabb_frontier_record(
        bvh_id,
        lower,
        upper,
        tokens,
        token_offset,
        token_capacity,
        root,
    )
    epoch[0] = wp.bvh_frontier_topology_epoch(bvh_id)
    resolved_root[0] = wp.bvh_frontier_root(bvh_id, root)


@wp.kernel
def inspect_frontier(
    bvh_id: wp.uint64,
    tokens: wp.array(dtype=int),
    count: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    hit_tags: wp.array(dtype=int),
    token_valid: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid < count[0]:
        token = tokens[tid]
        nodes[tid] = wp.bvh_frontier_token_node(token)
        if wp.bvh_frontier_token_is_hit(token):
            hit_tags[tid] = 1
        if wp.bvh_frontier_token_is_valid(bvh_id, token):
            token_valid[tid] = 1


@wp.kernel
def replay_frontier(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    tokens: wp.array(dtype=int),
    count: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid < count[0]:
        query = wp.bvh_query_aabb_frontier_token(bvh_id, lower, upper, tokens[tid])
        primitive = int(-1)
        while wp.bvh_query_next(query, primitive):
            wp.atomic_add(hits, primitive, 1)


@wp.kernel
def emit_hit_leaf_primitives(
    bvh_id: wp.uint64,
    tokens: wp.array(dtype=int),
    count: wp.array(dtype=int),
    hits: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid < count[0]:
        token = tokens[tid]
        if wp.bvh_frontier_token_is_hit(token):
            primitive_count = wp.bvh_frontier_token_primitive_count(bvh_id, token)
            for offset in range(primitive_count):
                primitive = wp.bvh_frontier_token_primitive_at(bvh_id, token, offset)
                if primitive >= 0:
                    wp.atomic_add(hits, primitive, 1)


@wp.kernel
def read_frontier_metadata(
    bvh_id: wp.uint64,
    root: int,
    epoch: wp.array(dtype=int),
    resolved_root: wp.array(dtype=int),
):
    epoch[0] = wp.bvh_frontier_topology_epoch(bvh_id)
    resolved_root[0] = wp.bvh_frontier_root(bvh_id, root)


@wp.kernel
def check_frontier_gate(
    bvh_id: wp.uint64,
    recorded_id: wp.uint64,
    recorded_epoch: int,
    recorded_root: int,
    root: int,
    valid: wp.array(dtype=int),
):
    valid[0] = 0
    if wp.bvh_frontier_is_valid(bvh_id, recorded_id, recorded_epoch, recorded_root, root):
        valid[0] = 1


@wp.kernel
def roll_frontier_atomic(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    input_tokens: wp.array(dtype=int),
    input_count: wp.array(dtype=int),
    output_tokens: wp.array(dtype=int),
    output_counts: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
    segment_capacity: int,
    errors: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid < input_count[0]:
        root = wp.bvh_frontier_token_node(input_tokens[tid])
        appended = wp.bvh_query_aabb_frontier_record_atomic(
            bvh_id,
            lower,
            upper,
            output_tokens,
            output_counts,
            overflow,
            0,
            segment_capacity,
            root,
        )
        if appended < 0:
            wp.atomic_add(errors, 0, 1)


@wp.kernel
def record_frontier_atomic_root(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    output_tokens: wp.array(dtype=int),
    output_counts: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
    segment_capacity: int,
    root: int,
    appended: wp.array(dtype=int),
):
    appended[0] = wp.bvh_query_aabb_frontier_record_atomic(
        bvh_id,
        lower,
        upper,
        output_tokens,
        output_counts,
        overflow,
        0,
        segment_capacity,
        root,
    )


@wp.kernel
def consume_frontier_or_root(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    tokens: wp.array(dtype=int),
    count: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
    segment_capacity: int,
    hits: wp.array(dtype=int),
    root_fallbacks: wp.array(dtype=int),
):
    if overflow[0] != 0 or count[0] < 1 or count[0] > segment_capacity:
        root_fallbacks[0] = 1
        query = wp.bvh_query_aabb(bvh_id, lower, upper)
        primitive = int(-1)
        while wp.bvh_query_next(query, primitive):
            wp.atomic_add(hits, primitive, 1)
    else:
        for token_index in range(count[0]):
            query = wp.bvh_query_aabb_frontier_token(bvh_id, lower, upper, tokens[token_index])
            primitive = int(-1)
            while wp.bvh_query_next(query, primitive):
                wp.atomic_add(hits, primitive, 1)


@wp.kernel
def check_malformed_frontier_metadata(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    tokens: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    overflow: wp.array(dtype=int),
    results: wp.array(dtype=int),
):
    invalid_token = int(0x7FFFFFFF)
    results[0] = wp.bvh_query_aabb_frontier_record(bvh_id, lower, upper, tokens, -1, 1, -1)
    results[1] = wp.bvh_query_aabb_frontier_record(bvh_id, lower, upper, tokens, 0, 1, invalid_token)
    results[2] = wp.bvh_query_aabb_frontier_record_atomic(
        bvh_id,
        lower,
        upper,
        tokens,
        counts,
        overflow,
        0,
        0,
        -1,
    )
    results[3] = wp.bvh_frontier_root(bvh_id, invalid_token)
    results[4] = wp.bvh_frontier_token_primitive_count(bvh_id, invalid_token)
    results[5] = wp.bvh_frontier_token_primitive_at(bvh_id, invalid_token, 0)
    results[6] = 0
    if wp.bvh_frontier_token_is_valid(bvh_id, invalid_token):
        results[6] = 1


def make_linear_bounds(num_bounds=16):
    lowers = np.zeros((num_bounds, 3), dtype=np.float32)
    uppers = np.zeros((num_bounds, 3), dtype=np.float32)
    for i in range(num_bounds):
        lowers[i] = (2.0 * i, 0.2 * (i % 3), 0.15 * (i % 2))
        uppers[i] = lowers[i] + (0.75, 0.65, 0.55)
    return lowers, uppers


def expected_hits(lowers, uppers, query_lower, query_upper):
    return np.logical_and(
        np.all(lowers <= np.asarray(query_upper)[None, :], axis=1),
        np.all(np.asarray(query_lower)[None, :] <= uppers, axis=1),
    ).astype(np.int32)


def create_bvh(lowers, uppers, device, leaf_size=1, constructor="sah"):
    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)
    bvh = wp.Bvh(device_lowers, device_uppers, constructor=constructor, leaf_size=leaf_size)
    return bvh, device_lowers, device_uppers


def record_complete_frontier(bvh, query_lower, query_upper, capacity, device, strided=False):
    if strided:
        token_storage = wp.array(np.full(capacity * 2, -777, dtype=np.int32), dtype=int, device=device)
        tokens = token_storage[::2]
    else:
        token_storage = wp.empty(capacity, dtype=int, device=device)
        tokens = token_storage

    count = wp.zeros(1, dtype=int, device=device)
    epoch = wp.zeros(1, dtype=int, device=device)
    resolved_root = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        record_frontier,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            tokens,
            0,
            capacity,
            -1,
            count,
            epoch,
            resolved_root,
        ],
        device=device,
    )
    return tokens, token_storage, count, epoch, resolved_root


def assert_frontier_exact(test, bvh, lower, upper, tokens, count, num_bounds, device, expected):
    replay_hits = wp.zeros(num_bounds, dtype=int, device=device)
    root_hits = wp.zeros(num_bounds, dtype=int, device=device)
    wp.launch(
        replay_frontier,
        dim=len(tokens),
        inputs=[bvh.id, lower, upper, tokens, count, replay_hits],
        device=device,
    )
    wp.launch(query_aabb_hits, dim=1, inputs=[bvh.id, lower, upper, root_hits], device=device)

    replay_np = replay_hits.numpy()
    root_np = root_hits.numpy()
    np.testing.assert_array_equal(replay_np, expected)
    np.testing.assert_array_equal(root_np, expected)
    np.testing.assert_array_equal(np.flatnonzero(replay_np), np.flatnonzero(expected))
    test.assertLessEqual(int(replay_np.max(initial=0)), 1, "Frontier replay returned a duplicate primitive")


def inspect_complete_frontier(test, bvh, tokens, count, capacity, device):
    nodes = wp.full(capacity, -1, dtype=int, device=device)
    hit_tags = wp.zeros(capacity, dtype=int, device=device)
    token_valid = wp.zeros(capacity, dtype=int, device=device)
    wp.launch(
        inspect_frontier,
        dim=capacity,
        inputs=[bvh.id, tokens, count, nodes, hit_tags, token_valid],
        device=device,
    )
    count_value = int(count.numpy()[0])
    test.assertGreater(count_value, 0)
    nodes_np = nodes.numpy()[:count_value]
    test.assertEqual(len(np.unique(nodes_np)), count_value, "A complete frontier must not repeat subtree roots")
    np.testing.assert_array_equal(token_valid.numpy()[:count_value], np.ones(count_value, dtype=np.int32))
    return nodes_np, hit_tags.numpy()[:count_value]


def check_gate(bvh, recorded_id, recorded_epoch, recorded_root, root, device):
    valid = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        check_frontier_gate,
        dim=1,
        inputs=[bvh.id, recorded_id, recorded_epoch, recorded_root, root, valid],
        device=device,
    )
    return int(valid.numpy()[0])


def test_root_frontier_exact_and_strided(test, device):
    lowers, uppers = make_linear_bounds()
    bvh, _, _ = create_bvh(lowers, uppers, device)
    query_lower = wp.vec3(1.75, -0.1, -0.1)
    query_upper = wp.vec3(4.20, 1.0, 1.0)
    expected = expected_hits(lowers, uppers, query_lower, query_upper)

    tokens, token_storage, count, epoch, resolved_root = record_complete_frontier(
        bvh,
        query_lower,
        query_upper,
        len(lowers),
        device,
        strided=True,
    )
    count_value = int(count.numpy()[0])
    test.assertGreater(count_value, 1)
    test.assertGreater(int(epoch.numpy()[0]), 0)
    test.assertGreaterEqual(int(resolved_root.numpy()[0]), 0)
    np.testing.assert_array_equal(
        token_storage.numpy()[1::2],
        np.full(len(lowers), -777, dtype=np.int32),
        err_msg="Recording through a positive-stride view overwrote its gaps",
    )

    inspect_complete_frontier(test, bvh, tokens, count, len(lowers), device)
    assert_frontier_exact(test, bvh, query_lower, query_upper, tokens, count, len(lowers), device, expected)

    # A complete terminal cut must also cover subtrees pruned by the query
    # that created it. Reuse the same tokens for a disjoint query region.
    alternate_lower = wp.vec3(19.75, -0.1, -0.1)
    alternate_upper = wp.vec3(22.20, 1.0, 1.0)
    assert_frontier_exact(
        test,
        bvh,
        alternate_lower,
        alternate_upper,
        tokens,
        count,
        len(lowers),
        device,
        expected_hits(lowers, uppers, alternate_lower, alternate_upper),
    )


def test_atomic_frontier_refit_and_overflow(test, device):
    lowers, uppers = make_linear_bounds()
    num_bounds = len(lowers)
    bvh, device_lowers, device_uppers = create_bvh(lowers, uppers, device)
    query_lower = wp.vec3(1.75, -0.1, -0.1)
    query_upper = wp.vec3(4.20, 1.0, 1.0)

    tokens_a, _, count_a, epoch, resolved_root = record_complete_frontier(
        bvh,
        query_lower,
        query_upper,
        num_bounds,
        device,
    )
    recorded_epoch = int(epoch.numpy()[0])
    recorded_root = int(resolved_root.numpy()[0])

    stage_1_lowers = lowers.copy()
    stage_1_uppers = uppers.copy()
    stage_1_lowers[0] = (2.50, 0.0, 0.0)
    stage_1_uppers[0] = (3.10, 0.5, 0.5)
    stage_1_lowers[1] = (30.0, 0.0, 0.0)
    stage_1_uppers[1] = (30.6, 0.5, 0.5)
    stage_1_lowers[7] = (3.25, 0.0, 0.0)
    stage_1_uppers[7] = (3.85, 0.5, 0.5)
    device_lowers.assign(stage_1_lowers)
    device_uppers.assign(stage_1_uppers)
    bvh.refit()

    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 1)

    tokens_b = wp.empty(num_bounds, dtype=int, device=device)
    count_b = wp.zeros(1, dtype=int, device=device)
    overflow_b = wp.zeros(1, dtype=int, device=device)
    errors = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        roll_frontier_atomic,
        dim=num_bounds,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            tokens_a,
            count_a,
            tokens_b,
            count_b,
            overflow_b,
            num_bounds,
            errors,
        ],
        device=device,
    )
    test.assertEqual(int(errors.numpy()[0]), 0)
    test.assertEqual(int(overflow_b.numpy()[0]), 0)
    test.assertLessEqual(int(count_b.numpy()[0]), num_bounds)
    inspect_complete_frontier(test, bvh, tokens_b, count_b, num_bounds, device)
    assert_frontier_exact(
        test,
        bvh,
        query_lower,
        query_upper,
        tokens_b,
        count_b,
        num_bounds,
        device,
        expected_hits(stage_1_lowers, stage_1_uppers, query_lower, query_upper),
    )

    # Roll the cache a second time using ping-pong storage. This moves every
    # prior hit away and activates a primitive from a previously pruned branch.
    stage_2_lowers = lowers.copy()
    stage_2_uppers = uppers.copy()
    stage_2_lowers[:, 0] += 50.0
    stage_2_uppers[:, 0] += 50.0
    stage_2_lowers[11] = (3.0, 0.0, 0.0)
    stage_2_uppers[11] = (3.6, 0.5, 0.5)
    device_lowers.assign(stage_2_lowers)
    device_uppers.assign(stage_2_uppers)
    bvh.refit()
    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 1)

    count_a.zero_()
    overflow_a = wp.zeros(1, dtype=int, device=device)
    errors.zero_()
    wp.launch(
        roll_frontier_atomic,
        dim=num_bounds,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            tokens_b,
            count_b,
            tokens_a,
            count_a,
            overflow_a,
            num_bounds,
            errors,
        ],
        device=device,
    )
    test.assertEqual(int(errors.numpy()[0]), 0)
    test.assertEqual(int(overflow_a.numpy()[0]), 0)
    inspect_complete_frontier(test, bvh, tokens_a, count_a, num_bounds, device)
    assert_frontier_exact(
        test,
        bvh,
        query_lower,
        query_upper,
        tokens_a,
        count_a,
        num_bounds,
        device,
        expected_hits(stage_2_lowers, stage_2_uppers, query_lower, query_upper),
    )

    # Restore the original bounds before sizing a root cut for the deliberate
    # one-short fixed segment.
    device_lowers.assign(lowers)
    device_uppers.assign(uppers)
    bvh.refit()
    _full_tokens, _, full_count, _, _ = record_complete_frontier(
        bvh,
        query_lower,
        query_upper,
        num_bounds,
        device,
    )
    required = int(full_count.numpy()[0])
    test.assertGreater(required, 1)
    one_short = required - 1
    partial_tokens = wp.full(one_short, -1, dtype=int, device=device)
    attempted_count = wp.zeros(1, dtype=int, device=device)
    overflow = wp.zeros(1, dtype=int, device=device)
    appended = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        record_frontier_atomic_root,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            partial_tokens,
            attempted_count,
            overflow,
            one_short,
            -1,
            appended,
        ],
        device=device,
    )
    test.assertEqual(int(appended.numpy()[0]), required)
    test.assertEqual(int(attempted_count.numpy()[0]), required, "Required count must remain unclamped")
    test.assertEqual(int(overflow.numpy()[0]), 1)

    fallback_hits = wp.zeros(num_bounds, dtype=int, device=device)
    root_fallbacks = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        consume_frontier_or_root,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            partial_tokens,
            attempted_count,
            overflow,
            one_short,
            fallback_hits,
            root_fallbacks,
        ],
        device=device,
    )
    test.assertEqual(int(root_fallbacks.numpy()[0]), 1, "An overflowed partial cut must be discarded")
    np.testing.assert_array_equal(
        fallback_hits.numpy(),
        expected_hits(lowers, uppers, query_lower, query_upper),
    )

    # Nonpositive counts are never consumable complete cuts, even when the
    # overflow flag is clear.
    empty_count = wp.zeros(1, dtype=int, device=device)
    clean_overflow = wp.zeros(1, dtype=int, device=device)
    empty_fallback_hits = wp.zeros(num_bounds, dtype=int, device=device)
    empty_root_fallbacks = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        consume_frontier_or_root,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            partial_tokens,
            empty_count,
            clean_overflow,
            one_short,
            empty_fallback_hits,
            empty_root_fallbacks,
        ],
        device=device,
    )
    test.assertEqual(int(empty_root_fallbacks.numpy()[0]), 1)
    np.testing.assert_array_equal(
        empty_fallback_hits.numpy(),
        expected_hits(lowers, uppers, query_lower, query_upper),
    )

    # A later successful append does not clear a prior overflow; the caller
    # must explicitly zero this sticky transaction flag before reuse.
    sticky_tokens = wp.empty(num_bounds, dtype=int, device=device)
    sticky_count = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        record_frontier_atomic_root,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            sticky_tokens,
            sticky_count,
            overflow,
            num_bounds,
            -1,
            appended,
        ],
        device=device,
    )
    test.assertEqual(int(overflow.numpy()[0]), 1)
    test.assertEqual(int(sticky_count.numpy()[0]), required)


def test_frontier_gate_and_metadata(test, device):
    lowers, uppers = make_linear_bounds()
    num_bounds = len(lowers)
    bvh, device_lowers, device_uppers = create_bvh(lowers, uppers, device)
    other_bvh, _, _ = create_bvh(lowers + 100.0, uppers + 100.0, device)
    query_lower = wp.vec3(1.75, -0.1, -0.1)
    query_upper = wp.vec3(4.20, 1.0, 1.0)
    tokens, _, count, epoch, resolved_root = record_complete_frontier(
        bvh,
        query_lower,
        query_upper,
        num_bounds,
        device,
    )
    recorded_epoch = int(epoch.numpy()[0])
    recorded_root = int(resolved_root.numpy()[0])
    nodes, _ = inspect_complete_frontier(test, bvh, tokens, count, num_bounds, device)
    alternate_roots = nodes[nodes != recorded_root]
    test.assertGreater(len(alternate_roots), 0)
    alternate_root = int(alternate_roots[0])

    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 1)
    test.assertEqual(check_gate(other_bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 0)
    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, alternate_root, device), 0)

    moved_lowers = lowers.copy()
    moved_uppers = uppers.copy()
    moved_lowers[:, 1] += 3.0
    moved_uppers[:, 1] += 3.0
    device_lowers.assign(moved_lowers)
    device_uppers.assign(moved_uppers)
    bvh.refit()
    refit_epoch = wp.zeros(1, dtype=int, device=device)
    refit_root = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        read_frontier_metadata,
        dim=1,
        inputs=[bvh.id, -1, refit_epoch, refit_root],
        device=device,
    )
    test.assertEqual(int(refit_epoch.numpy()[0]), recorded_epoch)
    test.assertEqual(int(refit_root.numpy()[0]), recorded_root)
    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 1)

    bvh.rebuild()
    rebuilt_epoch = wp.zeros(1, dtype=int, device=device)
    rebuilt_root = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        read_frontier_metadata,
        dim=1,
        inputs=[bvh.id, -1, rebuilt_epoch, rebuilt_root],
        device=device,
    )
    test.assertNotEqual(int(rebuilt_epoch.numpy()[0]), recorded_epoch)
    test.assertEqual(check_gate(bvh, bvh.id, recorded_epoch, recorded_root, -1, device), 0)

    malformed_results = wp.zeros(7, dtype=int, device=device)
    malformed_counts = wp.zeros(1, dtype=int, device=device)
    malformed_overflow = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        check_malformed_frontier_metadata,
        dim=1,
        inputs=[
            bvh.id,
            query_lower,
            query_upper,
            tokens,
            malformed_counts,
            malformed_overflow,
            malformed_results,
        ],
        device=device,
    )
    np.testing.assert_array_equal(malformed_results.numpy(), np.array((-1, -1, -1, -1, 0, -1, 0), dtype=np.int32))
    test.assertEqual(int(malformed_overflow.numpy()[0]), 1)


def test_frontier_direct_leaf_candidates(test, device):
    lowers, uppers = make_linear_bounds(4)
    query_lower = wp.vec3(2.10, -0.1, -0.1)
    query_upper = wp.vec3(2.20, 1.0, 1.0)
    expected = expected_hits(lowers, uppers, query_lower, query_upper)
    test.assertEqual(int(expected.sum()), 1)

    for leaf_size in (1, 4):
        with test.subTest(leaf_size=leaf_size):
            bvh, _, _ = create_bvh(lowers, uppers, device, leaf_size=leaf_size)
            tokens, _, count, _, _ = record_complete_frontier(
                bvh,
                query_lower,
                query_upper,
                len(lowers),
                device,
            )
            _, hit_tags = inspect_complete_frontier(test, bvh, tokens, count, len(lowers), device)
            test.assertGreater(int(hit_tags.sum()), 0)

            exact = wp.zeros(len(lowers), dtype=int, device=device)
            direct = wp.zeros(len(lowers), dtype=int, device=device)
            wp.launch(
                replay_frontier,
                dim=len(tokens),
                inputs=[bvh.id, query_lower, query_upper, tokens, count, exact],
                device=device,
            )
            wp.launch(
                emit_hit_leaf_primitives,
                dim=len(tokens),
                inputs=[bvh.id, tokens, count, direct],
                device=device,
            )
            exact_np = exact.numpy()
            direct_np = direct.numpy()
            np.testing.assert_array_equal(exact_np, expected)
            test.assertFalse(np.any(np.logical_and(expected != 0, direct_np == 0)), "Direct leaf emission missed a hit")
            test.assertLessEqual(int(direct_np.max(initial=0)), 1)
            if leaf_size == 1:
                np.testing.assert_array_equal(direct_np, expected)
            else:
                test.assertGreater(int(direct_np.sum()), int(expected.sum()))


def test_cuda_graph_rebuild_invalidates_frontier(test, device):
    lowers, uppers = make_linear_bounds(32)
    bvh, _, _ = create_bvh(lowers, uppers, device, constructor="lbvh")
    old_epoch = wp.zeros(1, dtype=int, device=device)
    old_root = wp.zeros(1, dtype=int, device=device)
    wp.launch(read_frontier_metadata, dim=1, inputs=[bvh.id, -1, old_epoch, old_root], device=device)
    epoch_0 = int(old_epoch.numpy()[0])
    root_0 = int(old_root.numpy()[0])

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        bvh.rebuild()

    wp.capture_launch(capture.graph)
    test.assertEqual(check_gate(bvh, bvh.id, epoch_0, root_0, -1, device), 0)
    epoch_1 = wp.zeros(1, dtype=int, device=device)
    root_1 = wp.zeros(1, dtype=int, device=device)
    wp.launch(read_frontier_metadata, dim=1, inputs=[bvh.id, -1, epoch_1, root_1], device=device)
    epoch_1_value = int(epoch_1.numpy()[0])
    test.assertNotEqual(epoch_1_value, epoch_0)

    wp.capture_launch(capture.graph)
    test.assertEqual(check_gate(bvh, bvh.id, epoch_1_value, int(root_1.numpy()[0]), -1, device), 0)
    epoch_2 = wp.zeros(1, dtype=int, device=device)
    root_2 = wp.zeros(1, dtype=int, device=device)
    wp.launch(read_frontier_metadata, dim=1, inputs=[bvh.id, -1, epoch_2, root_2], device=device)
    test.assertNotEqual(int(epoch_2.numpy()[0]), epoch_1_value)


devices = get_test_devices()
cuda_devices_with_mempool = get_cuda_test_devices_with_mempool()


class TestBvhFrontier(unittest.TestCase):
    pass


add_function_test(
    TestBvhFrontier,
    "test_root_frontier_exact_and_strided",
    test_root_frontier_exact_and_strided,
    devices=devices,
)
add_function_test(
    TestBvhFrontier,
    "test_atomic_frontier_refit_and_overflow",
    test_atomic_frontier_refit_and_overflow,
    devices=devices,
)
add_function_test(
    TestBvhFrontier,
    "test_frontier_gate_and_metadata",
    test_frontier_gate_and_metadata,
    devices=devices,
)
add_function_test(
    TestBvhFrontier,
    "test_frontier_direct_leaf_candidates",
    test_frontier_direct_leaf_candidates,
    devices=devices,
)
add_function_test(
    TestBvhFrontier,
    "test_cuda_graph_rebuild_invalidates_frontier",
    test_cuda_graph_rebuild_invalidates_frontier,
    devices=cuda_devices_with_mempool,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
