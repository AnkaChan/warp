"""Compare regular, cached, fused and peeling AABB queries."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from closest_point import benchmark
from provenance import source_record

import warp as wp
from warp.examples.benchmarks.benchmark_bvh_queries import make_aabb_data

VERSION = "ebvh-aabb-probe-v3"
print(VERSION, flush=True)


def make_probe(mode):
    selection = {
        "root": "auto query = wp::bvh_query_aabb(id, low, high, -1);",
        "walk": "auto query = wp::bvh_query_aabb_exclusive(id, low, high, seed);",
        "cached": "auto query = wp::bvh_query_aabb_exclusive_cached(id, low, high, node);",
        "oracle": "auto query = wp::bvh_query_aabb(id, low, high, node);",
        "fused": "auto query = wp::bvh_query_aabb_exclusive_update(id, low, high, node, false);",
        "fused_refine": "auto query = wp::bvh_query_aabb_exclusive_update(id, low, high, node, true);",
        "peeling": "auto query = wp::bvh_query_aabb_exclusive_cached_peeling(id, low, high, node);",
        "bottom_up": "auto query = wp::bvh_query_aabb_exclusive_cached_bottom_up(id, low, high, node);",
        "update": """
            node = wp::bvh_find_exclusive_containment(bvh, low, high, node);
            auto query = wp::bvh_query_aabb_exclusive_cached(id, low, high, node);
        """,
        "refine": """
            node = wp::bvh_find_exclusive_containment(bvh, low, high, node);
            for (int iteration = 0; iteration < 32; ++iteration) {
                const auto lower = wp::bvh_load_node(bvh.node_lowers, node);
                if (lower.b) break;
                const auto upper = wp::bvh_load_node(bvh.node_uppers, node);
                const int left = lower.i, right = upper.i;
                if (wp::bvh_exclusive_contains_strict(wp::bvh_get_exclusive_node(bvh, left), low, high))
                    node = left;
                else if (wp::bvh_exclusive_contains_strict(wp::bvh_get_exclusive_node(bvh, right), low, high))
                    node = right;
                else break;
            }
            auto query = wp::bvh_query_aabb_exclusive_cached(id, low, high, node);
        """,
    }[mode]
    snippet = (
        """
        const wp::BVH bvh = wp::bvh_get(id);
    """
        + selection
        + """
        int hit = -1, count = 0;
        unsigned checksum = 0;
        while (wp::bvh_query_next(query, hit, FLT_MAX)) {
            checksum += static_cast<unsigned>(hit);
            if (capacity > 0 && count < capacity) hit_ids[i * capacity + count] = hit;
            ++count;
        }
        counts[i] = count;
        sums[i] = checksum;
        next_nodes[i] = node;
    """
    )

    @wp.func_native(snippet)
    def native_query(
        id: wp.uint64,
        low: wp.vec3,
        high: wp.vec3,
        seed: int,
        node: int,
        counts: wp.array(dtype=int),
        sums: wp.array(dtype=wp.uint32),
        next_nodes: wp.array(dtype=int),
        hit_ids: wp.array(dtype=int),
        i: int,
        capacity: int,
    ): ...

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def kernel(
        id: wp.uint64,
        lows: wp.array(dtype=wp.vec3),
        highs: wp.array(dtype=wp.vec3),
        seeds: wp.array(dtype=int),
        nodes: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        sums: wp.array(dtype=wp.uint32),
        next_nodes: wp.array(dtype=int),
        hit_ids: wp.array(dtype=int),
        capacity: int,
    ):
        i = wp.tid()
        native_query(
            id,
            lows[i],
            highs[i],
            seeds[i],
            nodes[i],
            counts,
            sums,
            next_nodes,
            hit_ids,
            i,
            capacity,
        )

    return kernel


@wp.kernel
def initialize_nodes(
    id: wp.uint64,
    lows: wp.array(dtype=wp.vec3),
    highs: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
):
    i = wp.tid()
    nodes[i] = wp.bvh_query_aabb_exclusive_node(id, lows[i], highs[i], seeds[i])


@wp.func_native("""
    const auto bvh = wp::bvh_get(id);
    int depth;
    wp::bvh_find_exclusive_containment(bvh, low, high, node, &depth);
    int maximum = bvh.max_depth_ptr ? *bvh.max_depth_ptr : bvh.max_depth;
    int remaining = maximum - 1 - depth;
    return wp::vec3(static_cast<float>(depth), static_cast<float>(remaining),
                    static_cast<float>(depth < 0 || remaining < 0 || remaining > BVH_CACHED_QUERY_STACK_SIZE));
""")
def stack_diagnostic(
    id: wp.uint64, low: wp.vec3, high: wp.vec3, node: int
) -> wp.vec3: ...


@wp.kernel
def diagnose(
    id: wp.uint64,
    lows: wp.array(dtype=wp.vec3),
    highs: wp.array(dtype=wp.vec3),
    nodes: wp.array(dtype=int),
    stats: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    stats[i] = stack_diagnostic(id, lows[i], highs[i], nodes[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=1000000)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument(
        "--modes", default="root,walk,cached,oracle,fused,fused_refine,peeling"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    wp.init()
    device = wp.get_device("cuda:0")
    rng = np.random.default_rng(12345)
    lo, hi, qlo, qhi, seeds = make_aabb_data(args.primitives, args.queries, rng)
    bvh = wp.Bvh(
        wp.array(lo, dtype=wp.vec3, device=device),
        wp.array(hi, dtype=wp.vec3, device=device),
        constructor="sah",
        leaf_size=1,
        enable_exclusive=True,
    )
    qlo_wp, qhi_wp = [wp.array(x, dtype=wp.vec3, device=device) for x in (qlo, qhi)]
    seeds_wp = wp.array(seeds, dtype=int, device=device)
    nodes = wp.empty(args.queries, dtype=int, device=device)
    wp.launch(
        initialize_nodes,
        args.queries,
        [bvh.id, qlo_wp, qhi_wp, seeds_wp, nodes],
        device=device,
    )
    commands, outputs = {}, {}
    kernels = {}
    stats = wp.empty(args.queries, dtype=wp.vec3, device=device)
    wp.launch(
        diagnose, args.queries, [bvh.id, qlo_wp, qhi_wp, nodes, stats], device=device
    )
    stats_np = stats.numpy()
    for mode in args.modes.split(","):
        kernel = make_probe(mode)
        kernels[mode] = kernel
        out = [
            wp.empty(args.queries, dtype=dtype, device=device)
            for dtype in (int, wp.uint32, int)
        ]
        ids = wp.empty(1, dtype=int, device=device)
        commands[mode] = wp.launch(
            kernel,
            args.queries,
            [bvh.id, qlo_wp, qhi_wp, seeds_wp, nodes, *out, ids, 0],
            device=device,
            record_cmd=True,
        )
        commands[mode].launch()
        outputs[mode] = out
    root_counts, root_sums = [x.numpy() for x in outputs["root"][:2]]
    for out in outputs.values():
        np.testing.assert_array_equal(out[0].numpy(), root_counts)
        np.testing.assert_array_equal(out[1].numpy(), root_sums)
    # Compare complete primitive ID sets against independent NumPy overlap.
    sample = rng.choice(args.queries, min(32, args.queries), replace=False)
    validation_capacity = int(root_counts[sample].max()) + 1
    validation_inputs = [
        wp.array(x[sample], dtype=wp.vec3, device=device) for x in (qlo, qhi)
    ]
    validation_seeds = wp.array(seeds[sample], dtype=int, device=device)
    validation_nodes = wp.array(nodes.numpy()[sample], dtype=int, device=device)
    for mode, kernel in kernels.items():
        out = [
            wp.empty(len(sample), dtype=dtype, device=device)
            for dtype in (int, wp.uint32, int)
        ]
        ids = wp.empty(len(sample) * validation_capacity, dtype=int, device=device)
        wp.launch(
            kernel,
            len(sample),
            [
                bvh.id,
                *validation_inputs,
                validation_seeds,
                validation_nodes,
                *out,
                ids,
                validation_capacity,
            ],
            device=device,
        )
        actual_counts, actual_ids = out[0].numpy(), ids.numpy().reshape(len(sample), -1)
        for j, i in enumerate(sample):
            expected = np.flatnonzero(np.all((lo <= qhi[i]) & (hi >= qlo[i]), axis=1))
            np.testing.assert_array_equal(
                np.sort(actual_ids[j, : actual_counts[j]]), expected
            )
    print("CORRECT: all counts/checksums and sampled complete sets", flush=True)
    timings = benchmark(commands, args, device)
    record = {
        "version": VERSION,
        **source_record(),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "run_id": os.environ.get("EBVH_RUN_ID"),
        "timings": timings,
        "mean_hits": float(root_counts.mean()),
        "set_checks": len(sample) * len(kernels),
        "mean_depth": float(stats_np[:, 0].mean()),
        "stackless_fallback_fraction": float(stats_np[:, 2].mean()),
    }
    with args.output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
