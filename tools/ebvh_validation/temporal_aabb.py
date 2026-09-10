"""Time complete temporal AABB state updates across deforming frames."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from aabb_probe import diagnose, initialize_nodes, make_aabb_data, make_probe
from closest_point import benchmark, make_sheet
from provenance import source_record
from temporal_cp import PairCommand

import warp as wp

VERSION = "ebvh-temporal-aabb-v4"
print(VERSION, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("scatter", "sheet"), default="scatter")
    parser.add_argument("--primitives", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    wp.init()
    device = wp.get_device("cuda:0")
    rng = np.random.default_rng(54321)
    if args.scene == "scatter":
        lo, hi, qlo, qhi, seeds = make_aabb_data(args.primitives, args.queries, rng)
        vertices, indices = None, None
    else:
        vertices, indices, queries, seeds = make_sheet(
            args.primitives, args.queries, rng
        )
        triangles = vertices[indices].reshape(-1, 3, 3)
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        radius = 10.0 / (round(np.sqrt(len(lo) / 2)))
        qlo, qhi = queries - radius, queries + radius
    lowers, uppers = [wp.array(x, dtype=wp.vec3, device=device) for x in (lo, hi)]
    bvh = wp.Bvh(lowers, uppers, constructor="sah", leaf_size=1, enable_exclusive=True)
    ordinary_bvh = wp.Bvh(lowers, uppers, constructor="sah", leaf_size=1)
    query_lowers, query_uppers = [
        wp.array(x, dtype=wp.vec3, device=device) for x in (qlo, qhi)
    ]
    seed_wp = wp.array(seeds, dtype=int, device=device)
    caches = {
        name: wp.empty(args.queries, dtype=int, device=device)
        for name in ("legacy", "fused", "periodic")
    }
    for cache in caches.values():
        wp.launch(
            initialize_nodes,
            args.queries,
            [bvh.id, query_lowers, query_uppers, seed_wp, cache],
            device=device,
        )
    arrays = {
        name: [
            wp.empty(args.queries, dtype=d, device=device)
            for d in (int, wp.uint32, int)
        ]
        for name in ("root", "ordinary_root", "cached_query", "fused", "periodic")
    }
    ids = wp.empty(1, dtype=int, device=device)
    diagnostics = wp.empty(args.queries, dtype=wp.vec3, device=device)

    def command(mode, cache_name, output_name, tree=bvh):
        return wp.launch(
            make_probe(mode),
            args.queries,
            [
                tree.id,
                query_lowers,
                query_uppers,
                seed_wp,
                caches[cache_name],
                *arrays[output_name],
                ids,
                0,
            ],
            device=device,
            record_cmd=True,
        )

    root = command("root", "legacy", "root")
    ordinary_root = command("root", "legacy", "ordinary_root", ordinary_bvh)
    cached = command("cached", "legacy", "cached_query")
    fused = command("fused", "fused", "fused")
    periodic_plain = command("fused", "periodic", "periodic")
    periodic_refine = command("fused_refine", "periodic", "periodic")
    next_legacy = wp.empty(args.queries, dtype=int, device=device)
    refresh = wp.launch(
        initialize_nodes,
        args.queries,
        [bvh.id, query_lowers, query_uppers, seed_wp, next_legacy],
        device=device,
        record_cmd=True,
    )
    legacy_full = PairCommand(cached, refresh)
    for cmd in (
        root,
        ordinary_root,
        cached,
        fused,
        periodic_plain,
        periodic_refine,
        legacy_full,
    ):
        cmd.launch()
    wp.synchronize_device(device)
    records = []
    for frame in range(1, args.frames + 1):
        phase = frame * 0.15
        if vertices is None:
            movement = np.zeros_like(lo)
            movement[:, 2] = 0.03 * np.sin(0.7 * ((lo[:, 0] + hi[:, 0]) * 0.5) + phase)
            current_lo, current_hi = lo + movement, hi + movement
        else:
            moved = vertices.copy()
            moved[:, 2] += 0.03 * np.sin(0.7 * vertices[:, 0] + phase)
            tri = moved[indices].reshape(-1, 3, 3)
            current_lo, current_hi = tri.min(axis=1), tri.max(axis=1)
        query_motion = np.zeros_like(qlo)
        query_motion[:, 2] = 0.03 * np.sin(
            0.7 * ((qlo[:, 0] + qhi[:, 0]) * 0.5) + phase
        )
        query_motion[:, 0] = 0.008 * np.sin(phase)
        current_qlo, current_qhi = qlo + query_motion, qhi + query_motion
        lowers.assign(current_lo)
        uppers.assign(current_hi)
        query_lowers.assign(current_qlo)
        query_uppers.assign(current_qhi)
        wp.synchronize_device(device)
        refit_times = {}
        trees = [("exclusive", bvh), ("ordinary", ordinary_bvh)]
        for name, tree in trees if frame % 2 == 0 else reversed(trees):
            start = time.perf_counter()
            tree.refit()
            wp.synchronize_device(device)
            refit_times[name] = (time.perf_counter() - start) * 1000.0
        periodic = periodic_refine if frame % 8 == 0 else periodic_plain
        commands = {
            "root": root,
            "ordinary_root": ordinary_root,
            "cached_query": cached,
            "cached_with_refresh": legacy_full,
            "fused": fused,
            "fused_periodic": periodic,
        }
        for cmd in commands.values():
            cmd.launch()
        wp.launch(
            diagnose,
            args.queries,
            [bvh.id, query_lowers, query_uppers, caches["legacy"], diagnostics],
            device=device,
        )
        diagnostic_values = diagnostics.numpy()
        reference = [x.numpy() for x in arrays["root"][:2]]
        for out in arrays.values():
            np.testing.assert_array_equal(out[0].numpy(), reference[0])
            np.testing.assert_array_equal(out[1].numpy(), reference[1])
        if frame in (1, args.frames):
            for i in rng.choice(args.queries, min(16, args.queries), replace=False):
                expected = np.flatnonzero(
                    np.all(
                        (current_lo <= current_qhi[i]) & (current_hi >= current_qlo[i]),
                        axis=1,
                    )
                )
                if reference[0][i] != len(expected) or reference[1][i] != np.uint32(
                    expected.sum() % 2**32
                ):
                    raise AssertionError("Brute AABB reference mismatch")
        timings = benchmark(
            commands,
            argparse.Namespace(
                batch=50,
                seconds=1.5 / args.frames,
                repeats=1,
                burn_seconds=2.0 if frame == 1 else 0.05,
            ),
            device,
        )[0]
        records.append(
            {
                "frame": frame,
                "refit_ms": refit_times["exclusive"],
                "ordinary_refit_ms": refit_times["ordinary"],
                "cached_stackless_fraction": float(diagnostic_values[:, 2].mean()),
                "cached_mean_depth": float(diagnostic_values[:, 0].mean()),
                "mean_hits": float(reference[0].mean()),
                "cache_changed_fraction": float(
                    np.mean(arrays["fused"][2].numpy() != caches["fused"].numpy())
                ),
                "timing": timings,
            }
        )
        # Ping-pong state: timed replays do not feed same-frame outputs back in.
        wp.copy(caches["legacy"], next_legacy)
        wp.copy(caches["fused"], arrays["fused"][2])
        wp.copy(caches["periodic"], arrays["periodic"][2])
        print("FRAME", frame, flush=True)
    record = {
        "version": VERSION,
        **source_record(),
        "run_id": os.environ.get("EBVH_RUN_ID"),
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "frames": records,
    }
    with args.output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
