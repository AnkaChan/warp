"""Validate exact triangle CP arms with paired CUDA graph timing.

Run through run_claimed.sh. Oracle starts are an explicitly labeled upper bound.
No result from this reconstructed workload is a bit-for-bit P08 reproduction.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import warp as wp
from warp.examples.benchmarks.benchmark_bvh_queries import _morton_order, make_mesh_data

VERSION = "ebvh-exact-cp-v5"
ARMS = ("root", "warm", "walk", "cached", "gated", "oracle")
print(f"[EBVH] {VERSION}", flush=True)


@wp.func_native("""
    const wp::Mesh mesh = wp::mesh_get(id);
    if (node < 0 || node >= mesh.bvh.num_nodes) return -1;
    return wp::bvh_exclusive_node_depth(wp::bvh_get_exclusive_node(mesh.bvh, node));
""")
def node_depth(id: wp.uint64, node: int) -> int: ...


@wp.func_native("""
    const wp::Mesh mesh = wp::mesh_get(id);
    float d2 = 1.0e12f;
    int face = -1;
    float v = 0.0f, w = 0.0f;
    int leaf = wp::mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed, point, d2, face, v, w);
    // Benchmark-only oracle: node was certified on these exact inputs outside timing.
    if (node != leaf)
        wp::mesh_query_point_no_sign_traverse<false, true>(mesh, node, leaf, point, d2, face, v, w);
    return wp::vec3(static_cast<float>(face), 1.0f-v-w, v);
""")
def oracle_query(id: wp.uint64, point: wp.vec3, seed: int, node: int) -> wp.vec3: ...


@wp.kernel
def find_nodes(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    seeds: wp.array(dtype=int),
    nodes: wp.array(dtype=int),
    depths: wp.array(dtype=int),
):
    i = wp.tid()
    node = wp.mesh_query_point_no_sign_exclusive_node(mesh, points[i], 1.0e6, seeds[i])
    nodes[i] = node
    depths[i] = node_depth(mesh, node)


def make_kernel(mode):
    @wp.kernel(module="unique", module_options={"enable_backward": False, "fast_math": True})
    def query(
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        seeds: wp.array(dtype=int),
        nodes: wp.array(dtype=int),
        faces: wp.array(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        face = int(-1)
        u = float(0.0)
        v = float(0.0)
        if wp.static(mode == "root"):
            result = wp.mesh_query_point_no_sign(mesh, points[i], 1.0e6)
            face, u, v = int(result.face), float(result.u), float(result.v)
        elif wp.static(mode == "warm"):
            result = wp.mesh_query_point_no_sign_seeded(mesh, points[i], 1.0e6, seeds[i])
            face, u, v = int(result.face), float(result.u), float(result.v)
        elif wp.static(mode == "walk"):
            result = wp.mesh_query_point_no_sign_exclusive(mesh, points[i], 1.0e6, seeds[i])
            face, u, v = int(result.face), float(result.u), float(result.v)
        elif wp.static(mode == "cached"):
            result = wp.mesh_query_point_no_sign_exclusive_cached(mesh, points[i], 1.0e6, seeds[i], nodes[i])
            face, u, v = int(result.face), float(result.u), float(result.v)
        elif wp.static(mode == "gated"):
            if node_depth(mesh, nodes[i]) >= 8:
                result = wp.mesh_query_point_no_sign_exclusive_cached(mesh, points[i], 1.0e6, seeds[i], nodes[i])
                face, u, v = int(result.face), float(result.u), float(result.v)
            else:
                result = wp.mesh_query_point_no_sign_seeded(mesh, points[i], 1.0e6, seeds[i])
                face, u, v = int(result.face), float(result.u), float(result.v)
        elif wp.static(mode == "oracle"):
            oracle = oracle_query(mesh, points[i], seeds[i], nodes[i])
            face, u, v = int(oracle[0]), oracle[1], oracle[2]
        faces[i] = face
        positions[i] = wp.mesh_eval_position(mesh, face, u, v)

    return query


def make_sheet(n, nq, rng):
    side = int(round(np.sqrt(n / 2.0))) + 1
    axis = np.linspace(-10, 10, side, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    z = 0.25 * np.sin(0.7 * x) * np.cos(0.6 * y)
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    a = (np.arange(side - 1)[:, None] * side + np.arange(side - 1)[None, :]).ravel()
    triangles = np.stack(
        (
            np.stack((a, a + side, a + 1), axis=1),
            np.stack((a + side, a + side + 1, a + 1), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3)
    seeds = rng.integers(0, len(triangles), nq, dtype=np.int32)
    queries = vertices[triangles[seeds]].mean(axis=1)
    queries += rng.uniform(-0.3, 0.3, (nq, 3)).astype(np.float32) * (20.0 / (side - 1))
    order = _morton_order(queries)
    return vertices, triangles.astype(np.int32).ravel(), queries[order], seeds[order]


def brute_distance(point, vertices, indices):
    """Float64 projection onto triangle interiors and all three edge segments."""
    best = float("inf")
    for start in range(0, len(indices), 300_000):
        tri = vertices[indices[start : start + 300_000]].reshape(-1, 3, 3).astype(np.float64)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac, ap = b - a, c - a, point - a
        dot = lambda x, y: np.einsum("ij,ij->i", x, y)
        aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        pa, pc = dot(ap, ab), dot(ap, ac)
        determinant = aa * cc - bb * bb
        u = np.divide(pa * cc - pc * bb, determinant, out=np.zeros_like(pa), where=determinant > 0)
        v = np.divide(pc * aa - pa * bb, determinant, out=np.zeros_like(pc), where=determinant > 0)
        interior = (determinant > 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
        delta = ap - u[:, None] * ab - v[:, None] * ac
        d2 = np.where(interior, dot(delta, delta), np.inf)
        for edge_start, edge_end in ((a, b), (b, c), (c, a)):
            edge = edge_end - edge_start
            denom = dot(edge, edge)
            t = np.clip(
                np.divide(
                    dot(point - edge_start, edge),
                    denom,
                    out=np.zeros_like(denom),
                    where=denom > 0,
                ),
                0,
                1,
            )
            delta = point - edge_start - t[:, None] * edge
            d2 = np.minimum(d2, dot(delta, delta))
        best = min(best, float(d2.min()))
    return np.sqrt(best)


def timed_graph(graph, batch, device):
    start = time.perf_counter()
    wp.capture_launch(graph)
    wp.synchronize_device(device)
    return (time.perf_counter() - start) * 1000.0 / batch


def benchmark(commands, args, device):
    graphs = {}
    for name, command in commands.items():
        command.launch()
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(args.batch):
                command.launch()
        graphs[name] = capture.graph
    wp.synchronize_device(device)
    names = list(graphs)
    burn_start = time.perf_counter()
    while time.perf_counter() - burn_start < getattr(args, "burn_seconds", 2.0):
        for name in names:
            wp.capture_launch(graphs[name])
        wp.synchronize_device(device)
    repeats = []
    for repeat in range(args.repeats):
        samples = {name: [] for name in names}
        elapsed = dict.fromkeys(names, 0.0)
        cycle = 0
        while min(elapsed.values()) < args.seconds:
            offset = (cycle + repeat) % len(names)
            order = names[offset:] + names[:offset]
            if (cycle + repeat) % 2:
                order.reverse()
            for name in order:
                ms = timed_graph(graphs[name], args.batch, device)
                samples[name].append(ms)
                elapsed[name] += ms * args.batch * 0.001
            cycle += 1
        means = {name: float(np.mean(samples[name])) for name in names}
        record = {
            name: {
                "mean_ms": means[name],
                "std_ms": float(np.std(samples[name])),
                "median_ms": float(np.median(samples[name])),
                "samples": len(samples[name]),
                "measured_seconds": elapsed[name],
                "speedup": means["root"] / means[name],
                "samples_ms": samples[name],
            }
            for name in names
        }
        repeats.append(record)
        print(
            "TIMING",
            repeat,
            {k: round(v["speedup"], 3) for k, v in record.items()},
            flush=True,
        )
    return repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("scatter", "sheet"), default="scatter")
    parser.add_argument("--triangles", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000000)
    parser.add_argument("--constructor", choices=("sah", "lbvh", "median"), default="sah")
    parser.add_argument("--leaf-size", type=int, default=1)
    parser.add_argument("--block-dim", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--brute-samples", type=int, default=32)
    parser.add_argument("--unsorted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Run through run_claimed.sh to claim a GPU")
    wp.init()
    device = wp.get_device("cuda:0")
    wp.set_module_options({"enable_backward": False, "fast_math": True})
    rng = np.random.default_rng(args.seed)
    generator = make_mesh_data if args.scene == "scatter" else make_sheet
    vertices, indices, points, seeds = generator(args.triangles, args.queries, rng)
    if args.unsorted:
        order = rng.permutation(args.queries)
        points, seeds = points[order], seeds[order]
    vertices_wp = wp.array(vertices, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices, dtype=int, device=device)
    start = time.perf_counter()
    mesh = wp.Mesh(
        vertices_wp,
        indices_wp,
        bvh_constructor=args.constructor,
        bvh_leaf_size=args.leaf_size,
        enable_exclusive=True,
    )
    wp.synchronize_device(device)
    build_ms = (time.perf_counter() - start) * 1000
    points_wp = wp.array(points, dtype=wp.vec3, device=device)
    seeds_wp = wp.array(seeds, dtype=int, device=device)
    nodes = wp.empty(args.queries, dtype=int, device=device)
    depths = wp.empty_like(nodes)
    init_command = wp.launch(
        find_nodes,
        dim=args.queries,
        inputs=[mesh.id, points_wp, seeds_wp, nodes, depths],
        device=device,
        block_dim=args.block_dim,
        record_cmd=True,
    )
    init_command.launch()
    depth_np = depths.numpy()
    arms = ARMS
    outputs = {}
    commands = {}
    for name in arms:
        faces = wp.empty(args.queries, dtype=int, device=device)
        positions = wp.empty(args.queries, dtype=wp.vec3, device=device)
        kernel = make_kernel(name)
        commands[name] = wp.launch(
            kernel,
            dim=args.queries,
            inputs=[mesh.id, points_wp, seeds_wp, nodes, faces, positions],
            device=device,
            block_dim=args.block_dim,
            record_cmd=True,
        )
        commands[name].launch()
        outputs[name] = (faces, positions)
    root_dist = np.linalg.norm(outputs["root"][1].numpy().astype(np.float64) - points, axis=1)
    root_faces = outputs["root"][0].numpy()
    correctness = {}
    for name in arms:
        faces = outputs[name][0].numpy()
        positions = outputs[name][1].numpy()
        distance = np.linalg.norm(positions.astype(np.float64) - points, axis=1)
        np.testing.assert_allclose(distance, root_dist, rtol=2e-5, atol=4e-6)
        if not np.all((faces >= 0) & (faces < len(indices) // 3)):
            raise AssertionError(f"Invalid face returned by {name}")
        correctness[name] = {
            "max_distance_error": float(np.max(np.abs(distance - root_dist))),
            "face_differences": int(np.count_nonzero(faces != root_faces)),
        }
    sample_ids = rng.choice(args.queries, min(args.brute_samples, args.queries), replace=False)
    brute = np.array([brute_distance(points[i].astype(np.float64), vertices, indices) for i in sample_ids])
    np.testing.assert_allclose(root_dist[sample_ids], brute, rtol=2e-5, atol=4e-6)
    print("CORRECT", correctness, "brute_samples", len(sample_ids), flush=True)
    timings = benchmark(commands, args, device)
    init_command.launch()
    with wp.ScopedCapture(device=device) as capture:
        for _ in range(args.batch):
            init_command.launch()
    initialization_ms = [timed_graph(capture.graph, args.batch, device) for _ in range(10)]
    record = {
        "version": VERSION,
        "run_id": os.environ.get("EBVH_RUN_ID"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_sha256": hashlib.sha256(Path("warp/bin/warp.so").read_bytes()).hexdigest(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "actual_triangles": len(indices) // 3,
        "gpu": device.name,
        "architecture": device.arch,
        "clock_locked": os.environ.get("EBVH_CLOCK_LOCKED") == "1",
        "requested_clock_mhz": os.environ.get("EBVH_CLOCK_MHZ", "2490"),
        "build_ms": build_ms,
        "cache_initialization_ms": initialization_ms,
        "cache_kind": "same-query precomputed, revalidated except oracle",
        "depth_mean": float(np.mean(depth_np)),
        "depth_ge8_fraction": float(np.mean(depth_np >= 8)),
        "correctness": correctness,
        "brute_samples": len(sample_ids),
        "brute_max_error": float(np.max(np.abs(root_dist[sample_ids] - brute))) if len(sample_ids) else None,
        "timings": timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print("RESULT", args.output, flush=True)


if __name__ == "__main__":
    main()
