"""Compare closest-hit ray prototypes using genuinely previous-query seeds."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import warp as wp
from closest_point import benchmark, make_mesh_data


def brute_ray(start, direction, max_t, vertices):
    """Independent float64 Moller-Trumbore nearest intersection, chunked."""
    best = float(max_t)
    found = False
    for begin in range(0, len(vertices), 300000):
        tri = vertices[begin:begin + 300000].reshape(-1, 3, 3).astype(np.float64)
        edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        det = np.einsum("ij,ij->i", edge1, p)
        inverse = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-15)
        translated = start - tri[:, 0]
        u = np.einsum("ij,ij->i", translated, p) * inverse
        q = np.cross(translated, edge1)
        v = q @ direction * inverse
        t = np.einsum("ij,ij->i", edge2, q) * inverse
        valid = (np.abs(det) > 1e-15) & (u >= 0) & (v >= 0) & (u+v <= 1) & (t >= 0) & (t < best)
        if valid.any():
            best = float(t[valid].min())
            found = True
    return found, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triangles", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=200000)
    parser.add_argument("--length", type=float, default=0.1)
    parser.add_argument("--seconds", type=float, default=1.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    wp.init()
    from warp.tests.geometry import test_mesh_ray_peeling as ray_tests
    wp.set_module_options({"enable_backward": False}, module=ray_tests)
    device = wp.get_device("cuda:0")
    rng = np.random.default_rng(314159)
    vertices, indices, _, own_seeds = make_mesh_data(args.triangles, args.queries, rng)
    centers = vertices.reshape(-1, 3, 3).mean(axis=1)
    starts = centers[own_seeds].copy()
    starts[:, 2] += args.length
    directions = np.tile(np.array((0, 0, -1), dtype=np.float32), (args.queries, 1))
    max_ts = np.full(args.queries, args.length * 1.5, dtype=np.float32)
    mesh = wp.Mesh(wp.array(vertices, dtype=wp.vec3, device=device), wp.array(indices, dtype=int, device=device),
                   bvh_constructor="sah", bvh_leaf_size=1, enable_exclusive=True)
    prior = ray_tests.run_query(ray_tests.query_rays_root, mesh, starts, directions, max_ts, own_seeds, None, device)
    # Pull the actual nearest face from a prior ray; own_seeds are not ground truth.
    # The test helper returns a tuple documented by its implementation.
    prior_seeds = prior[1].astype(np.int32)
    segment_nodes = ray_tests.find_nodes(mesh, starts, directions, max_ts, prior_seeds, device)
    endpoint_nodes = ray_tests.find_endpoint_nodes(mesh, starts, directions, max_ts, prior_seeds, device)
    starts[:, 0] += 0.0005
    start_wp = wp.array(starts, dtype=wp.vec3, device=device)
    direction_wp = wp.array(directions, dtype=wp.vec3, device=device)
    max_wp = wp.array(max_ts, dtype=float, device=device)
    seed_wp = wp.array(prior_seeds, dtype=int, device=device)
    segment_wp = wp.array(segment_nodes, dtype=int, device=device)
    endpoint_wp = wp.array(endpoint_nodes, dtype=int, device=device)
    common = [mesh.id, start_wp, direction_wp, max_wp]
    modes = {
        "root": (ray_tests.query_rays_root, common),
        "warm": (ray_tests.query_rays_seeded, [*common, seed_wp]),
        "walk": (ray_tests.query_rays_exclusive, [*common, seed_wp]),
        "cached": (ray_tests.query_rays_exclusive_cached, [*common, seed_wp, segment_wp]),
        "bottom_up": (ray_tests.query_rays_exclusive_cached_bottom_up, [*common, seed_wp, endpoint_wp]),
        "peeling": (ray_tests.query_rays_exclusive_cached_peeling, [*common, seed_wp, endpoint_wp]),
    }
    commands, outputs = {}, {}
    for name, (kernel, inputs) in modes.items():
        out = [wp.empty(args.queries, dtype=d, device=device) for d in (int, int, float, wp.vec2, float, wp.vec3)]
        commands[name] = wp.launch(kernel, args.queries, [*inputs, *out], device=device, record_cmd=True)
        commands[name].launch()
        outputs[name] = out
    root = [x.numpy() for x in outputs["root"]]
    for name, out in outputs.items():
        ray_tests.assert_query_equal(root, [x.numpy() for x in out])
    for i in rng.choice(args.queries, min(16, args.queries), replace=False):
        hit, t = brute_ray(starts[i].astype(np.float64), directions[i].astype(np.float64), max_ts[i], vertices)
        if bool(root[0][i]) != hit:
            raise AssertionError("Brute ray hit mismatch")
        if hit:
            np.testing.assert_allclose(root[2][i], t, rtol=2e-5, atol=2e-6)
    print("CORRECT: prior seeds, moved queries, all attributes, 16 brute checks", flush=True)
    timings = benchmark(commands, args, device)
    record = {"version": "ebvh-ray-prototypes-v1", "run_id": os.environ.get("EBVH_RUN_ID"),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "seed_kind": "actual prior-ray closest face before a 0.0005 x translation",
              "hit_fraction": float(np.mean(root[0])),
              "changed_face_fraction": float(np.mean(root[1] != prior_seeds)), "timings": timings}
    with args.output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
