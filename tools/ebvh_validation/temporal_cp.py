"""Validate previous-frame exact CP seeds and cache state on deforming geometry."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import warp as wp
from closest_point import benchmark, brute_distance, find_nodes, make_kernel, make_mesh_data, make_sheet


class PairCommand:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def launch(self):
        self.first.launch()
        self.second.launch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("scatter", "sheet"), default="scatter")
    parser.add_argument("--triangles", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=100000)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--constructor", choices=("sah", "lbvh"), default="sah")
    parser.add_argument("--leaf-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Use run_claimed.sh")
    wp.init()
    wp.set_module_options({"enable_backward": False, "fast_math": True})
    device = wp.get_device("cuda:0")
    rng = np.random.default_rng(8675309)
    generator = make_mesh_data if args.scene == "scatter" else make_sheet
    base_vertices, indices, base_points, seed_ids = generator(args.triangles, args.queries, rng)
    vertices = wp.array(base_vertices, dtype=wp.vec3, device=device)
    mesh = wp.Mesh(vertices, wp.array(indices, dtype=int, device=device), bvh_constructor=args.constructor,
                   bvh_leaf_size=args.leaf_size, enable_exclusive=True)
    points = wp.array(base_points, dtype=wp.vec3, device=device)
    seeds = wp.array(seed_ids, dtype=int, device=device)
    nodes = wp.empty(args.queries, dtype=int, device=device)
    new_nodes = wp.empty_like(nodes)
    depths = wp.empty_like(nodes)
    outputs = {name: (wp.empty_like(nodes), wp.empty(args.queries, dtype=wp.vec3, device=device))
               for name in ("root", "warm", "walk", "cached", "gated")}
    commands = {name: wp.launch(make_kernel(name), dim=args.queries,
                               inputs=[mesh.id, points, seeds, nodes, *out], device=device, record_cmd=True)
                for name, out in outputs.items()}
    initialize = wp.launch(find_nodes, dim=args.queries, inputs=[mesh.id, points, seeds, nodes, depths],
                           device=device, record_cmd=True)
    refresh = wp.launch(find_nodes, dim=args.queries,
                        inputs=[mesh.id, points, outputs["cached"][0], new_nodes, depths],
                        device=device, record_cmd=True)
    commands["root"].launch()
    wp.copy(seeds, outputs["root"][0])
    initialize.launch()
    for command in commands.values():
        command.launch()
    refresh.launch()
    wp.synchronize_device(device)
    # Include cache maintenance using HEAD's existing API, without pretending a
    # separately computed containment node is free in a temporal application.
    commands["cached_with_refresh"] = PairCommand(commands["cached"], refresh)
    frame_records = []
    for frame in range(1, args.frames + 1):
        phase = frame * 0.12
        moved_vertices = base_vertices.copy()
        moved_vertices[:, 2] += 0.025 * np.sin(0.7 * base_vertices[:, 0] + phase)
        moved_points = base_points.copy()
        moved_points[:, 2] += 0.025 * np.sin(0.7 * base_points[:, 0] + phase)
        moved_points[:, 0] += 0.006 * np.sin(phase)
        vertices.assign(moved_vertices)
        points.assign(moved_points)
        wp.synchronize_device(device)
        start = time.perf_counter()
        mesh.refit()
        wp.synchronize_device(device)
        refit_ms = (time.perf_counter() - start) * 1000.0
        for command in commands.values():
            command.launch()
        reference = np.linalg.norm(outputs["root"][1].numpy().astype(np.float64) - moved_points, axis=1)
        max_error = 0.0
        for out in outputs.values():
            distance = np.linalg.norm(out[1].numpy().astype(np.float64) - moved_points, axis=1)
            np.testing.assert_allclose(distance, reference, rtol=2e-5, atol=4e-6)
            max_error = max(max_error, float(np.max(np.abs(distance - reference))))
        if frame in (1, args.frames):
            for i in rng.choice(args.queries, 8, replace=False):
                expected = brute_distance(moved_points[i].astype(np.float64), moved_vertices, indices)
                np.testing.assert_allclose(reference[i], expected, rtol=2e-5, atol=4e-6)
        timing_args = argparse.Namespace(batch=50, seconds=1.5 / args.frames, repeats=1)
        timing = benchmark(commands, timing_args, device)[0]
        # The timed path writes separate next-state buffers; inputs remain fixed
        # during graph replay. Feed the actual last-frame result into the next frame.
        commands["cached"].launch()
        refresh.launch()
        frame_records.append({"frame": frame, "refit_ms": refit_ms, "max_error": max_error,
                              "changed_seed_fraction": float(np.mean(outputs["cached"][0].numpy() != seeds.numpy())),
                              "changed_cache_fraction": float(np.mean(new_nodes.numpy() != nodes.numpy())),
                              "depth_mean": float(np.mean(depths.numpy())), "timing": timing})
        wp.copy(seeds, outputs["cached"][0])
        wp.copy(nodes, new_nodes)
        print("FRAME", frame, "max_error", max_error, flush=True)
    record = {"version": "ebvh-temporal-cp-v1", "run_id": os.environ.get("EBVH_RUN_ID"),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cache_kind": "previous-frame exact result, full leaf-walk refresh included in cached_with_refresh",
              "frames": frame_records}
    with args.output.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
