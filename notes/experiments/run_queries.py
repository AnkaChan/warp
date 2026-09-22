"""Isolated, deterministic ASV scene probe. Run with uv; see --help."""

# Imports must follow source selection so every process uses the requested checkout.
# ruff: noqa: PLC0415

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warp-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["point", "ray", "aabb"])
    parser.add_argument("--assets", nargs="+", default=["bunny", "rocks"])
    parser.add_argument("--constructors", nargs="+", default=["lbvh", "cubql"])
    parser.add_argument("--leaves", nargs="+", type=int, default=[0, 8])
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    root = args.warp_root.resolve()
    task_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(task_root / "asv"))
    os.environ["WARP_CACHE_PATH"] = str(args.cache.resolve())
    import numpy as np
    from benchmarks import spatial_query as spatial
    from benchmarks.mesh_ray_variants import get_mesh_ray_kernel

    import warp as wp

    if Path(wp.__file__).resolve().parent != root / "warp":
        raise RuntimeError(f"Imported the wrong Warp checkout: {wp.__file__}")
    wp.init()
    device = wp.get_device("cuda:0")
    wp.set_device(device)

    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(root), *cmd], text=True).strip()

    metadata = {
        "kind": "provenance",
        "warp_file": wp.__file__,
        "version": wp.__version__,
        "commit": git("rev-parse", "HEAD"),
        "diff": git("diff", "HEAD", "--", "warp/native"),
        "mesh_sha256": digest((root / "warp/native/mesh.h").read_bytes()),
        "bvh_sha256": digest((root / "warp/native/bvh.h").read_bytes()),
        "native_sha256": digest((root / "warp/bin/warp.so").read_bytes()),
        "harness_sha256": digest(Path(__file__).read_bytes()),
        "ray_benchmark_sha256": digest((task_root / "asv/benchmarks/mesh_ray_variants.py").read_bytes()),
        "cache": wp.config.kernel_cache_dir,
        "device": device.name,
        "arch": device.arch,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "block_size": 256,
        "seed": 42,
        "samples": args.samples,
        "warmup": args.warmup,
        "telemetry": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:

        def emit(record):
            print(json.dumps(record), flush=True)
            out.write(json.dumps(record) + "\n")
            out.flush()

        emit(metadata)
        start = wp.Event(device, enable_timing=True)
        end = wp.Event(device, enable_timing=True)

        def measure(case, graph, arrays, launches=10):
            for _ in range(args.warmup):
                wp.capture_launch(graph)
            wp.synchronize_device(device)
            values = []
            for _ in range(args.samples):
                wp.record_event(start)
                wp.capture_launch(graph)
                wp.record_event(end)
                values.append(wp.get_event_elapsed_time(start, end) / launches)
            snapshots = [array.numpy() for array in arrays]
            emit(
                {
                    "kind": "measurement",
                    "case": case,
                    "ms": values,
                    "median_ms": statistics.median(values),
                    "sha256": [digest(array.tobytes()) for array in snapshots],
                    "sum": [float(np.sum(array, dtype=np.float64)) for array in snapshots],
                }
            )
            return snapshots

        for constructor in args.constructors:
            for leaf in args.leaves:
                if "point" in args.families:
                    for asset in args.assets:
                        spatial.seed = 42
                        scene = spatial.MeshQuery()
                        scene.setup(leaf, asset, constructor)
                        for variant, cmd in [("unsigned", scene.cmd_no_sign), ("signed", scene.cmd_signed)]:
                            scene.query_closest_points.zero_()
                            with wp.ScopedCapture(device=device) as capture:
                                for _ in range(10):
                                    cmd.launch()
                            measure(
                                ["point", asset, constructor, leaf, variant],
                                capture.graph,
                                [scene.query_closest_points],
                            )
                if "ray" in args.families:
                    spatial.seed = 42
                    scene = spatial.BvhRayQuery()
                    scene.setup(args.resolution, leaf, "cuda", constructor)
                    reference = None
                    for variant in ["closest", "any", "ordered", "count", "sign"]:
                        kernel = get_mesh_ray_kernel(variant)
                        inputs = [
                            scene.mesh.id,
                            scene.camera,
                            scene.mesh_pos,
                            scene.mesh_rot,
                            args.resolution,
                            args.resolution,
                        ]
                        wp.launch(
                            kernel, scene.num_rays, inputs=inputs, outputs=[scene.rays], device=device, block_dim=256
                        )
                        with wp.ScopedCapture(device=device) as capture:
                            for _ in range(10):
                                wp.launch(
                                    kernel,
                                    scene.num_rays,
                                    inputs=inputs,
                                    outputs=[scene.rays],
                                    device=device,
                                    block_dim=256,
                                )
                        data = measure(
                            ["ray", "bunny10", constructor, leaf, variant, args.resolution], capture.graph, [scene.rays]
                        )[0][:, 0]
                        if variant == "closest":
                            reference = data
                        elif variant in ("any", "ordered"):
                            np.testing.assert_array_equal(data, reference)
                        else:
                            np.testing.assert_array_equal(data != 0, reference != 0)
                if "aabb" in args.families:
                    spatial.seed = 42
                    scene = spatial.BvhAABBQuery()
                    scene.setup(0.008, leaf, "cuda", constructor)
                    for variant, graph in [
                        ("bvh", scene.cuda_graph_bvh_aabb_vs_aabb),
                        ("mesh", scene.cuda_graph_mesh_aabb_vs_aabb),
                    ]:
                        measure(
                            ["aabb", "bunny10", constructor, leaf, variant, 0.008],
                            graph,
                            [scene.vertex_colliding_triangles_count],
                        )


if __name__ == "__main__":
    main()
