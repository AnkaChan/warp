"""Benchmark comparing Warp SDF vs pytorch_volumetric SDF.

Uses the same mesh assets and query strategy as Warp's ASV benchmark
(asv/benchmarks/spatial_query.py): bunny, bear, rocks meshes with
1M random query points sampled within the mesh bounding box.

Experiments:
  1. Direct mesh query timing (signed & unsigned distance)
  2. Accuracy comparison (SDF values between libraries)
  3. Scaling with number of query points
  4. CachedSDF (voxel grid) vs Warp direct query
  5. SDF cross-section visualization
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WARP_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(WARP_ROOT))

# ---------------------------------------------------------------------------
# Imports - Warp
# ---------------------------------------------------------------------------
import warp as wp

wp.init()
wp.set_module_options({"enable_backward": False})

import warp.examples

# ---------------------------------------------------------------------------
# Imports - PyTorch / pytorch_volumetric
# ---------------------------------------------------------------------------
import torch
import open3d as o3d
from pytorch_volumetric import MeshSDF, CachedSDF, MeshObjectFactory, ObjectFactory, SDFQuery

# ---------------------------------------------------------------------------
# Constants (matching Warp's ASV benchmark)
# ---------------------------------------------------------------------------
ASSETS = ["bunny", "bear", "rocks"]
SEED = 42
NUM_QUERY_POINTS_DEFAULT = 1_000_000
QUERY_POINT_SWEEP = [1_000, 10_000, 100_000, 1_000_000]
WARMUP_ITERS = 2
BENCH_ITERS = 5
CACHED_SDF_RESOLUTION = 0.01
PV_MESHSDF_MAX_PTS = 10_000  # PV MeshSDF is CPU-only via Open3D, cap query count
PV_MESHSDF_ITERS = 3


# ===================================================================
# Warp kernels (same as spatial_query.py)
# ===================================================================
@wp.kernel
def warp_mesh_query_no_sign(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    out_dist: wp.array(dtype=wp.float32),
    out_face: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, query_d_max)
    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out_dist[tid] = wp.length(p - cp)
        out_face[tid] = query.face
    else:
        out_dist[tid] = -1.0
        out_face[tid] = -1


@wp.kernel
def warp_mesh_query_signed(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    out_dist: wp.array(dtype=wp.float32),
    out_face: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point(mesh, p, query_d_max)
    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out_dist[tid] = query.sign * wp.length(p - cp)
        out_face[tid] = query.face
    else:
        out_dist[tid] = -1.0
        out_face[tid] = -1


# ===================================================================
# Helpers
# ===================================================================
def load_usd_mesh(asset_name):
    """Load mesh from USD file, return (points_np, indices_np)."""
    from pxr import Usd, UsdGeom

    asset_dir = warp.examples.get_asset_directory()
    stage = Usd.Stage.Open(os.path.join(asset_dir, f"{asset_name}.usd"))
    mesh_geom = UsdGeom.Mesh(stage.GetPrimAtPath(f"/root/{asset_name}"))
    points = np.array(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return points, indices


def make_query_points(bounding_box, n_points, seed):
    """Generate random query points within the bounding box."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(bounding_box[0], bounding_box[1], size=(n_points, 3)).astype(np.float32)
    return pts


def make_o3d_mesh(points_np, indices_np):
    """Create Open3D TriangleMesh from numpy arrays."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(indices_np.reshape(-1, 3).astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


class DirectMeshFactory(ObjectFactory):
    """Minimal ObjectFactory subclass that takes vertices/indices directly."""

    def __init__(self, points_np, indices_np, **kwargs):
        o3d_mesh = make_o3d_mesh(points_np, indices_np)
        super().__init__(name="direct", mesh=o3d_mesh, **kwargs)

    def make_collision_obj(self, z, rgba=None):
        return None, None

    def get_mesh_resource_filename(self):
        return ""


# ===================================================================
# Benchmark runners
# ===================================================================
def bench_warp_query(mesh_wp, query_pts_wp, n_pts, device, signed=True, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time Warp mesh_query_point on given device. Returns (mean_ms, std_ms, dist_np)."""
    out_dist = wp.zeros(n_pts, dtype=wp.float32, device=device)
    out_face = wp.zeros(n_pts, dtype=wp.int32, device=device)
    kernel = warp_mesh_query_signed if signed else warp_mesh_query_no_sign

    # Warmup
    for _ in range(warmup):
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, query_pts_wp, 1e7, out_dist, out_face], device=device)
    wp.synchronize_device(device)

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, query_pts_wp, 1e7, out_dist, out_face], device=device)
        wp.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times)), out_dist.numpy()


def bench_pv_mesh_sdf(factory, query_pts_np, device_str="cpu", warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time pytorch_volumetric MeshSDF query. Returns (mean_ms, std_ms, dist_np)."""
    pts_tensor = torch.tensor(query_pts_np, dtype=torch.float32, device=device_str)
    mesh_sdf = MeshSDF(factory)

    # Warmup
    for _ in range(warmup):
        dist, grad = mesh_sdf(pts_tensor.unsqueeze(0))
    if device_str != "cpu":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dist, grad = mesh_sdf(pts_tensor.unsqueeze(0))
        if device_str != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times)), float(np.std(times)), dist.squeeze(0).detach().cpu().numpy()


def bench_pv_cached_sdf(factory, query_pts_np, resolution, device_str="cpu",
                        warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Time pytorch_volumetric CachedSDF. Returns (construct_ms, mean_query_ms, std_query_ms, dist_np)."""
    mesh_sdf = MeshSDF(factory)
    bb = factory.bounding_box(padding=0.1)

    # Construction timing
    t0 = time.perf_counter()
    cached = CachedSDF(
        "bench", resolution, bb, mesh_sdf,
        device=device_str, clean_cache=True,
        cache_path=str(RESULTS_DIR / "_tmp_cache.pkl"),
    )
    construct_ms = (time.perf_counter() - t0) * 1000.0

    pts_tensor = torch.tensor(query_pts_np, dtype=torch.float32, device=device_str)

    # Warmup
    for _ in range(warmup):
        dist, grad = cached(pts_tensor.unsqueeze(0))
    if device_str != "cpu":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dist, grad = cached(pts_tensor.unsqueeze(0))
        if device_str != "cpu":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    return construct_ms, float(np.mean(times)), float(np.std(times)), dist.squeeze(0).detach().cpu().numpy()


# ===================================================================
# Experiments
# ===================================================================
def experiment_1_timing(results):
    """Direct mesh query timing across assets, signed & unsigned."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Direct Mesh Query Timing")
    print("=" * 80)

    cuda_dev = "cuda:0"
    n_pts = NUM_QUERY_POINTS_DEFAULT

    for asset in ASSETS:
        print(f"\n--- {asset} ---")
        points_np, indices_np = load_usd_mesh(asset)
        bb = np.array([points_np.min(axis=0), points_np.max(axis=0)])
        query_pts_np = make_query_points(bb, n_pts, SEED)
        n_tris = len(indices_np) // 3

        print(f"  Vertices: {len(points_np)}, Triangles: {n_tris}, Query points: {n_pts}")

        # Warp GPU
        mesh_wp_gpu = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3, device=cuda_dev),
            indices=wp.array(indices_np, dtype=int, device=cuda_dev),
        )
        qpts_wp_gpu = wp.array(query_pts_np, dtype=wp.vec3, device=cuda_dev)

        # Warp CPU
        mesh_wp_cpu = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3, device="cpu"),
            indices=wp.array(indices_np, dtype=int, device="cpu"),
        )
        qpts_wp_cpu = wp.array(query_pts_np, dtype=wp.vec3, device="cpu")

        # pytorch_volumetric factory
        factory = DirectMeshFactory(points_np, indices_np)

        for signed in [False, True]:
            label = "signed" if signed else "unsigned"
            print(f"\n  [{label} distance]")

            # Warp GPU
            mean_ms, std_ms, warp_gpu_dist = bench_warp_query(
                mesh_wp_gpu, qpts_wp_gpu, n_pts, cuda_dev, signed=signed
            )
            print(f"    Warp GPU:           {mean_ms:10.2f} ms  (std {std_ms:.2f})")
            results[f"exp1_{asset}_{label}_warp_gpu_ms"] = mean_ms
            results[f"exp1_{asset}_{label}_warp_gpu_std"] = std_ms

            # Warp CPU
            mean_ms, std_ms, warp_cpu_dist = bench_warp_query(
                mesh_wp_cpu, qpts_wp_cpu, n_pts, "cpu", signed=signed, warmup=1, iters=2
            )
            print(f"    Warp CPU:           {mean_ms:10.2f} ms  (std {std_ms:.2f})")
            results[f"exp1_{asset}_{label}_warp_cpu_ms"] = mean_ms
            results[f"exp1_{asset}_{label}_warp_cpu_std"] = std_ms

            # pytorch_volumetric MeshSDF (always CPU internally via Open3D)
            pv_n = min(n_pts, PV_MESHSDF_MAX_PTS)
            pv_query = query_pts_np[:pv_n]
            mean_ms, std_ms, pv_dist = bench_pv_mesh_sdf(
                factory, pv_query, device_str="cpu", warmup=1, iters=PV_MESHSDF_ITERS
            )
            scaled_ms = mean_ms * (n_pts / pv_n)
            print(f"    PV MeshSDF (CPU):   {mean_ms:10.2f} ms  ({pv_n} pts, ~{scaled_ms:.0f} ms at {n_pts})")
            results[f"exp1_{asset}_{label}_pv_meshsdf_ms"] = mean_ms
            results[f"exp1_{asset}_{label}_pv_meshsdf_std"] = std_ms
            results[f"exp1_{asset}_{label}_pv_meshsdf_npts"] = pv_n

            # Accuracy comparison (on shared subset)
            if signed:
                # Compare Warp GPU vs PV on the shared subset
                warp_subset = warp_gpu_dist[:pv_n]
                mae = float(np.mean(np.abs(warp_subset - pv_dist)))
                sign_agree = float(np.mean(np.sign(warp_subset) == np.sign(pv_dist)))
                print(f"    Accuracy (Warp GPU vs PV): MAE={mae:.6f}, sign agreement={sign_agree:.4f}")
                results[f"exp1_{asset}_accuracy_mae"] = mae
                results[f"exp1_{asset}_accuracy_sign_agree"] = sign_agree

        del mesh_wp_gpu, mesh_wp_cpu, qpts_wp_gpu, qpts_wp_cpu


def experiment_2_scaling(results):
    """Scaling with number of query points."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Scaling with Query Point Count")
    print("=" * 80)

    asset = "bunny"
    cuda_dev = "cuda:0"
    points_np, indices_np = load_usd_mesh(asset)
    bb = np.array([points_np.min(axis=0), points_np.max(axis=0)])

    factory = DirectMeshFactory(points_np, indices_np)

    for n_pts in QUERY_POINT_SWEEP:
        query_pts_np = make_query_points(bb, n_pts, SEED)
        print(f"\n  N={n_pts:>10,}")

        # Warp GPU signed
        mesh_wp = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3, device=cuda_dev),
            indices=wp.array(indices_np, dtype=int, device=cuda_dev),
        )
        qpts_wp = wp.array(query_pts_np, dtype=wp.vec3, device=cuda_dev)
        mean_ms, std_ms, _ = bench_warp_query(mesh_wp, qpts_wp, n_pts, cuda_dev, signed=True)
        print(f"    Warp GPU signed:    {mean_ms:10.2f} ms")
        results[f"exp2_warp_gpu_{n_pts}"] = mean_ms

        # Warp CPU signed (skip 1M - too slow)
        if n_pts <= 100_000:
            mesh_wp_cpu = wp.Mesh(
                points=wp.array(points_np, dtype=wp.vec3, device="cpu"),
                indices=wp.array(indices_np, dtype=int, device="cpu"),
            )
            qpts_wp_cpu = wp.array(query_pts_np, dtype=wp.vec3, device="cpu")
            mean_ms, std_ms, _ = bench_warp_query(mesh_wp_cpu, qpts_wp_cpu, n_pts, "cpu", signed=True, iters=3)
            print(f"    Warp CPU signed:    {mean_ms:10.2f} ms")
            results[f"exp2_warp_cpu_{n_pts}"] = mean_ms

        # pytorch_volumetric (cap at PV_MESHSDF_MAX_PTS)
        pv_n = min(n_pts, PV_MESHSDF_MAX_PTS)
        pv_query = query_pts_np[:pv_n]
        mean_ms, _, _ = bench_pv_mesh_sdf(factory, pv_query, device_str="cpu", warmup=1, iters=PV_MESHSDF_ITERS)
        print(f"    PV MeshSDF:         {mean_ms:10.2f} ms  ({pv_n} pts)")
        results[f"exp2_pv_meshsdf_{n_pts}"] = mean_ms
        results[f"exp2_pv_meshsdf_actual_npts_{n_pts}"] = pv_n

        del mesh_wp, qpts_wp


def experiment_3_cached(results):
    """CachedSDF (voxel grid) vs Warp direct query."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: CachedSDF vs Warp Direct Query")
    print("=" * 80)

    cuda_dev = "cuda:0"
    n_pts = NUM_QUERY_POINTS_DEFAULT

    # Only use bunny for CachedSDF (construction is extremely slow for larger meshes)
    asset = "bunny"
    print(f"\n--- {asset} (CachedSDF resolution={CACHED_SDF_RESOLUTION}) ---")
    points_np, indices_np = load_usd_mesh(asset)
    bb = np.array([points_np.min(axis=0), points_np.max(axis=0)])
    query_pts_np = make_query_points(bb, n_pts, SEED)

    factory = DirectMeshFactory(points_np, indices_np)

    # Warp GPU signed
    mesh_wp = wp.Mesh(
        points=wp.array(points_np, dtype=wp.vec3, device=cuda_dev),
        indices=wp.array(indices_np, dtype=int, device=cuda_dev),
    )
    qpts_wp = wp.array(query_pts_np, dtype=wp.vec3, device=cuda_dev)
    warp_mean, warp_std, warp_dist = bench_warp_query(mesh_wp, qpts_wp, n_pts, cuda_dev, signed=True)
    print(f"  Warp GPU signed:    {warp_mean:10.2f} ms")

    # CachedSDF CPU
    construct_ms, mean_ms, std_ms, cached_cpu_dist = bench_pv_cached_sdf(
        factory, query_pts_np, CACHED_SDF_RESOLUTION, device_str="cpu", iters=5
    )
    print(f"  PV CachedSDF CPU:   {mean_ms:10.2f} ms  (construct: {construct_ms:.0f} ms)")
    results[f"exp3_{asset}_cached_cpu_ms"] = mean_ms
    results[f"exp3_{asset}_cached_cpu_construct_ms"] = construct_ms

    # CachedSDF GPU (if torch CUDA available)
    if torch.cuda.is_available():
        construct_ms, mean_ms, std_ms, cached_gpu_dist = bench_pv_cached_sdf(
            factory, query_pts_np, CACHED_SDF_RESOLUTION, device_str="cuda:0", iters=5
        )
        print(f"  PV CachedSDF GPU:   {mean_ms:10.2f} ms  (construct: {construct_ms:.0f} ms)")
        results[f"exp3_{asset}_cached_gpu_ms"] = mean_ms
        results[f"exp3_{asset}_cached_gpu_construct_ms"] = construct_ms

        # Accuracy vs Warp
        mae_gpu = float(np.mean(np.abs(warp_dist - cached_gpu_dist)))
        print(f"  CachedSDF GPU vs Warp MAE: {mae_gpu:.6f}")
        results[f"exp3_{asset}_cached_gpu_vs_warp_mae"] = mae_gpu

    results[f"exp3_{asset}_warp_gpu_ms"] = warp_mean
    del mesh_wp, qpts_wp


def experiment_4_cross_section(results):
    """Generate SDF cross-section slices for visual comparison."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: SDF Cross-Section Visualization")
    print("=" * 80)

    asset = "bunny"
    cuda_dev = "cuda:0"
    points_np, indices_np = load_usd_mesh(asset)
    bb = np.array([points_np.min(axis=0), points_np.max(axis=0)])
    center = (bb[0] + bb[1]) / 2
    extent = (bb[1] - bb[0]).max()

    # Create a grid of points on the XY plane at Z=center
    res = 256
    x = np.linspace(bb[0][0] - 0.1, bb[1][0] + 0.1, res)
    y = np.linspace(bb[0][1] - 0.1, bb[1][1] + 0.1, res)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, center[2])
    grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    n_grid = len(grid_pts)

    print(f"  Grid: {res}x{res} = {n_grid} points at z={center[2]:.3f}")

    # Warp GPU
    mesh_wp = wp.Mesh(
        points=wp.array(points_np, dtype=wp.vec3, device=cuda_dev),
        indices=wp.array(indices_np, dtype=int, device=cuda_dev),
    )
    qpts_wp = wp.array(grid_pts, dtype=wp.vec3, device=cuda_dev)
    out_dist = wp.zeros(n_grid, dtype=wp.float32, device=cuda_dev)
    out_face = wp.zeros(n_grid, dtype=wp.int32, device=cuda_dev)
    wp.launch(warp_mesh_query_signed, dim=n_grid,
              inputs=[mesh_wp.id, qpts_wp, 1e7, out_dist, out_face], device=cuda_dev)
    wp.synchronize_device(cuda_dev)
    warp_sdf = out_dist.numpy().reshape(res, res)

    # pytorch_volumetric MeshSDF
    factory = DirectMeshFactory(points_np, indices_np)
    mesh_sdf = MeshSDF(factory)
    pts_tensor = torch.tensor(grid_pts, dtype=torch.float32)
    pv_dist_val, pv_grad = mesh_sdf(pts_tensor.unsqueeze(0))
    pv_sdf = pv_dist_val.squeeze(0).detach().cpu().numpy().reshape(res, res)

    # Save raw data for plotting
    np.savez(
        str(RESULTS_DIR / "cross_section_data.npz"),
        x=x, y=y, warp_sdf=warp_sdf, pv_sdf=pv_sdf, center_z=center[2],
    )
    print("  Saved cross-section data.")

    del mesh_wp, qpts_wp


# ===================================================================
# Plot generation
# ===================================================================
def generate_plots(results):
    """Generate all benchmark plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Plot 1: Timing bar chart per asset (Exp 1) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_idx, label in enumerate(["signed", "unsigned"]):
        ax = axes[ax_idx]
        assets_found = [a for a in ASSETS if f"exp1_{a}_{label}_warp_gpu_ms" in results]
        x = np.arange(len(assets_found))
        width = 0.2

        warp_gpu = [results.get(f"exp1_{a}_{label}_warp_gpu_ms", 0) for a in assets_found]
        warp_cpu = [results.get(f"exp1_{a}_{label}_warp_cpu_ms", 0) for a in assets_found]
        pv_n = [results.get(f"exp1_{a}_{label}_pv_meshsdf_npts", 100000) for a in assets_found]
        pv_raw = [results.get(f"exp1_{a}_{label}_pv_meshsdf_ms", 0) for a in assets_found]
        # Scale PV to 1M points
        pv_scaled = [pv_raw[i] * (NUM_QUERY_POINTS_DEFAULT / pv_n[i]) for i in range(len(assets_found))]

        bars1 = ax.bar(x - width, warp_gpu, width, label="Warp GPU", color="#1f77b4")
        bars2 = ax.bar(x, warp_cpu, width, label="Warp CPU", color="#ff7f0e")
        bars3 = ax.bar(x + width, pv_scaled, width, label="PV MeshSDF (CPU)*", color="#2ca02c")

        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{label.capitalize()} Distance Query ({NUM_QUERY_POINTS_DEFAULT:,} pts)")
        ax.set_xticks(x)
        ax.set_xticklabels(assets_found)
        ax.legend()
        ax.set_yscale("log")

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    fig.suptitle("Experiment 1: Direct Mesh Query Timing (1M query points)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "exp1_timing.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved exp1_timing.png")

    # ---- Plot 2: Scaling (Exp 2) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ns = QUERY_POINT_SWEEP
    warp_gpu_times = [results.get(f"exp2_warp_gpu_{n}", None) for n in ns]
    warp_cpu_times = [results.get(f"exp2_warp_cpu_{n}", None) for n in ns]
    pv_times_raw = [results.get(f"exp2_pv_meshsdf_{n}", None) for n in ns]
    pv_actual_n = [results.get(f"exp2_pv_meshsdf_actual_npts_{n}", n) for n in ns]
    # Scale PV times to requested N
    pv_times = [
        (pv_times_raw[i] * ns[i] / pv_actual_n[i]) if pv_times_raw[i] is not None else None
        for i in range(len(ns))
    ]

    def plot_line(ax, xs, ys, label, color, marker):
        valid = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if valid:
            vx, vy = zip(*valid)
            ax.plot(vx, vy, marker=marker, label=label, color=color, linewidth=2)

    plot_line(ax, ns, warp_gpu_times, "Warp GPU (signed)", "#1f77b4", "o")
    plot_line(ax, ns, warp_cpu_times, "Warp CPU (signed)", "#ff7f0e", "s")
    plot_line(ax, ns, pv_times, "PV MeshSDF (CPU)*", "#2ca02c", "^")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Query Points")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Experiment 2: Scaling with Query Point Count (bunny)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(str(RESULTS_DIR / "exp2_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved exp2_scaling.png")

    # ---- Plot 3: CachedSDF vs Warp (Exp 3) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Query time comparison
    ax = axes[0]
    methods = ["Warp GPU\n(BVH)", "PV CachedSDF\nCPU", "PV CachedSDF\nGPU"]
    vals = [
        results.get("exp3_bunny_warp_gpu_ms", 0),
        results.get("exp3_bunny_cached_cpu_ms", 0),
        results.get("exp3_bunny_cached_gpu_ms", 0),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(methods, vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Query Time (ms)")
    ax.set_title(f"Query Time ({NUM_QUERY_POINTS_DEFAULT:,} pts, bunny)")

    # Construction time
    ax = axes[1]
    construct_vals = [
        0,  # Warp has no construction phase
        results.get("exp3_bunny_cached_cpu_construct_ms", 0) / 1000,
        results.get("exp3_bunny_cached_gpu_construct_ms", 0) / 1000,
    ]
    bars = ax.bar(methods, construct_vals, color=colors)
    for bar, v in zip(bars, construct_vals):
        ax.annotate(f"{v:.1f}s", xy=(bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Construction Time (s)")
    ax.set_title(f"Construction Time (res={CACHED_SDF_RESOLUTION}, bunny)")

    fig.suptitle("Experiment 3: CachedSDF vs Warp Direct Query", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "exp3_cached.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved exp3_cached.png")

    # ---- Plot 4: Cross-section (Exp 4) ----
    data_path = RESULTS_DIR / "cross_section_data.npz"
    if data_path.exists():
        data = np.load(str(data_path))
        x, y = data["x"], data["y"]
        warp_sdf, pv_sdf = data["warp_sdf"], data["pv_sdf"]
        center_z = float(data["center_z"])

        vmax = np.percentile(np.abs(warp_sdf), 95)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].imshow(warp_sdf, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
                              cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"Warp SDF (z={center_z:.3f})")
        axes[0].contour(x, y, warp_sdf, levels=[0], colors="k", linewidths=1)
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(pv_sdf, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
                              cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"PV MeshSDF (z={center_z:.3f})")
        axes[1].contour(x, y, pv_sdf, levels=[0], colors="k", linewidths=1)
        plt.colorbar(im1, ax=axes[1])

        diff = warp_sdf - pv_sdf
        dmax = np.percentile(np.abs(diff), 95)
        im2 = axes[2].imshow(diff, extent=[x[0], x[-1], y[0], y[-1]], origin="lower",
                              cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[2].set_title("Difference (Warp - PV)")
        plt.colorbar(im2, ax=axes[2])

        fig.suptitle("Experiment 4: SDF Cross-Section (bunny)", fontsize=14)
        plt.tight_layout()
        plt.savefig(str(RESULTS_DIR / "exp4_cross_section.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved exp4_cross_section.png")

    # ---- Summary Table ----
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Asset':<10} {'Method':<25} {'Signed (ms)':<15} {'Unsigned (ms)':<15}"
    print(header)
    print("-" * len(header))
    for asset in ASSETS:
        for method, key in [
            ("Warp GPU", "warp_gpu"),
            ("Warp CPU", "warp_cpu"),
            ("PV MeshSDF (CPU)", "pv_meshsdf"),
        ]:
            s = results.get(f"exp1_{asset}_signed_{key}_ms", "N/A")
            u = results.get(f"exp1_{asset}_unsigned_{key}_ms", "N/A")
            s_str = f"{s:.2f}" if isinstance(s, float) else s
            u_str = f"{u:.2f}" if isinstance(u, float) else u
            print(f"  {asset:<8} {method:<25} {s_str:<15} {u_str:<15}")


# ===================================================================
# Main
# ===================================================================
def main():
    # Unbuffered printing for progress visibility
    import functools
    global print
    print = functools.partial(__builtins__.__dict__["print"], flush=True)

    results = {}
    print(f"Warp version: {wp.config.version}")
    print(f"PyTorch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    print(f"Open3D version: {o3d.__version__}")
    if torch.cuda.is_available():
        print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")

    # Pre-compile warp kernels
    print("\nPre-compiling Warp kernels...")
    wp.load_module(device="cuda:0")
    wp.load_module(device="cpu")

    experiment_1_timing(results)
    experiment_2_scaling(results)
    experiment_3_cached(results)
    experiment_4_cross_section(results)

    # Save results
    results_path = RESULTS_DIR / "benchmark_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
