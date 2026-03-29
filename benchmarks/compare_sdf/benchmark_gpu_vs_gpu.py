"""Fair GPU-vs-GPU comparison: Warp BVH vs pytorch_volumetric CachedSDF.

pytorch_volumetric's actual GPU strategy is a precomputed dense 3D voxel grid
stored as a GPU tensor (CachedSDF). This script compares that against Warp's
native CUDA BVH queries across multiple resolutions to show the
accuracy/speed/memory tradeoff.

Also includes the fair CPU-vs-CPU comparison: Warp CPU vs Open3D Embree CPU.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
WARP_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(WARP_ROOT))

import warp as wp
wp.init()
wp.set_module_options({"enable_backward": False})
import warp.examples

import torch
import open3d as o3d
from pytorch_volumetric import MeshSDF, CachedSDF, ObjectFactory

import functools
_builtin_print = print
print = functools.partial(_builtin_print, flush=True)

# ===================================================================
# Constants
# ===================================================================
SEED = 42
NUM_QUERY_POINTS = 1_000_000
WARMUP = 2
ITERS = 5
CACHED_RESOLUTIONS = [0.05, 0.02, 0.01, 0.005, 0.002]
PV_CPU_MAX_PTS = 10_000  # PV MeshSDF is CPU-only, cap for speed
ASSETS = ["bunny", "bear", "rocks"]


# ===================================================================
# Warp kernels
# ===================================================================
@wp.kernel
def warp_sdf_signed(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    out_dist: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point(mesh, p, query_d_max)
    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out_dist[tid] = query.sign * wp.length(p - cp)
    else:
        out_dist[tid] = query_d_max


@wp.kernel
def warp_sdf_unsigned(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    out_dist: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, query_d_max)
    if query.result:
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        out_dist[tid] = wp.length(p - cp)
    else:
        out_dist[tid] = query_d_max


# ===================================================================
# Helpers
# ===================================================================
def load_usd_mesh(asset_name):
    from pxr import Usd, UsdGeom
    asset_dir = warp.examples.get_asset_directory()
    stage = Usd.Stage.Open(os.path.join(asset_dir, f"{asset_name}.usd"))
    mesh_geom = UsdGeom.Mesh(stage.GetPrimAtPath(f"/root/{asset_name}"))
    points = np.array(mesh_geom.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    return points, indices


def make_query_points(bb, n_points, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(bb[0], bb[1], size=(n_points, 3)).astype(np.float32)


class DirectMeshFactory(ObjectFactory):
    def __init__(self, points_np, indices_np):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points_np.astype(np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(indices_np.reshape(-1, 3).astype(np.int32))
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        super().__init__(name="direct", mesh=mesh)

    def make_collision_obj(self, z, rgba=None):
        return None, None

    def get_mesh_resource_filename(self):
        return ""


def time_warp_gpu(mesh_wp, qpts_wp, n_pts, signed=True):
    kernel = warp_sdf_signed if signed else warp_sdf_unsigned
    out = wp.zeros(n_pts, dtype=wp.float32, device="cuda:0")
    for _ in range(WARMUP):
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, qpts_wp, 1e7, out], device="cuda:0")
    wp.synchronize_device("cuda:0")
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, qpts_wp, 1e7, out], device="cuda:0")
        wp.synchronize_device("cuda:0")
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), out.numpy()


def time_warp_cpu(mesh_wp, qpts_wp, n_pts, signed=True):
    kernel = warp_sdf_signed if signed else warp_sdf_unsigned
    out = wp.zeros(n_pts, dtype=wp.float32, device="cpu")
    for _ in range(WARMUP):
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, qpts_wp, 1e7, out], device="cpu")
    wp.synchronize_device("cpu")
    times = []
    for _ in range(min(ITERS, 2)):
        t0 = time.perf_counter()
        wp.launch(kernel, dim=n_pts, inputs=[mesh_wp.id, qpts_wp, 1e7, out], device="cpu")
        wp.synchronize_device("cpu")
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), out.numpy()


def time_pv_meshsdf(factory, query_pts_np, n_pts):
    """PV MeshSDF (Open3D Embree CPU)."""
    mesh_sdf = MeshSDF(factory)
    pts = torch.tensor(query_pts_np[:n_pts], dtype=torch.float32)
    for _ in range(WARMUP):
        mesh_sdf(pts.unsqueeze(0))
    times = []
    for _ in range(min(ITERS, 3)):
        t0 = time.perf_counter()
        dist, grad = mesh_sdf(pts.unsqueeze(0))
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), dist.squeeze(0).detach().cpu().numpy()


def time_pv_cached_gpu(factory, query_pts_np, resolution):
    """PV CachedSDF on GPU. Returns (construct_ms, query_mean_ms, query_std_ms, dist, memory_mb)."""
    mesh_sdf = MeshSDF(factory)
    bb = factory.bounding_box(padding=0.1)

    t0 = time.perf_counter()
    cached = CachedSDF(
        "bench", resolution, bb, mesh_sdf,
        device="cuda:0", clean_cache=True,
        cache_path=str(RESULTS_DIR / "_tmp_cache.pkl"),
    )
    torch.cuda.synchronize()
    construct_ms = (time.perf_counter() - t0) * 1000

    # Memory: voxel grid + gradient grid
    mem_bytes = cached.voxels.raw_data.nelement() * cached.voxels.raw_data.element_size()
    mem_bytes += cached.voxels_grad.nelement() * cached.voxels_grad.element_size()
    memory_mb = mem_bytes / (1024 * 1024)
    grid_shape = tuple(cached.voxels.raw_data.shape)

    pts = torch.tensor(query_pts_np, dtype=torch.float32, device="cuda:0")
    for _ in range(WARMUP):
        cached(pts.unsqueeze(0))
    torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        dist, grad = cached(pts.unsqueeze(0))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    dist_np = dist.squeeze(0).detach().cpu().numpy()
    return construct_ms, float(np.mean(times)), float(np.std(times)), dist_np, memory_mb, grid_shape


# ===================================================================
# Main benchmark
# ===================================================================
def main():
    results = {}

    print(f"Warp {wp.config.version}, PyTorch {torch.__version__}, Open3D {o3d.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    wp.load_module(device="cuda:0")
    wp.load_module(device="cpu")

    for asset in ASSETS:
        print(f"\n{'='*80}")
        print(f"  ASSET: {asset}")
        print(f"{'='*80}")

        points_np, indices_np = load_usd_mesh(asset)
        bb = np.array([points_np.min(axis=0), points_np.max(axis=0)])
        n_tris = len(indices_np) // 3
        query_pts_np = make_query_points(bb, NUM_QUERY_POINTS, SEED)
        factory = DirectMeshFactory(points_np, indices_np)

        print(f"  Vertices={len(points_np)}, Triangles={n_tris}")
        print(f"  Bounding box: {bb[0]} -> {bb[1]}")
        print(f"  Query points: {NUM_QUERY_POINTS:,}")

        # ----- CPU vs CPU (fair comparison) -----
        print(f"\n  --- CPU vs CPU (signed distance, {PV_CPU_MAX_PTS:,} pts) ---")

        mesh_wp_cpu = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3, device="cpu"),
            indices=wp.array(indices_np, dtype=int, device="cpu"),
        )
        qpts_cpu = wp.array(query_pts_np[:PV_CPU_MAX_PTS], dtype=wp.vec3, device="cpu")
        warp_cpu_ms, warp_cpu_std, warp_cpu_dist = time_warp_cpu(
            mesh_wp_cpu, qpts_cpu, PV_CPU_MAX_PTS, signed=True
        )
        print(f"    Warp CPU (BVH, single-thread): {warp_cpu_ms:8.2f} ms")
        results[f"{asset}_cpu_warp_ms"] = warp_cpu_ms

        pv_cpu_ms, pv_cpu_std, pv_cpu_dist = time_pv_meshsdf(factory, query_pts_np, PV_CPU_MAX_PTS)
        print(f"    PV MeshSDF (Embree, multi-thread): {pv_cpu_ms:8.2f} ms")
        results[f"{asset}_cpu_pv_ms"] = pv_cpu_ms

        cpu_ratio = warp_cpu_ms / pv_cpu_ms
        print(f"    Ratio (Warp/PV): {cpu_ratio:.1f}x {'(PV faster)' if cpu_ratio > 1 else '(Warp faster)'}")
        results[f"{asset}_cpu_ratio"] = cpu_ratio

        # CPU accuracy
        mae_cpu = float(np.mean(np.abs(warp_cpu_dist - pv_cpu_dist)))
        sign_agree = float(np.mean(np.sign(warp_cpu_dist) == np.sign(pv_cpu_dist)))
        print(f"    Accuracy: MAE={mae_cpu:.6f}, sign agreement={sign_agree:.4f}")
        results[f"{asset}_cpu_mae"] = mae_cpu
        results[f"{asset}_cpu_sign_agree"] = sign_agree

        # ----- GPU vs GPU (fair comparison) -----
        print(f"\n  --- GPU vs GPU (signed distance, {NUM_QUERY_POINTS:,} pts) ---")

        # Warp GPU BVH
        mesh_wp_gpu = wp.Mesh(
            points=wp.array(points_np, dtype=wp.vec3, device="cuda:0"),
            indices=wp.array(indices_np, dtype=int, device="cuda:0"),
        )
        qpts_gpu = wp.array(query_pts_np, dtype=wp.vec3, device="cuda:0")
        warp_gpu_ms, warp_gpu_std, warp_gpu_dist = time_warp_gpu(
            mesh_wp_gpu, qpts_gpu, NUM_QUERY_POINTS, signed=True
        )
        print(f"    Warp GPU (BVH):      {warp_gpu_ms:8.2f} ms  (std {warp_gpu_std:.2f})")
        results[f"{asset}_gpu_warp_ms"] = warp_gpu_ms

        # Warp GPU memory: BVH nodes + mesh data
        # Approximate: BVH is ~2*n_tris nodes, each ~32 bytes
        warp_mem_mb = (2 * n_tris * 32 + len(points_np) * 12 + len(indices_np) * 4) / (1024 * 1024)
        print(f"    Warp GPU memory:     ~{warp_mem_mb:.1f} MB (BVH + mesh)")
        results[f"{asset}_gpu_warp_mem_mb"] = warp_mem_mb

        # PV CachedSDF at multiple resolutions
        # Estimate voxel count and skip if > 2GB to avoid OOM
        MAX_GRID_MB = 2000  # Cap at ~2GB voxel grid
        bb_extent = bb[1] - bb[0]
        print(f"\n    PV CachedSDF GPU at different resolutions:")
        print(f"    {'Res':>8s}  {'Grid':>20s}  {'Query (ms)':>12s}  {'Construct':>12s}  {'Mem (MB)':>10s}  {'MAE vs Warp':>12s}  {'Sign Agree':>12s}")
        print(f"    {'-'*98}")

        for res in CACHED_RESOLUTIONS:
            est_voxels = np.prod(np.ceil(bb_extent / res + 20).astype(int))  # +padding
            est_mb = est_voxels * 4 * 4 / (1024 * 1024)  # float32 * (sdf + 3 grad)
            if est_mb > MAX_GRID_MB:
                print(f"    {res:>8.3f}  SKIPPED (est. {est_mb:.0f} MB > {MAX_GRID_MB} MB cap)")
                continue
            try:
                construct_ms, query_ms, query_std, cached_dist, mem_mb, grid_shape = \
                    time_pv_cached_gpu(factory, query_pts_np, res)

                mae = float(np.mean(np.abs(warp_gpu_dist - cached_dist)))
                sa = float(np.mean(np.sign(warp_gpu_dist) == np.sign(cached_dist)))

                shape_str = "x".join(str(s) for s in grid_shape)
                construct_str = f"{construct_ms/1000:.1f}s" if construct_ms > 1000 else f"{construct_ms:.0f}ms"
                print(f"    {res:>8.3f}  {shape_str:>20s}  {query_ms:>10.2f}ms  {construct_str:>12s}  {mem_mb:>8.1f}MB  {mae:>12.6f}  {sa:>12.4f}")

                results[f"{asset}_cached_{res}_query_ms"] = query_ms
                results[f"{asset}_cached_{res}_construct_ms"] = construct_ms
                results[f"{asset}_cached_{res}_mem_mb"] = mem_mb
                results[f"{asset}_cached_{res}_mae"] = mae
                results[f"{asset}_cached_{res}_sign_agree"] = sa
                results[f"{asset}_cached_{res}_grid"] = shape_str

            except Exception as e:
                print(f"    {res:>8.3f}  SKIPPED: {e}")
                break

        del mesh_wp_gpu, qpts_gpu, mesh_wp_cpu, qpts_cpu

    # Cleanup temp
    tmp_cache = RESULTS_DIR / "_tmp_cache.pkl"
    if tmp_cache.exists():
        tmp_cache.unlink()

    # Save results
    with open(str(RESULTS_DIR / "gpu_vs_gpu_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    generate_plots(results)
    print("\nDone!")


def generate_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Plot 1: CPU vs CPU bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    assets_found = [a for a in ASSETS if f"{a}_cpu_warp_ms" in results]
    x = np.arange(len(assets_found))
    width = 0.3
    warp_vals = [results[f"{a}_cpu_warp_ms"] for a in assets_found]
    pv_vals = [results[f"{a}_cpu_pv_ms"] for a in assets_found]

    b1 = ax.bar(x - width/2, warp_vals, width, label="Warp CPU (BVH, single-thread)", color="#1f77b4")
    b2 = ax.bar(x + width/2, pv_vals, width, label="PV MeshSDF (Embree, multi-thread)", color="#2ca02c")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"CPU vs CPU: Signed Distance ({PV_CPU_MAX_PTS:,} query pts)")
    ax.set_xticks(x)
    ax.set_xticklabels(assets_found)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "fair_cpu_vs_cpu.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fair_cpu_vs_cpu.png")

    # ---- Plot 2: GPU query time - Warp BVH vs CachedSDF at different resolutions ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, asset in enumerate(ASSETS):
        ax = axes[idx]
        resolutions = []
        cached_times = []
        cached_maes = []
        for res in CACHED_RESOLUTIONS:
            key = f"{asset}_cached_{res}_query_ms"
            if key in results:
                resolutions.append(res)
                cached_times.append(results[key])
                cached_maes.append(results[f"{asset}_cached_{res}_mae"])

        if not resolutions:
            continue

        warp_ms = results.get(f"{asset}_gpu_warp_ms", 0)

        # Query time
        color_cached = "#2ca02c"
        ax.plot(resolutions, cached_times, "o-", color=color_cached, label="PV CachedSDF GPU", linewidth=2, markersize=8)
        ax.axhline(warp_ms, color="#1f77b4", linestyle="--", linewidth=2, label=f"Warp GPU BVH ({warp_ms:.1f}ms)")

        ax.set_xscale("log")
        ax.set_xlabel("Voxel Resolution")
        ax.set_ylabel("Query Time (ms)")
        ax.set_title(f"{asset}")
        ax.legend(fontsize=8)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Annotate MAE on each point
        for r, t, m in zip(resolutions, cached_times, cached_maes):
            ax.annotate(f"MAE={m:.4f}", (r, t), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", color="gray")

    fig.suptitle(f"GPU vs GPU: Query Time vs Voxel Resolution ({NUM_QUERY_POINTS:,} pts)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "fair_gpu_query_vs_resolution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fair_gpu_query_vs_resolution.png")

    # ---- Plot 3: Accuracy vs Speed tradeoff ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, asset in enumerate(ASSETS):
        ax = axes[idx]
        query_times = []
        maes = []
        mems = []
        labels = []
        for res in CACHED_RESOLUTIONS:
            key = f"{asset}_cached_{res}_query_ms"
            if key in results:
                query_times.append(results[key])
                maes.append(results[f"{asset}_cached_{res}_mae"])
                mems.append(results[f"{asset}_cached_{res}_mem_mb"])
                labels.append(f"res={res}")

        if not query_times:
            continue

        warp_ms = results.get(f"{asset}_gpu_warp_ms", 0)

        # Scatter: x=query_time, y=MAE, size=memory
        sizes = [max(20, m * 2) for m in mems]  # scale marker size by memory
        sc = ax.scatter(query_times, maes, s=sizes, c=mems, cmap="YlOrRd",
                        edgecolors="black", linewidths=0.5, zorder=5)
        plt.colorbar(sc, ax=ax, label="Memory (MB)")

        # Warp reference point (MAE=0, at its query time)
        ax.scatter([warp_ms], [0], s=200, c="#1f77b4", marker="*", zorder=10,
                   label=f"Warp GPU BVH ({warp_ms:.1f}ms)")

        for t, m, lab in zip(query_times, maes, labels):
            ax.annotate(lab, (t, m), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)

        ax.set_xlabel("Query Time (ms)")
        ax.set_ylabel("MAE vs Ground Truth (Warp)")
        ax.set_title(f"{asset}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Accuracy vs Speed Tradeoff ({NUM_QUERY_POINTS:,} pts)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "fair_accuracy_vs_speed.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fair_accuracy_vs_speed.png")

    # ---- Plot 4: Construction cost vs resolution ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for asset in ASSETS:
        resolutions = []
        construct_times = []
        for res in CACHED_RESOLUTIONS:
            key = f"{asset}_cached_{res}_construct_ms"
            if key in results:
                resolutions.append(res)
                construct_times.append(results[key] / 1000)  # to seconds
        if resolutions:
            ax.plot(resolutions, construct_times, "o-", label=asset, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Voxel Resolution")
    ax.set_ylabel("Construction Time (s)")
    ax.set_title("CachedSDF Construction Cost")
    ax.legend()
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for asset in ASSETS:
        resolutions = []
        mem_vals = []
        for res in CACHED_RESOLUTIONS:
            key = f"{asset}_cached_{res}_mem_mb"
            if key in results:
                resolutions.append(res)
                mem_vals.append(results[key])
        if resolutions:
            ax.plot(resolutions, mem_vals, "o-", label=asset, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Voxel Resolution")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("CachedSDF Memory Usage")
    ax.legend()
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle("CachedSDF Overhead vs Resolution", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "fair_construction_cost.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fair_construction_cost.png")


if __name__ == "__main__":
    main()
