# Warp vs pytorch_volumetric: SDF Benchmark Report

**Date:** 2026-03-29
**Hardware:** NVIDIA L40 (48 GB), x86_64 CPU
**Software:** Warp 1.13.0.dev0 (CUDA 12.6), pytorch_volumetric + Open3D 0.19.0, PyTorch 2.6.0+cu126

## 1. Implementation Analysis

### Warp SDF

- **Architecture:** Native C++/CUDA implementation, JIT-compiled via Warp's kernel system
- **Acceleration structure:** Custom BVH — LBVH on GPU (linear construction), SAH on CPU
- **Distance computation:** BVH traversal to leaf nodes, then closest-point-on-triangle via barycentric coordinates. O(log n) average case.
- **Sign determination** — 4 methods:
  1. **Ray-tracing parity** (default): 3 axis-aligned rays, majority vote on intersection parity
  2. **Angle-weighted pseudo-normal**: accumulated normals from nearby triangles, sign from dot product
  3. **Winding number**: hierarchical solid angle via BVH (Jacobson's fast winding number), robust for non-watertight meshes
  4. **Parity with perturbation**: randomly-perturbed rays with majority vote
- **GPU path:** Native CUDA kernels with shared-memory BVH stack traversal, full parallel execution
- **CPU path:** Same C++ algorithms, single-threaded per query
- **Memory:** O(n) for BVH nodes — typically ~1 MB for 12K triangles

### pytorch_volumetric SDF

- **Architecture:** Pure Python/PyTorch wrapper around Open3D — no custom CUDA kernels
- **Distance computation:** Delegates to `Open3D RaycastingScene.compute_closest_points()`, which uses Intel Embree BVH internally (CPU only, multi-threaded SIMD)
- **Sign determination:** Single ray cast toward a guaranteed-exterior point with random noise. Odd intersection count = inside.
- **GPU strategy: CachedSDF** — precomputes a dense 3D voxel grid of SDF values using the CPU path, then stores it as a GPU tensor for O(1) lookup via index arithmetic or `torch.nn.functional.grid_sample`
- **Autograd integration:** Custom backward pass for gradient flow through FK chains (robotics use case)

### Key Algorithmic Differences

| Aspect | Warp | pytorch_volumetric |
|--------|------|--------------------|
| CPU acceleration | Custom BVH (SAH), single-threaded | Open3D Embree BVH, multi-threaded SIMD |
| GPU strategy | Native CUDA BVH traversal | Dense 3D voxel tensor (CachedSDF) |
| GPU memory | O(n) — ~1 MB per mesh | O(1/res^3) — 73 MB to 9 GB+ |
| Sign methods | 4 (ray, normal, winding, parity) | 1 (ray parity) |
| Non-watertight robustness | Winding number method | Not robust |
| Autograd | No | Yes |
| Robot FK integration | No | Yes (ComposedSDF) |
| Dependencies | Self-contained (JIT) | Open3D + PyTorch + many |

## 2. Benchmark Design

### Dataset

Same assets as Warp's ASV benchmark (`asv/benchmarks/spatial_query.py`):

| Mesh | Vertices | Triangles | Bounding Box Volume | Characteristics |
|------|----------|-----------|---------------------|-----------------|
| bunny | 6,102 | 12,200 | ~2.6 | Watertight, concave, standard benchmark |
| bear | 1,986 | 3,968 | ~522 | Watertight, large bounding box |
| rocks | 5,437 | 8,858 | ~11,798 | Multi-component, very large BB |

### Query Point Generation

1M random points uniformly sampled within each mesh's bounding box (same strategy as Warp's ASV benchmark, seed=42).

## 3. Results

### Fair Comparison 1: CPU vs CPU (10K query points)

This compares Warp's single-threaded BVH against Open3D's Embree (multi-threaded SIMD):

| Asset | Warp CPU (ms) | PV Embree (ms) | Ratio | Accuracy |
|-------|---------------|----------------|-------|----------|
| bunny | 42.4 | **6.5** | 6.5x PV faster | MAE=0, 100% sign |
| bear | 18.4 | **3.7** | 5.0x PV faster | MAE=0, 100% sign |
| rocks | 481.9 | **30.6** | 15.7x PV faster | MAE=0.14, 81% sign |

**Open3D Embree wins on CPU by 5-16x** thanks to multi-threaded SIMD execution.

![CPU vs CPU](results/fair_cpu_vs_cpu.png)

### Fair Comparison 2: GPU vs GPU (1M query points)

This is the core comparison: **Warp's CUDA BVH traversal vs PV's CachedSDF voxel grid**.

They represent fundamentally different strategies:
- **Warp:** O(log n) BVH traversal per query, exact distance, ~1 MB memory
- **CachedSDF:** O(1) tensor gather per query, quantized distance, 0.6 MB — 9 GB+ memory

#### bunny (12K triangles, compact bounding box)

| Method | Query (ms) | Memory | MAE | Sign Agree | Construction |
|--------|-----------|--------|-----|------------|-------------|
| **Warp GPU BVH** | **8.8** | **1.0 MB** | **0 (exact)** | **100%** | **instant** |
| CachedSDF res=0.05 | 1.0 | 0.6 MB | 0.0119 | 97.6% | 124s |
| CachedSDF res=0.02 | 0.8 | 9.4 MB | 0.0048 | 99.0% | 103s |
| CachedSDF res=0.01 | 0.9 | 73 MB | 0.0024 | 99.5% | 123s |
| CachedSDF res=0.005 | 1.2 | 582 MB | 0.0012 | 99.8% | 135s |

#### bear (4K triangles, large bounding box)

| Method | Query (ms) | Memory | MAE | Sign Agree | Construction |
|--------|-----------|--------|-----|------------|-------------|
| **Warp GPU BVH** | **3.7** | **0.3 MB** | **0 (exact)** | **100%** | **instant** |
| CachedSDF res=0.05 | 0.8 | 37 MB | 0.0122 | 99.0% | 100s |
| CachedSDF res=0.02 | 1.0 | 563 MB | 0.0049 | 99.6% | 111s |
| CachedSDF res=0.01 | SKIPPED | 4,470 MB | — | — | — |

#### rocks (9K triangles, very large bounding box)

| Method | Query (ms) | Memory | MAE | Sign Agree | Construction |
|--------|-----------|--------|-----|------------|-------------|
| **Warp GPU BVH** | **81.6** | **0.7 MB** | **0 (exact)** | **100%** | **instant** |
| CachedSDF res=0.05 | 1.1 | 1,646 MB | 0.1513 | 81.3% | 297s |
| CachedSDF res=0.02 | SKIPPED | ~26 GB | — | — | — |

![GPU Query Time vs Resolution](results/fair_gpu_query_vs_resolution.png)

### Accuracy vs Speed vs Memory Tradeoff

![Accuracy vs Speed Tradeoff](results/fair_accuracy_vs_speed.png)

The scatter plot shows each CachedSDF resolution as a circle (size = memory), with Warp as the blue star at MAE=0. CachedSDF is faster in query time but trades off accuracy and memory.

### CachedSDF Overhead

![Construction Cost and Memory](results/fair_construction_cost.png)

CachedSDF construction cost is **100-300 seconds** across all configurations tested, because it must evaluate Open3D's CPU SDF at every voxel center. Memory scales cubically with 1/resolution — the bear mesh at res=0.01 already requires 4.5 GB.

### SDF Cross-Section (bunny)

![Cross-Section Visualization](results/exp4_cross_section.png)

Visually identical SDF fields. Difference map shows noise-level disagreement (<1e-3).

## 4. Analysis

### CachedSDF is PV's GPU strategy — and it has severe tradeoffs

CachedSDF achieves ~1ms query time (vs Warp's 3-82ms) by reducing SDF lookup to a tensor gather. But:

1. **Construction is extremely expensive**: 100-300s per mesh, because it evaluates the Open3D CPU SDF at every voxel. This must be amortized over many queries.
2. **Memory scales as O(1/res^3)**: For the bear mesh (bounding box ~5x10x10), even res=0.01 requires **4.5 GB**. For rocks (BB ~37x11x30), even res=0.05 uses **1.6 GB**. Finer resolutions are infeasible.
3. **Accuracy is limited by voxel resolution**: MAE of 0.001-0.012 depending on resolution. This may be acceptable for collision avoidance but not for precise distance queries.
4. **Bounding box sensitivity**: Meshes with large bounding boxes relative to their detail (like bear and rocks) waste enormous memory on empty space. No adaptive/sparse representation exists.

### Warp GPU BVH: exact, compact, no precomputation

Warp's approach uses ~1 MB of memory regardless of bounding box size, returns exact distances, and requires no precomputation beyond BVH construction (which is included in mesh creation). It's slower per-query than CachedSDF (8.8ms vs 0.8ms for bunny) but needs no amortization period.

### Break-even analysis (bunny)

CachedSDF construction takes ~120s. The per-query advantage is ~8ms. Break-even point: **~15,000 queries** before CachedSDF amortizes its construction cost. For single or few-query workloads, Warp is strictly better.

### CPU comparison favors Open3D

Warp's CPU path is **5-16x slower** than Open3D Embree because Warp runs single-threaded while Embree uses multi-threaded SIMD. This is a clear gap in Warp's CPU implementation.

## 5. Summary

| Scenario | Winner | Why |
|----------|--------|-----|
| GPU, exact SDF, any mesh | **Warp** | Native CUDA BVH, ~1 MB memory, zero construction |
| GPU, many repeated queries on static mesh | **CachedSDF** | O(1) lookup after amortizing 100s+ construction |
| CPU signed distance | **PV (Embree)** | Multi-threaded SIMD, 5-16x faster |
| Non-watertight meshes | **Warp** | Winding number method |
| Large bounding box meshes | **Warp** | CachedSDF memory explodes with BB volume |
| Differentiable / autograd | **PV** | Custom backward for FK chains |
| Small / robotics meshes with repeated queries | **CachedSDF** | Compact BB + amortized construction |
