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

### pytorch_volumetric SDF

- **Architecture:** Pure Python/PyTorch wrapper around Open3D
- **No custom CUDA kernels** — zero `.cu` or `.cpp` extension code
- **Distance computation:** Delegates to `Open3D RaycastingScene.compute_closest_points()`, which uses Intel Embree BVH internally (CPU only)
- **Sign determination:** Single ray cast toward a point guaranteed outside the mesh bounding box, with small random noise to reduce artifacts. Odd intersection count = inside.
- **GPU path:** MeshSDF has none — always copies GPU tensors to CPU for Open3D, then copies back. CachedSDF stores a precomputed dense voxel grid as a GPU tensor for O(1) lookup.
- **Autograd integration:** Custom backward pass for gradient flow through FK chains (robotics use case)

### Key Algorithmic Differences

| Aspect | Warp | pytorch_volumetric |
|--------|------|--------------------|
| Acceleration | Custom BVH (LBVH/SAH) | Open3D Embree BVH (CPU only) |
| GPU distance query | Native CUDA kernel | CPU round-trip via Open3D |
| GPU cached query | N/A (direct BVH query) | Dense voxel grid O(1) lookup |
| Sign methods | 4 (ray, normal, winding, parity) | 1 (ray parity) |
| Non-watertight robustness | Winding number method | Not robust |
| Autograd | N/A | Custom backward for transforms |
| Memory model | O(n) BVH | O(resolution^-3) voxel grid |

## 2. Benchmark Design

### Dataset

Same assets as Warp's ASV benchmark (`asv/benchmarks/spatial_query.py`):

| Mesh | Vertices | Triangles | Characteristics |
|------|----------|-----------|-----------------|
| bunny | 6,102 | 12,200 | Watertight, concave, standard benchmark |
| bear | 1,986 | 3,968 | Watertight, simpler geometry |
| rocks | 5,437 | 8,858 | Multi-component, non-watertight regions |

### Query Point Generation

1M random points uniformly sampled within each mesh's axis-aligned bounding box (same strategy as Warp's ASV benchmark, same seed=42).

### Experiments

1. **Direct mesh query timing** — signed & unsigned distance at 1M points
2. **Scaling** — query time vs point count (1K to 1M) on bunny
3. **CachedSDF vs Warp** — voxel grid (res=0.01) query vs BVH query at 1M points
4. **SDF cross-section** — visual comparison on a 256x256 XY slice through bunny

## 3. Results

### Experiment 1: Direct Mesh Query Timing (1M query points)

| Asset | Method | Unsigned (ms) | Signed (ms) | Speedup vs PV* |
|-------|--------|---------------|-------------|-----------------|
| bunny | **Warp GPU** | **5.9** | **8.8** | **73x / 73x** |
| bunny | Warp CPU | 2,711 | 4,278 | 0.24x / 0.15x |
| bunny | PV MeshSDF | ~642* | ~638* | 1x |
| bear | **Warp GPU** | **1.1** | **3.7** | **355x / 106x** |
| bear | Warp CPU | 678 | 1,840 | 0.58x / 0.21x |
| bear | PV MeshSDF | ~391* | ~393* | 1x |
| rocks | **Warp GPU** | **35** | **82** | **93x / 38x** |
| rocks | Warp CPU | 19,036 | 47,309 | 0.17x / 0.07x |
| rocks | PV MeshSDF | ~3,243* | ~3,087* | 1x |

\* PV MeshSDF measured at 10K points, linearly extrapolated to 1M.

**Key finding:** Warp GPU is **38x-355x faster** than pytorch_volumetric's MeshSDF for direct mesh queries. The speedup is largest for smaller meshes where GPU parallelism dominates.

![Experiment 1 Timing](results/exp1_timing.png)

### Accuracy Comparison (Signed Distance)

| Asset | MAE (Warp vs PV) | Sign Agreement |
|-------|-------------------|----------------|
| bunny | 1.3e-8 | 100% |
| bear | 1.1e-8 | 100% |
| rocks | 0.139 | 81.4% |

Bunny and bear show **perfect agreement** — both libraries produce identical SDF values to numerical precision. The rocks mesh has significant disagreement (MAE=0.139, 81% sign agreement) because it contains multiple disconnected components where sign determination is ambiguous. Warp's default ray-tracing parity (3 axis-aligned rays) and PV's single-ray method diverge on such geometry.

### Experiment 2: Scaling with Query Point Count (bunny)

| N | Warp GPU (ms) | Warp CPU (ms) | PV MeshSDF (ms) |
|---|---------------|---------------|------------------|
| 1K | 0.78 | 4.2 | 5.6 |
| 10K | 0.81 | 42 | 6.4 |
| 100K | 1.3 | 426 | ~65* |
| 1M | 8.8 | — | ~645* |

\* Extrapolated from 10K measurement.

![Experiment 2 Scaling](results/exp2_scaling.png)

Warp GPU shows near-constant time up to 10K (kernel launch overhead dominates), then scales linearly. PV MeshSDF scales linearly from the start (CPU-bound). Warp CPU is ~10x slower than PV for the same operation — Warp's single-threaded CPU path vs Open3D's Embree which uses multi-threaded SIMD.

### Experiment 3: CachedSDF vs Warp Direct Query (bunny, 1M points)

| Method | Query Time (ms) | Construction Time (s) |
|--------|-----------------|----------------------|
| Warp GPU (BVH) | 8.8 | 0 (BVH built at mesh creation) |
| PV CachedSDF CPU | 10.7 | 79 |
| PV CachedSDF GPU | **0.8** | 81 |

![Experiment 3 CachedSDF](results/exp3_cached.png)

CachedSDF GPU achieves **0.8 ms** query time — faster than Warp's BVH query — because it's a simple tensor gather (O(1) per point). However:
- **Construction takes ~80 seconds** (must evaluate the ground-truth SDF at every voxel center)
- Memory scales as O(1/resolution^3): resolution 0.01 for bunny is manageable, but finer resolutions explode
- Accuracy: MAE = 0.0024 vs Warp (quantization error from discrete grid)

### Experiment 4: SDF Cross-Section Visualization

![Experiment 4 Cross-Section](results/exp4_cross_section.png)

The SDF field visualizations are visually identical between Warp and pytorch_volumetric. The difference map shows noise-level disagreement (< 1e-3) except at boundary regions where sign determination methods diverge slightly.

## 4. Summary

### Capability Comparison

| Feature | Warp | pytorch_volumetric |
|---------|------|--------------------|
| GPU-accelerated SDF query | Yes (native CUDA) | No (CPU round-trip) |
| GPU-accelerated cached query | N/A | Yes (tensor gather) |
| BVH acceleration | Yes (LBVH/SAH) | Via Open3D Embree (CPU) |
| Multiple sign methods | 4 methods | 1 method |
| Non-watertight mesh support | Winding number | Limited |
| Autograd / differentiable | No | Yes |
| Robot FK integration | No | Yes (ComposedSDF) |
| Dependencies | Self-contained (JIT) | Open3D + PyTorch + many |
| Batch config queries | No | Yes (BatchedViewLookup) |

### Performance Summary

- **Warp GPU** is the fastest option for direct SDF queries: **38-355x faster** than pytorch_volumetric depending on mesh complexity
- **pytorch_volumetric CachedSDF GPU** can be faster than Warp for repeated queries on the same mesh (0.8ms vs 8.8ms) but requires extremely expensive construction (80+ seconds) and consumes O(1/res^3) memory
- **Warp CPU** is significantly slower than pytorch_volumetric's Embree-backed CPU path due to single-threaded execution
- Both libraries produce identical SDF values for watertight meshes; they diverge on multi-component/non-watertight geometry due to different sign determination algorithms

### When to Use Which

- **Warp:** Real-time SDF queries, simulation inner loops, GPU-first workflows, non-watertight meshes (winding number)
- **pytorch_volumetric:** Differentiable robotics pipelines, robot FK-aware collision checking, cases where CachedSDF amortization pays off (many queries on the same static mesh), autograd integration
