# EBVH validation drivers

These research benchmarks validate the `f812c377` closest-point implementation and
compare temporal AABB updates with the peeling prototypes. They are not CI timing
tests. The September 2026 reports and raw JSONL/telemetry are in:

```text
/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation
```

## Environment on the assigned L40 VM

Run from the `ebvh-query-accel` worktree. `setup_cuda.py` installs SHA256-verified
NVIDIA CUDA 12.6.3 redistributables locally; it does not install a system toolkit.

```bash
uv python install
uv sync --extra dev
uv run tools/ebvh_validation/setup_cuda.py
PM_PACKAGES_ROOT=/home/horde/.cache/packman \
  WARP_CUDA_PATH="$PWD/_build/ebvh-cuda-12.6.3" \
  uv run build_lib.py --quick -j 6
```

The quick build requires a sufficiently recent driver. The measured VM uses
570.158.01, with CUDA 12.8 driver support.

Use `run_claimed.sh` for every GPU run. It claims one GPU, logs clocks and power,
and restores clocks/releases the claim on exit. It contains this VM's environment
and task-directory paths. Use a new `EBVH_CACHE_TAG` after native changes; do not
clear another process's kernel cache. The 2490 MHz request hit the 300 W power
limit; 1050 MHz was verified stable under sustained load.

## Runs

```bash
# Exact-triangle CP reproduction and independent-process repeats.
EBVH_CLOCK_MHZ=1050 EBVH_CACHE_TAG=validation \
  bash tools/ebvh_validation/run_claimed.sh \
  uv run tools/ebvh_validation/run_stable_cp.py

# AABB, closest-hit ray and actual moving-frame comparisons.
EBVH_CLOCK_MHZ=1050 EBVH_CACHE_TAG=intersection-v2 \
  bash tools/ebvh_validation/run_claimed.sh \
  uv run tools/ebvh_validation/run_intersection_suite.py

# Two additional independent processes for the core AABB arms.
EBVH_CLOCK_MHZ=1050 EBVH_CACHE_TAG=intersection-v2 \
  bash tools/ebvh_validation/run_claimed.sh \
  uv run tools/ebvh_validation/run_intersection_suite.py \
  --group aabb --passes 2 --aabb-modes root,cached,fused,fused_refine

uv run tools/ebvh_validation/summarize_intersections.py \
  /home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation
```

JSONL output is append-only. Preserve previous files or select an explicit output
with the individual drivers when starting a separate experiment. Use `--help` on
`closest_point.py`, `cp_controls.py`, `aabb_probe.py`, `ray_benchmark.py`,
`temporal_cp.py` and `temporal_aabb.py` for case parameters.

## Interpretation

- Exact CP leaf tests use triangles, unlike the archived P08 AABB proxy. The
  original P08 source and real collision geometry were unavailable on this VM.
- Static precomputed cache/oracle arms isolate query cost. Temporal drivers keep
  previous-frame inputs separate from current outputs during graph replay and
  measure cache maintenance. AABB temporal runs also compare ordinary BVH refits.
- Ray seeds/caches come from a previous ray before query motion, but their
  initialization and ongoing maintenance are outside the timed ray kernels.
- All arms perform matching output work. Static cases use three interleaved
  timing rounds, each accumulating at least 1.5 seconds per arm after warmup.
  Temporal runs accumulate that budget across 25 frames, with a 2-second initial
  warmup and 50 ms warmup on subsequent frames.
- The regular root iterator must remain independent of peeling dispatch. See
  [the design note](../../design/ebvh-temporal-query-update.md) for the measured
  regression that motivated distinct iterator types.

Correctness coverage lives in `warp/tests/geometry/test_bvh_temporal_update.py`,
`test_bvh_exclusive.py`, `test_bvh_aabb_peeling.py` and `test_mesh_ray_peeling.py`.
Use `unittest`; no test asserts a performance ratio.
