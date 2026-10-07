# Architecture compiler audit, October 7, 2026

This directory contains offline NVRTC compilation artifacts, not GPU execution results. No CUDA driver is initialized by the compiler helper. The CUDA 13 matrix has 300 compilations and 540 kernel entries; CUDA 12.6 has 350 compilations and 630 kernel entries. All succeeded and reported zero spill loads/stores. `verification-summary.json` records independently checked totals and target lists.

Each `resources.json` records the exact native header hashes, source hash, compiler library/version/hash, options, launch bounds, and 256-thread block size. Each target has raw PTX, assembled CUBIN, assembler log, physical registers, stack, spill and static shared-memory counts. Actual ELF SM and PTX target/entry names were verified, along with artifact hashes. PTX virtual register declarations are not used for occupancy.

| Label | Native source |
|---|---|
| pinned-main | September baseline `061b4d01292b0d493f77a82c0767ef8e5d75e9a6` |
| candidate | Local compact point plus scoped ray candidate, header hash in manifests; original read-only count retained in this snapshot |
| latest-upstream | October main `b5ea46659b86efb8f1b511904a3b5c254cd13d70`, including Eric's rollbacks |
| point-parent | Exact pre-#1844 parent `6c24bab2ba9d63df790748c6b2e97a6ff6e3037b` |
| ray-parent | Exact pre-#1840 parent `64c6cacd6b9f106db9af0dabf5a591c0e41bdc80` |
| count-selected | Separate architecture-scoped ordinary Thor count-load option now selected by the user; only the count kernel is compiled |

The matrices compile identical saved benchmark-generated CUDA sources against each native revision. They do not import Warp, and do not include a build or runtime test of October main. The point source also contains auxiliary AABB/collision kernels; the occupancy audit includes these and distinguishes them from point queries.

`batch-jobs.json` lists the source/native combinations. `compile_resources_used.py` preserves the exact compiler helper used. Reproduce a cell from the worktree root with a new output directory:

```bash
uv run --no-sync python notes/experiments/compile_resources.py \
  --source notes/experiments/resources-final/closest/source.cu \
  --native .worktrees/upstream-20261007/warp/native \
  --nvrtc /absolute/path/to/libnvrtc.so.13 \
  --all-supported --emit-ptx --out /new/empty/output/directory
```

Repeat with CUDA 12.6's library for the legacy target matrix. Default unsuffixed targets are covered; optional architecture/family suffixes `a` and `f` are not. Native headers must be kept pinned when replaying the commands. Neither remote Jetson access nor toolkit installation is needed for this compilation.

`ptx-inventory-cuda13.json` records per-entry static local declarations and memory instruction counts (not dynamically executed instruction counts). `ray-kernel-text-cuda13.json` shows that only the five deliberately changed path/SM pairs differ from the September baseline; all other 55 ray machine-code comparisons are identical. `ray-kernel-text-cuda126.json` verifies all 70 ray path/SM pairs are identical to that baseline. Count-selection is a separate candidate and is not included in those original-candidate identity claims.

The independent agent's occupancy output is in `occupancy-review/resources-annotated.json`. The root agent independently rebuilt the NVIDIA-header-based calculator, checked 14 literal occupancy boundary cases, and repeated the audit into `occupancy-review/root-verified.json`. See `../occupancy-review-20261007.md` for findings, hardware specifications, assumptions, and the explicit SM 88 physical-limit gap. These are theoretical occupancy limits at 256 threads and zero dynamic shared memory, not measured achieved occupancy or runtime performance.
