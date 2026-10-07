# Rebase validation, 2026-10-07

The branch was rebased onto freshly fetched upstream `416985fed1f214c71c7f29ca6013a222b574549c`, including Eric's two query rollbacks and the GH-1991 SAH depth cap. The previous published tip `ed1cca9ceabba6f0ffabd5551c86d7d5449a8138` is preserved locally as `ankac/warp-bvh-regression-astra-before-rebase-20261007`.

Conflict resolution restores the compact shared point core and the architecture-specific ray load helper, while retaining newer Cppcheck diagnostic casts and all other upstream native changes. The selected Thor count-intersections option is now part of the ray commit: closest-hit uses ordinary loads on SM 100/103/110; any-hit, ordered, and count use ordinary loads on SM 110. Other targets retain read-only loads. Point and ray changes remain separate commits.

The [independent review](rebase-review-20261007.md) found no unintended upstream reversions. `bvh.cpp`, `bvh.h`, and the new SAH regression test remain identical to the fetched base. The final native header SHA-256 is `6a9adcdd361830c5c95bc8ebd73c9e8d9d5091d864f990ea802dfa00a13fba3d`.

## Builds and correctness

Both the candidate and a separately named baseline worktree at the exact new main were rebuilt with the existing CUDA 12.6.85 toolkit, `--quick --no-use-libmathdx`, and separate private build/kernel caches. Driver 570.158.01 supports CUDA 12.8, satisfying the quick-build constraint. No shared native installation, driver, toolkit, or clock setting was changed.

The candidate imported Warp from this worktree and reported version `1.19.0.dev0`. Its rebuilt `warp.so` SHA-256 is `d31920a91f0fa95dd4ff6d7c313bf3ed1bf251590da9a34fd8702f9b473f0635`. Validation ran after the structural rebase at `8582277411ea8d2f775a2f6acd6d46c325929500`; subsequent updates affect commit messages, notes, and changelog wording, not native code.

The affected CPU/CUDA suite passed **88 tests in 75.995 seconds**, including `TestBvh.test_sah_depth_fits_query_stack`, with this command:

```bash
WARP_CACHE_PATH=/absolute/private/cache/tests uv run --extra dev \
  -m warp.tests -s autodetect -k TestMeshQueryPoint -k TestMeshQueryRay \
  -k TestMeshQueryAABBMethods -k TestBvh
```

All six compiler-report helper tests passed. Changed-file pre-commit checks and `git diff --check` passed. The full Warp test suite was not run.

## Fresh compiler checks

The identical saved query CUDA sources were compiled against both the rebased candidate and new main: six modules, all 12 integer architectures advertised by NVRTC 13.0.88 and all 14 advertised by NVRTC 12.6.85. All **312 PTX/CUBIN compilations** succeeded, covering **572 kernel/target resource rows**, with no reported spills. Actual ELF SM fields, PTX targets, and artifact hashes were independently checked.

Compared with the pre-rebase candidates and October baseline, every executable kernel text section is byte-identical except the deliberately changed CUDA 13 SM 110 count kernel. Its register count changes from 63 to 64, with the same 192-byte stack and four-block residency at 256 threads. No other registers, stack, spill, or shared-memory resource tuple changes. See [resource verification](rebase-20261007/resource-verification.json).

The existing signed-point CUDA 13 occupancy concern remains: the compact candidate uses 65 registers versus the original parent's 63 on SM 75/80/86/87/89, allowing three rather than four 256-thread blocks. No Thor runtime recovery is claimed. Optional `a`/`f` architecture suffixes were not audited, and SM 88 physical occupancy remains unverified.

## Runtime benchmark scope and limitation

A bounded CuBQL matrix covered bunny/rocks signed and unsigned point queries, all five ray variants, and both AABB paths, with default and eight-primitive leaves. Three repetitions alternated main/candidate order (AB/BA/AB), using 256-thread blocks, seed 42, 1080-squared rays, ten warmup graphs, and fifteen timing samples of ten launches each.

All **22 output-checksum cells matched across all six runs**. As before, signed-point benchmark checksums cover faces/barycentrics, with signs covered by unit tests; AABB benchmark outputs are counts, with hit sets covered by the existing tests. [Benchmark provenance and checksum verification](rebase-20261007/benchmark-correctness.json) records every imported checkout, native/header hash, and run fingerprint.

**The new timings are excluded from performance conclusions.** The allocation helper granted an exclusive advisory lock, but other compute processes were already using that GPU outside the claim system. The timing repeat spread reached 59.31%, and those processes' start times predate the benchmark. All four GPUs had other compute processes present when investigated, so no further timing run was attempted. Other jobs were not disturbed, and this task's GPU claims were released. Raw timings and diagnostic telemetry remain local under `rebase-20261007/`.

Historical September timings retain their original provenance. A fresh isolated L40 timing comparison and Thor runtime validation remain outstanding; they are not prerequisites for concluding that the rebase and its functional integration passed the checks above.
