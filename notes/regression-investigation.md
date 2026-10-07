# BVH/mesh query regression investigation

**October 7 update:** upstream is now `b5ea46659`, after Eric fully reverted #1844 (`e6a149729`) and partially rolled back #1840 (`45c001d471`). The latter restores ordinary index loads for closest-hit, any-hit, ordered, and count-intersections on all architectures; AABB, closest-sign, and #1843 packed-leaf cursors remain. See the [fresh upstream audit](experiments/upstream-reverts-20261007.md). The September timing tables below remain measurements against their pinned September sources, not October main.

The user has selected avoiding regressions relative to the original parent for count-intersections. The separate Thor ordinary-load option is therefore the selected local count candidate; October upstream already uses ordinary loads for this path. The original local candidate commits and September handoff archive are preserved as historical artifacts. Fresh PTX/CUBIN and occupancy findings are in the [October architecture review](experiments/occupancy-review-20261007.md).

Local investigation on 2026-09-21–22, NVIDIA L40, driver 570.158.01. **Thor runtime validation remains outstanding.** These are independently reviewable local candidates, not a release decision. No remote changes were made.

## Primary evidence and pinned sources

Read both issue bodies, every comment (paginated: three on #1840, two on #1844), both timelines, and the relevant upstream native-code history. Raw responses are in `evidence/`. A fresh `upstream/main` fetch remained at `061b4d01292b0d493f77a82c0767ef8e5d75e9a6`; neither regression had a later fix or revert. The original integrations were GitLab MRs !2880 and !2897, as recorded in the commits; these GitHub numbers are issues, not PRs.

| Source | Exact commit |
|---|---|
| #1840 parent | `64c6cacd6b9f106db9af0dabf5a591c0e41bdc80` |
| #1840 integration | `fcc86dd5b90f2ef45dc96b21acac23e8ec4c52fe` |
| #1844 parent | `6c24bab2ba9d63df790748c6b2e97a6ff6e3037b` |
| #1844 integration | `f005e099ce47d57829452a10532616be92ca9e45` |
| Current-main baseline | `061b4d01292b0d493f77a82c0767ef8e5d75e9a6` |
| Compact point candidate | `e16106bac` |
| Scoped ray policy, benchmark and tests | `f0f1710f9` |
| Exact point revert candidate | `572d2063a`, branch `ankac/astra-point-release-revert` |

Eric's [point-query follow-up](https://github.com/NVIDIA/warp/issues/1844#issuecomment-5750618281) reports Thor unsigned regressions of 2.5–9.5%, with a 128-to-384-byte stack increase and unchanged occupancy. His [initial ray follow-up](https://github.com/NVIDIA/warp/issues/1840#issuecomment-5750645507) isolates four loads and the 62-to-69-register closest-hit cliff. His [later ray refinement](https://github.com/NVIDIA/warp/issues/1840#issuecomment-5750888119) supersedes the blanket path-only advice: use path and architecture evidence, preserve AABB and closest-sign, and treat count-intersections as a tradeoff.

## Isolation and measurement protocol

Each original revision and diagnostic candidate has a named feature-branch checkout under `.worktrees/`, with separately built native libraries. The primary candidate uses this worktree. Every execution selects and verifies the imported Warp checkout and records its path/version, commit, native-header hashes, library hash, native diff, harness hashes, cache path, device UUID information, and block size. No shared Warp installation or native binaries were changed. Build cache clearing is redirected to private `WARP_CACHE_PATH` directories; benchmark and test caches are separate. Other agents' worktrees were only read.

Native/runtime compiler: existing CUDA **12.6.85**, exposed through a private symlink view of `/home/horde/cuda-conda-12.6`. Quick builds are valid with this driver's CUDA 12.8 support. `--no-use-libmathdx` disables unrelated tile-math dependencies. Runtime kernels use Warp's automatic PTX output and the installed driver JIT. This is **not** the CUDA 13 runtime configuration used on Thor.

Resource compiler: existing **NVRTC 13.0.88**, not installed or replaced during this task. Cross-compilation produces actual CUBINs for all six requested targets: SM 89, 100, 103, 110, 120, 121. No requested target is unsupported by this compiler. The helper saves compiler/options/source/header hashes, PTXAS registers/stack/spills, block size 256, and validates the actual ELF architecture. CUDA 13.4 and Blackwell runtime hardware remain validation gaps.

Runtime measurements use a GPU exclusively claimed through `gpu-claim.sh occupy`, fixed seed 42, ASV's bunny/rocks point scenes and ten-bunny replicated ray/AABB scene, 200,000 point/AABB queries, and 1080×1080 camera rays. Each cell has ten warmup graphs and fifteen CUDA-event samples; each graph contains ten queries and timings are **milliseconds per query launch**. Three repetitions alternate source order (AB/BA/AB, or balanced three-way orders for main/compact/revert). GPU clocks were not modified. Full raw samples and repeat spreads are retained; small differences should not be overinterpreted.

Checksums cover point faces/barycentrics, full ray benchmark outputs, and AABB hit counts. They match across the historical and candidate comparisons. Signed benchmark output does not include the sign; sign semantics are checked by the unit tests instead. Unit tests also compare geometric results against independent references. AABB benchmark checksums are counts, not full hit-set proofs; the core AABB tests provide that coverage.

## #1844: compact continuation versus release revert

The merged traversal stores a 64-bit packed payload plus a float bound for each of 32 far continuations. The compact candidate stores only a 32-bit far-node index. The near child stays in registers; popped far nodes reload their bounds and recompute the distance. The radius comparisons (`<` when descending/updating, `>` when rejecting popped nodes), right-first distance ties, sliver handling, and existing stack-capacity guard are preserved. Specialized normal-sign and furthest-point code is unchanged.

This deliberately exchanges extra far-node loads/computation for a smaller local stack. Resource counts prove the storage reduction. They do **not** prove that local-stack traffic alone caused Thor's slowdown; no Thor profiling or runtime measurement was available.

CUDA 13 resources below are `registers / stack bytes`, with zero spills throughout:

| Target | Unsigned parent → integration → compact | Signed parent → integration → compact |
|---|---|---|
| SM 89 | 54/128 → 56/384 → 56/128 | 63/256 → 65/384 → 65/128 |
| SM 100, 103, 110 | 55/128 → 59/384 → 59/128 | 57/256 → 60/384 → 60/128 |
| SM 120, 121 | 54/128 → 56/384 → 57/128 | 55/256 → 60/384 → 59/128 |

L40 unsigned candidate versus current main, median of three run medians:

| Scene | LBVH default | LBVH leaf 8 | CuBQL default | CuBQL leaf 8 |
|---|---:|---:|---:|---:|
| Bunny | -0.12% | +0.62% | -2.66% | +0.05% |
| Rocks | -6.46% | -4.69% | -14.45% | -5.71% |

Signed changes range from -0.29% to -6.28%. Repeat spreads reach 5.83% in one compact rocks cell; several small improvements are within observed variation. The +0.62% bunny cell is retained in the report, not hidden by an aggregate. Full [compact](experiments/point-compact-comparison.md), [original parent/integration](experiments/point-original-comparison.md), and [revert](experiments/point-revert-comparison.md) tables include all signed/unsigned cells and repeat spreads.

The exact revert restores the old point implementation and is separately built/tested. In this L40 matrix it costs up to 10.79% relative to current main, while improving the rocks/CuBQL/leaf-8 unsigned case by 3.87%. It remains the lower-design-change release option recommended by Eric for Thor. The compact candidate retains the shared core and has promising local results, but should not be substituted for that release decision without Thor validation.

## #1840: path-specific load policy

The AABB optimization and ray regressions come from different parts of the same commit. All candidates preserve eager packed-leaf AABB loads and leave closest-sign alone.

Three diagnostic variants changed only the affected ray leaf loops: one ordinary primitive-index load, three ordinary triangle-index loads, or all four ordinary loads. CUDA 13 closest-hit on SM 110 remains at 69 registers for either partial change; all four ordinary loads restore 62. Any-hit similarly changes from 63 to 56 only with all four changed. Thus the minimal tested change that removes the closest-hit register cliff is the whole four-load group, not either subset.

| Closest-hit target | Parent registers | Integration | Scoped candidate | Stack / spills |
|---|---:|---:|---:|---|
| SM 89 | 64 | 63 | 63 | 320 B / 0 |
| SM 100, 103, 110 | 62 | 69 | 62 | 320 B / 0 |
| SM 120, 121 | 63 | 61 | 61 | 320 B / 0 |

The candidate uses ordinary loads for closest-hit on SM 100/103/110, and any-hit/ordered only on SM 110. All other paths/targets retain read-only loads. At 256 threads, the affected closest-hit register budget returns from three to four register-limited blocks per SM. This is compiler-resource evidence for SM 100/103, and additionally Eric's runtime evidence for Thor; it is not a runtime validation on B200/B300.

The ordered and closest-sign benchmark wrappers have different resource totals from Eric's exact probes; their parent/candidate comparisons are recorded for these wrappers without claiming those absolute counts reproduce his. All 30 path/target combinations of the scoped candidate match the selected parent or integration resource tuple, including stack/spills.

The L40 exact parent/integration comparison reproduces packed-leaf raw BVH AABB gains of 17.77–19.65% for LBVH/CuBQL and 18.40% for SAH; mesh AABB improves 6.53–8.59%. The five ray variants are approximately neutral with CUDA 12.6/driver JIT (-1.07% to +0.60%). See the full [ray/AABB table](experiments/ray-original-comparison.md) and [SAH table](experiments/aabb-sah-comparison.md). This compiler/runtime distinction matters: it does not contradict Eric's CUDA 13 Thor result or establish preservation of every historical L40 timing gain.

A separate three-repeat comparison of main against the all-ordinary diagnostic found CuBQL/default closest-hit +2.71% and any-hit +1.86% on L40; most other changes were below 1%. This supports retaining the existing loads on L40 rather than globally reverting them. See [all-ordinary results](experiments/ray-ordinary-v-main.md). A final interleaved current-main/scoped-policy check of all five CuBQL/leaf-8 ray variants and both AABB paths ranged from -0.37% to +0.28%, with identical output hashes ([table](experiments/ray-policy-v-main.md)). The full final ray/AABB matrix also ran three times as a functional benchmark smoke check; those additional runs were not used as an interleaved performance comparison.

For all five benchmark ray kernels, the CUDA 13 **SM 89 machine-code `.text` sections are byte-identical to pristine current main**. This directly checks preservation of those L40 kernel instructions. `compare_kernel_text.py` reproduces the comparison; `sm89-kernel-text-comparison.json` records the section hashes and sizes. Whole CUBIN hashes differ, so the claim is specifically about executable kernel sections.

Count-intersections stays read-only in the main candidate. The separate `1840-count-ordinary-thor-option.patch` switches its four loads only on SM 110. Eric's data says that option removes packed-leaf losses but sacrifices the 9.91% CuBQL/default gain. Both options are retained for review; no release priority has been assumed. Split-load resource experiments alone cannot tell whether either subset preserves that runtime gain on Thor.

## Correctness and remaining validation

The compact candidate passed 85 targeted tests across CPU/CUDA BVH, mesh AABB, point and ray classes. Coverage includes normal/parity/winding signs where supported, furthest-point, adjoints, packed leaves, and existing brute-force comparisons. New radius tests check exact/just-above radius boundaries, zero radius, nearest-face choice, and agreement across shared point variants. New native ordered/closest-sign tests check hit/miss results, face/distance, strict max-t rejection and closest-sign behavior on CPU/CUDA. CuBQL winding-number remains unsupported and is skipped intentionally.

The new functional checks were run on the corresponding unmodified paths before native changes. The performance regression reproducers are the isolated timing and compiler-resource comparisons; no timing assertions were added to unit tests.

An independent final code review found no semantic blocker. The new equidistant-point test checks consistency across variants; it does not pin a particular historical face ID. Preservation of the right-first tie order was checked directly against the unchanged comparison/descent logic and the benchmark face hashes.

The exact release revert passed its nine existing point tests on CPU/CUDA. The all-ordinary ray diagnostic passed 43 targeted tests, including the new internal ray-variant checks. The final combined candidate passed **87 targeted tests** in 43.882 seconds, and all five new ASV class setup/timing entry points passed a direct smoke run. The full Warp test suite was not run; validation was scoped to the affected query families. Changed-file pre-commit checks passed.

The final CUDA 13 resource reports are in `experiments/resources-final/`: all 36 source/target compilations succeeded, their actual CUBIN architectures match the requested targets, and the compiled header SHA-256 is `4f557762185da68deaac5c5b74596ab59a9086bd25ab95fdadc9c1df81f20de0`. The optional count policy was also compiled separately on all six targets: it restores SM 110 to 64 registers and leaves the other targets' resource counts unchanged; its stack remains 192 bytes with zero spills. These reports are compiler evidence, not Blackwell execution.

No Thor, B200/B300, or SM 120/121 runtime measurements were made. No clocks, drivers, toolkits, remote resources, issues, PRs or assignments were modified. Native stack bounds are unchanged and exercised by the existing BVH tests; no CUDA memory-sanitizer run or new proof for malformed/deeper-than-supported trees is claimed.

## Reproduce or hand off to Thor

`thor-handoff.tar.gz` contains these notes, the four alternative patches, the frozen benchmark/support files and assets, replay scripts, and final CUDA 13 source/resource artifacts. Extract it at the root of a separate **driver worktree** at the pinned main revision; keep the revision checkouts being tested separate from that driver worktree. Run the harness from the driver worktree and select each tested checkout with `--warp-root`. The archive does not include native libraries or CUDA toolkits: build each tested checkout locally on Thor.

1. Use named feature-branch worktrees at the exact parents, integrations, and current-main commit above. Apply the compact or exact-revert patch independently; apply the ray policy independently. The count option applies on top of the ray policy. Do not apply both point alternatives together.
2. Build each source checkout with its own native output: `uv run build_lib.py --cuda-path <toolkit> --no-use-libmathdx`. Use `--quick` only when the installed driver supports that toolkit. Set a separate `WARP_CACHE_PATH` even during builds. On Thor use the installed CUDA 13 toolkit and record the driver/compiler versions; do not change clocks for comparability with this investigation.
3. Install benchmark dependencies in a private environment: `uv run --extra benchmark --with asv-runner python notes/experiments/run_queries.py --help`. The harness imports Warp only after selecting `--warp-root`. Its ASV support files and both assets must accompany the handoff.
4. Claim a free GPU with the local allocation helper. For each revision, run the command below with a separate cache/output. Use three repetitions, reversing revision order on the second repetition. Keep the GPU claim for the entire interleaved comparison, then release it.

```bash
uv run --extra benchmark --with asv-runner python notes/experiments/run_queries.py \
  --warp-root /absolute/path/to/revision \
  --cache /absolute/private/cache/revision \
  --output notes/experiments/revision-r1.jsonl \
  --families point ray aabb --samples 15 --warmup 10 --resolution 1080 \
  --constructors lbvh cubql --leaves 0 8 --assets bunny rocks

uv run python notes/experiments/compare_runs.py \
  'notes/experiments/before-r*.jsonl' 'notes/experiments/after-r*.jsonl'

WARP_CACHE_PATH=/absolute/private/cache/tests uv run --extra dev \
  -m warp.tests -s autodetect -k TestMeshQueryPoint -k TestMeshQueryRay \
  -k TestMeshQueryAABBMethods -k TestBvh
```

5. Cross-compile the saved JIT sources with the resource helper. This performs no GPU execution. Supply the local NVRTC library explicitly; use `--targets` to limit the matrix when necessary. Unsupported targets are saved as errors, not silently retargeted.

```bash
uv run python notes/experiments/compile_resources.py \
  --source /absolute/path/to/saved/kernel.cu \
  --native /absolute/path/to/revision/warp/native \
  --nvrtc /absolute/path/to/libnvrtc.so.13 \
  --out notes/experiments/resources-revision
```

The helper preloads NVRTC builtins and disables the CUDA 13 compiler disk cache so verbose resource diagnostics are present. Early failed builtins loads and early ELF-ABI parser reports are retained in `resources-main-point`, `resources-main-*-v2`, and related diagnostic directories; use the successful exact parent/integration reports and `resources-final/` for conclusions. A transient `resources-main-v3` batch actually used the compact header (its saved hash exposes this); it is excluded from the pristine-main evidence.
