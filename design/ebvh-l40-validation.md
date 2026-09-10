# EBVH validation on L40, September 2026

The new AABB cache update improves an existing temporal EBVH pipeline. The
closest-point validation supports warm seeding, but does not reproduce the
archived 3–4.8x scattered-query result. Peeling is not a winning direction in the
tested workloads. These conclusions apply to reconstructed synthetic inputs;
the original P08 source and real collision geometry were unavailable.

Tabulated ratios are speedups: baseline time divided by candidate time. Values
above 1 mean faster; milliseconds are labeled separately.

## Closest point

The query implementation is the `f812c377` base. Exact point-to-triangle leaf
tests replace the archive's point-to-AABB proxy. All six query arms agree on
closest distance across the primary 14,400,075 input-query evaluations, using
rtol=2e-5 / atol=4e-6. Six hundred sampled primary queries agree with independent
float64 triangle/edge projection; maximum discrepancy is 4.5e-7 world units.
Tied faces need not have the same identity.

At verified 1050 MHz, representative query-only speedups against root are:

| Scene | Triangles | Warm | Revalidated cache | Archived warm | Archived cache |
|---|---:|---:|---:|---:|---:|
| scatter | 10,000 | 1.556x | 2.277x | 1.52x | — |
| scatter | 100,000 | 1.779x | 1.891x | 1.43x | — |
| scatter | 1,000,000 | 1.586x | 1.446x | 1.60x | 4.82x |
| scatter | 10,000,000 | 1.061x | 1.042x | 1.59x | — |
| sheet | 996,872 | 1.394x | 1.216x | 1.60x | 1.42x |

The 1M scatter/sheet headline cases have three independent fixed-clock
processes. The full primary sweep also has three independent processes per case
at a requested 2490 MHz, which telemetry showed was power-throttled. Ordering
matters: the unsorted 1M sheet sensitivity reaches 3.163x cached, while unsorted
scatter reaches 2.044x. Neither reproduces the archived scattered-scene headline.
A unified depth gate still trails warm-only at the tested 1M headline cases.

Temporal CP tests feed actual previous-frame faces/nodes through 25 deforming
frames. Cached query speedups of 1.49–1.79x become 0.84–0.93x when followed by the
existing full leaf-based cache refresh. Those temporal CP runs use the
power-capped maximum-clock request; they are not fixed-frequency measurements.

## AABB implementation and results

`bvh_query_aabb_exclusive_update` validates once, writes back its containment
node and uses the ordinary traversal stack. Optional refinement searches deeper
containing children. Invalid or stale caches retain exact overlap behavior.
See [the design](ebvh-temporal-query-update.md).

Static SAH trees, leaf size 1, 1M Morton-ordered coherent queries per pass:

| Bounds | Root ms | Old cached / root | New update / root | Refined update / root |
|---:|---:|---:|---:|---:|
| 10,000 | 0.126427 | 2.746x | 1.645x | 1.499x |
| 100,000 | 0.259609 | 0.454x | 1.114x | 1.052x |
| 1,000,000 | 0.775687 | 0.226x | 0.959x | 0.930x |

Each core arm has three independent processes and three interleaved timing rounds
per process. New-update speedup ranges are 1.6444–1.6451x, 1.1132–1.1140x and
0.9586–0.9587x. At 1M bounds the new update is 4.25x faster than the old cached
query, while remaining slightly slower than root. The old small stack is useful
when all queries fit, as in the 10K static case.

Three independent processes per temporal case, 100K tracks, 25 refit frames:

| Scene | Update / root, query only | Update vs old cached + refresh, including refit | Update vs ordinary BVH, including refit |
|---|---:|---:|---:|
| scatter / 10K bounds | 1.128x | 1.569x | 0.677x |
| scatter / 1M bounds | 0.989x | 1.550x | 0.364x |
| sheet / 999,698 bounds | 0.964x | 1.212x | 0.273x |

The extra metadata refit costs prevent a win over an ordinary BVH at this query
count. Keep the feature opt-in for pipelines already maintaining EBVH metadata,
or measure amortization across many query batches. Refining every eight frames
does not beat the default update on these trajectories. GPU-processing totals
exclude host geometry generation, uploads and application scheduling; refit
totals use matched steady frames 2–25.

## Peeling comparison and regression avoided

The imported AABB peeling dispatch initially slowed the 1M ordinary root query
from 0.787 ms to 1.351 ms. A distinct internal iterator type removes peeling from
ordinary dispatch and restores the original performance. The losing prototype
remains available as an experimental comparison.

AABB peeling achieves only 0.50x, 0.30x and 0.18x root on the static cases.
Closest-hit ray tests use actual prior-ray seeds/caches followed by origin motion;
each of four cases has three timing rounds and 200K hit-producing rays per pass.

| Triangles | Origin offset / max t | Cached segment / root | Endpoint peeling / root |
|---:|---|---:|---:|
| 10,000 | 0.1 / 0.15 | 1.110x | 0.470x |
| 10,000 | 20 / 30 | 0.792x | 0.281x |
| 1,000,000 | 0.1 / 0.15 | 0.837x | 0.303x |
| 1,000,000 | 20 / 30 | 0.823x | 0.189x |

Ray initialization and ongoing cache maintenance are excluded. The small cached
ray gain is not an end-to-end recommendation. The frontier branch was inspected
but not merged or benchmarked; no frontier speedup is claimed.

## Evidence and reproduction

- L40 sm_89, driver 570.158.01, locally installed CUDA 12.6.3. Final intersection
  telemetry has 1,719 samples above 10% GPU utilization, all at 1050 MHz SM and
  9001 MHz memory. Only the claimed GPU was used; clocks/claim were restored.
- Compilation, correctness checks and warmup precede timing. Static graphs have
  100 kernel launches, rotated/reversed arm order and at least 1.5 seconds of
  execution per arm per round. Temporal inputs and output caches are separate;
  graph replays cannot converge the same-frame cache.
- All 54 focused CPU/CUDA tests pass. Static AABB counts/checksums agree across
  all 9M input-query evaluations, plus 1,440 sampled complete hit sets checked
  against NumPy. Temporal counts/checksums agree on 22.5M input-query evaluations.
  Ray attributes agree across all arms, plus 64 independent float64 brute checks.
- Native build, generated stubs, documentation and changed-file pre-commit
  checks pass. Final documentation build has no warnings.

Versioned drivers and commands are in
[tools/ebvh_validation](../tools/ebvh_validation/README.md). Detailed reports,
raw JSONL samples, native hashes and telemetry are in the assigned task folder:

```text
/home/horde/Code/AI-Docs/AI-Logs/Newton/tasks/ebvh-validation/
  2026-09-10-plan.md
  2026-09-10-environment.md
  2026-09-10-cp-validation.md
  2026-09-10-intersection-validation.md
  2026-09-10-intersection-summary.txt
```
