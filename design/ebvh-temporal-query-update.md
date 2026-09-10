# Temporal Exclusive BVH query state

Status: Experiment on `ankac/ebvh-query-accel` (September 2026)

## Motivation

An Exclusive BVH can certify that every primitive intersecting an AABB query is
inside a subtree. The existing cached query revalidates the previous containment
node, but does not expose its replacement when motion requires climbing. A caller
that separately recomputes a containment node from a primitive seed pays another
dependent parent walk. The CP validation also measured this integration pitfall:
cached queries were faster, while query plus a complete leaf-based cache refresh
was slower than root traversal on the tested temporal sequences.

The archived AABB oracle-start measurements bound the opportunity at about
1.05–1.84x on scattered workloads. Connected self-collision workloads have little
prefix work to remove. The target here is coherent AABB streams with useful
containment, not a promised speedup on every BVH query.

## Design

Add `bvh_query_aabb_exclusive_update(id, low, high, cached_node, refine=False)`.
The node argument is an input/output variable, following the existing query-next
index convention. The constructor finds a valid current containment node once,
writes it back, and initializes the existing cached iterator from that certified
node. This shares the small-stack traversal and safe fallback with the current API.
The caller stores the resulting node for its next query.

The default path only climbs. It adds no child search to successful cache reuse.
Optional refinement descends through strictly containing child exclusive boxes;
it can recover deeper starts after an expanded query shrinks or initialize a
cache from the root. Refinement is bounded and uses the current tree. Applications
can request it periodically rather than paying for two child-box probes on every
stable frame. Benchmarks must include cache writes and the chosen refinement
schedule.

No new persistent native allocation or BVH layout change is needed. Cache storage
is one caller-owned integer per query. Refit and rebuild require revalidation of
current exclusive bounds; old topology identity alone is never a certificate.
Invalid indices and missing metadata fall back to the root. Strict containment
retains the ordinary inclusive AABB boundary behavior.

## Alternatives

- AABB and ray peeling prototypes on `origin/ankac/ebvh-peeling` collect results
  while walking and trim only residuals that retain an exact representable shape.
  Benchmark these separately; declining a middle cut is required for correctness.
- Temporal frontiers on `origin/ankac/ebvh-frontier-proto` cache both rejected and
  hit terminal nodes. They can reuse more traversal work, but require identity and
  topology-epoch checks, bounded token storage, overflow fallback, and recording
  traffic. Their cost is not bounded by the single-node oracle, so they should not
  inherit its projected speedups. They remain a separate research line.
- A closest-hit ray certificate must contain the complete active finite segment.
  A cached endpoint alone cannot justify skipping the complement. The existing
  endpoint-peeling prototype handles this distinction and is the ray comparison
  candidate.

## Validation

Require full primitive-set equality against NumPy brute force across refits,
rebuilds, packed leaves, invalid cache indices, missing metadata and clipping
boundaries. Refinement must recover leaf certificates from a root cache on
separated bounds. Performance tests use a claimed GPU, observed clock telemetry,
warmup, interleaved CUDA graph samples and explicit cache-maintenance costs.
No performance ratio is asserted in a unit test.
