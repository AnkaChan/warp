# Payload-Pair BVH Traversal for Volume Queries

**Status**: Proposed

## Motivation

The stack-based BVH traversal used by AABB and sphere queries (`wp.bvh_query_aabb`,
`wp.bvh_query_sphere`, `wp.mesh_query_aabb`, `wp.mesh_query_sphere`) spends a large
fraction of its time on stack traffic and node reloads: every pushed node index must
be popped and its two 16-byte halves reloaded from global memory before it can be
tested or descended into. On an L40 this overhead is 20-30% of total query time at
scene scale.

This change restructures the volume-query traversal loop around three cooperating
mechanisms. Isolated measurements (see Performance) show the mechanisms are **not
independently beneficial** -- the first is a measured regression on its own -- so they
ship as one unit:

1. **Test-at-parent.** When an internal node is processed, both children's AABBs are
   loaded and tested immediately, and only passing children proceed. Total AABB loads
   are unchanged (each visited node is still loaded exactly once, just one level
   earlier), but failing children no longer consume a stack slot, a pop, and a loop
   iteration. On its own this is a *regression* (+6% to +41%): without the mechanisms
   below, every passing child is loaded a second time on pop to recover its payload.
2. **Register-carried near child.** Of the passing children, one continues immediately
   as the next node; its 63-bit payload (child indices / leaf range + leaf flag) is
   carried in registers (`cur_node`/`have_node`), never touching the stack.
3. **Payload pairs.** The deferred far child is pushed as *two* 32-bit stack slots
   holding its packed payload (top slot tagged with bit 31; node indices are 31-bit,
   so plain index entries are unambiguous). Popping a pair reconstructs the node with
   zero memory accesses. This is where most of the gain lives: without it the far
   child must be reloaded on pop, which costs back most of what test-at-parent saved
   (measured: mixed results, up to +8% on packed-leaf trees).

Directed queries (ray, capsule) are unaffected; they use the stackless skip-link
traversal and have no stack at all.

## Requirements

| ID  | Requirement                                                            | Priority | Notes |
| --- | ---------------------------------------------------------------------- | -------- | ----- |
| R1  | Identical query results (bitwise hit sets) for every constructor       | Must     | traversal *order* is explicitly not part of the API contract |
| R2  | No stack overflow / dropped results for any constructor-produced tree  | Must     | see the depth-budget proof below |
| R3  | No regression for ray/capsule/closest-point queries                    | Must     | verified per-kind |
| R4  | Graph-capture-safe on in-place LBVH rebuilds                           | Must     | depth lives behind a device pointer |

**Non-goals**: changing the 32-byte packed node layout, the shared-memory stack
placement (measured 2.7x faster than a local stack), or the public query API.

## Design

### The overflow hazard and the depth budget

A pair costs two slots where an index costs one, so pairs halve the effective
capacity of the 32-slot stack. The constructors hard-terminate tree construction at
depth `BVH_QUERY_STACK_SIZE` (32), which makes the classic one-slot-per-entry budget
*exactly* safe -- but an all-pairs traversal of a legal depth-32 tree can demand up to
31 pending entries = 62 slots. **Overflow here does not crash; it silently drops far
children, i.e. missing query results.** A degenerate tree with diagonally
exponentially spaced boxes reproduces a dropped hit on the SAH constructor; it is now
a regression test (`test_bvh_degenerate_deep_tree`).

The guard: record the tree depth `max_depth` (root = 1) at construction, and push
pairs only while the current stack occupancy is at most `64 - 2*max_depth` slots;
above that, fall back to plain index entries.

*Proof sketch.* Pending stack entries never exceed `max_depth - 1` (one deferred far
child per internal node on the current root-leaf path). Pairs occupy the bottom of
the stack (LIFO: entries below the limit were pushed first and popped last), so live
pairs never exceed `(64 - 2*max_depth)/2 + 1 = 33 - max_depth`. Since every entry is
one slot plus one extra slot per live pair, worst-case usage is
`(max_depth - 1) + (33 - max_depth) = 32 = BVH_QUERY_STACK_SIZE`, for every
`max_depth <= 32`. Unknown, out-of-range,
or grouped-build depths (grouped host builds restart the depth counter per group)
yield a negative limit, disabling pairs entirely and restoring the exactly-safe
index-only encoding.

*Why pairs sit at the stack bottom rather than the top:* the payload value of an
entry tracks its stack **residence time**, not its node depth. LIFO pops deep entries
while their cache lines are still L1-hot (an index reload is nearly free); shallow
entries wait longest and go cold. Placing pairs deep-first measured 5-28% *slower*.

### max_depth tracking

- Host builders (SAH/median/LBVH) already track `bvh.max_depth` during construction.
- The GPU LBVH builder records depth via `atomicMax` in `mark_packed_leaf_nodes`,
  writing through `bvh.max_depth_ptr` -- a device pointer, so an in-place
  graph-captured rebuild updates the depth without host round trips, and
  `bvh_query_pair_limit()` reads it per query via `__ldg`.
- cuBQL trees record depth during the conversion walk.
- Host-to-device copies allocate the pointer and seed it from the host value.
- Refit does not change topology, so the recorded depth stays valid.

### Alternatives Considered

- **Test-on-pop (status quo).** Simpler, no budget invariant, but leaves the 20-30%
  on the table. It remains the fallback semantics when pairs are disabled.
- **Test-at-parent with index-only pushes.** Measured +6% to +41% slower than
  test-on-pop everywhere (double-loads with no compensation); not a viable
  intermediate.
- **Wider stack slots (64-bit).** Doubles the 32KB shared-memory slab per block,
  which is already the occupancy limiter.
- **Local-memory stacks.** Measured 2.7-6x slower in iterator structs (defeats
  scalar replacement; the whole query state spills to local memory).
- **All-pairs without the register-carried near child.** Measured equal to the full
  design on the benchmark scene (the near child's stack round trip through shared
  memory is nearly free). Kept the register carry for now because it is the
  extensively validated form; dropping it is a candidate simplification tracked
  separately -- it would remove the `cur_node`/`have_node` state machine at no
  measured cost.

## Performance

Measured on an L40, 122k-triangle scene (10x-replicated bunny), 200k queries,
lbvh/cubql/sah constructors, leaf sizes 1 and 8; each step measured in landing
order with bitwise-identical results:

| Variant                              | Scene AABB vs test-on-pop | Dense micro (AABB single) |
| ------------------------------------ | ------------------------- | ------------------------- |
| test-at-parent alone                 | +6% to +27%               | +34%                      |
| + register-carried near child        | -25% to +8% (mixed)       | +12%                      |
| + payload pairs (this design)        | **-20% to -30%**          | **-4%**                   |

Sphere queries gain 26-30% at scene scale through the same loop. CPU AABB queries
measure 15-21% faster for the full bundle in landing position (the
test-at-parent/register-carry savings dominate; an earlier series on a different
baseline measured the pair encoding alone as a ~4-5% CPU cost, but no CPU-gating is
warranted at the bundle level). Measured CPU run-to-run noise on this harness is
about +/-3%.

## Testing Strategy

- `test_bvh_degenerate_deep_tree`: a diagonally exponentially spaced box chain that
  drives SAH/LBVH trees to the construction depth bound; fails (dropped hit) without
  the depth budget, passes with it. Runs for all available constructors, CPU + GPU.
- Existing BVH/mesh query suites (AABB, ray, sphere, capsule, grouped subtree
  queries, tile variants) pass unchanged.
- Benchmark harness verifies bitwise-identical hit sets against the previous
  traversal across 48 scene cases (3 constructors x 2 leaf sizes x all query kinds).
