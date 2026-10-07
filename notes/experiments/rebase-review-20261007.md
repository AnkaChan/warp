# Independent rebase review, 2026-10-07

No unintended upstream reversions or semantic integration blockers were found.
The UTC clock reported 2026-10-07 at review start. Reviewed base:
`416985fed1f214c71c7f29ca6013a222b574549c`; reviewed rebased HEAD:
`8582277411ea8d2f775a2f6acd6d46c325929500`. The original local tip was
`ed1cca9ceabba6f0ffabd5551c86d7d5449a8138`.

The rebased point and ray changes remain separate commits, `5384109a7` and
`014a8d1ca`. Only `warp/native/mesh.h` differs from the new upstream under
`warp/native/`. Its reviewed SHA-256 is
`6a9adcdd361830c5c95bc8ebd73c9e8d9d5091d864f990ea802dfa00a13fba3d`.

## Preserved upstream changes

- `bvh.cpp:533` retains the GH-1991 SAH cutoff at
  `depth >= BVH_QUERY_STACK_SIZE - 1`. The independent per-group depth budget
  at `bvh.cpp:208` is also retained. Neither construction nor group-root
  accounting was changed by conflict resolution.
- The new `TestBvh.test_sah_depth_fits_query_stack` at `test_bvh.py:1106`
  is byte-identical to upstream. It covers the 31-level ungrouped limit,
  independent grouped-subtree budgets, and group-root query isolation.
- The Cppcheck fixes remain: `bvh_load_node` uses
  `reinterpret_cast<const float4*>`, and both mesh point/velocity diagnostic
  print calls cast the mesh ID to `unsigned long long`.
- All other native updates from upstream, including tile and Clang changes,
  are present. There is no diff from upstream in `bvh.cpp`, `bvh.h`, or
  `test_bvh.py`.

## Traversal and policy checks

The compact point core at `mesh.h:136` is unchanged from the reviewed local
candidate. It retains strict triangle-distance updates, strict radius hit
acceptance, near-first traversal with right-child-first ties, and the
far-stack bounds check. Pop-time bounds are recomputed against the shrinking
radius. Sign classification, sign-normal traversal, furthest-point traversal,
and adjoints are unchanged. The four shared-core callers remain unsigned,
default signed, parity, and winding-number point queries.

The ray helper at `mesh.h:1136` uses ordinary loads for closest-hit on
SM 100/103/110 and for any-hit, ordered, and count-intersections on SM 110.
Count's four substitutions at `mesh.h:1416` implement the user's selected
Thor policy. All other targets use the existing read-only helper; CPU loads
remain ordinary through that helper. Closest-sign and AABB code are unchanged.
Ray traversal, strict `max_t`, root arguments, triangle ordering, and count
accumulation are unchanged.

The exact native diff from the old candidate consists only of the four
selected count loads, the corresponding helper comment, and the two upstream
diagnostic casts. This check confirms that conflict resolution did not rewrite
unrelated query behavior.

## Coverage and remaining limits

The two added query-test sections and the five-path ASV benchmark are
byte-identical to the original candidate. Radius tests exercise boundary
rejection and cross-variant face agreement; native ray tests exercise ordered
and closest-sign results and strict ray limits. Timing stays in benchmarks.
The radius test still allows either equidistant face instead of independently
pinning the historical tie winner. That pre-existing coverage limitation is
unchanged by the rebase.

The upstream SAH regression test should be included in the root's rebuilt
validation. Grouped global-root traversal retains its documented depth
limitation; the compact stack's overflow guard is not proof of exhaustive
traversal for arbitrarily deep grouped trees.

The known CUDA 13 signed-point register cliff versus the original parent and
reverted upstream remains a candidate limitation, not a rebase conflict.
Rebasing does not invalidate that warning or establish Thor runtime recovery.
Previous compiler and timing artifacts retain their original source/header
fingerprints; they must not be relabeled as measurements of this rebased tree.

This review used read-only diffs and source inspection. `git diff --check
416985fed..858227741` passed. Builds and CPU/CUDA tests are owned by the root
agent and are not claimed as completed by this report. No source edits, git
mutations, native builds, or GPU execution were performed by the reviewer.
