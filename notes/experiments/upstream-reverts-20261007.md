# Upstream rollback audit — 2026-10-07

Fetched `upstream/main` from `https://github.com/NVIDIA/warp.git` on 2026-10-07. The fetched tip is `b5ea46659b86efb8f1b511904a3b5c254cd13d70` (2026-10-07 17:50:40 UTC), 52 commits after the original investigation baseline `061b4d01292b0d493f77a82c0767ef8e5d75e9a6`. No local branch, checkout, source edit, or GPU resource was changed by this audit.

## Count and scope

For Anka's BVH/mesh-query series, **two original merged changes were rolled back, by two Eric Shi commits: one full revert and one partial rollback**. Only one commit is a literal `This reverts commit` operation. It would be inaccurate to describe this as two entire PRs reverted, or three changes because the ray rollback also edits the GH-1843 changelog fragment.

| Original change | Eric's rollback on main | Date (UTC) | Extent |
| --- | --- | --- | --- |
| [`f005e099ce47d57829452a10532616be92ca9e45`](https://github.com/NVIDIA/warp/commit/f005e099ce47d57829452a10532616be92ca9e45), “Share one shrinking-radius core across mesh closest-point queries”, GH-1844, original GitLab MR !2897 | [`e6a1497290018224de9128adccad43c386c50eca`](https://github.com/NVIDIA/warp/commit/e6a1497290018224de9128adccad43c386c50eca), “Revert shared mesh query core”, GitLab MR !3045 | September 24, 03:09:09 (September 23 PDT) | Full revert of the original shared traversal and its changelog fragment, for the 1.18 release candidate. |
| [`fcc86dd5b90f2ef45dc96b21acac23e8ec4c52fe`](https://github.com/NVIDIA/warp/commit/fcc86dd5b90f2ef45dc96b21acac23e8ec4c52fe), “Use read-only loads in BVH and mesh query hot paths”, GH-1840, original GitLab MR !2880 | [`45c001d47199f83ae11d8ae798ba7da3e7b2ce31`](https://github.com/NVIDIA/warp/commit/45c001d47199f83ae11d8ae798ba7da3e7b2ce31), “Restore mesh ray-query occupancy”, GitLab MR !3046 | September 25, 03:10:06 (September 24 PDT) | Partial rollback: ordinary primitive/index loads in closest-hit, any-hit, ordered, and count-intersections. Applies to all architectures; there are no SM guards. |

The ray rollback retains AABB eager/read-only loads and closest-sign read-only loads. GH-1843's separate packed-leaf scalar-cursor optimization (`a22a1f38e2bdd814748e527553fa22a5866bd2db`, original MR !2883) is retained. Its fragment was reformatted, not reverted. The later release synchronization consumes GH-1840/GH-1843 fragments into the final changelog; fragment deletion there is not another code revert.

## What this means for our local candidates

- Our local exact-revert candidate `572d2063a` has a byte-identical `warp/native/mesh.h` to upstream's point-revert commit `e6a149729`. The release-safe point mitigation is already upstream. The compact-continuation candidate remains a separate, unpublished follow-up optimization and is not in current main.
- Current main already uses the user's preferred parent-preserving count-intersections choice: ordinary loads. Our earlier local architecture-scoped ray candidate would retain read-only loads on architectures without the observed regression and initially left count alone. Any resumed implementation needs to start from the actual new baseline and deliberately compare against upstream's broader ordinary-load policy.
- `warp/native/mesh.h` and `warp/native/bvh.h` have no changes between the ray rollback `45c001d471` and the freshly fetched tip. Before that rollback, `4eecc3c13` made Cppcheck-oriented casts/initialization changes; these are not performance rollbacks.
- Our September L40 performance tables compare against the pinned September baseline/parents, not this October tip. They must not be described as current-main measurements without revalidation.

## Other recent regression corrections, not counted as Anka BVH reverts

The entire 52-commit post-baseline history was inspected, plus a broader August–October revert/regression message search:

- [`c59167bbec2d78e1bd07e7af3d714ecce241d2ed`](https://github.com/NVIDIA/warp/commit/c59167bbec2d78e1bd07e7af3d714ecce241d2ed), “Preserve BSR and FEM performance [GH-1918]”, Eric, September 25 UTC: selects truncating division in proven-nonnegative hot paths while retaining correct signed floor semantics. The original correctness change was by Aamir Ahmed (`7784b3e3`), not Anka. This is a targeted performance repair, not a full revert.
- `15c295616`, “Remove misplaced external NVCC test”, Eric, September 23 UTC: removes an environment-dependent test from Eric's own GH-1920 change; runtime regression coverage remains.
- `9cd151ec0`, “Restore Marching Cubes bounds compatibility”, Eric, September 23 UTC: restores deprecated API aliases after the geometry migration; not a reversal of the BVH work or a performance revert.

No additional rollback of Anka's BVH/mesh-query series was found in this interval. This is not an all-time count of every change Eric has amended across the repository.

## Fresh primary evidence

Saved raw bodies, **all** comments and timelines using paginated `gh api` calls under `notes/evidence/latest-20261007/`. GH-1840 has three comments and GH-1844 two, with no newer comments since September 20. Both issues remain closed. Their timelines now reference the rollback commits. Commit-associated GitHub PR endpoints return no PRs; GitLab MR identifiers above are from the commit messages.

Eric's latest [GH-1840 comment](https://github.com/NVIDIA/warp/issues/1840#issuecomment-5750888119) recommended path/architecture-specific exceptions, including ordinary count loads on Thor when avoiding regressions relative to the parent is the priority. The **landed code is broader**: ordinary loads for all four affected ray paths on all architectures. Our report describes the code as landed, rather than assuming it exactly implements that earlier comment.

Eric's [GH-1844 comment](https://github.com/NVIDIA/warp/issues/1844#issuecomment-5750618281) recommended the release revert and compact continuations later. His revert commit attributes the Thor regression to additional stack traffic; our independent evidence still distinguishes deterministic resource expansion from a fully profiled causal breakdown.

Reproducible read-only checks:

```bash
git fetch upstream main
git log --format=fuller 061b4d01292b0d493f77a82c0767ef8e5d75e9a6..b5ea46659b86efb8f1b511904a3b5c254cd13d70
git show e6a1497290018224de9128adccad43c386c50eca
git show 45c001d47199f83ae11d8ae798ba7da3e7b2ce31
git diff 572d2063a e6a149729 -- warp/native/mesh.h
git diff 45c001d471 b5ea46659 -- warp/native/mesh.h warp/native/bvh.h
```

Both final diffs are empty. No posts, issue-state changes, pushes, or merge requests were made.
