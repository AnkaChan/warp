# Independent occupancy review, 2026-10-07

The compact point candidate retains an occupancy regression relative to the
original point parent and today's upstream: CUDA 13 signed queries use
65 rather than 63 registers on SM 75, 80, 86, 87, and 89. At 256 threads,
this reduces theoretical residency from four blocks to three. It is inherited
from the shared traversal, not newly introduced by the compact stack. It must
remain flagged when the acceptance criterion is avoiding regressions versus
the original parent. No candidate residency decrease versus the September
pinned main was found. CUDA 12.6 has no candidate residency decrease against
any audited reference.

The selected Thor count-intersections ordinary-load option changes 63 to
64 registers and remains at four blocks. This implements the user's priority
of avoiding the reported packed-leaf regressions versus the ray parent; it
gives up the reported default-leaf count benefit. It is still a local candidate
requiring Thor runtime validation.

## Scope and provenance

The review independently read all 50 new manifests: 300 CUDA 13.0.88 and
350 CUDA 12.6.85 compilations, with 1,170 kernel/target resource rows.
All saved CUBIN/PTX hashes, actual ELF target fields, PTX targets and entry
sets match their manifests. Raw logs agree with the saved shared-memory
counts. The 108 overlapping module/target reports from September agree with
the fresh CUDA 13 registers, stack, and spills. Original evidence was not edited.

| Label | Commit |
|---|---|
| September pinned main | `061b4d01292b0d493f77a82c0767ef8e5d75e9a6` |
| Local candidate checkout | `5fb0dbce37f006f63579dbd3360187711f6d4d61` |
| Latest upstream | `b5ea46659b86efb8f1b511904a3b5c254cd13d70` |
| Original point parent | `6c24bab2ba9d63df790748c6b2e97a6ff6e3037b` |
| Original ray parent | `64c6cacd6b9f106db9af0dabf5a591c0e41bdc80` |

The candidate `mesh.h` SHA-256 remains
`4f557762185da68deaac5c5b74596ab59a9086bd25ab95fdadc9c1df81f20de0`.
The separately selected count candidate is an edited checkout based on
`061b4d012`, with `mesh.h` SHA-256
`4e52384c56be1af9fd1b7d359ab9b30db3c25f5f077452f16bbab18b64f43db6`.
Header fingerprints, rather than checkout commit alone, identify these builds.

Latest upstream includes `e6a1497290018224de9128adccad43c386c50eca`
(point-core revert) and `45c001d47199f83ae11d8ae798ba7da3e7b2ce31`
(ordinary loads in closest, any, ordered, and count rays). Consequently,
comparing only against September main would conceal a current-upstream
occupancy disadvantage in the compact signed-point candidate.

## Occupancy calculation

`occupancy_check.cpp` calls NVIDIA's standalone `cudaOccMaxActiveBlocksPerMultiprocessor`
from the locally available CUDA Runtime 13.0.96 header. It loads no CUDA driver.
The header is
`/home/horde/.cache/uv/archive-v0/ggiamPOaiv0ZNdo0wCNmT/nvidia/cu13/include/cuda_occupancy.h`,
SHA-256 `163b8f296ef2a33c799e2dc8c525b37166da9217bcc6a3c1d26b1c10aca5e22b`.
The calculation includes register allocation granularity, register partitions,
per-block and per-SM limits, shared memory, warp limits, and barriers.
Inputs are 256 threads, zero dynamic shared memory, default carveout, the
reported static shared memory, and zero barriers as reported in every log.

NVIDIA's calculator allocates registers in units of 256 per warp for these
targets. Modern SMs have four register partitions; SM 60 has two. For the
four-partition case, each partition has 16,384 registers. With eight warps
per block, the register limit is:

```text
allocated registers/warp = round_up(32 * registers/thread, 256)
register-limited blocks = floor(4 * floor(16384 / allocated registers/warp) / 8)
```

This explicitly respects partitioning; dividing 65,536 by the raw per-block
register count is not the calculation used. For example, 63/64 registers
allocate 16,384 registers/block and allow four blocks. A count of 65 allocates
18,432 and allows three. Counts 69 and 72 also allow three. The 80/81 boundary
reduces three blocks to two. These examples were run against NVIDIA's API.
Register allocation granularity and the need to account for other resources
are also explained in the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#calculating-occupancy).

| Targets | Maximum warps/SM | Maximum blocks/SM | Registers/SM | Shared memory/SM |
|---|---:|---:|---:|---:|
| SM 75 | 32 | 16 | 65,536 | 64 KiB |
| SM 80 | 64 | 32 | 65,536 | 164 KiB |
| SM 86 | 48 | 16 | 65,536 | 100 KiB |
| SM 87, Orin | 48 | 16 | 65,536 | 164 KiB |
| SM 89, L40 | 48 | 24 | 65,536 | 100 KiB |
| SM 90, 100, 103 | 64 | 32 | 65,536 | 228 KiB |
| SM 110, Thor | 48 | 24 | 65,536 | 228 KiB |
| SM 120, 121 | 48 | 24 | 65,536 | 100 KiB |

These limits come from the [current CUDA Programming Guide tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html#features-and-technical-specifications),
with HTML column spans checked explicitly. Device mappings come from
[NVIDIA's GPU capability list](https://developer.nvidia.com/cuda/gpus).
The helper also covers SM 50/52/53/60/61/62/70/72 using the
[CUDA 12.6 specification table](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#features-and-technical-specifications),
including the 32,768-register block limit on SM 53 and SM 62.

SM 88 is supported by NVRTC 13, and NVIDIA groups it with SM 87/Orin in the
[Tile IR support matrix](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html).
The physical-limit table does not provide an SM 88 row. Its compiler resources
are retained, but the helper returns occupancy unavailable instead of assuming
SM 87 capacities.

The query kernels have 1,024 bytes of actual static shared memory for Warp's
`tile_shared_storage_t::smem_base[256]`, visible in PTX. The calculator adds
the separate 1 KiB driver reservation on SM 80 and later. This allocation
does not limit these queries. NVIDIA documents the reservation in the
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#unified-shared-memory-l1-texture-cache).
The stack is thread-local memory, not shared memory or registers. Reducing
the stack cannot by itself restore a register-limited resident block.

## Flagged point results

The following CUDA 13 register counts are parent / pinned main / candidate.
Latest upstream matches the original parent for these two query kernels.

| Targets | Unsigned registers | Signed registers | Signed blocks: parent/main/candidate |
|---|---|---|---|
| SM 75 | 56 / 59 / 58 | 63 / 65 / 65 | 4 / 3 / 3 |
| SM 80 | 55 / 57 / 58 | 63 / 65 / 65 | 4 / 3 / 3 |
| SM 86, 87, 89 | 54 / 56 / 56 | 63 / 65 / 65 | 4 / 3 / 3 |
| SM 88 | 54 / 56 / 56 | 63 / 65 / 65 | unavailable |
| SM 90 | 54 / 57 / 58 | 61 / 63 / 63 | 4 / 4 / 4 |
| SM 100, 103, 110 | 55 / 59 / 59 | 57 / 60 / 60 | 4 / 4 / 4 |
| SM 120, 121 | 54 / 56 / 57 | 55 / 60 / 59 | 4 / 4 / 4 |

The signed cliff is 100% to 75% theoretical occupancy on SM 75, 50% to 37.5%
on SM 80, and 66.7% to 50% on SM 86/87/89. The candidate reduces both point
stacks to 128 bytes, compared with 384 on pinned main. Original unsigned
stack was 128 bytes and signed stack 256 bytes. All spill counts remain zero.
Unsigned register increases on SM 80/90/120/121 versus pinned main do not
cross a 256-thread residency threshold, but remain register-pressure changes.

CUDA 12.6 produces different signed counts: candidate 68 versus pinned-main
67 and parent 65 on SM 86/87/89, all three blocks; SM 72/Xavier stays 71
registers and three blocks. On SM 50/52/53/60/61/62, compact signed queries
reduce 81 to 72 registers versus pinned main, restoring two blocks to three.
SM 60's original signed count is 65, so its candidate still uses seven more
registers without crossing another residency threshold. No CUDA 12.6
candidate residency decrease was found against parent, pinned main, or latest
upstream. This is not evidence that the CUDA 13 signed cliff is harmless.

## Rays and count priority

| CUDA 13 path/target | Pinned main registers | Candidate registers | Blocks before/after |
|---|---:|---:|---|
| Closest SM 100, 103 | 69 | 62 | 3 / 4 |
| Closest SM 110 | 69 | 62 | 3 / 4 |
| Any SM 110 | 63 | 56 | 4 / 4 |
| Ordered SM 110 | 63 | 64 | 4 / 4 |
| Selected count SM 110 | 63 | 64 | 4 / 4 |

Closest occupancy rises from 37.5% to 50% on SM 100/103, and from 50% to
66.7% on Thor. Any-hit and ordered regressions cannot be attributed to a
resident-block cliff in these wrappers. Their load-policy rationale also
depends on Eric's runtime observations. Count 63 and 64 allocate the same
16,384 registers per block, so selecting ordinary Thor count loads adds no
occupancy penalty here. It restores the original parent's count resource
tuple (64 registers, 192-byte stack, zero spills). Count on other targets,
closest-sign, and AABB paths retain their previous policy.

For the selected count option, all other target resource tuples are unchanged;
CUDA 12.6 cannot target Thor and does not compile that branch. Avoiding the
reported count regression is a workload choice, not a proof that every Thor
scene is faster. Prior Eric measurements reported roughly a 9.91% default
CuBQL count benefit and packed-leaf losses of 4.25–6.17% from read-only loads.

## Other flags and limits

- No spills were reported in any of the 1,170 kernel/target rows. Barriers
  were zero throughout, and shared-memory sizes did not increase.
- Auxiliary collision kernels in the spatial module use 33,792 static shared
  bytes; their occupancy can be shared-memory limited. Some register counts
  differ versus the older point parent, but none reduces calculated residency.
  They are included in the machine-readable audit rather than silently omitted.
- PTX 9.0 from CUDA 13 and PTX 8.5 from CUDA 12.6 were checked independently.
  PTX virtual register declarations are not physical register occupancy counts;
  the corresponding CUBIN assembler reports supply those counts.
- Compiling SM 87 or SM 110 does not validate loading that PTX on an installed
  Jetson driver. These checks executed no GPU kernel and measured no runtime
  occupancy or performance.
- The policy is chosen by the compile-time architecture. A lower virtual PTX
  target used on newer hardware keeps the lower-target policy. Mixed-GPU
  defaults or `ptx_target_arch` overrides therefore require a separate check.
- All integer targets advertised by each compiler were covered. Optional
  architecture/family suffixes `a` and `f` were not audited.

## Reproduction

```bash
g++ -std=c++17 -O2 -Wall -Wextra \
  -I /home/horde/.cache/uv/archive-v0/ggiamPOaiv0ZNdo0wCNmT/nvidia/cu13/include \
  notes/experiments/occupancy_check.cpp \
  -o notes/experiments/architecture-audit-20261007/occupancy-review/occupancy_check
uv run --no-sync notes/experiments/audit_occupancy.py \
  --root notes/experiments/architecture-audit-20261007 \
  --calculator notes/experiments/architecture-audit-20261007/occupancy-review/occupancy_check \
  --out notes/experiments/architecture-audit-20261007/occupancy-review/resources-annotated.json
```

The output records all input-manifest hashes, all resource rows and occupancy
limits, and every changed comparison. The ten reported decreasing comparisons
are the same signed-point cliff on five SMs against two references. The root
agent independently compiled the helper and repeated the audit. No production
native source was changed by this review.
