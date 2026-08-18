// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "intersect.h"

#ifdef __CUDA_ARCH__
#define BVH_SHARED_STACK 1
#else
#define BVH_SHARED_STACK 0
#endif

#define SAH_NUM_BUCKETS (16)
#define USE_LOAD4
#define BVH_QUERY_STACK_SIZE (32)

#define BVH_CONSTRUCTOR_SAH (0)
#define BVH_CONSTRUCTOR_MEDIAN (1)
#define BVH_CONSTRUCTOR_LBVH (2)
#define BVH_CONSTRUCTOR_CUBQL (-1)

#ifndef WP_BVH_BLOCK_DIM
#define WP_BVH_BLOCK_DIM 256
#endif

namespace wp {

// std_min() / std_max() follow C++ std::min() / std::max() semantics: first-argument wins on tie and on unordered
// comparison. Faster than wp::min / wp::max when inputs are known to be finite (BVH/mesh builders, refit loops, AABB
// queries on real geometry).
template <typename T> CUDA_CALLABLE inline T std_min(T a, T b) { return (b < a) ? b : a; }
template <typename T> CUDA_CALLABLE inline T std_max(T a, T b) { return (b > a) ? b : a; }

template <unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> std_min(const vec_t<Length, Type>& a, const vec_t<Length, Type>& b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = std_min(a[i], b[i]);
    }
    return ret;
}

template <unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> std_max(const vec_t<Length, Type>& a, const vec_t<Length, Type>& b)
{
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = std_max(a[i], b[i]);
    }
    return ret;
}

struct bounds3 {
    CUDA_CALLABLE inline bounds3()
        : lower(FLT_MAX)
        , upper(-FLT_MAX)
    {
    }

    CUDA_CALLABLE inline bounds3(const vec3& lower, const vec3& upper)
        : lower(lower)
        , upper(upper)
    {
    }

    CUDA_CALLABLE inline vec3 center() const { return 0.5f * (lower + upper); }
    CUDA_CALLABLE inline vec3 edges() const { return upper - lower; }

    CUDA_CALLABLE inline void expand(float r)
    {
        lower -= vec3(r);
        upper += vec3(r);
    }

    CUDA_CALLABLE inline void expand(const vec3& r)
    {
        lower -= r;
        upper += r;
    }

    CUDA_CALLABLE inline bool empty() const
    {
        return lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2];
    }

    CUDA_CALLABLE inline bool overlaps(const vec3& p) const
    {
        if (p[0] < lower[0] || p[1] < lower[1] || p[2] < lower[2] || p[0] > upper[0] || p[1] > upper[1]
            || p[2] > upper[2]) {
            return false;
        } else {
            return true;
        }
    }

    CUDA_CALLABLE inline bool overlaps(const bounds3& b) const
    {
        if (lower[0] > b.upper[0] || lower[1] > b.upper[1] || lower[2] > b.upper[2] || upper[0] < b.lower[0]
            || upper[1] < b.lower[1] || upper[2] < b.lower[2]) {
            return false;
        } else {
            return true;
        }
    }

    CUDA_CALLABLE inline bool overlaps(const vec3& b_lower, const vec3& b_upper) const
    {
        if (lower[0] > b_upper[0] || lower[1] > b_upper[1] || lower[2] > b_upper[2] || upper[0] < b_lower[0]
            || upper[1] < b_lower[1] || upper[2] < b_lower[2]) {
            return false;
        } else {
            return true;
        }
    }

    CUDA_CALLABLE inline void add_point(const vec3& p)
    {
        lower = std_min(lower, p);
        upper = std_max(upper, p);
    }

    CUDA_CALLABLE inline void add_bounds(const vec3& lower_other, const vec3& upper_other)
    {
        // lower_other will only impact the lower of the new bounds
        // upper_other will only impact the upper of the new bounds
        // this costs only half of the computation of adding lower_other and upper_other separately
        lower = std_min(lower, lower_other);
        upper = std_max(upper, upper_other);
    }

    CUDA_CALLABLE inline float area() const
    {
        vec3 e = upper - lower;
        return 2.0f * (e[0] * e[1] + e[0] * e[2] + e[1] * e[2]);
    }

    vec3 lower;
    vec3 upper;
};

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const vec3& b)
{
    return bounds3(std_min(a.lower, b), std_max(a.upper, b));
}

CUDA_CALLABLE inline bounds3 bounds_union(const bounds3& a, const bounds3& b)
{
    return bounds3(std_min(a.lower, b.lower), std_max(a.upper, b.upper));
}

CUDA_CALLABLE inline bounds3 bounds_intersection(const bounds3& a, const bounds3& b)
{
    return bounds3(std_max(a.lower, b.lower), std_min(a.upper, b.upper));
}

struct BVHPackedNodeHalf {
    float x;
    float y;
    float z;
    // For non-leaf nodes:
    // - 'lower.i' represents the index of the left child node.
    // - 'upper.i' represents the index of the right child node.
    //
    // For leaf nodes:
    // - 'lower.i' indicates the start index of the primitives in 'primitive_indices'.
    // - 'upper.i' indicates the index just after the last primitive in 'primitive_indices'
    unsigned int i : 31;
    unsigned int b : 1;
};

struct BVH {
    BVHPackedNodeHalf* node_lowers;
    BVHPackedNodeHalf* node_uppers;

    // used for fast refits
    int* node_parents;
    int* node_counts;
    // reordered primitive indices corresponds to the ordering of leaf nodes
    int* primitive_indices;
    // skip links for the stackless ray traversal: node_escapes[i] is the node
    // to continue at once node i's subtree is finished or culled (the right
    // sibling of the nearest ancestor, including i itself, that is a left
    // child; -1 past the root). Recomputed whenever topology changes.
    int* node_escapes;

    // maximum node depth counting the root as depth 1 (see also max_depth_ptr)
    int max_depth;
    int max_nodes;
    int num_nodes;
    // since we use packed leaf nodes, the number of them is no longer the number of items, but variable
    int num_leaf_nodes;

    // pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
    // representing the root of the tree, this is not always the first node
    // for bottom-up builders
    int* root;

    // maximum node depth of the tree counting the root as depth 1, stored
    // behind a device pointer so that in-place (graph-captured) rebuilds can
    // update it; null for host-resident trees, whose max_depth field is
    // authoritative. Consumed by bvh_query_pair_limit().
    int* max_depth_ptr;

    // item bounds are not owned by the BVH but by the caller
    vec3* item_lowers;
    vec3* item_uppers;
    int* item_groups;
    int num_items;
    int leaf_size;
    int constructor_type;

    // cuda context
    void* context;
};


CUDA_CALLABLE inline BVHPackedNodeHalf make_node(const vec3& bound, int child, bool leaf)
{
    BVHPackedNodeHalf n;
    n.x = bound[0];
    n.y = bound[1];
    n.z = bound[2];
    n.i = (unsigned int)child;
    n.b = (unsigned int)(leaf ? 1 : 0);

    return n;
}

// variation of make_node through volatile pointers used in build_hierarchy
CUDA_CALLABLE inline void make_node(volatile BVHPackedNodeHalf* n, const vec3& bound, int child, bool leaf)
{
    n->x = bound[0];
    n->y = bound[1];
    n->z = bound[2];
    n->i = (unsigned int)child;
    n->b = (unsigned int)(leaf ? 1 : 0);
}

#ifdef __CUDA_ARCH__
__device__ inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index)
{
#ifdef USE_LOAD4
    float4 f4 = __ldg((const float4*)(nodes) + index);
    return (const wp::BVHPackedNodeHalf&)f4;
    // return  (const wp::BVHPackedNodeHalf&)(*((const float4*)(nodes)+index));
#else
    return nodes[index];
#endif  // USE_LOAD4
}

// read-only loads for the remaining BVH/mesh query inputs (primitive indices,
// item bounds, mesh vertices); plain pointer dereferences compile to generic
// loads because the arrays are reached through a descriptor pointer, whereas
// __ldg uses the read-only data path
__device__ inline int bvh_load_int(const int* data, int index) { return __ldg(data + index); }

__device__ inline vec3 bvh_load_vec3(const vec3* data, int index)
{
    const float* p = reinterpret_cast<const float*>(data + index);
    return vec3(__ldg(p + 0), __ldg(p + 1), __ldg(p + 2));
}
#else
inline wp::BVHPackedNodeHalf bvh_load_node(const wp::BVHPackedNodeHalf* nodes, int index) { return nodes[index]; }

inline int bvh_load_int(const int* data, int index) { return data[index]; }

inline vec3 bvh_load_vec3(const vec3* data, int index) { return data[index]; }
#endif  // __CUDACC__

// escape target of `node`: walk up until we are a left child, then jump to
// the right sibling; -1 when the walk exits the root. Parents of reachable
// nodes are always internal, so their lower.i is a child index; the bounds
// and step guards keep the walk robust for uninitialized or muted nodes,
// whose escapes are never read by a traversal.
CUDA_CALLABLE inline int bvh_compute_node_escape(
    const BVHPackedNodeHalf* node_lowers,
    const BVHPackedNodeHalf* node_uppers,
    const int* node_parents,
    int node,
    int num_nodes
)
{
    int cur = node;
    for (int step = 0; step < num_nodes; ++step) {
        const int parent = node_parents[cur];
        if (parent < 0 || parent >= num_nodes)
            return -1;
        if (!node_lowers[parent].b && int(node_lowers[parent].i) == cur)
            return int(node_uppers[parent].i);
        cur = parent;
    }
    return -1;
}

CUDA_CALLABLE inline int clz(int x)
{
    int n;
    if (x == 0)
        return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1)
        ;
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n)
{
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits
template <int dim> CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z)
{
    uint32_t ux = clamp(int(x * dim), 0, dim - 1);
    uint32_t uy = clamp(int(y * dim), 0, dim - 1);
    uint32_t uz = clamp(int(z * dim), 0, dim - 1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

// making the class accessible from python

CUDA_CALLABLE inline BVH bvh_get(uint64_t id) { return *(BVH*)(id); }

CUDA_CALLABLE inline int bvh_get_num_bounds(uint64_t id)
{
    BVH bvh = bvh_get(id);
    return bvh.num_items;
}

CUDA_CALLABLE inline int get_leaf_group(const BVH& bvh, int leaf)
{
    if (!bvh.item_groups)
        return 0;
    return bvh.item_groups[bvh.primitive_indices[bvh.node_lowers[leaf].i]];
}

CUDA_CALLABLE inline int lower_bound_group(const BVH& bvh, int group)
{
    int lo = 0;
    int hi = bvh.num_leaf_nodes;

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (get_leaf_group(bvh, mid) < group) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo == bvh.num_leaf_nodes || (get_leaf_group(bvh, lo)) != group)
        return -1;

    return lo;
}

CUDA_CALLABLE inline int upper_bound_group(const BVH& bvh, int group)
{
    int lo = 0;
    int hi = bvh.num_leaf_nodes;

    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (get_leaf_group(bvh, mid) <= group) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    return lo;
}

CUDA_CALLABLE inline uint64_t bvh_query_node_pack(const BVHPackedNodeHalf& lower, const BVHPackedNodeHalf& upper)
{
    return (uint64_t(lower.b) << 62) | (uint64_t(upper.i) << 31) | uint64_t(lower.i);
}

CUDA_CALLABLE inline uint64_t bvh_query_node_load(const BVH& bvh, int node_index)
{
    const BVHPackedNodeHalf lower = bvh_load_node(bvh.node_lowers, node_index);
    const BVHPackedNodeHalf upper = bvh_load_node(bvh.node_uppers, node_index);
    return bvh_query_node_pack(lower, upper);
}

CUDA_CALLABLE inline bool bvh_query_node_is_leaf(uint64_t node) { return (node >> 62) != 0; }

CUDA_CALLABLE inline int bvh_query_node_lower_payload(uint64_t node) { return int(node & 0x7fffffffu); }

CUDA_CALLABLE inline int bvh_query_node_upper_payload(uint64_t node) { return int((node >> 31) & 0x7fffffffu); }

// Compile-time query-kind tag used by bvh_query_test and the traversal
// skeletons. Not stored in the struct; selection happens at Warp codegen time
// via distinct Python types (BvhQuery / BvhQueryRay / BvhQueryCapsule /
// BvhQuerySphere).
enum class BvhQueryKind : uint8_t { AABB = 0, RAY = 1, SPHERE = 2 };

// Far children pushed onto the 32-bit traversal stack are stored as a tagged
// pair of slots holding the node's packed payload, so popping them needs no
// memory access: the top slot has bit 31 set and holds the upper payload, the
// slot below it holds the lower payload and the leaf flag. When fewer than
// two slots are free the node index is pushed instead (bit 31 clear, indices
// are 31-bit) and the payload is re-loaded on pop.
CUDA_CALLABLE inline int bvh_query_stack_slot_lo(const BVHPackedNodeHalf& lower)
{
    return int(lower.i | (unsigned(lower.b) << 31));
}

CUDA_CALLABLE inline int bvh_query_stack_slot_hi(const BVHPackedNodeHalf& upper) { return int(upper.i | 0x80000000u); }

CUDA_CALLABLE inline uint64_t bvh_query_stack_unpack(unsigned slot_lo, unsigned slot_hi)
{
    return (uint64_t(slot_lo >> 31) << 62) | (uint64_t(slot_hi & 0x7fffffffu) << 31) | uint64_t(slot_lo & 0x7fffffffu);
}

// Largest stack occupancy at which a far child may still be pushed as a
// two-slot payload pair. A traversal never has more pending stack entries
// than internal nodes on a root-leaf path (max_depth - 1), and stopping pair
// pushes above this limit bounds live pairs so that total slot usage stays
// within BVH_QUERY_STACK_SIZE for every tree the constructors can produce
// (they hard-terminate at the stack depth). An unknown or out-of-range depth
// yields a negative limit, which disables pairs entirely and falls back to
// the exactly-safe one-slot-per-entry index encoding.
CUDA_CALLABLE inline int bvh_query_pair_limit(const BVH& bvh)
{
    int max_depth = bvh.max_depth;
#ifdef __CUDA_ARCH__
    if (bvh.max_depth_ptr)
        max_depth = __ldg(bvh.max_depth_ptr);
#else
    if (bvh.max_depth_ptr)
        max_depth = *bvh.max_depth_ptr;
#endif
    // grouped host builds restart the depth counter per group, so their
    // recorded depth is not a root-leaf bound
    if (max_depth < 1 || max_depth > BVH_QUERY_STACK_SIZE + 1 || bvh.item_groups)
        max_depth = BVH_QUERY_STACK_SIZE + 1;
    return std_min(64 - 2 * max_depth, BVH_QUERY_STACK_SIZE - 2);
}

CUDA_CALLABLE inline int lca(int node_a, int node_b, const int* parent)
{
    int da = 0, db = 0;
    for (int t = node_a; t != -1; t = parent[t])
        ++da;
    for (int t = node_b; t != -1; t = parent[t])
        ++db;

    if (da > db) {
        int diff = da - db;
        while (diff-- && node_a != -1)
            node_a = parent[node_a];
    } else if (db > da) {
        int diff = db - da;
        while (diff-- && node_b != -1)
            node_b = parent[node_b];
    }

    while (node_a != node_b) {
        if (node_a == -1 || node_b == -1)
            return -1;
        node_a = parent[node_a];
        node_b = parent[node_b];
    }
    return node_a;  // either the LCA or -1
}

// this function requires all the leaf nodes to be stored as the first bvh.num_leaf_nodes nodes
// and sorted by their group ids
CUDA_CALLABLE inline int bvh_get_group_root(uint64_t id, int group_id)
{
    BVH bvh = bvh_get(id);
    // locate first leaf of the current group
    const int first = lower_bound_group(bvh, group_id);
    if (first < 0)
        return -1;

    // find the first leaf with a greater group id to locate the last leaf of the current group
    const int last = upper_bound_group(bvh, group_id) - 1;

    return lca(first, last, bvh.node_parents);
}

// represents a strided stack in shared memory
// so each level of the stack is stored contiguously
// across the block
struct bvh_stack_t {
    CUDA_CALLABLE inline int operator[](int depth) const { return ptr[depth * WP_TILE_BLOCK_DIM]; }
    CUDA_CALLABLE inline int& operator[](int depth) { return ptr[depth * WP_TILE_BLOCK_DIM]; }

    int* ptr;
};

// stores state required to traverse the BVH nodes that
// overlap with a query AABB.
struct bvh_query_t {
    CUDA_CALLABLE bvh_query_t()
        : bvh()
        , stack()
        , count(0)
        , input_lower()
        , input_upper()
        , bounds_nr(0)
        , prim_cur(0)
        , prim_end(0)
        , cur_node(0)
        , have_node(false)
        , cursor(-1)
        , end_cursor(-1)
        , pair_limit(-1)
        , last_query_valid(true)
        , radius(0.0f)
        , radius_sq(0.0f)
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline bvh_query_t& operator+=(const bvh_query_t& other) { return *this; }

    BVH bvh;

    // BVH traversal stack of node indices; every entry passed its AABB/ray
    // test before being pushed.
    // On CUDA the stack lives in shared memory: keeping an array out of this
    // struct lets the compiler keep the remaining members in registers.
#if BVH_SHARED_STACK
    bvh_stack_t stack;
#else
    int stack[BVH_QUERY_STACK_SIZE];
#endif

    int count;

    // primitive range of the packed leaf currently being enumerated;
    // when prim_cur < prim_end the query resumes mid-leaf on the next
    // bvh_query_next() call
    int prim_cur;
    int prim_end;

    // packed payload (see bvh_query_node_pack()) of the node to process next,
    // valid when have_node is set; it already passed its intersection test
    // (AABB traversal only)
    uint64_t cur_node;
    bool have_node;

    // stackless ray traversal: next node to visit; finished when
    // cursor == end_cursor (the escape target of the query root's subtree)
    int cursor;
    int end_cursor;

    // stack occupancy up to which far children may be pushed as two-slot
    // payload pairs (see bvh_query_pair_limit())
    int pair_limit;

    // inputs
    wp::vec3 input_lower;  // start for ray
    wp::vec3 input_upper;  // dir for ray

    int bounds_nr;
    // Tracks whether the most recent bvh_query_next() / tile_bvh_query_next() call
    // produced a valid index. Seeded to true on construction so an initial
    // tile_query_valid() check (before any next() call) reports valid.
    bool last_query_valid;
    // Minkowski-offset: sphere radius, or ray inflation radius (0 => plain ray / aabb).
    float radius;
    float radius_sq;  // pre-computed radius*radius for sphere/capsule node tests
};

// Node/primitive overlap test, specialized per query kind at compile time: `if constexpr`
// discards the untaken branches, so each instantiation folds to a dispatch-free test.
// The ray variants also apply the max_dist predicate (closed endpoint for capsules,
// half-open for plain rays, matching the original behavior of each).
template <BvhQueryKind QUERY_KIND, bool HAS_RADIUS>
CUDA_CALLABLE inline bool
bvh_query_test(const bvh_query_t& query, const vec3& node_lower, const vec3& node_upper, const float& max_dist)
{
    if constexpr (QUERY_KIND == BvhQueryKind::SPHERE) {
        // exact sphere-AABB node test using pre-computed radius_sq
        return intersect_sphere_aabb(query.input_lower, query.radius_sq, node_lower, node_upper);
    } else if constexpr (QUERY_KIND == BvhQueryKind::RAY) {
        float t = FLT_MAX;
        if constexpr (HAS_RADIUS) {
            // Capsule: inflate bounds by radius and use the robust slab test so axis-aligned
            // directions (rcp_dir = +-inf) correctly handle tangent slabs instead of 0*inf = NaN.
            bool hit = intersect_ray_aabb_robust(
                query.input_lower,
                vec3(1.0f / query.input_upper[0], 1.0f / query.input_upper[1], 1.0f / query.input_upper[2]),
                query.input_upper, node_lower - vec3(query.radius), node_upper + vec3(query.radius), t
            );
            return hit && !(t > max_dist);
        } else {
            // Plain ray (radius == 0): original slab test, identical to pre-PR behavior.
            bool hit = intersect_ray_aabb(query.input_lower, query.input_upper, node_lower, node_upper, t);
            return hit && !(t >= max_dist);
        }
    } else {
        return intersect_aabb_aabb(query.input_lower, query.input_upper, node_lower, node_upper);
    }
}

CUDA_CALLABLE inline bvh_query_t bvh_query_common(uint64_t id, const vec3& lower, const vec3& upper)
{
    bvh_query_t query;
    query.bounds_nr = -1;
    query.bvh = bvh_get(id);
    query.input_lower = lower;
    query.input_upper = upper;
    return query;
}

// The shared-memory traversal stack is used by volume and capsule queries;
// plain ray queries remain stackless. Kept in a single non-template function
// so every stack-based query shares one block-wide allocation.
CUDA_CALLABLE inline void bvh_query_alloc_stack(bvh_query_t& query)
{
#if BVH_SHARED_STACK
    __shared__ int stack[BVH_QUERY_STACK_SIZE * WP_TILE_BLOCK_DIM];
    query.stack.ptr = &stack[threadIdx.x];
#endif
}

// Volume queries (AABB, sphere) traverse with pre-tested stack entries and a
// register-carried current node, so the root is tested here.
template <BvhQueryKind QUERY_KIND, bool HAS_RADIUS>
CUDA_CALLABLE inline void bvh_query_init_volume(bvh_query_t& query, int root)
{
    const BVH& bvh = query.bvh;
    bvh_query_alloc_stack(query);

    const int root_index = (root == -1) ? *bvh.root : root;

    const BVHPackedNodeHalf root_lower = bvh_load_node(bvh.node_lowers, root_index);
    const BVHPackedNodeHalf root_upper = bvh_load_node(bvh.node_uppers, root_index);

    query.pair_limit = bvh_query_pair_limit(bvh);

    if (bvh_query_test<QUERY_KIND, HAS_RADIUS>(
            query, reinterpret_cast<const vec3&>(root_lower), reinterpret_cast<const vec3&>(root_upper), FLT_MAX
        )) {
        query.cur_node = bvh_query_node_pack(root_lower, root_upper);
        query.have_node = true;
    }
}

// Plain ray queries traverse stackless via the skip links.
CUDA_CALLABLE inline void bvh_query_init_directed(bvh_query_t& query, int root)
{
    const BVH& bvh = query.bvh;
    const int root_index = (root == -1) ? *bvh.root : root;
    query.cursor = root_index;
    // the traversal ends once the cursor escapes the query root's subtree
    // (-1 when the query root is the tree root)
    query.end_cursor = bvh_load_int(bvh.node_escapes, root_index);
}

CUDA_CALLABLE inline bvh_query_t bvh_query(uint64_t id, const vec3& lower, const vec3& upper, int root)
{
    bvh_query_t query = bvh_query_common(id, lower, upper);
    bvh_query_init_volume<BvhQueryKind::AABB, false>(query, root);
    return query;
}

CUDA_CALLABLE inline bvh_query_t bvh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper, int root)
{
    return bvh_query(id, lower, upper, root);
}

CUDA_CALLABLE inline bvh_query_t bvh_query_ray(uint64_t id, const vec3& start, const vec3& dir, int root)
{
    bvh_query_t query = bvh_query_common(id, start, 1.0f / dir);
    bvh_query_init_directed(query, root);
    return query;
}

CUDA_CALLABLE inline bvh_query_t
bvh_query_capsule(uint64_t id, const vec3& start, const vec3& dir, float radius, int root)
{
    // Like the plain ray, input_upper stores the reciprocal direction;
    // bvh_query_test<RAY, true> derives the raw direction from it.
    bvh_query_t query = bvh_query_common(id, start, 1.0f / dir);
    query.radius = max(radius, 0.0f);
    query.radius_sq = query.radius * query.radius;
    bvh_query_alloc_stack(query);
    query.stack[0] = root == -1 ? *query.bvh.root : root;
    query.count = 1;
    return query;
}

// Sphere query: returns all bounds within `radius` of `center` using an exact
// sphere-AABB node test.
CUDA_CALLABLE inline bvh_query_t bvh_query_sphere(uint64_t id, const vec3& center, float radius, int root)
{
    bvh_query_t query = bvh_query_common(id, center, center);
    // the radius must be set before the volume init tests the root
    query.radius = max(radius, 0.0f);
    query.radius_sq = query.radius * query.radius;
    bvh_query_init_volume<BvhQueryKind::SPHERE, false>(query, root);
    return query;
}

// Volume traversal (AABB, sphere): a single flat loop; every iteration either
// emits one primitive from the packed leaf currently being enumerated, or
// processes the node carried over in registers / popped from the shared
// stack. Children are tested at the parent; the near child rides in
// registers and far children go on the stack as tagged payload pairs while
// the tree depth proves they fit (see bvh_query_pair_limit), so pops need no
// memory access.
template <BvhQueryKind QUERY_KIND, bool HAS_RADIUS>
CUDA_CALLABLE inline bool bvh_query_next_volume(bvh_query_t& query, int& index, const float& max_dist)
{
    BVH bvh = query.bvh;

    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = bvh_load_int(bvh.primitive_indices, query.prim_cur++);

            // load the item bounds eagerly so the test below compiles to one
            // predicate chain instead of a branch per component
            const vec3 item_lower = bvh_load_vec3(bvh.item_lowers, primitive_index);
            const vec3 item_upper = bvh_load_vec3(bvh.item_uppers, primitive_index);

            if (bvh_query_test<QUERY_KIND, HAS_RADIUS>(query, item_lower, item_upper, max_dist)) {
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }
            continue;
        }

        if (!query.have_node) {
            if (!query.count)
                return false;

            const unsigned top = unsigned(query.stack[--query.count]);
            if (top & 0x80000000u) {
                // payload pair: the node is reconstructed without any memory access
                query.cur_node = bvh_query_stack_unpack(unsigned(query.stack[--query.count]), top);
            } else {
                // index entry: it already passed its test, so the AABB part
                // of this load is unused and no re-test is needed
                query.cur_node = bvh_query_node_load(bvh, int(top));
            }
        }

        const uint64_t node = query.cur_node;
        query.have_node = false;

        if (bvh_query_node_is_leaf(node)) {
            const int start = bvh_query_node_lower_payload(node);
            const int end = bvh_query_node_upper_payload(node);

            // fast path when the leaf contains exactly one primitive: its
            // AABB is the leaf node's AABB, which already passed its test
            if (end - start == 1) {
                const int primitive_index = bvh_load_int(bvh.primitive_indices, start);
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }

            // packed leaf: enumerate its primitives one per loop iteration,
            // without re-loading the leaf node
            query.prim_cur = start;
            query.prim_end = end;
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(node);
        const int right_index = bvh_query_node_upper_payload(node);

        const BVHPackedNodeHalf left_lower = bvh_load_node(bvh.node_lowers, left_index);
        const BVHPackedNodeHalf left_upper = bvh_load_node(bvh.node_uppers, left_index);
        const BVHPackedNodeHalf right_lower = bvh_load_node(bvh.node_lowers, right_index);
        const BVHPackedNodeHalf right_upper = bvh_load_node(bvh.node_uppers, right_index);

        const bool hit_left = bvh_query_test<QUERY_KIND, HAS_RADIUS>(
            query, reinterpret_cast<const vec3&>(left_lower), reinterpret_cast<const vec3&>(left_upper), max_dist
        );
        const bool hit_right = bvh_query_test<QUERY_KIND, HAS_RADIUS>(
            query, reinterpret_cast<const vec3&>(right_lower), reinterpret_cast<const vec3&>(right_upper), max_dist
        );

        if (hit_left) {
            query.cur_node = bvh_query_node_pack(left_lower, left_upper);
            query.have_node = true;
            if (hit_right) {
                // pairs start below pair_limit so that slot usage can never
                // exceed the stack for constructor-produced trees; the final
                // guard only matters for depths beyond the construction bound
                if (query.count <= query.pair_limit) {
                    query.stack[query.count++] = bvh_query_stack_slot_lo(right_lower);
                    query.stack[query.count++] = bvh_query_stack_slot_hi(right_upper);
                } else if (query.count < BVH_QUERY_STACK_SIZE) {
                    query.stack[query.count++] = right_index;
                }
            }
        } else if (hit_right) {
            query.cur_node = bvh_query_node_pack(right_lower, right_upper);
            query.have_node = true;
        }
    }
}

// Directed traversal (plain ray): stackless skip-link walk. Every visited
// node is loaded and tested exactly once; a culled or finished subtree jumps
// to its precomputed escape node, an overlapped internal node descends to its
// left child. State is two scalar cursors, so directed queries need no
// traversal stack, no shared memory, and have no depth limit.
template <bool HAS_RADIUS>
CUDA_CALLABLE inline bool bvh_query_next_directed(bvh_query_t& query, int& index, const float& max_dist)
{
    BVH bvh = query.bvh;

    for (;;) {
        // emit remaining primitives of the current packed leaf, one per iteration
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = bvh_load_int(bvh.primitive_indices, query.prim_cur++);

            // load the item bounds eagerly so the tests below compile to one
            // predicate chain instead of a branch per component
            const vec3 item_lower = bvh_load_vec3(bvh.item_lowers, primitive_index);
            const vec3 item_upper = bvh_load_vec3(bvh.item_uppers, primitive_index);

            if (bvh_query_test<BvhQueryKind::RAY, HAS_RADIUS>(query, item_lower, item_upper, max_dist)) {
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }
            continue;
        }

        if (query.cursor == query.end_cursor)
            return false;

        const int node_index = query.cursor;

        BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
        BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

        if (!bvh_query_test<BvhQueryKind::RAY, HAS_RADIUS>(
                query, reinterpret_cast<vec3&>(node_lower), reinterpret_cast<vec3&>(node_upper), max_dist
            )) {
            // subtree culled: jump over it
            query.cursor = bvh_load_int(bvh.node_escapes, node_index);
            continue;
        }

        if (node_lower.b) {
            const int start = node_lower.i;
            const int end = node_upper.i;

            // the leaf is fully handled by the emission loop; move the cursor on
            query.cursor = bvh_load_int(bvh.node_escapes, node_index);

            // fast path when the leaf contains exactly one primitive: its
            // AABB is the leaf node's AABB, which just passed the test above
            if (end - start == 1) {
                const int primitive_index = bvh_load_int(bvh.primitive_indices, start);
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }

            query.prim_cur = start;
            query.prim_end = end;
            continue;
        }

        // overlapped internal node: descend to the left child
        query.cursor = node_lower.i;
    }
}

// Capsule queries use a test-on-pop stack so a caller may tighten max_dist
// between iterator calls. Packed-leaf items are still tested individually.
CUDA_CALLABLE inline bool bvh_query_next_capsule(bvh_query_t& query, int& index, const float& max_dist)
{
    BVH bvh = query.bvh;

    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = bvh_load_int(bvh.primitive_indices, query.prim_cur++);
            const vec3 item_lower = bvh_load_vec3(bvh.item_lowers, primitive_index);
            const vec3 item_upper = bvh_load_vec3(bvh.item_uppers, primitive_index);

            if (bvh_query_test<BvhQueryKind::RAY, true>(query, item_lower, item_upper, max_dist)) {
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }
            continue;
        }

        if (!query.count)
            return false;

        const int node_index = query.stack[--query.count];
        const BVHPackedNodeHalf node_lower = bvh_load_node(bvh.node_lowers, node_index);
        const BVHPackedNodeHalf node_upper = bvh_load_node(bvh.node_uppers, node_index);

        if (!bvh_query_test<BvhQueryKind::RAY, true>(
                query, reinterpret_cast<const vec3&>(node_lower), reinterpret_cast<const vec3&>(node_upper), max_dist
            )) {
            continue;
        }

        if (node_lower.b) {
            const int start = node_lower.i;
            const int end = node_upper.i;
            if (end - start == 1) {
                const int primitive_index = bvh_load_int(bvh.primitive_indices, start);
                index = primitive_index;
                query.bounds_nr = primitive_index;
                return true;
            }

            query.prim_cur = start;
            query.prim_end = end;
            continue;
        }

        query.stack[query.count++] = node_upper.i;
        query.stack[query.count++] = node_lower.i;
    }
}

// Per-kind public iterators. Each is a thin wrapper around the shared template
// skeletons; the caller's Python type determines which one gets emitted, so
// NVCC sees only one traversal loop per kernel -- no runtime dispatch.

// AABB query iterator (backward-compatible; BvhQuery type, existing callers unchanged)
CUDA_CALLABLE inline bool bvh_query_next(bvh_query_t& query, int& index, const float& max_dist)
{
    return bvh_query_next_volume<BvhQueryKind::AABB, false>(query, index, max_dist);
}

// Plain ray iterator (BvhQueryRay type)
CUDA_CALLABLE inline bool bvh_query_ray_next(bvh_query_t& query, int& index, const float& max_dist)
{
    return bvh_query_next_directed<false>(query, index, max_dist);
}

// Capsule iterator (BvhQueryCapsule type)
CUDA_CALLABLE inline bool bvh_query_capsule_next(bvh_query_t& query, int& index, const float& max_dist)
{
    return bvh_query_next_capsule(query, index, max_dist);
}

// Sphere iterator (BvhQuerySphere type)
CUDA_CALLABLE inline bool bvh_query_sphere_next(bvh_query_t& query, int& index, const float& max_dist)
{
    return bvh_query_next_volume<BvhQueryKind::SPHERE, false>(query, index, max_dist);
}

CUDA_CALLABLE inline int iter_next(bvh_query_t& query) { return query.bounds_nr; }

CUDA_CALLABLE inline bool iter_cmp(bvh_query_t& query)
{
    float max_dist = FLT_MAX;
    bool finished = bvh_query_next(query, query.bounds_nr, max_dist);
    return finished;
}

CUDA_CALLABLE inline bvh_query_t iter_reverse(const bvh_query_t& query)
{
    // can't reverse BVH queries, users should not rely on traversal ordering
    return query;
}

CUDA_CALLABLE bool bvh_get_descriptor(uint64_t id, BVH& bvh);
CUDA_CALLABLE void bvh_add_descriptor(uint64_t id, const BVH& bvh);
CUDA_CALLABLE void bvh_rem_descriptor(uint64_t id);


void bvh_create_host(
    vec3* lowers, vec3* uppers, int num_items, int constructor_type, int* groups, int leaf_size, BVH& bvh
);
// (re)compute node_escapes for a host-resident tree, allocating it if needed
void bvh_compute_escapes_host(BVH& bvh);
void bvh_destroy_host(wp::BVH& bvh);
void bvh_refit_host(wp::BVH& bvh);
void cubql_bvh_create_host(vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh);
void cubql_bvh_destroy_host(BVH& bvh);
void cubql_bvh_refit_host(BVH& bvh);
void cubql_bvh_rebuild_host(BVH& bvh);
// reorder a top-down-constructed bvh so its structure accords with a bottom-up tree:
// all of its leaves nodes are stored as the first bvh.num_leaf_nodes nodes
void reorder_top_down_bvh(BVH& bvh_host);

#if WP_ENABLE_CUDA

void bvh_create_device(
    void* context,
    vec3* lowers,
    vec3* uppers,
    int num_items,
    int constructor_type,
    int* groups,
    int leaf_size,
    BVH& bvh_device_on_host
);
void bvh_destroy_device(BVH& bvh);
void bvh_refit_device(BVH& bvh);
// Copy a host-built BVH to the device. Reorders leaves to the front unless the
// BVH is grouped or constructed by cuBQL (those layouts must be preserved).
void copy_host_tree_to_device(void* context, BVH& bvh_host, BVH& bvh_device_on_host);
void cubql_bvh_create_device(
    void* context, vec3* lowers, vec3* uppers, int num_items, int leaf_size, BVH& bvh_device_on_host
);
void cubql_bvh_destroy_device(BVH& bvh);
// Returns true on success and false when refit fails; callers should propagate failures.
bool cubql_bvh_refit_device(BVH& bvh);
void cubql_bvh_rebuild_device(BVH& bvh);

#endif  // WP_ENABLE_CUDA

}  // namespace wp


#include "tile_bvh.h"
