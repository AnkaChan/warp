// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "array.h"
#include "bvh.h"
#include "intersect.h"
#include "rand.h"
#include "solid_angle.h"

#define BVH_DEBUG 0

namespace wp {

struct Mesh {
    array_t<vec3> points;
    array_t<vec3> velocities;

    array_t<int> indices;

    vec3* lowers;
    vec3* uppers;

    SolidAngleProps* solid_angle_props;

    int num_points;
    int num_tris;

    BVH bvh;

    void* context;
    float average_edge_length;

    inline CUDA_CALLABLE Mesh(int id = 0)
    {
        // for backward a = 0 initialization syntax
        lowers = nullptr;
        uppers = nullptr;
        num_points = 0;
        num_tris = 0;
        context = nullptr;
        solid_angle_props = nullptr;
        average_edge_length = 0.0f;
        bvh = BVH {};
    }

    inline CUDA_CALLABLE Mesh(
        array_t<vec3> points,
        array_t<vec3> velocities,
        array_t<int> indices,
        int num_points,
        int num_tris,
        void* context = nullptr
    )
        : points(points)
        , velocities(velocities)
        , indices(indices)
        , num_points(num_points)
        , num_tris(num_tris)
        , context(context)
    {
        lowers = nullptr;
        uppers = nullptr;
        solid_angle_props = nullptr;
        average_edge_length = 0.0f;
        bvh = BVH {};
    }
};

CUDA_CALLABLE inline Mesh mesh_get(uint64_t id) { return *(Mesh*)(id); }

CUDA_CALLABLE inline int mesh_get_group_root(uint64_t id, int group_id)
{
    Mesh* mesh = (Mesh*)(id);
    return bvh_get_group_root((uint64_t)&mesh->bvh, group_id);
}


CUDA_CALLABLE inline Mesh& operator+=(Mesh& a, const Mesh& b)
{
    // dummy operator needed for adj_select involving meshes
    return a;
}

CUDA_CALLABLE inline float distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
    const float dx = min(upper[0], max(lower[0], p[0])) - p[0];
    const float dy = min(upper[1], max(lower[1], p[1])) - p[1];
    const float dz = min(upper[2], max(lower[2], p[2])) - p[2];
    return dx * dx + dy * dy + dz * dz;
}

CUDA_CALLABLE inline float furthest_distance_to_aabb_sq(const vec3& p, const vec3& lower, const vec3& upper)
{
    // X-axis
    float dist_lower_x = fabs(p[0] - lower[0]);
    float dist_upper_x = fabs(p[0] - upper[0]);
    float corner_diff_x = (dist_lower_x > dist_upper_x) ? dist_lower_x : dist_upper_x;

    // Y-axis
    float dist_lower_y = fabs(p[1] - lower[1]);
    float dist_upper_y = fabs(p[1] - upper[1]);
    float corner_diff_y = (dist_lower_y > dist_upper_y) ? dist_lower_y : dist_upper_y;

    // Z-axis
    float dist_lower_z = fabs(p[2] - lower[2]);
    float dist_upper_z = fabs(p[2] - upper[2]);
    float corner_diff_z = (dist_lower_z > dist_upper_z) ? dist_lower_z : dist_upper_z;

    // Calculate and return the distance
    return corner_diff_x * corner_diff_x + corner_diff_y * corner_diff_y + corner_diff_z * corner_diff_z;
}

CUDA_CALLABLE inline int
mesh_query_ray_count_intersections(uint64_t id, const vec3& start, const vec3& dir, int root = -1);
CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p);
CUDA_CALLABLE inline float
mesh_query_inside_parity(uint64_t id, const vec3& p, const vec3 base_dir, int n_sample, float perturbation_scale);

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool
mesh_query_point(uint64_t id, const vec3& point, float max_dist, float& inside, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;

    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }

#if BVH_DEBUG

            tests++;

            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);

            if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist * max_dist) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif

        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }


#if BVH_DEBUG
    printf("%d\n", tests);

    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;

    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;

    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = sqrtf(min_dist_sq);

        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );

        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }

        fclose(f);
    }
#endif

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        inside = mesh_query_inside_ray_tracing(id, point);

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    int n_sample = 1,
    float perturbation_scale = 0.1f
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;

    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }

#if BVH_DEBUG

            tests++;

            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);

            if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist * max_dist) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif

        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }


#if BVH_DEBUG
    printf("%d\n", tests);

    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;

    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;

    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = sqrtf(min_dist_sq);

        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );

        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }

        fclose(f);
    }
#endif

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        inside = mesh_query_inside_parity(id, point, vec3(1.f, 1.f, 1.f), n_sample, perturbation_scale);

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool
mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;

    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
                const int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
                const int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
                const int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

                const vec3 p = bvh_load_vec3(mesh.points, i);
                const vec3 q = bvh_load_vec3(mesh.points, j);
                const vec3 r = bvh_load_vec3(mesh.points, k);
                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }

#if BVH_DEBUG

            tests++;

            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);

            if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist * max_dist) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif

        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }


#if BVH_DEBUG
    printf("%d\n", tests);

    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;

    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;

    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = sqrtf(min_dist_sq);

        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );

        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }

        fclose(f);
    }
#endif

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline void mesh_query_point_no_sign_update_primitive(
    const Mesh& mesh,
    int primitive_index,
    const vec3& point,
    float& min_dist_sq,
    int& min_face,
    float& min_v,
    float& min_w
)
{
    const int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
    const int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
    const int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

    const vec3 p = bvh_load_vec3(mesh.points, i);
    const vec3 q = bvh_load_vec3(mesh.points, j);
    const vec3 r = bvh_load_vec3(mesh.points, k);
    vec3 e0 = q - p;
    vec3 e1 = r - p;
    vec3 e2 = r - q;
    vec3 normal = cross(e0, e1);

    // Match mesh_query_point_no_sign() by ignoring sliver triangles. Keep the
    // same arithmetic as the stock query: squaring this comparison can
    // overflow for large but otherwise valid triangles.
    if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
        return;

    vec2 barycentric = closest_point_to_triangle(p, q, r, point);
    float triangle_u = barycentric[0];
    float triangle_v = barycentric[1];
    float triangle_w = 1.f - triangle_u - triangle_v;
    vec3 closest = triangle_u * p + triangle_v * q + triangle_w * r;

    float dist_sq = length_sq(closest - point);

    if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        min_v = triangle_v;
        min_w = triangle_w;
        min_face = primitive_index;
    }
}

CUDA_CALLABLE inline void mesh_query_point_no_sign_update_leaf(
    const Mesh& mesh,
    int start,
    int end,
    const vec3& point,
    float& min_dist_sq,
    int& min_face,
    float& min_v,
    float& min_w
)
{
    for (int primitive_counter = start; primitive_counter < end; ++primitive_counter) {
        const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
        mesh_query_point_no_sign_update_primitive(mesh, primitive_index, point, min_dist_sq, min_face, min_v, min_w);
    }
}

template <bool TestStartNode, bool SkipLeaf>
CUDA_CALLABLE inline void mesh_query_point_no_sign_traverse(
    const Mesh& mesh,
    int start_node,
    int skip_leaf,
    const vec3& point,
    float& min_dist_sq,
    int& min_face,
    float& min_v,
    float& min_w
)
{
    int stack[BVH_QUERY_STACK_SIZE];
    int count = 0;

    const BVHPackedNodeHalf start_lower = bvh_load_node(mesh.bvh.node_lowers, start_node);
    const BVHPackedNodeHalf start_upper = bvh_load_node(mesh.bvh.node_uppers, start_node);
    if (TestStartNode
        && distance_to_aabb_sq(
               point, reinterpret_cast<const vec3&>(start_lower), reinterpret_cast<const vec3&>(start_upper)
           ) > min_dist_sq) {
        return;
    }

    uint64_t node = bvh_query_node_pack(start_lower, start_upper);
    bool have_node = true;

    // Carry the near child in registers and store only far-child indices.
    // Popped far children are re-tested against the current shrinking radius;
    // the packed leaf payload is passed through without reloading the leaf.
    while (have_node || count) {
        if (!have_node) {
            const int node_index = stack[--count];
            const BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
            const BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

            const float node_dist_sq = distance_to_aabb_sq(
                point, reinterpret_cast<const vec3&>(lower), reinterpret_cast<const vec3&>(upper)
            );
            if (node_dist_sq > min_dist_sq)
                continue;

            node = bvh_query_node_pack(lower, upper);
        }
        have_node = false;

        if (bvh_query_node_is_leaf(node)) {
            mesh_query_point_no_sign_update_leaf(
                mesh, bvh_query_node_lower_payload(node), bvh_query_node_upper_payload(node), point, min_dist_sq,
                min_face, min_v, min_w
            );
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(node);
        const int right_index = bvh_query_node_upper_payload(node);
        const BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        const BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        const BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        const BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        const float left_dist_sq = distance_to_aabb_sq(
            point, reinterpret_cast<const vec3&>(left_lower), reinterpret_cast<const vec3&>(left_upper)
        );
        const float right_dist_sq = distance_to_aabb_sq(
            point, reinterpret_cast<const vec3&>(right_lower), reinterpret_cast<const vec3&>(right_upper)
        );

        // Ties visit the right child first, matching the existing traversal.
        if (left_dist_sq < right_dist_sq) {
            if (right_dist_sq < min_dist_sq && (!SkipLeaf || right_index != skip_leaf))
                stack[count++] = right_index;
            if (left_dist_sq < min_dist_sq && (!SkipLeaf || left_index != skip_leaf)) {
                node = bvh_query_node_pack(left_lower, left_upper);
                have_node = true;
            }
        } else {
            if (left_dist_sq < min_dist_sq && (!SkipLeaf || left_index != skip_leaf))
                stack[count++] = left_index;
            if (right_dist_sq < min_dist_sq && (!SkipLeaf || right_index != skip_leaf)) {
                node = bvh_query_node_pack(right_lower, right_upper);
                have_node = true;
            }
        }
    }
}

CUDA_CALLABLE inline int mesh_query_point_no_sign_initialize_seed_leaf(
    const Mesh& mesh, int seed_face, const vec3& point, float& min_dist_sq, int& min_face, float& min_v, float& min_w
)
{
    if (seed_face < 0 || seed_face >= mesh.num_tris)
        return -1;

    const int seed_leaf = bvh_get_primitive_leaf(mesh.bvh, seed_face);
    if (seed_leaf < 0 || seed_leaf >= mesh.bvh.num_nodes)
        return -1;

    const BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, seed_leaf);
    if (!lower.b)
        return -1;

    const BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, seed_leaf);
    const int start = int(lower.i);
    const int end = int(upper.i);
    if (end - start == 1) {
        mesh_query_point_no_sign_update_primitive(mesh, seed_face, point, min_dist_sq, min_face, min_v, min_w);
    } else {
        for (int primitive_counter = start; primitive_counter < end; ++primitive_counter) {
            const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, primitive_counter);
            mesh_query_point_no_sign_update_primitive(
                mesh, primitive_index, point, min_dist_sq, min_face, min_v, min_w
            );
        }
    }
    return seed_leaf;
}

CUDA_CALLABLE inline bool
mesh_query_point_no_sign_exclusive_contains_distance(const BVHExclusiveNode& node, const vec3& point, float distance_sq)
{
    // A radius-r sphere lies strictly inside the exclusive box exactly when
    // r is smaller than all six point-to-plane margins. The minimum of the
    // rounded float differences is monotone; stepping that minimum down by
    // one representable value therefore cannot enlarge the exact margin.
    // Squaring a binary32 value is exact in binary64, so the final strict
    // comparison stays conservative without a square root.
    union FloatBits {
        float f;
        uint32_t u;
    };

    const float margin_0 = point[0] - node.lower_x;
    const float margin_1 = point[1] - node.lower_y;
    const float margin_2 = point[2] - node.lower_z;
    const float margin_3 = node.upper_x - point[0];
    const float margin_4 = node.upper_y - point[1];
    const float margin_5 = node.upper_z - point[2];
    if (!(distance_sq >= 0.0f && margin_0 > 0.0f && margin_1 > 0.0f && margin_2 > 0.0f && margin_3 > 0.0f
          && margin_4 > 0.0f && margin_5 > 0.0f))
        return false;

    FloatBits margin;
    margin.f = min(min(min(margin_0, margin_1), min(margin_2, margin_3)), min(margin_4, margin_5));
    margin.u--;
    const double margin_d = double(margin.f);
    return double(distance_sq) < margin_d * margin_d;
}

// Revalidate a candidate containment node against the current Exclusive BVH.
// A cached node may be stale after a refit or rebuild, so every record and
// parent index is range-checked before it is read. Only current strict sphere
// containment is accepted as a traversal certificate.
CUDA_CALLABLE inline int mesh_query_point_no_sign_find_exclusive_containment(
    const Mesh& mesh, const vec3& point, float distance_sq, int candidate_node
)
{
    if (!bvh_has_exclusive(mesh.bvh) || candidate_node < 0 || candidate_node >= mesh.bvh.num_nodes)
        return *mesh.bvh.root;

    int node_index = candidate_node;
    while (node_index >= 0 && node_index < mesh.bvh.num_nodes) {
        const BVHExclusiveNode exclusive_node = bvh_get_exclusive_node(mesh.bvh, node_index);
        if (bvh_exclusive_node_depth(exclusive_node) >= 0
            && mesh_query_point_no_sign_exclusive_contains_distance(exclusive_node, point, distance_sq))
            return node_index;

        const int parent = bvh_exclusive_node_parent(exclusive_node);
        if (parent == -1)
            break;
        if (parent < 0 || parent >= mesh.bvh.num_nodes)
            return *mesh.bvh.root;
        node_index = parent;
    }
    return *mesh.bvh.root;
}

CUDA_CALLABLE inline bool mesh_query_point_no_sign_finish(
    float min_dist_sq,
    float max_dist_sq,
    const int& min_face,
    const float& min_v,
    const float& min_w,
    int& face,
    float& u,
    float& v
)
{
    if (min_dist_sq < max_dist_sq) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;
        return true;
    }
    return false;
}

// Returns true if there is a point (strictly) < distance max_dist. The exact
// triangles in seed_face's packed leaf initialize the branch-and-bound radius.
CUDA_CALLABLE inline bool mesh_query_point_no_sign_seeded(
    uint64_t id, const vec3& point, float max_dist, int seed_face, int& face, float& u, float& v
)
{
    Mesh mesh = mesh_get(id);
    const float max_dist_sq = max_dist * max_dist;
    float min_dist_sq = max_dist_sq;
    int min_face;
    float min_v;
    float min_w;

    const int seed_leaf
        = mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed_face, point, min_dist_sq, min_face, min_v, min_w);
    if (seed_leaf == -1)
        return mesh_query_point_no_sign(id, point, max_dist, face, u, v);

    const int root = *mesh.bvh.root;
    if (root != seed_leaf)
        mesh_query_point_no_sign_traverse<false, true>(
            mesh, root, seed_leaf, point, min_dist_sq, min_face, min_v, min_w
        );

    return mesh_query_point_no_sign_finish(min_dist_sq, max_dist_sq, min_face, min_v, min_w, face, u, v);
}

// Returns true if there is a point (strictly) < distance max_dist. The exact
// seed-leaf result defines a shrinking sphere; strict containment by an
// exclusive box certifies that traversal may begin at that box's node.
CUDA_CALLABLE inline bool mesh_query_point_no_sign_exclusive(
    uint64_t id, const vec3& point, float max_dist, int seed_face, int& face, float& u, float& v
)
{
    Mesh mesh = mesh_get(id);
    const float max_dist_sq = max_dist * max_dist;
    float min_dist_sq = max_dist_sq;
    int min_face;
    float min_v;
    float min_w;

    // Packed leaves may hold more than the seed face. Every primitive must be
    // tested to obtain the tightest exact upper bound for both leaf_size=1 and
    // larger leaves.
    const int seed_leaf
        = mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed_face, point, min_dist_sq, min_face, min_v, min_w);
    if (seed_leaf == -1)
        return mesh_query_point_no_sign(id, point, max_dist, face, u, v);

    const int start_node = mesh_query_point_no_sign_find_exclusive_containment(mesh, point, min_dist_sq, seed_leaf);

    if (start_node != seed_leaf)
        mesh_query_point_no_sign_traverse<false, true>(
            mesh, start_node, seed_leaf, point, min_dist_sq, min_face, min_v, min_w
        );

    return mesh_query_point_no_sign_finish(min_dist_sq, max_dist_sq, min_face, min_v, min_w, face, u, v);
}

// Return the current Exclusive BVH node that certifies a seed-leaf exact
// distance bound. The global root is the safe result for invalid seeds or
// meshes without Exclusive BVH metadata.
CUDA_CALLABLE inline int
mesh_query_point_no_sign_exclusive_node(uint64_t id, const vec3& point, float max_dist, int seed_face)
{
    const Mesh mesh = mesh_get(id);
    const int root = *mesh.bvh.root;
    if (!bvh_has_exclusive(mesh.bvh))
        return root;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;
    const int seed_leaf
        = mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed_face, point, min_dist_sq, min_face, min_v, min_w);
    if (seed_leaf == -1)
        return root;

    return mesh_query_point_no_sign_find_exclusive_containment(mesh, point, min_dist_sq, seed_leaf);
}

// Reuse a previously certified node while preserving correctness across
// refits and rebuilds. The candidate is accepted only when its current
// exclusive box strictly contains the exact seed-leaf distance sphere;
// otherwise its current parent chain is searched and traversal falls back to
// the root.
CUDA_CALLABLE inline bool mesh_query_point_no_sign_exclusive_cached(
    uint64_t id, const vec3& point, float max_dist, int seed_face, int cached_node, int& face, float& u, float& v
)
{
    const Mesh mesh = mesh_get(id);
    const float max_dist_sq = max_dist * max_dist;
    float min_dist_sq = max_dist_sq;
    int min_face;
    float min_v;
    float min_w;

    const int seed_leaf
        = mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed_face, point, min_dist_sq, min_face, min_v, min_w);
    if (seed_leaf == -1)
        return mesh_query_point_no_sign(id, point, max_dist, face, u, v);

    const int start_node = mesh_query_point_no_sign_find_exclusive_containment(mesh, point, min_dist_sq, cached_node);
    if (start_node != seed_leaf)
        mesh_query_point_no_sign_traverse<false, true>(
            mesh, start_node, seed_leaf, point, min_dist_sq, min_face, min_v, min_w
        );

    return mesh_query_point_no_sign_finish(min_dist_sq, max_dist_sq, min_face, min_v, min_w, face, u, v);
}

// returns true if there is a point (strictly) > distance min_dist
CUDA_CALLABLE inline bool
mesh_query_furthest_point_no_sign(uint64_t id, const vec3& point, float min_dist, int& face, float& u, float& v)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = min_dist * min_dist;
    int max_face;
    float max_v;
    float max_w;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;

    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = furthest_distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));

        // if maximum distance to this node is less than our existing furthest max then skip
        if (node_dist_sq < min_dist_sq) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = furthest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq > min_dist_sq) {
                    min_dist_sq = dist_sq;
                    max_v = v;
                    max_w = w;
                    max_face = primitive_index;
                }
            }

#if BVH_DEBUG

            tests++;

            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);

            if (distance_to_aabb_sq(point, b.lower, b.upper) > max_dist * max_dist) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif

        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = furthest_distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = furthest_distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq > right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] > min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] > min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }


#if BVH_DEBUG
    printf("%d\n", tests);

    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;

    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;

    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = sqrtf(min_dist_sq);

        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );

        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }

        fclose(f);
    }
#endif

    // check if we found a point, and write outputs
    if (min_dist_sq > min_dist * min_dist) {
        u = 1.0f - max_v - max_w;
        v = max_v;
        face = max_face;

        return true;
    } else {
        return false;
    }
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    const float epsilon = 1e-3f
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;
    int count = 1;
    float min_dist = max_dist;
    int min_face;
    float min_v;
    float min_w;
    vec3 accumulated_angle_weighted_normal;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;
    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif
    float epsilon_min_dist = mesh.average_edge_length * epsilon;
    float epsilon_min_dist_sq = epsilon_min_dist * epsilon_min_dist;

    while (count) {
        const int nodeIndex = stack[--count];
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist)) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                float e0_norm_sq = dot(e0, e0);
                float e1_norm_sq = dot(e1, e1);
                float e2_norm_sq = dot(e2, e2);
                if (length(normal) / (e0_norm_sq + e1_norm_sq + e2_norm_sq) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;
                float dist = sqrtf(length_sq(c - point));
                if (dist < min_dist + epsilon_min_dist) {
                    float weight = 0.0f;
                    vec3 cp = c - p;
                    vec3 cq = c - q;
                    vec3 cr = c - r;
                    float len_cp_sq = length_sq(cp);
                    float len_cq_sq = length_sq(cq);
                    float len_cr_sq = length_sq(cr);

                    // Check if near vertex
                    if (len_cp_sq < epsilon_min_dist_sq) {
                        // Vertex 0 is the closest feature
                        weight = acosf(dot(normalize(e0), normalize(e1)));
                    } else if (len_cq_sq < epsilon_min_dist_sq) {
                        // Vertex 1 is the closest feature
                        weight = acosf(dot(normalize(e2), normalize(-e0)));
                    } else if (len_cr_sq < epsilon_min_dist_sq) {
                        // Vertex 2 is the closest feature
                        weight = acosf(dot(normalize(-e1), normalize(-e2)));
                    } else {
                        float e0cp = dot(e0, cp);
                        float e2cq = dot(e2, cq);
                        float e1cp = dot(e1, cp);

                        if ((len_cp_sq * e0_norm_sq - e0cp * e0cp < epsilon_min_dist_sq * e0_norm_sq)
                            || (len_cq_sq * e2_norm_sq - e2cq * e2cq < epsilon_min_dist_sq * e2_norm_sq)
                            || (len_cp_sq * e1_norm_sq - e1cp * e1cp < epsilon_min_dist_sq * e1_norm_sq)) {
                            // One of the edge
                            weight = 3.14159265359f;  // PI
                        } else {
                            weight = 2.0f * 3.14159265359f;  // 2*PI
                        }
                    }

                    if (dist > min_dist - epsilon_min_dist) {
                        // Treat as equal
                        accumulated_angle_weighted_normal += weight * normalize(normal);
                        if (dist < min_dist) {
                            min_dist = dist;
                            min_v = v;
                            min_w = w;
                            min_face = primitive_index;
                        }
                    } else {
                        // Less
                        min_dist = dist;
                        min_v = v;
                        min_w = w;
                        min_face = primitive_index;
                        accumulated_angle_weighted_normal = weight * normalize(normal);
                    }
                }
            }
#if BVH_DEBUG
            tests++;
            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);
            if (distance_to_aabb_sq(point, b.lower, b.upper)
                < (max_dist + epsilon_min_dist) * (max_dist + epsilon_min_dist)) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif
        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist))
                stack[count++] = child_indices[0];

            if (child_dist[1] < (min_dist + epsilon_min_dist) * (min_dist + epsilon_min_dist))
                stack[count++] = child_indices[1];
        }
    }
#if BVH_DEBUG
    printf("%d\n", tests);
    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;
    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;
    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = min_dist;
        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );
        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }
        fclose(f);
    }
#endif
    // check if we found a point, and write outputs
    if (min_dist < max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;
        // determine inside outside using ray-cast parity check
        // inside = mesh_query_inside(id, point);
        int i = mesh.indices[min_face * 3 + 0];
        int j = mesh.indices[min_face * 3 + 1];
        int k = mesh.indices[min_face * 3 + 2];
        vec3 p = mesh.points[i];
        vec3 q = mesh.points[j];
        vec3 r = mesh.points[k];
        vec3 closest_point = p * u + q * v + r * min_w;
        if (dot(accumulated_angle_weighted_normal, point - closest_point) > 0.0) {
            inside = 1.0f;
        } else {
            inside = -1.0f;
        }
        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline float solid_angle_iterative(uint64_t id, const vec3& p, const float accuracy_sq)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    int at_child[BVH_QUERY_STACK_SIZE];  // 0 for left, 1 for right, 2 for done
    float angle[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;
    at_child[0] = 0;

    int count = 1;
    angle[0] = 0.0f;

    while (count) {
        const int nodeIndex = stack[count - 1];
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        const int left_index = lower.i;
        const int right_index = upper.i;
        if (lower.b) {
            // compute closest point on tri
            const int start = left_index;
            const int end = right_index;
            angle[count - 1] = 0.f;
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];
                angle[count - 1] += robust_solid_angle(mesh.points[i], mesh.points[j], mesh.points[k], p);
                // printf("Leaf %d, got %f\n", leaf_index, my_data[count - 1]);
            }
            count--;
        } else {
            // See if I have to descend
            if (at_child[count - 1] == 0) {
                // First visit
                bool des
                    = evaluate_node_solid_angle(p, &mesh.solid_angle_props[nodeIndex], angle[count - 1], accuracy_sq);

                // printf("Non-Leaf %d, got %f\n", nodeIndex, angle[count - 1]);
                if (des) {
                    // Go left
                    stack[count] = left_index;
                    at_child[count - 1] = 1;
                    angle[count] = 0.0f;
                    at_child[count] = 0;
                    count++;
                } else {
                    // Does not descend done
                    count--;
                }
            } else if (at_child[count - 1] == 1) {
                // Add data to parent
                angle[count - 1] += angle[count];
                // Go right
                stack[count] = right_index;
                at_child[count - 1] = 2;
                angle[count] = 0.0f;
                at_child[count] = 0;
                count++;
            } else {
                // Descend both sides already
                angle[count - 1] += angle[count];
                count--;
            }
        }
    }
    return angle[0];
}

CUDA_CALLABLE inline float mesh_query_winding_number(uint64_t id, const vec3& p, const float accuracy)
{
    float angle = solid_angle_iterative(id, p, accuracy * accuracy);
    return angle * 0.07957747154;  // divided by 4 PI
}

// returns true if there is a point (strictly) < distance max_dist
CUDA_CALLABLE inline bool mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float& inside,
    int& face,
    float& u,
    float& v,
    const float accuracy,
    const float winding_number_threshold
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    stack[0] = *mesh.bvh.root;

    int count = 1;

    float min_dist_sq = max_dist * max_dist;
    int min_face;
    float min_v;
    float min_w;

#if BVH_DEBUG
    int tests = 0;
    int secondary_culls = 0;

    std::vector<int> test_history;
    std::vector<vec3> test_centers;
    std::vector<vec3> test_extents;
#endif

    while (count) {
        const int nodeIndex = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, nodeIndex);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, nodeIndex);

        // re-test distance
        float node_dist_sq
            = distance_to_aabb_sq(point, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z));
        if (node_dist_sq > min_dist_sq) {
#if BVH_DEBUG
            secondary_culls++;
#endif
            continue;
        }

        const int left_index = lower.i;
        const int right_index = upper.i;

        if (lower.b) {
            const int start = left_index;
            const int end = right_index;
            // loops through primitives in the leaf
            for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                vec3 e0 = q - p;
                vec3 e1 = r - p;
                vec3 e2 = r - q;
                vec3 normal = cross(e0, e1);

                // sliver detection
                if (length(normal) / (dot(e0, e0) + dot(e1, e1) + dot(e2, e2)) < 1.e-6f)
                    continue;

                vec2 barycentric = closest_point_to_triangle(p, q, r, point);
                float u = barycentric[0];
                float v = barycentric[1];
                float w = 1.f - u - v;
                vec3 c = u * p + v * q + w * r;

                float dist_sq = length_sq(c - point);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    min_v = v;
                    min_w = w;
                    min_face = primitive_index;
                }
            }
#if BVH_DEBUG

            tests++;

            bounds3 b;
            b = bounds_union(b, p);
            b = bounds_union(b, q);
            b = bounds_union(b, r);

            if (distance_to_aabb_sq(point, b.lower, b.upper) < max_dist * max_dist) {
                // if (dist_sq < max_dist*max_dist)
                test_history.push_back(left_index);
                test_centers.push_back(b.center());
                test_extents.push_back(b.edges());
            }
#endif

        } else {
            BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
            BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

            BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
            BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

            float left_dist_sq = distance_to_aabb_sq(
                point, vec3(left_lower.x, left_lower.y, left_lower.z), vec3(left_upper.x, left_upper.y, left_upper.z)
            );
            float right_dist_sq = distance_to_aabb_sq(
                point, vec3(right_lower.x, right_lower.y, right_lower.z),
                vec3(right_upper.x, right_upper.y, right_upper.z)
            );

            wp::vec2i child_indices;
            wp::vec2 child_dist;
            if (left_dist_sq < right_dist_sq) {
                child_indices = wp::vec2i(right_index, left_index);
                child_dist = wp::vec2(right_dist_sq, left_dist_sq);
            } else {
                child_indices = wp::vec2i(left_index, right_index);
                child_dist = wp::vec2(left_dist_sq, right_dist_sq);
            }

            if (child_dist[0] < min_dist_sq)
                stack[count++] = child_indices[0];

            if (child_dist[1] < min_dist_sq)
                stack[count++] = child_indices[1];
        }
    }


#if BVH_DEBUG
    printf("%d\n", tests);

    static int max_tests = 0;
    static vec3 max_point;
    static float max_point_dist = 0.0f;
    static int max_secondary_culls = 0;

    if (secondary_culls > max_secondary_culls)
        max_secondary_culls = secondary_culls;

    if (tests > max_tests) {
        max_tests = tests;
        max_point = point;
        max_point_dist = sqrtf(min_dist_sq);

        printf(
            "max_tests: %d max_point: %f %f %f max_point_dist: %f max_second_culls: %d\n", max_tests, max_point[0],
            max_point[1], max_point[2], max_point_dist, max_secondary_culls
        );

        FILE* f = fopen("test_history.txt", "w");
        for (int i = 0; i < test_history.size(); ++i) {
            fprintf(
                f, "%d, %f, %f, %f, %f, %f, %f\n", test_history[i], test_centers[i][0], test_centers[i][1],
                test_centers[i][2], test_extents[i][0], test_extents[i][1], test_extents[i][2]
            );
        }

        fclose(f);
    }
#endif

    // check if we found a point, and write outputs
    if (min_dist_sq < max_dist * max_dist) {
        u = 1.0f - min_v - min_w;
        v = min_v;
        face = min_face;

        // determine inside outside using ray-cast parity check
        if (!mesh.solid_angle_props) {
            inside = mesh_query_inside_ray_tracing(id, point);
        } else {
            float winding_number = mesh_query_winding_number(id, point, accuracy);
            inside = (winding_number > winding_number_threshold) ? -1.0f : 1.0f;
        }

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    vec3 adj_p, adj_q, adj_r;

    vec2 adj_uv(adj_u, adj_v);

    adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_seeded(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_exclusive(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_exclusive_cached(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    int cached_node,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    int& adj_cached_node,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_furthest_point_no_sign(
    uint64_t id,
    const vec3& point,
    float min_dist,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_min_dist,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    vec3 adj_p, adj_q, adj_r;

    vec2 adj_uv(adj_u, adj_v);

    adj_closest_point_to_triangle(p, q, r, point, adj_p, adj_q, adj_r, adj_point, adj_uv);  // Todo for Miles :>
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    int n_sample,
    float perturbation_scale,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    int& adj_n_sample,
    float& adj_perturbation_scale,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    const float epsilon,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    float& adj_epsilon,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    const float accuracy,
    const float winding_number_threshold,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    float& adj_accuracy,
    float& adj_winding_number_threshold,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}


// Stores the result of querying the closest point on a mesh.
struct mesh_query_point_t {
    CUDA_CALLABLE mesh_query_point_t()
        : result(false)
        , sign(0.0f)
        , face(0)
        , u(0.0f)
        , v(0.0f)
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_point_t& operator+=(const mesh_query_point_t& other)
    {
        result |= other.result;  // Use OR for bool accumulation
        sign += other.sign;
        face += other.face;
        u += other.u;
        v += other.v;
        return *this;
    }

    bool result;
    float sign;
    int face;
    float u;
    float v;
};


CUDA_CALLABLE inline void adj_mesh_query_point(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const float& inside,
    const int& face,
    const float& u,
    const float& v,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_inside,
    int& adj_face,
    float& adj_u,
    float& adj_v,
    bool& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, face, u, v, adj_id, adj_point, adj_max_dist, adj_face, adj_u, adj_v, adj_ret
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_ret.sign,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point(uint64_t id, const vec3& point, float max_dist)
{
    mesh_query_point_t query;
    query.result = mesh_query_point(id, point, max_dist, query.sign, query.face, query.u, query.v);
    return query;
}


CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_sign_parity(
    uint64_t id, const vec3& point, float max_dist, int n_sample = 1, float perturbation_scale = 0.1f
)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_parity(
        id, point, max_dist, query.sign, query.face, query.u, query.v, n_sample, perturbation_scale
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_no_sign(uint64_t id, const vec3& point, float max_dist)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_point_no_sign(id, point, max_dist, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_point_no_sign_seeded(uint64_t id, const vec3& point, float max_dist, int seed_face)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_point_no_sign_seeded(id, point, max_dist, seed_face, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_point_no_sign_exclusive(uint64_t id, const vec3& point, float max_dist, int seed_face)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_point_no_sign_exclusive(id, point, max_dist, seed_face, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_no_sign_exclusive_cached(
    uint64_t id, const vec3& point, float max_dist, int seed_face, int cached_node
)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_point_no_sign_exclusive_cached(
        id, point, max_dist, seed_face, cached_node, query.face, query.u, query.v
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_furthest_point_no_sign(uint64_t id, const vec3& point, float min_dist)
{
    mesh_query_point_t query;
    query.sign = 0.0;
    query.result = mesh_query_furthest_point_no_sign(id, point, min_dist, query.face, query.u, query.v);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t
mesh_query_point_sign_normal(uint64_t id, const vec3& point, float max_dist, const float epsilon = 1e-3f)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_normal(id, point, max_dist, query.sign, query.face, query.u, query.v, epsilon);
    return query;
}

CUDA_CALLABLE inline mesh_query_point_t mesh_query_point_sign_winding_number(
    uint64_t id, const vec3& point, float max_dist, float accuracy, float winding_number_threshold
)
{
    mesh_query_point_t query;
    query.result = mesh_query_point_sign_winding_number(
        id, point, max_dist, query.sign, query.face, query.u, query.v, accuracy, winding_number_threshold
    );
    return query;
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_parity(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int n_sample,
    float perturbation_scale,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_n_sample,
    float& adj_perturbation_scale,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_parity(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, n_sample, perturbation_scale, adj_id, adj_point,
        adj_max_dist, adj_ret.sign, adj_ret.face, adj_ret.u, adj_ret.v, adj_n_sample, adj_perturbation_scale,
        adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign(
    uint64_t id,
    const vec3& point,
    float max_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_no_sign(
        id, point, max_dist, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_ret.face, adj_ret.u,
        adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_seeded(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_no_sign_seeded(
        id, point, max_dist, seed_face, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_seed_face,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_exclusive(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_no_sign_exclusive(
        id, point, max_dist, seed_face, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist, adj_seed_face,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_no_sign_exclusive_cached(
    uint64_t id,
    const vec3& point,
    float max_dist,
    int seed_face,
    int cached_node,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    int& adj_seed_face,
    int& adj_cached_node,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_no_sign_exclusive_cached(
        id, point, max_dist, seed_face, cached_node, ret.face, ret.u, ret.v, adj_id, adj_point, adj_max_dist,
        adj_seed_face, adj_cached_node, adj_ret.face, adj_ret.u, adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_furthest_point_no_sign(
    uint64_t id,
    const vec3& point,
    float min_dist,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_min_dist,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_furthest_point_no_sign(
        id, point, min_dist, ret.face, ret.u, ret.v, adj_id, adj_point, adj_min_dist, adj_ret.face, adj_ret.u,
        adj_ret.v, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_normal(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float epsilon,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_epsilon,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_normal(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, epsilon, adj_id, adj_point, adj_max_dist, adj_ret.sign,
        adj_ret.face, adj_ret.u, adj_ret.v, adj_epsilon, adj_ret.result
    );
}

CUDA_CALLABLE inline void adj_mesh_query_point_sign_winding_number(
    uint64_t id,
    const vec3& point,
    float max_dist,
    float accuracy,
    float winding_number_threshold,
    const mesh_query_point_t& ret,
    uint64_t adj_id,
    vec3& adj_point,
    float& adj_max_dist,
    float& adj_accuracy,
    float& adj_winding_number_threshold,
    mesh_query_point_t& adj_ret
)
{
    adj_mesh_query_point_sign_winding_number(
        id, point, max_dist, ret.sign, ret.face, ret.u, ret.v, accuracy, winding_number_threshold, adj_id, adj_point,
        adj_max_dist, adj_ret.sign, adj_ret.face, adj_ret.u, adj_ret.v, adj_accuracy, adj_winding_number_threshold,
        adj_ret.result
    );
}

CUDA_CALLABLE inline vec3 mesh_query_ray_safe_dir(const vec3& dir)
{
    vec3 ray_dir = dir;
    if (ray_dir[0] == 0.0f)
        ray_dir[0] = 1.0e-20f;
    if (ray_dir[1] == 0.0f)
        ray_dir[1] = 1.0e-20f;
    if (ray_dir[2] == 0.0f)
        ray_dir[2] = 1.0e-20f;
    return ray_dir;
}

CUDA_CALLABLE inline bool mesh_query_ray_use_fast_aabb(const vec3& dir)
{
    return dir[0] != 0.0f && dir[1] != 0.0f && dir[2] != 0.0f;
}

CUDA_CALLABLE inline bool mesh_query_ray_intersect_aabb(
    const vec3& start,
    const vec3& dir,
    const vec3& rcp_dir,
    bool fast_aabb,
    const vec3& lower,
    const vec3& upper,
    float& t
)
{
    if (fast_aabb)
        return intersect_ray_aabb(start, rcp_dir, lower, upper, t);
    else
        return intersect_ray_aabb_robust(start, dir, rcp_dir, lower, upper, t);
}

CUDA_CALLABLE inline bool mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face,
    int root = -1
)
{
    Mesh mesh = mesh_get(id);

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_load(mesh.bvh, (root == -1) ? *mesh.bvh.root : root);

    vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);

    float min_t = max_t;
    int min_face;
    float min_u;
    float min_v;
    float min_sign = 1.0f;
    vec3 min_normal;
    bool hit = false;

    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            const int primitive_begin = bvh_query_node_lower_payload(cur_node);
            const int primitive_end = bvh_query_node_upper_payload(cur_node);
            // Leaf: test all primitives in the leaf.
            for (int pc = primitive_begin; pc < primitive_end; ++pc) {
                int primitive_index = mesh.bvh.primitive_indices[pc];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float tri_t, tri_u, tri_v, tri_sign;
                vec3 n;

                if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                    if (tri_t < min_t && tri_t >= 0.0f) {
                        min_t = tri_t;
                        min_face = primitive_index;
                        min_u = tri_u;
                        min_v = tri_v;
                        min_sign = tri_sign;
                        min_normal = n;
                        hit = true;
                    }
                }
            }
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
            continue;
        }

        // Inner node: load both children so we can sort by entry distance.
        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);

        BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float t0 = FLT_MAX;
        float t1 = FLT_MAX;
        const bool h0 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(left_lower.x, left_lower.y, left_lower.z),
                            vec3(left_upper.x, left_upper.y, left_upper.z), t0
                        )
            && t0 < min_t;
        const bool h1 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(right_lower.x, right_lower.y, right_lower.z),
                            vec3(right_upper.x, right_upper.y, right_upper.z), t1
                        )
            && t1 < min_t;

        if (h0 && h1) {
            const bool near_left = (t0 < t1);
            if (stack_size >= BVH_QUERY_STACK_SIZE)
                break;
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (h0) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (h1) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            // Neither child reachable; pop.
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
        }
    }

    if (hit) {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    } else {
        return false;
    }
}

// Internal closest-hit state shared by the seeded and Exclusive BVH ray
// experiments below. Keeping the unnormalized normal matches mesh_query_ray()
// until the final result is written.
struct mesh_query_ray_hit_state_t {
    CUDA_CALLABLE explicit mesh_query_ray_hit_state_t(float max_t)
        : min_t(max_t)
        , min_u(0.0f)
        , min_v(0.0f)
        , min_sign(1.0f)
        , min_normal()
        , min_face(-1)
        , hit(false)
        , ambiguous_tie(false)
    {
    }

    float min_t;
    float min_u;
    float min_v;
    float min_sign;
    vec3 min_normal;
    int min_face;
    bool hit;
    bool ambiguous_tie;
};

enum MeshQueryRayPeelingStatus {
    MESH_QUERY_RAY_PEEL_NONE = 0,
    MESH_QUERY_RAY_PEEL_TERMINAL = 1 << 0,
    MESH_QUERY_RAY_PEEL_PREFIX = 1 << 1,
    MESH_QUERY_RAY_PEEL_SUFFIX = 1 << 2,
    MESH_QUERY_RAY_PEEL_MIDDLE_DECLINED = 1 << 3,
    MESH_QUERY_RAY_PEEL_INVALID = 1 << 4,
};

CUDA_CALLABLE inline void mesh_query_ray_update_primitive(
    const Mesh& mesh,
    int primitive_index,
    const vec3& start,
    const vec3& dir,
    float interval_min,
    float interval_max,
    mesh_query_ray_hit_state_t& state
)
{
    const int i = bvh_load_int(mesh.indices, primitive_index * 3 + 0);
    const int j = bvh_load_int(mesh.indices, primitive_index * 3 + 1);
    const int k = bvh_load_int(mesh.indices, primitive_index * 3 + 2);

    const vec3 p = bvh_load_vec3(mesh.points, i);
    const vec3 q = bvh_load_vec3(mesh.points, j);
    const vec3 r = bvh_load_vec3(mesh.points, k);

    float tri_t, tri_u, tri_v, tri_sign;
    vec3 n;
    if (!intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n) || tri_t < interval_min
        || (tri_t >= interval_max && !(state.hit && interval_max == state.min_t && tri_t == state.min_t))) {
        return;
    }

    if (tri_t < state.min_t) {
        state.min_t = tri_t;
        state.min_face = primitive_index;
        state.min_u = tri_u;
        state.min_v = tri_v;
        state.min_sign = tri_sign;
        state.min_normal = n;
        state.hit = true;
        state.ambiguous_tie = false;
    } else if (state.hit && tri_t == state.min_t && primitive_index != state.min_face) {
        // A seeded traversal can visit equal-distance faces in a different
        // order from the stock root traversal. The caller reruns the stock
        // query when this is observed so face/sign/normal tie behavior stays
        // identical, not merely the hit distance.
        state.ambiguous_tie = true;
    }
}

CUDA_CALLABLE inline float mesh_query_ray_next_float_down(float value)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    if (value == 0.0f) {
        bits.u = 0x80000001u;
    } else if (value > 0.0f) {
        --bits.u;
    } else {
        ++bits.u;
    }
    return bits.f;
}

CUDA_CALLABLE inline float mesh_query_ray_next_float_up(float value)
{
    union FloatBits {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    if (value == 0.0f) {
        bits.u = 0x00000001u;
    } else if (value > 0.0f) {
        ++bits.u;
    } else {
        --bits.u;
    }
    return bits.f;
}

CUDA_CALLABLE inline void mesh_query_ray_update_leaf(
    const Mesh& mesh,
    int primitive_begin,
    int primitive_end,
    const vec3& start,
    const vec3& dir,
    float interval_min,
    float interval_max,
    mesh_query_ray_hit_state_t& state
)
{
    for (int pc = primitive_begin; pc < primitive_end; ++pc) {
        const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, pc);
        mesh_query_ray_update_primitive(mesh, primitive_index, start, dir, interval_min, interval_max, state);
    }
}

CUDA_CALLABLE inline bool mesh_query_ray_node_before_limit(float node_t, const mesh_query_ray_hit_state_t& state)
{
    // Stock traversal uses a strict limit. Once a seeded hit exists, retain
    // an equal-entry node long enough to detect exact-distance ties and fall
    // back to stock ordering.
    return node_t < state.min_t || (state.hit && node_t == state.min_t);
}

// Traverse one subtree with stock near/far ordering while optionally omitting
// an already evaluated packed seed leaf. TestStartNode is used for arbitrary
// cached roots; child bounds are tested exactly as in mesh_query_ray().
template <bool TestStartNode, bool SkipLeaf>
CUDA_CALLABLE inline void mesh_query_ray_traverse_seeded(
    const Mesh& mesh,
    int start_node,
    int skip_leaf,
    const vec3& start,
    const vec3& dir,
    float interval_min,
    mesh_query_ray_hit_state_t& state
)
{
    if (start_node < 0 || start_node >= mesh.bvh.num_nodes || (SkipLeaf && start_node == skip_leaf))
        return;

    const vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    const vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);

    const BVHPackedNodeHalf start_lower = bvh_load_node(mesh.bvh.node_lowers, start_node);
    const BVHPackedNodeHalf start_upper = bvh_load_node(mesh.bvh.node_uppers, start_node);
    if (TestStartNode) {
        float start_t = FLT_MAX;
        if (!mesh_query_ray_intersect_aabb(
                start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(start_lower),
                reinterpret_cast<const vec3&>(start_upper), start_t
            )
            || !mesh_query_ray_node_before_limit(start_t, state)) {
            return;
        }
    }

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_pack(start_lower, start_upper);

    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            mesh_query_ray_update_leaf(
                mesh, bvh_query_node_lower_payload(cur_node), bvh_query_node_upper_payload(cur_node), start, dir,
                interval_min, state.min_t, state
            );
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);
        const BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        const BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        const BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        const BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float left_t = FLT_MAX;
        float right_t = FLT_MAX;
        const bool hit_left = (!SkipLeaf || left_index != skip_leaf)
            && mesh_query_ray_intersect_aabb(
                                  start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(left_lower),
                                  reinterpret_cast<const vec3&>(left_upper), left_t
            )
            && mesh_query_ray_node_before_limit(left_t, state);
        const bool hit_right = (!SkipLeaf || right_index != skip_leaf)
            && mesh_query_ray_intersect_aabb(
                                   start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(right_lower),
                                   reinterpret_cast<const vec3&>(right_upper), right_t
            )
            && mesh_query_ray_node_before_limit(right_t, state);

        if (hit_left && hit_right) {
            const bool near_left = left_t < right_t;
            if (stack_size >= BVH_QUERY_STACK_SIZE) {
                state.ambiguous_tie = true;
                break;
            }
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (hit_left) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (hit_right) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
        }
    }
}

// Traverse a disjoint bottom-up frontier subtree over a residual ray
// interval. The already evaluated packed seed leaf is omitted wherever it
// occurs in that frontier. Proper complement nodes are disjoint from every
// peeled E-box interior, so peeling cannot remove one of their AABB tests;
// retaining the stock node test avoids paying exit-distance arithmetic for a
// cull that the Exclusive BVH invariant proves impossible. The residual is
// still applied to exact primitive candidates and can terminate the ascent.
CUDA_CALLABLE inline void mesh_query_ray_traverse_interval(
    const Mesh& mesh,
    int start_node,
    int skip_leaf,
    const vec3& start,
    const vec3& dir,
    const vec3& rcp_dir,
    bool fast_aabb,
    float interval_min,
    float interval_max,
    mesh_query_ray_hit_state_t& state
)
{
    if (start_node < 0 || start_node >= mesh.bvh.num_nodes || start_node == skip_leaf)
        return;

    const BVHPackedNodeHalf start_lower = bvh_load_node(mesh.bvh.node_lowers, start_node);
    const BVHPackedNodeHalf start_upper = bvh_load_node(mesh.bvh.node_uppers, start_node);
    if (start_node != *mesh.bvh.root) {
        float start_t = FLT_MAX;
        if (!mesh_query_ray_intersect_aabb(
                start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(start_lower),
                reinterpret_cast<const vec3&>(start_upper), start_t
            )
            || !mesh_query_ray_node_before_limit(start_t, state)) {
            return;
        }
    }

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_pack(start_lower, start_upper);
    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            mesh_query_ray_update_leaf(
                mesh, bvh_query_node_lower_payload(cur_node), bvh_query_node_upper_payload(cur_node), start, dir,
                interval_min, interval_max, state
            );
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);
        const BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        const BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        const BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        const BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float left_t = FLT_MAX;
        float right_t = FLT_MAX;
        const bool hit_left = left_index != skip_leaf
            && mesh_query_ray_intersect_aabb(
                                  start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(left_lower),
                                  reinterpret_cast<const vec3&>(left_upper), left_t
            )
            && mesh_query_ray_node_before_limit(left_t, state);
        const bool hit_right = right_index != skip_leaf
            && mesh_query_ray_intersect_aabb(
                                   start, dir, rcp_dir, fast_aabb, reinterpret_cast<const vec3&>(right_lower),
                                   reinterpret_cast<const vec3&>(right_upper), right_t
            )
            && mesh_query_ray_node_before_limit(right_t, state);

        if (hit_left && hit_right) {
            if (stack_size >= BVH_QUERY_STACK_SIZE) {
                state.ambiguous_tie = true;
                break;
            }
            const bool near_left = left_t < right_t;
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (hit_left) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (hit_right) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            if (stack_size == 0)
                break;
            cur_node = stack[--stack_size];
        }
    }
}

CUDA_CALLABLE inline bool mesh_query_ray_seed_direction_matches_stock(const vec3& dir)
{
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(dir[axis]))
            return false;
        if (dir[axis] != 0.0f && !isfinite(1.0f / dir[axis]))
            return false;
    }
    return true;
}

CUDA_CALLABLE inline int mesh_query_ray_initialize_seed_leaf(
    const Mesh& mesh, int seed_face, const vec3& start, const vec3& dir, mesh_query_ray_hit_state_t& state
)
{
    if (seed_face < 0 || seed_face >= mesh.num_tris || !bvh_has_exclusive(mesh.bvh)
        || !mesh_query_ray_seed_direction_matches_stock(dir))
        return -1;

    const int seed_leaf = bvh_get_primitive_leaf(mesh.bvh, seed_face);
    if (seed_leaf < 0 || seed_leaf >= mesh.bvh.num_nodes)
        return -1;

    const BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, seed_leaf);
    if (!lower.b)
        return -1;
    const BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, seed_leaf);
    mesh_query_ray_update_leaf(mesh, int(lower.i), int(upper.i), start, dir, 0.0f, state.min_t, state);
    return seed_leaf;
}

// Build an outward-rounded AABB for the active finite ray segment. The active
// endpoint is the exact seed-leaf hit when available, otherwise the caller's
// max_t endpoint. Stepping both bounds outward makes strict E-box containment
// fail closed near a clipping plane on both CPU and CUDA.
CUDA_CALLABLE inline bool mesh_query_ray_active_segment_bounds(
    const vec3& start,
    const vec3& dir,
    float max_t,
    const mesh_query_ray_hit_state_t& state,
    vec3& segment_lower,
    vec3& segment_upper
)
{
    const float segment_t = state.hit ? state.min_t : max_t;
    if (!isfinite(start) || !isfinite(dir) || !isfinite(segment_t) || segment_t < 0.0f)
        return false;

    for (int axis = 0; axis < 3; ++axis) {
        const float endpoint = fmaf(dir[axis], segment_t, start[axis]);
        if (!isfinite(endpoint))
            return false;
        const float lower = min(start[axis], endpoint);
        const float upper = max(start[axis], endpoint);
        segment_lower[axis] = mesh_query_ray_next_float_down(lower);
        segment_upper[axis] = mesh_query_ray_next_float_up(upper);
    }
    return true;
}

CUDA_CALLABLE inline bool mesh_query_ray_active_endpoint_bounds(
    const vec3& start,
    const vec3& dir,
    float max_t,
    const mesh_query_ray_hit_state_t& state,
    vec3& endpoint_lower,
    vec3& endpoint_upper
)
{
    const float endpoint_t = state.hit ? state.min_t : max_t;
    if (!isfinite(start) || !isfinite(dir) || !isfinite(endpoint_t) || endpoint_t < 0.0f)
        return false;

    for (int axis = 0; axis < 3; ++axis) {
        const float endpoint = fmaf(dir[axis], endpoint_t, start[axis]);
        if (!isfinite(endpoint))
            return false;
        endpoint_lower[axis] = mesh_query_ray_next_float_down(endpoint);
        endpoint_upper[axis] = mesh_query_ray_next_float_up(endpoint);
    }
    return true;
}

CUDA_CALLABLE inline bool mesh_query_ray_point_strictly_inside_exclusive(
    const vec3& start, const vec3& dir, float ray_t, const BVHExclusiveNode& node, bool& valid
)
{
    valid = true;
    for (int axis = 0; axis < 3; ++axis) {
        const float point = fmaf(dir[axis], ray_t, start[axis]);
        if (!isfinite(point)) {
            valid = false;
            return false;
        }
        const float point_lower = mesh_query_ray_next_float_down(point);
        const float point_upper = mesh_query_ray_next_float_up(point);
        const float box_lower = axis == 0 ? node.lower_x : (axis == 1 ? node.lower_y : node.lower_z);
        const float box_upper = axis == 0 ? node.upper_x : (axis == 1 ? node.upper_y : node.upper_z);
        if (!(point_lower > box_lower && point_upper < box_upper))
            return false;
    }
    return true;
}

// Classify one prefix or suffix covered by an exclusive box. Along the
// bottom-up path E(child) is contained by E(parent), and every proper
// complement subtree is disjoint from the peeled interior. Materializing the
// slab cut therefore cannot change a remaining node or primitive test; only
// the terminal case can remove work. Keep the residual endpoints unchanged
// and use this implicit peel to avoid division-heavy slab arithmetic on the
// common prefix/suffix path. A middle intersection is only diagnosed and is
// deliberately declined because its complement is two ray intervals.
CUDA_CALLABLE inline int mesh_query_ray_peel_exclusive_interval(
    const vec3& start, const vec3& dir, const BVHExclusiveNode& node, float& interval_min, float& interval_max
)
{
    if (bvh_exclusive_node_depth(node) < 0 || !isfinite(start) || !isfinite(dir) || !isfinite(interval_min)
        || !isfinite(interval_max) || interval_min < 0.0f || interval_max < interval_min) {
        return MESH_QUERY_RAY_PEEL_INVALID;
    }

    const vec3 box_lower = bvh_exclusive_node_lower(node);
    const vec3 box_upper = bvh_exclusive_node_upper(node);
    for (int axis = 0; axis < 3; ++axis) {
        if (!(box_lower[axis] < box_upper[axis]))
            return MESH_QUERY_RAY_PEEL_NONE;
    }

    bool lower_valid = false;
    bool upper_valid = false;
    const bool lower_inside
        = mesh_query_ray_point_strictly_inside_exclusive(start, dir, interval_min, node, lower_valid);
    const bool upper_inside
        = mesh_query_ray_point_strictly_inside_exclusive(start, dir, interval_max, node, upper_valid);
    if (!lower_valid || !upper_valid)
        return MESH_QUERY_RAY_PEEL_INVALID;
    if (lower_inside && upper_inside)
        return MESH_QUERY_RAY_PEEL_TERMINAL;
    if (lower_inside)
        return MESH_QUERY_RAY_PEEL_PREFIX;
    if (upper_inside)
        return MESH_QUERY_RAY_PEEL_SUFFIX;

    float box_entry = -FLT_MAX;
    float box_exit = FLT_MAX;
    for (int axis = 0; axis < 3; ++axis) {
        if (dir[axis] == 0.0f) {
            if (start[axis] <= box_lower[axis] || start[axis] >= box_upper[axis])
                return MESH_QUERY_RAY_PEEL_NONE;
        } else {
            const float t0 = (box_lower[axis] - start[axis]) / dir[axis];
            const float t1 = (box_upper[axis] - start[axis]) / dir[axis];
            box_entry = max(box_entry, min(t0, t1));
            box_exit = min(box_exit, max(t0, t1));
        }
    }

    if (!(box_entry < box_exit))
        return MESH_QUERY_RAY_PEEL_NONE;
    if (!(box_exit > interval_min && box_entry < interval_max))
        return MESH_QUERY_RAY_PEEL_NONE;
    return MESH_QUERY_RAY_PEEL_MIDDLE_DECLINED;
}

CUDA_CALLABLE inline int mesh_query_ray_find_exclusive_segment(
    const Mesh& mesh, const vec3& segment_lower, const vec3& segment_upper, int candidate_node
)
{
    const int root = *mesh.bvh.root;
    if (!bvh_has_exclusive(mesh.bvh) || candidate_node < 0 || candidate_node >= mesh.bvh.num_nodes)
        return root;

    int node_index = candidate_node;
    while (node_index >= 0 && node_index < mesh.bvh.num_nodes) {
        const BVHExclusiveNode exclusive_node = bvh_get_exclusive_node(mesh.bvh, node_index);
        if (bvh_exclusive_node_depth(exclusive_node) >= 0
            && bvh_exclusive_contains_strict(exclusive_node, segment_lower, segment_upper)) {
            return node_index;
        }

        const int parent = bvh_exclusive_node_parent(exclusive_node);
        if (parent == -1)
            break;
        if (parent < 0 || parent >= mesh.bvh.num_nodes)
            return root;
        node_index = parent;
    }
    return root;
}

CUDA_CALLABLE inline bool mesh_query_ray_finish_seeded(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    const mesh_query_ray_hit_state_t& state,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    if (state.ambiguous_tie)
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
    if (!state.hit)
        return false;

    t = state.min_t;
    u = state.min_u;
    v = state.min_v;
    sign = state.min_sign;
    normal = normalize(state.min_normal);
    face = state.min_face;
    return true;
}

CUDA_CALLABLE inline bool mesh_query_ray_seeded(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    const Mesh mesh = mesh_get(id);
    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    if (seed_leaf == -1)
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);

    mesh_query_ray_traverse_seeded<false, true>(mesh, *mesh.bvh.root, seed_leaf, start, dir, 0.0f, state);
    return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
}

CUDA_CALLABLE inline int
mesh_query_ray_exclusive_node(uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face)
{
    const Mesh mesh = mesh_get(id);
    const int root = *mesh.bvh.root;
    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    if (seed_leaf == -1)
        return root;

    vec3 segment_lower;
    vec3 segment_upper;
    if (!mesh_query_ray_active_segment_bounds(start, dir, max_t, state, segment_lower, segment_upper))
        return root;
    return mesh_query_ray_find_exclusive_segment(mesh, segment_lower, segment_upper, seed_leaf);
}

CUDA_CALLABLE inline int
mesh_query_ray_exclusive_endpoint_node(uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face)
{
    const Mesh mesh = mesh_get(id);
    const int root = *mesh.bvh.root;
    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    if (seed_leaf == -1)
        return root;

    vec3 endpoint_lower;
    vec3 endpoint_upper;
    if (!mesh_query_ray_active_endpoint_bounds(start, dir, max_t, state, endpoint_lower, endpoint_upper))
        return root;
    return mesh_query_ray_find_exclusive_segment(mesh, endpoint_lower, endpoint_upper, seed_leaf);
}

// Search a cached subtree and the disjoint sibling frontier to the root. With
// Peel enabled, each completed subtree classifies an implicit prefix/suffix
// peel and terminates once an ancestor E-box strictly contains both active
// endpoints. The no-peel specialization is the identical-topology control.
template <bool Peel>
CUDA_CALLABLE inline bool mesh_query_ray_exclusive_cached_bottom_up_impl(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    int cached_node,
    int& peeling_status,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    peeling_status = MESH_QUERY_RAY_PEEL_NONE;
    const Mesh mesh = mesh_get(id);
    const int root = *mesh.bvh.root;
    if (!bvh_has_exclusive(mesh.bvh) || root < 0 || root >= mesh.bvh.num_nodes || cached_node < 0
        || cached_node >= mesh.bvh.num_nodes || !isfinite(start) || !isfinite(dir) || !isfinite(max_t)
        || max_t < 0.0f) {
        peeling_status = MESH_QUERY_RAY_PEEL_INVALID;
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
    }

    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    BVHExclusiveNode current_exclusive = bvh_get_exclusive_node(mesh.bvh, cached_node);
    if (seed_leaf == -1 || bvh_exclusive_node_depth(current_exclusive) < 0) {
        peeling_status = MESH_QUERY_RAY_PEEL_INVALID;
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
    }

    float interval_min = 0.0f;
    float interval_max = state.min_t;
    const vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    const vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);
    int current = cached_node;
    mesh_query_ray_traverse_interval(
        mesh, current, seed_leaf, start, dir, rcp_dir, fast_aabb, interval_min, interval_max, state
    );
    if (state.ambiguous_tie) {
        peeling_status = MESH_QUERY_RAY_PEEL_INVALID;
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
    }
    interval_max = min(interval_max, state.min_t);

    if (Peel && current != root) {
        const int status
            = mesh_query_ray_peel_exclusive_interval(start, dir, current_exclusive, interval_min, interval_max);
        peeling_status |= status;
        if (status & MESH_QUERY_RAY_PEEL_INVALID)
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        if ((status & MESH_QUERY_RAY_PEEL_TERMINAL) || interval_min > interval_max)
            return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
    }

    for (int iteration = 0; current != root; ++iteration) {
        if (iteration >= mesh.bvh.num_nodes) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }

        const int parent = bvh_exclusive_node_parent(current_exclusive);
        if (parent < 0 || parent >= mesh.bvh.num_nodes) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }
        const BVHPackedNodeHalf parent_lower = bvh_load_node(mesh.bvh.node_lowers, parent);
        const BVHPackedNodeHalf parent_upper = bvh_load_node(mesh.bvh.node_uppers, parent);
        if (parent_lower.b) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }

        const int left = int(parent_lower.i);
        const int right = int(parent_upper.i);
        int sibling = -1;
        if (left == current && right != current)
            sibling = right;
        else if (right == current && left != current)
            sibling = left;
        if (sibling < 0 || sibling >= mesh.bvh.num_nodes) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }

        mesh_query_ray_traverse_interval(
            mesh, sibling, seed_leaf, start, dir, rcp_dir, fast_aabb, interval_min, interval_max, state
        );
        if (state.ambiguous_tie) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }
        interval_max = min(interval_max, state.min_t);
        current = parent;
        current_exclusive = bvh_get_exclusive_node(mesh.bvh, current);
        if (bvh_exclusive_node_depth(current_exclusive) < 0) {
            peeling_status |= MESH_QUERY_RAY_PEEL_INVALID;
            return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
        }

        if (Peel && current != root) {
            const int status
                = mesh_query_ray_peel_exclusive_interval(start, dir, current_exclusive, interval_min, interval_max);
            peeling_status |= status;
            if (status & MESH_QUERY_RAY_PEEL_INVALID)
                return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);
            if ((status & MESH_QUERY_RAY_PEEL_TERMINAL) || interval_min > interval_max)
                return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
        }
    }

    return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
}

CUDA_CALLABLE inline bool mesh_query_ray_exclusive_cached_bottom_up(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    int cached_node,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    int peeling_status;
    return mesh_query_ray_exclusive_cached_bottom_up_impl<false>(
        id, start, dir, max_t, seed_face, cached_node, peeling_status, t, u, v, sign, normal, face
    );
}

CUDA_CALLABLE inline bool mesh_query_ray_exclusive_cached_peeling(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    int cached_node,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    int peeling_status;
    return mesh_query_ray_exclusive_cached_bottom_up_impl<true>(
        id, start, dir, max_t, seed_face, cached_node, peeling_status, t, u, v, sign, normal, face
    );
}

CUDA_CALLABLE inline int mesh_query_ray_exclusive_cached_peeling_status(
    uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face, int cached_node
)
{
    int peeling_status;
    float t, u, v, sign;
    vec3 normal;
    int face;
    mesh_query_ray_exclusive_cached_bottom_up_impl<true>(
        id, start, dir, max_t, seed_face, cached_node, peeling_status, t, u, v, sign, normal, face
    );
    return peeling_status;
}

CUDA_CALLABLE inline int mesh_query_ray_exclusive_node_depth(uint64_t id, int node)
{
    const Mesh mesh = mesh_get(id);
    if (!bvh_has_exclusive(mesh.bvh) || node < 0 || node >= mesh.bvh.num_nodes)
        return -1;
    return bvh_exclusive_node_depth(bvh_get_exclusive_node(mesh.bvh, node));
}

CUDA_CALLABLE inline bool mesh_query_ray_exclusive(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    const Mesh mesh = mesh_get(id);
    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    if (seed_leaf == -1)
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);

    vec3 segment_lower;
    vec3 segment_upper;
    int start_node = *mesh.bvh.root;
    if (mesh_query_ray_active_segment_bounds(start, dir, max_t, state, segment_lower, segment_upper)) {
        start_node = mesh_query_ray_find_exclusive_segment(mesh, segment_lower, segment_upper, seed_leaf);
    }

    mesh_query_ray_traverse_seeded<false, true>(mesh, start_node, seed_leaf, start, dir, 0.0f, state);
    return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
}

CUDA_CALLABLE inline bool mesh_query_ray_exclusive_cached(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int seed_face,
    int cached_node,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face
)
{
    const Mesh mesh = mesh_get(id);
    mesh_query_ray_hit_state_t state(max_t);
    const int seed_leaf = mesh_query_ray_initialize_seed_leaf(mesh, seed_face, start, dir, state);
    if (seed_leaf == -1)
        return mesh_query_ray(id, start, dir, max_t, t, u, v, sign, normal, face);

    vec3 segment_lower;
    vec3 segment_upper;
    int start_node = *mesh.bvh.root;
    if (mesh_query_ray_active_segment_bounds(start, dir, max_t, state, segment_lower, segment_upper)) {
        start_node = mesh_query_ray_find_exclusive_segment(mesh, segment_lower, segment_upper, cached_node);
    }

    mesh_query_ray_traverse_seeded<false, true>(mesh, start_node, seed_leaf, start, dir, 0.0f, state);
    return mesh_query_ray_finish_seeded(id, start, dir, max_t, state, t, u, v, sign, normal, face);
}

CUDA_CALLABLE inline bool
mesh_query_ray_anyhit(uint64_t id, const vec3& start, const vec3& dir, float max_t, int root = -1)
{
    Mesh mesh = mesh_get(id);

    uint64_t stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    uint64_t cur_node = bvh_query_node_load(mesh.bvh, (root == -1) ? *mesh.bvh.root : root);

    vec3 ray_dir = mesh_query_ray_safe_dir(dir);
    vec3 rcp_dir(1.0f / ray_dir[0], 1.0f / ray_dir[1], 1.0f / ray_dir[2]);
    const bool fast_aabb = mesh_query_ray_use_fast_aabb(dir);

    while (true) {
        if (bvh_query_node_is_leaf(cur_node)) {
            const int primitive_begin = bvh_query_node_lower_payload(cur_node);
            const int primitive_end = bvh_query_node_upper_payload(cur_node);
            for (int pc = primitive_begin; pc < primitive_end; ++pc) {
                int primitive_index = mesh.bvh.primitive_indices[pc];
                int i = mesh.indices[primitive_index * 3 + 0];
                int j = mesh.indices[primitive_index * 3 + 1];
                int k = mesh.indices[primitive_index * 3 + 2];

                vec3 p = mesh.points[i];
                vec3 q = mesh.points[j];
                vec3 r = mesh.points[k];

                float tri_t, tri_u, tri_v, tri_sign;
                vec3 n;

                if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                    if (tri_t < max_t && tri_t >= 0.0f) {
                        return true;
                    }
                }
            }
            if (stack_size == 0)
                return false;
            cur_node = stack[--stack_size];
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(cur_node);
        const int right_index = bvh_query_node_upper_payload(cur_node);

        BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        float t0 = FLT_MAX;
        float t1 = FLT_MAX;
        const bool h0 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(left_lower.x, left_lower.y, left_lower.z),
                            vec3(left_upper.x, left_upper.y, left_upper.z), t0
                        )
            && t0 < max_t;
        const bool h1 = mesh_query_ray_intersect_aabb(
                            start, dir, rcp_dir, fast_aabb, vec3(right_lower.x, right_lower.y, right_lower.z),
                            vec3(right_upper.x, right_upper.y, right_upper.z), t1
                        )
            && t1 < max_t;

        if (h0 && h1) {
            const bool near_left = (t0 < t1);
            if (stack_size >= BVH_QUERY_STACK_SIZE)
                return false;
            const uint64_t left_node = bvh_query_node_pack(left_lower, left_upper);
            const uint64_t right_node = bvh_query_node_pack(right_lower, right_upper);
            stack[stack_size++] = near_left ? right_node : left_node;
            cur_node = near_left ? left_node : right_node;
        } else if (h0) {
            cur_node = bvh_query_node_pack(left_lower, left_upper);
        } else if (h1) {
            cur_node = bvh_query_node_pack(right_lower, right_upper);
        } else {
            if (stack_size == 0)
                return false;
            cur_node = stack[--stack_size];
        }
    }
}

CUDA_CALLABLE inline int mesh_query_ray_count_intersections(uint64_t id, const vec3& start, const vec3& dir, int root)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];

    stack[0] = root == -1 ? *mesh.bvh.root : root;
    int count = 1;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);

    int num_hit = 0;
    float temp_t;

    while (count) {
        const int node_index = stack[--count];

        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

        bool hit = intersect_ray_aabb_robust(
            start, dir, rcp_dir, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z), temp_t
        );

        if (hit) {
            if (lower.b) {
                const int start_index = lower.i;
                const int end_index = upper.i;
                // loops through primitives in the leaf
                for (int primitive_counter = start_index; primitive_counter < end_index; primitive_counter++) {
                    int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                    int i = mesh.indices[primitive_index * 3 + 0];
                    int j = mesh.indices[primitive_index * 3 + 1];
                    int k = mesh.indices[primitive_index * 3 + 2];

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float temp_t, temp_u, temp_v, temp_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, temp_t, temp_u, temp_v, temp_sign, &n)) {
                        if (temp_t >= 0.0f) {
                            num_hit++;
                        }
                    }
                }
            } else {
                stack[count++] = lower.i;
                stack[count++] = upper.i;
            }
        }
    }

    return num_hit;
}

template <typename T> CUDA_CALLABLE inline void _swap(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}

CUDA_CALLABLE inline bool mesh_query_ray_ordered(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float& t,
    float& u,
    float& v,
    float& sign,
    vec3& normal,
    int& face,
    int root = -1
)
{
    Mesh mesh = mesh_get(id);

    int stack[BVH_QUERY_STACK_SIZE];
    float stack_dist[BVH_QUERY_STACK_SIZE];

    stack[0] = root == -1 ? *mesh.bvh.root : root;
    stack_dist[0] = -FLT_MAX;

    int count = 1;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);

    float min_t = max_t;
    int min_face;
    float min_u;
    float min_v;
    float min_sign = 1.0f;
    vec3 min_normal;

    while (count) {
        count -= 1;

        const int node_index = stack[count];
        const float node_dist = stack_dist[count];

        if (node_dist < min_t) {
            int left_index = mesh.bvh.node_lowers[node_index].i;
            int right_index = mesh.bvh.node_uppers[node_index].i;
            bool leaf = mesh.bvh.node_lowers[node_index].b;

            if (leaf) {
                const int start_index = left_index;
                const int end_index = right_index;
                // loops through primitives in the leaf
                for (int primitive_counter = start_index; primitive_counter < end_index; primitive_counter++) {
                    int primitive_index = mesh.bvh.primitive_indices[primitive_counter];
                    int i = mesh.indices[primitive_index * 3 + 0];
                    int j = mesh.indices[primitive_index * 3 + 1];
                    int k = mesh.indices[primitive_index * 3 + 2];

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float temp_t, temp_u, temp_v, temp_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, temp_t, temp_u, temp_v, temp_sign, &n)) {
                        if (temp_t < min_t && temp_t >= 0.0f) {
                            min_t = temp_t;
                            min_face = primitive_index;
                            min_u = temp_u;
                            min_v = temp_v;
                            min_sign = temp_sign;
                            min_normal = n;
                        }
                    }
                }
            } else {
                BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
                BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);

                BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
                BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

                float left_dist = FLT_MAX;
                bool left_hit = intersect_ray_aabb_robust(
                    start, dir, rcp_dir, vec3(left_lower.x, left_lower.y, left_lower.z),
                    vec3(left_upper.x, left_upper.y, left_upper.z), left_dist
                );

                float right_dist = FLT_MAX;
                bool right_hit = intersect_ray_aabb_robust(
                    start, dir, rcp_dir, vec3(right_lower.x, right_lower.y, right_lower.z),
                    vec3(right_upper.x, right_upper.y, right_upper.z), right_dist
                );


                if (left_dist < right_dist) {
                    _swap(left_index, right_index);
                    _swap(left_dist, right_dist);
                    _swap(left_hit, right_hit);
                }

                if (left_hit && left_dist < min_t) {
                    stack[count] = left_index;
                    stack_dist[count] = left_dist;
                    count += 1;
                }

                if (right_hit && right_dist < min_t) {
                    stack[count] = right_index;
                    stack_dist[count] = right_dist;
                    count += 1;
                }
            }
        }
    }

    if (min_t < max_t) {
        // write outputs
        u = min_u;
        v = min_v;
        sign = min_sign;
        t = min_t;
        normal = normalize(min_normal);
        face = min_face;

        return true;
    } else {
        return false;
    }
}

CUDA_CALLABLE inline void adj_mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    float t,
    float u,
    float v,
    float sign,
    const vec3& n,
    int face,
    int root,
    uint64_t adj_id,
    vec3& adj_start,
    vec3& adj_dir,
    float& adj_max_t,
    float& adj_t,
    float& adj_u,
    float& adj_v,
    float& adj_sign,
    vec3& adj_n,
    int& adj_face,
    int& adj_root,
    bool& adj_ret
)
{

    Mesh mesh = mesh_get(id);

    // face is determined by BVH in forward pass
    int i = mesh.indices[face * 3 + 0];
    int j = mesh.indices[face * 3 + 1];
    int k = mesh.indices[face * 3 + 2];

    vec3 a = mesh.points[i];
    vec3 b = mesh.points[j];
    vec3 c = mesh.points[k];

    vec3 adj_a, adj_b, adj_c;

    adj_intersect_ray_tri_woop(
        start, dir, a, b, c, t, u, v, sign, n, adj_start, adj_dir, adj_a, adj_b, adj_c, adj_t, adj_u, adj_v, adj_sign,
        adj_n, adj_ret
    );
}

// Stores the result of querying the closest point on a mesh.
struct mesh_query_ray_t {
    CUDA_CALLABLE mesh_query_ray_t()
        : result(false)
        , sign(0.0f)
        , face(0)
        , t(0.0f)
        , u(0.0f)
        , v(0.0f)
        , normal()
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_ray_t& operator+=(const mesh_query_ray_t& other)
    {
        result |= other.result;  // Use OR for bool accumulation
        sign += other.sign;
        face += other.face;
        t += other.t;
        u += other.u;
        v += other.v;
        normal += other.normal;
        return *this;
    }

    float sign;
    int face;
    float t;
    float u;
    float v;
    vec3 normal;
    bool result;
};

CUDA_CALLABLE inline mesh_query_ray_t
mesh_query_ray(uint64_t id, const vec3& start, const vec3& dir, float max_t, int root)
{
    mesh_query_ray_t query;
    query.result
        = mesh_query_ray(id, start, dir, max_t, query.t, query.u, query.v, query.sign, query.normal, query.face, root);
    return query;
}

CUDA_CALLABLE inline mesh_query_ray_t
mesh_query_ray_seeded(uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face)
{
    mesh_query_ray_t query;
    query.result = mesh_query_ray_seeded(
        id, start, dir, max_t, seed_face, query.t, query.u, query.v, query.sign, query.normal, query.face
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_ray_t
mesh_query_ray_exclusive(uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face)
{
    mesh_query_ray_t query;
    query.result = mesh_query_ray_exclusive(
        id, start, dir, max_t, seed_face, query.t, query.u, query.v, query.sign, query.normal, query.face
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_ray_t mesh_query_ray_exclusive_cached(
    uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face, int cached_node
)
{
    mesh_query_ray_t query;
    query.result = mesh_query_ray_exclusive_cached(
        id, start, dir, max_t, seed_face, cached_node, query.t, query.u, query.v, query.sign, query.normal, query.face
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_ray_t mesh_query_ray_exclusive_cached_bottom_up(
    uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face, int cached_node
)
{
    mesh_query_ray_t query;
    query.result = mesh_query_ray_exclusive_cached_bottom_up(
        id, start, dir, max_t, seed_face, cached_node, query.t, query.u, query.v, query.sign, query.normal, query.face
    );
    return query;
}

CUDA_CALLABLE inline mesh_query_ray_t mesh_query_ray_exclusive_cached_peeling(
    uint64_t id, const vec3& start, const vec3& dir, float max_t, int seed_face, int cached_node
)
{
    mesh_query_ray_t query;
    query.result = mesh_query_ray_exclusive_cached_peeling(
        id, start, dir, max_t, seed_face, cached_node, query.t, query.u, query.v, query.sign, query.normal, query.face
    );
    return query;
}

CUDA_CALLABLE inline void adj_mesh_query_ray(
    uint64_t id,
    const vec3& start,
    const vec3& dir,
    float max_t,
    int root,
    const mesh_query_ray_t& ret,
    uint64_t adj_id,
    vec3& adj_start,
    vec3& adj_dir,
    float& adj_max_t,
    int& adj_root,
    mesh_query_ray_t& adj_ret
)
{
    adj_mesh_query_ray(
        id, start, dir, max_t, ret.t, ret.u, ret.v, ret.sign, ret.normal, ret.face, root, adj_id, adj_start, adj_dir,
        adj_max_t, adj_ret.t, adj_ret.u, adj_ret.v, adj_ret.sign, adj_ret.normal, adj_ret.face, adj_root, adj_ret.result
    );
}

// Flat-stack closest-hit traversal that returns only the sign of the closest
// hit. Used by mesh_query_inside_ray_tracing for the three axis-aligned probe
// rays: those rays penetrate the whole mesh and rarely allow pruning, so the
// eager-child-loading overhead of the near-far mesh_query_ray traversal is not
// worth paying. This function uses the classic push-both-children approach,
// which has half the BVH node loads per inner step.
CUDA_CALLABLE inline bool
mesh_query_ray_closest_sign(const Mesh& mesh, const vec3& start, const vec3& dir, float& out_sign)
{
    int stack[BVH_QUERY_STACK_SIZE];
    int stack_size = 0;
    int node_index = *mesh.bvh.root;

    vec3 rcp_dir(1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]);
    float min_t = FLT_MAX;
    float temp_t;
    bool hit = false;

    while (true) {
        BVHPackedNodeHalf lower = bvh_load_node(mesh.bvh.node_lowers, node_index);
        BVHPackedNodeHalf upper = bvh_load_node(mesh.bvh.node_uppers, node_index);

        if (intersect_ray_aabb_robust(
                start, dir, rcp_dir, vec3(lower.x, lower.y, lower.z), vec3(upper.x, upper.y, upper.z), temp_t
            )
            && temp_t < min_t) {
            if (lower.b) {
                for (int pc = lower.i; pc < upper.i; ++pc) {
                    int primitive_index = mesh.bvh.primitive_indices[pc];
                    int i = mesh.indices[primitive_index * 3 + 0];
                    int j = mesh.indices[primitive_index * 3 + 1];
                    int k = mesh.indices[primitive_index * 3 + 2];

                    vec3 p = mesh.points[i];
                    vec3 q = mesh.points[j];
                    vec3 r = mesh.points[k];

                    float tri_t, tri_u, tri_v, tri_sign;
                    vec3 n;

                    if (intersect_ray_tri_woop(start, dir, p, q, r, tri_t, tri_u, tri_v, tri_sign, &n)) {
                        if (tri_t >= 0.0f && tri_t < min_t) {
                            min_t = tri_t;
                            out_sign = tri_sign;
                            hit = true;
                        }
                    }
                }
            } else {
                stack[stack_size++] = lower.i;
                stack[stack_size++] = upper.i;
            }
        }

        if (stack_size == 0)
            break;
        node_index = stack[--stack_size];
    }
    return hit;
}

// determine if a point is inside (ret < 0 ) or outside the mesh (ret > 0) using ray tracing
CUDA_CALLABLE inline float mesh_query_inside_ray_tracing(uint64_t id, const vec3& p)
{
    Mesh mesh = mesh_get(id);

    int vote = 0;
    float sign;

    for (int i = 0; i < 3; ++i) {
        if (mesh_query_ray_closest_sign(mesh, p, vec3(float(i == 0), float(i == 1), float(i == 2)), sign) && sign < 0) {
            vote++;
        }
    }

    if (vote >= 2)
        return -1.0f;
    else
        return 1.0f;
}


// determine if a point is inside (ret < 0 ) or outside the mesh (ret > 0)
CUDA_CALLABLE inline float
mesh_query_inside_parity(uint64_t id, const vec3& p, const vec3 base_dir, int n_sample, float perturbation_scale)
{
    int vote = 0;

    // deterministic
    uint32_t rand_state = rand_init(42);

    for (int i = 0; i < n_sample; ++i) {

        vec3 dir;
        do {
            dir = base_dir
                + vec3(
                      randf(rand_state, -perturbation_scale, perturbation_scale),
                      randf(rand_state, -perturbation_scale, perturbation_scale),
                      randf(rand_state, -perturbation_scale, perturbation_scale)
                );
        } while (length_sq(dir) < 1e-8f);

        if (mesh_query_ray_count_intersections(id, p, dir) % 2) {
            vote++;
        }
    }

    if (vote * 2 >= n_sample)
        return -1.0f;
    else
        return 1.0f;
}

// stores state required to traverse the BVH nodes that
// overlap with a query AABB.
struct mesh_query_aabb_t {
    CUDA_CALLABLE mesh_query_aabb_t()
        : mesh()
        , stack()
        , count(0)
        , input_lower()
        , input_upper()
        , prim_cur(0)
        , prim_end(0)
        , cur_node(0)
        , have_node(false)
        , pair_limit(-1)
        , face(0)
        , last_query_valid(true)
    {
    }

    // Required for adjoint computations.
    CUDA_CALLABLE inline mesh_query_aabb_t& operator+=(const mesh_query_aabb_t& other) { return *this; }

    // Mesh ID
    Mesh mesh;

    // BVH traversal stack of node indices; every entry passed its AABB test
    // before being pushed.
    // On CUDA the stack lives in shared memory: keeping an array out of this
    // struct lets the compiler keep the remaining members in registers.
#if BVH_SHARED_STACK
    bvh_stack_t stack;
#else
    int stack[BVH_QUERY_STACK_SIZE];
#endif

    int count;

    // inputs
    wp::vec3 input_lower;
    wp::vec3 input_upper;

    // primitive range of the packed leaf currently being enumerated;
    // when prim_cur < prim_end the query resumes mid-leaf on the next
    // mesh_query_aabb_next() call
    int prim_cur;
    int prim_end;

    // packed payload (see bvh_query_node_pack()) of the node to process next,
    // valid when have_node is set; it already passed its AABB test
    uint64_t cur_node;
    bool have_node;

    // stack occupancy up to which far children may be pushed as two-slot
    // payload pairs (see bvh_query_pair_limit())
    int pair_limit;

    // Face
    int face;

    // Tracks whether the most recent mesh_query_aabb_next() / tile_mesh_query_aabb_next()
    // call produced a valid face index. Seeded to true so an initial tile_query_valid()
    // check (before any next() call) reports valid.
    bool last_query_valid;
};


CUDA_CALLABLE inline mesh_query_aabb_t mesh_query_aabb(uint64_t id, const vec3& lower, const vec3& upper)
{
    // initialize empty
    mesh_query_aabb_t query;
    query.face = -1;

    Mesh mesh = mesh_get(id);
    query.mesh = mesh;

#if BVH_SHARED_STACK
    __shared__ int stack[BVH_QUERY_STACK_SIZE * WP_TILE_BLOCK_DIM];
    query.stack.ptr = &stack[threadIdx.x];
#endif

    query.input_lower = lower;
    query.input_upper = upper;

    query.pair_limit = bvh_query_pair_limit(mesh.bvh);

    // Both stack entries and cur_node must have passed their AABB test
    // already, so test the root here.
    const int root_index = *mesh.bvh.root;
    const BVHPackedNodeHalf root_lower = bvh_load_node(mesh.bvh.node_lowers, root_index);
    const BVHPackedNodeHalf root_upper = bvh_load_node(mesh.bvh.node_uppers, root_index);

    if (intersect_aabb_aabb(
            lower, upper, reinterpret_cast<const vec3&>(root_lower), reinterpret_cast<const vec3&>(root_upper)
        )) {
        query.cur_node = bvh_query_node_pack(root_lower, root_upper);
        query.have_node = true;
    }

    return query;
}

CUDA_CALLABLE inline bool mesh_query_aabb_next(mesh_query_aabb_t& query, int& index)
{
    const Mesh mesh = query.mesh;

    // A single flat loop; every iteration either emits one primitive from
    // the packed leaf currently being enumerated, or processes one node.
    for (;;) {
        if (query.prim_cur < query.prim_end) {
            const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, query.prim_cur++);

            // Load the face bounds eagerly so the test below compiles to one
            // predicate chain instead of a branch per component.
            const vec3 face_lower = bvh_load_vec3(mesh.lowers, primitive_index);
            const vec3 face_upper = bvh_load_vec3(mesh.uppers, primitive_index);

            if (intersect_aabb_aabb(query.input_lower, query.input_upper, face_lower, face_upper)) {
                index = primitive_index;
                query.face = primitive_index;
                return true;
            }
            continue;
        }

        if (!query.have_node) {
            if (!query.count)
                return false;

            const unsigned top = unsigned(query.stack[--query.count]);
            if (top & 0x80000000u) {
                // Payload pair: reconstruct the node without any memory access.
                query.cur_node = bvh_query_stack_unpack(unsigned(query.stack[--query.count]), top);
            } else {
                // Index entry: it already passed its AABB test, so the AABB
                // part of this load is unused and no re-test is needed.
                query.cur_node = bvh_query_node_load(mesh.bvh, int(top));
            }
        }

        const uint64_t node = query.cur_node;
        query.have_node = false;

        if (bvh_query_node_is_leaf(node)) {
            const int start = bvh_query_node_lower_payload(node);
            const int end = bvh_query_node_upper_payload(node);

            // Fast path when the leaf contains exactly one primitive: its
            // AABB is the leaf node's AABB, which already passed its test.
            if (end - start == 1) {
                const int primitive_index = bvh_load_int(mesh.bvh.primitive_indices, start);
                index = primitive_index;
                query.face = primitive_index;
                return true;
            }

            // Packed leaf: enumerate its primitives one per loop iteration,
            // without re-loading the leaf node.
            query.prim_cur = start;
            query.prim_end = end;
            continue;
        }

        const int left_index = bvh_query_node_lower_payload(node);
        const int right_index = bvh_query_node_upper_payload(node);

        const BVHPackedNodeHalf left_lower = bvh_load_node(mesh.bvh.node_lowers, left_index);
        const BVHPackedNodeHalf left_upper = bvh_load_node(mesh.bvh.node_uppers, left_index);
        const BVHPackedNodeHalf right_lower = bvh_load_node(mesh.bvh.node_lowers, right_index);
        const BVHPackedNodeHalf right_upper = bvh_load_node(mesh.bvh.node_uppers, right_index);

        const bool hit_left = intersect_aabb_aabb(
            query.input_lower, query.input_upper, reinterpret_cast<const vec3&>(left_lower),
            reinterpret_cast<const vec3&>(left_upper)
        );
        const bool hit_right = intersect_aabb_aabb(
            query.input_lower, query.input_upper, reinterpret_cast<const vec3&>(right_lower),
            reinterpret_cast<const vec3&>(right_upper)
        );

        if (hit_left) {
            query.cur_node = bvh_query_node_pack(left_lower, left_upper);
            query.have_node = true;
            if (hit_right) {
                // Pair pushes stop at pair_limit so slot usage cannot exceed
                // the stack for constructor-produced trees.
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


CUDA_CALLABLE inline int iter_next(mesh_query_aabb_t& query) { return query.face; }

CUDA_CALLABLE inline bool iter_cmp(mesh_query_aabb_t& query)
{
    bool finished = mesh_query_aabb_next(query, query.face);
    return finished;
}

CUDA_CALLABLE inline mesh_query_aabb_t iter_reverse(const mesh_query_aabb_t& query)
{
    // can't reverse BVH queries, users should not rely on neighbor ordering
    return query;
}

CUDA_CALLABLE inline vec3 mesh_eval_position(uint64_t id, int tri, float u, float v)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    return p * u + q * v + r * (1.0f - u - v);
}

CUDA_CALLABLE inline vec3 mesh_eval_velocity(uint64_t id, int tri, float u, float v)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 vp = mesh.velocities[i];
    vec3 vq = mesh.velocities[j];
    vec3 vr = mesh.velocities[k];

    return vp * u + vq * v + vr * (1.0f - u - v);
}


CUDA_CALLABLE inline void adj_mesh_eval_position(
    uint64_t id,
    int tri,
    float u,
    float v,
    uint64_t& adj_id,
    int& adj_tri,
    float& adj_u,
    float& adj_v,
    const vec3& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return;

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    adj_u += (p[0] - r[0]) * adj_ret[0] + (p[1] - r[1]) * adj_ret[1] + (p[2] - r[2]) * adj_ret[2];
    adj_v += (q[0] - r[0]) * adj_ret[0] + (q[1] - r[1]) * adj_ret[1] + (q[2] - r[2]) * adj_ret[2];
}

CUDA_CALLABLE inline void adj_mesh_eval_velocity(
    uint64_t id,
    int tri,
    float u,
    float v,
    uint64_t& adj_id,
    int& adj_tri,
    float& adj_u,
    float& adj_v,
    const vec3& adj_ret
)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return;

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 vp = mesh.velocities[i];
    vec3 vq = mesh.velocities[j];
    vec3 vr = mesh.velocities[k];

    adj_u += (vp[0] - vr[0]) * adj_ret[0] + (vp[1] - vr[1]) * adj_ret[1] + (vp[2] - vr[2]) * adj_ret[2];
    adj_v += (vq[0] - vr[0]) * adj_ret[0] + (vq[1] - vr[1]) * adj_ret[1] + (vq[2] - vr[2]) * adj_ret[2];
}

CUDA_CALLABLE inline vec3 mesh_eval_face_normal(uint64_t id, int tri)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

    assert(tri < mesh.num_tris);

    int i = mesh.indices[tri * 3 + 0];
    int j = mesh.indices[tri * 3 + 1];
    int k = mesh.indices[tri * 3 + 2];

    vec3 p = mesh.points[i];
    vec3 q = mesh.points[j];
    vec3 r = mesh.points[k];

    return normalize(cross(q - p, r - p));
}

CUDA_CALLABLE inline void
adj_mesh_eval_face_normal(uint64_t id, int tri, uint64_t& adj_id, int& adj_tri, const vec3& adj_ret)
{
    // MISSINGADJOINT: backprop through normalize(cross(q-p, r-p)) to
    // mesh.points.grad slots for the three face vertex indices
}

CUDA_CALLABLE inline vec3 mesh_get_point(uint64_t id, int index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.points)
        return vec3();

#if FP_CHECK
    if (index >= mesh.num_tris * 3) {
        printf("mesh_get_point (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
        assert(0);
    }
#endif

    int i = mesh.indices[index];
    return mesh.points[i];
}

CUDA_CALLABLE inline void
adj_mesh_get_point(uint64_t id, int index, uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
    // MISSINGADJOINT: atomic-add adj_ret into mesh.points.grad[index] when the gradient
    // buffer is allocated
}

CUDA_CALLABLE inline vec3 mesh_get_velocity(uint64_t id, int index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.velocities)
        return vec3();

#if FP_CHECK
    if (index >= mesh.num_tris * 3) {
        printf("mesh_get_velocity (%llu, %d) out of bounds at %s:%d\n", id, index, __FILE__, __LINE__);
        assert(0);
    }
#endif

    int i = mesh.indices[index];
    return mesh.velocities[i];
}

CUDA_CALLABLE inline void
adj_mesh_get_velocity(uint64_t id, int index, uint64_t& adj_id, int& adj_index, const vec3& adj_ret)
{
    // MISSINGADJOINT: atomic-add adj_ret into mesh.velocities.grad[index] when the
    // gradient buffer is allocated
}

CUDA_CALLABLE inline int mesh_get_index(uint64_t id, int face_vertex_index)
{
    Mesh mesh = mesh_get(id);

    if (!mesh.indices)
        return -1;

    assert(face_vertex_index < mesh.num_tris * 3);

    return mesh.indices[face_vertex_index];
}

CUDA_CALLABLE bool mesh_get_descriptor(uint64_t id, Mesh& mesh);
CUDA_CALLABLE bool mesh_set_descriptor(uint64_t id, const Mesh& mesh);
CUDA_CALLABLE void mesh_add_descriptor(uint64_t id, const Mesh& mesh);
CUDA_CALLABLE void mesh_rem_descriptor(uint64_t id);

}  // namespace wp


#include "tile_mesh.h"
