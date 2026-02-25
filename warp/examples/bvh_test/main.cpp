#include "warp.h"

#include "bvh.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace wp;

// Generate random float in [lo, hi]
static float randf(float lo, float hi) { return lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo); }

void test_host_bvh(int num_items)
{
    printf("=== Host BVH Test (%d AABBs) ===\n", num_items);

    // Generate random AABBs
    std::vector<vec3> lowers(num_items);
    std::vector<vec3> uppers(num_items);

    for (int i = 0; i < num_items; i++) {
        float cx = randf(-10.0f, 10.0f);
        float cy = randf(-10.0f, 10.0f);
        float cz = randf(-10.0f, 10.0f);
        float hx = randf(0.1f, 1.0f);
        float hy = randf(0.1f, 1.0f);
        float hz = randf(0.1f, 1.0f);

        lowers[i] = vec3(cx - hx, cy - hy, cz - hz);
        uppers[i] = vec3(cx + hx, cy + hy, cz + hz);
    }

    // Create host BVH using SAH constructor
    BVH bvh;
    memset(&bvh, 0, sizeof(BVH));
    bvh_create_host(lowers.data(), uppers.data(), num_items, BVH_CONSTRUCTOR_SAH, nullptr, 1, bvh);

    printf("  BVH created successfully\n");
    printf("  num_items:      %d\n", bvh.num_items);
    printf("  num_nodes:      %d\n", bvh.num_nodes);
    printf("  num_leaf_nodes: %d\n", bvh.num_leaf_nodes);
    printf("  max_depth:      %d\n", bvh.max_depth);
    printf("  root index:     %d\n", *bvh.root);

    // Print root bounds
    BVHPackedNodeHalf& root_lower = bvh.node_lowers[*bvh.root];
    BVHPackedNodeHalf& root_upper = bvh.node_uppers[*bvh.root];
    printf(
        "  root bounds:    (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", root_lower.x, root_lower.y, root_lower.z,
        root_upper.x, root_upper.y, root_upper.z
    );

    // Simple overlap query: test a point against the BVH manually
    vec3 query_point(0.0f, 0.0f, 0.0f);
    int overlap_count = 0;
    for (int i = 0; i < num_items; i++) {
        if (query_point[0] >= lowers[i][0] && query_point[0] <= uppers[i][0] && query_point[1] >= lowers[i][1]
            && query_point[1] <= uppers[i][1] && query_point[2] >= lowers[i][2] && query_point[2] <= uppers[i][2]) {
            overlap_count++;
        }
    }
    printf("  AABBs containing origin (brute force): %d\n", overlap_count);

    // Cleanup
    bvh_destroy_host(bvh);
    printf("  BVH destroyed\n\n");
}

#if WP_ENABLE_CUDA

void test_device_bvh(int num_items)
{
    printf("=== Device BVH Test (%d AABBs) ===\n", num_items);

    // Generate random AABBs on host
    std::vector<vec3> h_lowers(num_items);
    std::vector<vec3> h_uppers(num_items);

    for (int i = 0; i < num_items; i++) {
        float cx = randf(-10.0f, 10.0f);
        float cy = randf(-10.0f, 10.0f);
        float cz = randf(-10.0f, 10.0f);
        float hx = randf(0.1f, 1.0f);
        float hy = randf(0.1f, 1.0f);
        float hz = randf(0.1f, 1.0f);

        h_lowers[i] = vec3(cx - hx, cy - hy, cz - hz);
        h_uppers[i] = vec3(cx + hx, cy + hy, cz + hz);
    }

    // Copy to device
    vec3* d_lowers = nullptr;
    vec3* d_uppers = nullptr;
    size_t size = num_items * sizeof(vec3);

    void* context = wp_cuda_context_get_current();
    d_lowers = (vec3*)wp_alloc_device(context, size);
    d_uppers = (vec3*)wp_alloc_device(context, size);
    wp_memcpy_h2d(context, d_lowers, h_lowers.data(), size);
    wp_memcpy_h2d(context, d_uppers, h_uppers.data(), size);

    // Create device BVH (uses LBVH constructor on GPU)
    uint64_t bvh_id = wp_bvh_create_device(context, d_lowers, d_uppers, num_items, BVH_CONSTRUCTOR_LBVH, nullptr, 1);

    if (bvh_id == 0) {
        printf("  ERROR: Failed to create device BVH\n\n");
        wp_free_device(context, d_lowers);
        wp_free_device(context, d_uppers);
        return;
    }

    // Retrieve the host-side descriptor for inspection
    BVH bvh_desc;
    if (bvh_get_descriptor(bvh_id, bvh_desc)) {
        printf("  Device BVH created successfully\n");
        printf("  num_items:      %d\n", bvh_desc.num_items);
        printf("  num_nodes:      %d\n", bvh_desc.num_nodes);
        printf("  num_leaf_nodes: %d\n", bvh_desc.num_leaf_nodes);
        printf("  max_depth:      %d\n", bvh_desc.max_depth);
    } else {
        printf("  Device BVH created (id=%llu), descriptor not available on host\n", (unsigned long long)bvh_id);
    }

    // Cleanup
    wp_bvh_destroy_device(bvh_id);
    wp_free_device(context, d_lowers);
    wp_free_device(context, d_uppers);
    printf("  Device BVH destroyed\n\n");
}

#endif  // WP_ENABLE_CUDA

int main()
{
    srand((unsigned int)time(nullptr));

    printf("Warp BVH Test\n");
    printf("=============\n\n");

    // Initialize warp runtime
    wp_init(nullptr);

    // Test host BVH
    test_host_bvh(100);
    test_host_bvh(10000);

#if WP_ENABLE_CUDA
    // Initialize CUDA
    int cuda_devices = wp_cuda_device_get_count();
    if (cuda_devices > 0) {
        printf("Found %d CUDA device(s)\n\n", cuda_devices);
        void* ctx = wp_cuda_device_get_primary_context(0);
        wp_cuda_context_set_current(ctx);

        test_device_bvh(100);
        test_device_bvh(10000);
    } else {
        printf("No CUDA devices found, skipping device tests\n\n");
    }
#else
    printf("CUDA not enabled, skipping device tests\n\n");
#endif

    printf("Done!\n");
    return 0;
}
