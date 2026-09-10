"""Measure a depth gate sharing one seed evaluation and one traversal body."""

import warp as wp
import closest_point as cp


@wp.func_native("""
    const wp::Mesh mesh = wp::mesh_get(id);
    float d2 = 1.0e12f;
    int face = -1;
    float v = 0.0f, w = 0.0f;
    const int leaf = wp::mesh_query_point_no_sign_initialize_seed_leaf(mesh, seed, point, d2, face, v, w);
    int start = *mesh.bvh.root;
    if (node >= 0 && node < mesh.bvh.num_nodes &&
        wp::bvh_exclusive_node_depth(wp::bvh_get_exclusive_node(mesh.bvh, node)) >= 8)
        start = wp::mesh_query_point_no_sign_find_exclusive_containment(mesh, point, d2, node);
    if (start != leaf)
        wp::mesh_query_point_no_sign_traverse<false, true>(mesh, start, leaf, point, d2, face, v, w);
    return wp::vec3(static_cast<float>(face), 1.0f-v-w, v);
""")
def gated_query(id: wp.uint64, point: wp.vec3, seed: int, node: int) -> wp.vec3: ...


@wp.kernel(module="unique", module_options={"enable_backward": False, "fast_math": True})
def gated_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), seeds: wp.array(dtype=int),
                 nodes: wp.array(dtype=int), faces: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result = gated_query(mesh, points[i], seeds[i], nodes[i])
    face = int(result[0])
    faces[i] = face
    positions[i] = wp.mesh_eval_position(mesh, face, result[1], result[2])


if __name__ == "__main__":
    original_factory = cp.make_kernel
    cp.make_kernel = lambda mode: gated_kernel if mode == "gated" else original_factory(mode)
    cp.VERSION = "ebvh-cp-unified-gate-v1"
    print("[EBVH]", cp.VERSION, flush=True)
    cp.main()
