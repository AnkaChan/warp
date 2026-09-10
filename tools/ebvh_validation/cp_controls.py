"""Disentangle root traversal changes and the unified CP depth policy."""

import warp as wp
import closest_point as cp
from cp_policy import gated_kernel


@wp.func_native("""
    const wp::Mesh mesh = wp::mesh_get(id);
    float d2 = 1.0e12f;
    int face = -1;
    float v = 0.0f, w = 0.0f;
    wp::mesh_query_point_no_sign_traverse<true, false>(mesh, *mesh.bvh.root, -1, point, d2, face, v, w);
    return wp::vec3(static_cast<float>(face), 1.0f-v-w, v);
""")
def unseeded_query(id: wp.uint64, point: wp.vec3) -> wp.vec3: ...


@wp.kernel(module="unique", module_options={"enable_backward": False, "fast_math": True})
def unseeded_kernel(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), seeds: wp.array(dtype=int),
                    nodes: wp.array(dtype=int), faces: wp.array(dtype=int), positions: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    result = unseeded_query(mesh, points[i])
    face = int(result[0])
    faces[i] = face
    positions[i] = wp.mesh_eval_position(mesh, face, result[1], result[2])


if __name__ == "__main__":
    original_factory = cp.make_kernel

    def factory(mode):
        if mode == "optimized_root":
            return unseeded_kernel
        if mode == "gated":
            return gated_kernel
        return original_factory(mode)

    cp.make_kernel = factory
    cp.ARMS = ("root", "optimized_root", "warm", "walk", "cached", "gated", "oracle")
    cp.VERSION = "ebvh-cp-controls-v1"
    print("[EBVH]", cp.VERSION, flush=True)
    cp.main()
