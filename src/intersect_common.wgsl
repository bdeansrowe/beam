// intersect_common.wgsl — BVH traversal logic shared by canvas and bloom intersect variants.
// NOT a standalone compute shader — no @compute entry point.
// Composed into each intersect_variant_*.wgsl pipeline at pipeline creation time.
// Bindings (bvh_nodes, tlas_instances, spheres, hit_records) are declared in intersect.wgsl
// and are globally visible within the concatenated module.

// ── BVH traversal — writes one HitRecord per ray into hit_records[idx] ────────
fn traverse_bvh(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32, idx: u32) {
    var best_t       = tmax;
    var best_prim    = 0u;
    var best_face_fw = 0u;
    var did_hit      = false;

    let num_instances = arrayLength(&tlas_instances);

    for (var inst_idx = 0u; inst_idx < num_instances; inst_idx++) {
        let inst         = tlas_instances[inst_idx];
        let local_origin = (inst.transform * vec4<f32>(origin, 1.0)).xyz;
        let local_dir    = (inst.transform * vec4<f32>(dir,    0.0)).xyz;
        let local_inv    = 1.0 / local_dir;

        // Stack-based BLAS traversal — 32 entries per CLAUDE.md
        var stack:     array<u32, 32>;
        var stack_ptr: i32 = 0;
        stack[0]  = inst.blas_offset;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr];
            let node     = bvh_nodes[node_idx];

            if !aabb_hit(node, local_origin, local_inv, tmin, best_t) { continue; }

            if node.node_type == NODE_LEAF_SPHERE {
                let sidx = node_sphere_index(node);
                let t    = sphere_hit(spheres[sidx], local_origin, local_dir, tmin, best_t);
                if t > 0.0 {
                    let hit_pos  = local_origin + t * local_dir;
                    let out_n    = normalize(hit_pos - spheres[sidx].center_radius.xyz);
                    did_hit      = true;
                    best_t       = t;
                    best_prim    = sidx;
                    best_face_fw = select(0u, 1u, dot(local_dir, out_n) < 0.0);
                }
            } else {
                // NODE_INTERNAL: push right child first so left is popped first (LIFO)
                let rc = node_right_child(node);
                let lc = node_left_child(node);
                if rc != INVALID_NODE && stack_ptr < 32 {
                    stack[stack_ptr] = rc;
                    stack_ptr += 1;
                }
                if lc != INVALID_NODE && stack_ptr < 32 {
                    stack[stack_ptr] = lc;
                    stack_ptr += 1;
                }
            }
        }
    }

    if did_hit {
        hit_records[idx] = HitRecord(best_t, best_prim, vec2<f32>(0.0), best_face_fw, 0u, 0u, 0u);
    } else {
        hit_records[idx] = HitRecord(F32_MAX, 0u, vec2<f32>(0.0), 0u, 0u, 0u, 0u);
    }
}
