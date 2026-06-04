// ── Bindings ──────────────────────────────────────────────────────────────────
// group(0) = scene-global resources (BVH, geometry, materials)
@group(0) @binding(0) var<storage, read> bvh_nodes      : array<BvhNode>;
@group(0) @binding(1) var<storage, read> tlas_instances : array<TlasInstance>;
@group(0) @binding(2) var<storage, read> spheres        : array<Sphere>;
// Step 5.5 — declared, not yet used
@group(0) @binding(3) var<storage, read> vertices       : array<Vertex>;
@group(0) @binding(4) var<storage, read> geometry       : array<TriangleRecord>;
// Step 6 — declared, not yet used
@group(0) @binding(5) var<storage, read> materials      : array<Material>;
// Step 7 — declared, not yet used
@group(0) @binding(6) var<storage, read> lights         : array<LightUniform>;
// B07a — declared, not yet used
@group(0) @binding(7) var<uniform>       frame_data     : FrameUniform;

// group(1) = per-pass resources
@group(1) @binding(0) var<storage, read_write>  rays        : array<Ray>;
@group(1) @binding(1) var<storage, read_write>  scratch_buf : array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> hit_records : array<HitRecord>;

// ── BVH traversal — writes one HitRecord per ray into hit_records[idx] ────────
// Internal state uses anonymous locals; the old traversal-local HitRecord struct
// (t, normal, hit: bool) is gone — replaced by the buffer HitRecord above.
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

// ── Main ──────────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= frame_data.dim_x || py >= frame_data.dim_y { return; }

    let idx = py * frame_data.dim_x + px;

    // Terminated-ray early exit (sentinel written on miss or by roulette_pass).
    if rays[idx].direction.w < 0.0 {
        hit_records[idx] = HitRecord(F32_MAX, 0u, vec2<f32>(0.0), 0u, 0u, 0u, 0u);
        return;
    }

    let ray = rays[idx];

    traverse_bvh(
        ray.origin.xyz,
        ray.direction.xyz,
        ray.origin.w,
        ray.direction.w,
        idx,
    );

    // Miss: add background × path throughput; mark ray terminated so later bounces skip it.
    if hit_records[idx].t >= F32_MAX {
        let tp = rays[idx].throughput;
        scratch_buf[idx] += BACKGROUND * vec4<f32>(tp[0], tp[1], tp[2], 1.0);
        rays[idx].direction.w = -1.0;
    }
}
