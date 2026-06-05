// shade_direct.wgsl — direct lighting (NEE) kernel.
// Fires a shadow ray toward the point light and writes direct illumination to scratch_buf.
// Composed with shade_common.wgsl at pipeline creation:
//   common_common.wgsl (BvhNode, TlasInstance, Sphere, Material, LightUniform, helpers) +
//   shade_common.wgsl  (BG0 bindings 2-6, BG1 hit_records + scratch_buf, utilities) +
//   shade_direct.wgsl  (BG0 bindings 0-1, BG1 rays, entry point)

// BG0 bindings 0 and 1 — needed for shadow ray BVH traversal.
// Bindings 2-6 are declared by shade_common.wgsl (prepended).
@group(0) @binding(0) var<storage, read> bvh_nodes:     array<BvhNode>;
@group(0) @binding(1) var<storage, read> tlas_instances: array<TlasInstance>;

@group(1) @binding(2) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= frame_data.dim_x || py >= frame_data.dim_y { return; }

    let idx = py * frame_data.dim_x + px;
    let hit = hit_records[idx];

    // Skip background — already written by intersect kernel.
    if hit.t >= F32_MAX { return; }

    let ray     = rays[idx];
    let mat_id  = select(spheres[hit.prim_idx].back_material_id,
                         spheres[hit.prim_idx].front_material_id,
                         hit.face_forward == 1u);
    let mat     = materials[mat_id];

    // Glass transmits rather than scatters — NEE at a glass surface is physically wrong.
    if mat.material_type == MAT_GLASS { return; }

    let hit_pos = hit_position(ray, hit.t);
    let normal  = interpolate_normal(hit, ray);

    // ── Shadow ray ────────────────────────────────────────────────────────────
    let light      = lights[0];
    let light_pos  = light.position.xyz;
    let to_light   = light_pos - hit_pos;
    let dist       = length(to_light);
    let shadow_dir = to_light / dist;
    let shadow_origin = offset_ray_origin(hit_pos, normal);
    let shadow_tmax   = dist - 1e-4;

    // ── Any-hit BVH traversal with glass transmittance ────────────────────────
    var transmittance  = vec3<f32>(1.0);
    var shadow_blocked = false;

    let num_instances = arrayLength(&tlas_instances);
    for (var inst_idx = 0u; inst_idx < num_instances; inst_idx++) {
        if shadow_blocked { break; }
        let inst         = tlas_instances[inst_idx];
        let local_origin = (inst.transform * vec4<f32>(shadow_origin, 1.0)).xyz;
        let local_dir    = (inst.transform * vec4<f32>(shadow_dir,    0.0)).xyz;
        let local_inv    = 1.0 / local_dir;

        var stack:     array<u32, 32>;
        var stack_ptr: i32 = 1;
        stack[0] = inst.blas_offset;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr];
            let node     = bvh_nodes[node_idx];

            if !aabb_hit(node, local_origin, local_inv, 1e-4, shadow_tmax) { continue; }

            if node.node_type == NODE_LEAF_SPHERE {
                let sidx = node_sphere_index(node);
                let sph  = spheres[sidx];

                // Compute both sphere intersections.
                let oc   = local_origin - sph.center_radius.xyz;
                let a    = dot(local_dir, local_dir);
                let h    = dot(oc, local_dir);
                let c    = dot(oc, oc) - sph.center_radius.w * sph.center_radius.w;
                let disc = h * h - a * c;
                if disc < 0.0 { continue; }
                let sq     = sqrt(disc);
                let t_near = (-h - sq) / a;
                let t_far  = (-h + sq) / a;

                // Sphere must overlap segment [1e-4, shadow_tmax].
                if t_far < 1e-4 || t_near > shadow_tmax { continue; }

                // Material from which face the shadow ray enters.
                let s_mat_id = select(sph.back_material_id,
                                      sph.front_material_id,
                                      t_near >= 1e-4);
                let s_mat = materials[s_mat_id];

                if s_mat.material_type == MAT_GLASS {
                    let path_len = min(t_far, shadow_tmax) - max(t_near, 1e-4);
                    transmittance *= exp(-s_mat.absorption.rgb * max(0.0, path_len));
                    if transmittance.x < 0.001 && transmittance.y < 0.001 && transmittance.z < 0.001 {
                        shadow_blocked = true;
                        break;
                    }
                } else {
                    // Opaque surface: fully blocked.
                    shadow_blocked = true;
                    break;
                }
            } else {
                // NODE_INTERNAL: push children (right first so left is popped first).
                let rc = node_right_child(node);
                let lc = node_left_child(node);
                if rc != INVALID_NODE && stack_ptr < 32 { stack[stack_ptr] = rc; stack_ptr += 1; }
                if lc != INVALID_NODE && stack_ptr < 32 { stack[stack_ptr] = lc; stack_ptr += 1; }
            }
        }
    }

    // ── Direct illumination — scaled by path throughput, added to frame accumulator ──
    let tp      = ray.throughput;
    let n_dot_l = max(0.0, dot(normal, shadow_dir));
    let falloff = 1.0 / (dist * dist);
    let direct  = select(
        vec3<f32>(0.0),
        light.color.rgb * light.intensity * n_dot_l * falloff * transmittance,
        !shadow_blocked,
    ) * vec3<f32>(tp[0], tp[1], tp[2]);

    scratch_buf[idx] += vec4<f32>(direct, 0.0);
}
