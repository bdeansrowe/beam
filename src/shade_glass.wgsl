// shade_glass.wgsl — glass BSDF: Schlick Fresnel, Snell refraction, medium stack, Beer's law.
// Composed with shade_common.wgsl at pipeline creation: shade_common is prepended,
// so all structs, bindings, and utility functions are already in scope here.

@group(1) @binding(2) var<storage, read_write> rays: array<Ray>;

// ── Schlick Fresnel approximation ─────────────────────────────────────────────
// r0 computed via multiplication to avoid pow() with a potentially negative base.
fn schlick(cos_theta: f32, n1: f32, n2: f32) -> f32 {
    let t  = (n1 - n2) / (n1 + n2);
    let r0 = t * t;
    return r0 + (1.0 - r0) * pow(1.0 - cos_theta, 5.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx = py * dims.x + px;
    let hit = hit_records[idx];

    // Skip misses and non-glass hits.
    if hit.t >= F32_MAX { return; }
    let mat_id = select(spheres[hit.prim_idx].back_material_id,
                        spheres[hit.prim_idx].front_material_id,
                        hit.face_forward == 1u);
    let mat = materials[mat_id];
    if mat.material_type != MAT_GLASS { return; }

    var ray = rays[idx];

    // ── D2 — Medium stack underflow guard ─────────────────────────────────────
    // Back-face hit requires depth >= 2: depth-1 is current medium (glass),
    // depth-2 is the medium being re-entered (air or outer dielectric).
    if hit.face_forward == 0u && ray.medium_depth < 2u {
        textureStore(hdr_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(1.0, 0.0, 1.0, 1.0));
        return;
    }

    // ── n1 / n2 ───────────────────────────────────────────────────────────────
    let n1 = ray.medium_stack[ray.medium_depth - 1u].ior;
    var n2: f32;
    if hit.face_forward == 1u {
        n2 = mat.ior;                                         // entering: material IOR
    } else {
        n2 = ray.medium_stack[ray.medium_depth - 2u].ior;    // exiting: medium below
    }

    // ── Surface normal (face-forward by interpolate_normal convention) ────────
    let normal    = interpolate_normal(hit, ray);
    let cos_theta = clamp(-dot(ray.direction.xyz, normal), 0.0, 1.0);

    // ── D4 — TIR check ────────────────────────────────────────────────────────
    let eta    = n1 / n2;
    let sin2_t = eta * eta * (1.0 - cos_theta * cos_theta);

    var out_dir:    vec3<f32>;
    var is_reflect: bool;

    if sin2_t > 1.0 {
        // Total internal reflection — hard reflect, no roulette, no stack change.
        out_dir    = reflect(ray.direction.xyz, normal);
        is_reflect = true;
    } else {
        // ── D3 — Schlick Fresnel + Russian roulette ────────────────────────────
        let F    = schlick(cos_theta, n1, n2);
        let rand = f32(hash_u32(idx * 0x9e3779b9u + 1u) & 0x00ffffffu) / f32(0x01000000u);
        if rand < F {
            out_dir    = reflect(ray.direction.xyz, normal);
            is_reflect = true;
        } else {
            // ── Snell refraction ───────────────────────────────────────────────
            let cos_t = sqrt(1.0 - sin2_t);
            out_dir   = normalize(eta * ray.direction.xyz + (eta * cos_theta - cos_t) * normal);
            is_reflect = false;
        }
    }

    // ── D2 — Push / pop medium stack (refracted paths only) ───────────────────
    if !is_reflect {
        if hit.face_forward == 1u {
            if ray.medium_depth < 4u {
                ray.medium_stack[ray.medium_depth] = MediumEntry(mat_id, mat.ior);
                ray.medium_depth += 1u;
            }
        } else {
            ray.medium_depth -= 1u;
        }
    }

    // ── D5 — Beer's law absorption (refracted path only) ──────────────────────
    var throughput = vec3<f32>(1.0);
    if !is_reflect {
        throughput = exp(-mat.absorption.rgb * hit.t);
    }
    let color = throughput * mat.base_color.rgb;

    // ── Write updated ray back for future multi-bounce ─────────────────────────
    let hit_pos       = hit_position(ray, hit.t);
    let offset_normal = select(-normal, normal, is_reflect);
    ray.origin        = vec4<f32>(offset_ray_origin(hit_pos, offset_normal), ray.origin.w);
    ray.direction     = vec4<f32>(out_dir, ray.direction.w);
    rays[idx]         = ray;

    // ── Single-bounce output: throughput × base_color × background ────────────
    // When multi-bounce arrives this write moves to the accumulation kernel.
    textureStore(hdr_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(color * BACKGROUND.rgb, 1.0));
}
