// shade_diffuse.wgsl — Lambertian shading kernel.
// Composed with shade_common.wgsl at pipeline creation: shade_common is prepended,
// so all structs, bindings, and utility functions are already in scope here.

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx = py * dims.x + px;
    let hit = hit_records[idx];

    // Skip misses and non-diffuse hits.
    if hit.t >= F32_MAX { return; }
    // Step 6b — sphere material hardcoded to index 0; see B04 parley
    let mat = materials[0u];
    if mat.material_type != MAT_DIFFUSE { return; }

    let ray    = rays[idx];
    let normal = interpolate_normal(hit, ray);

    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let ndotl     = max(0.0, dot(normal, light_dir));
    let color     = mat.base_color.rgb * ndotl;

    textureStore(hdr_out, vec2<i32>(i32(px), i32(py)), vec4<f32>(color, 1.0));
}
