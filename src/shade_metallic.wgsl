// shade_metallic.wgsl — perfect mirror shading kernel.
// Composed with shade_common.wgsl at pipeline creation: shade_common is prepended,
// so all structs, bindings, and utility functions are already in scope here.

@group(1) @binding(2) var<storage, read> rays: array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(accum_buf);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx = py * dims.x + px;
    let hit = hit_records[idx];

    // Skip misses and non-metallic hits.
    if hit.t >= F32_MAX { return; }
    let mat_id = select(spheres[hit.prim_idx].back_material_id,
                        spheres[hit.prim_idx].front_material_id,
                        hit.face_forward == 1u);
    let mat = materials[mat_id];
    if mat.material_type != MAT_METALLIC { return; }

    let ray    = rays[idx];
    let normal = interpolate_normal(hit, ray);

    // reflect(ray.direction.xyz, normal) will drive HDR env lookup once env map exists.
    // Stand-in: uniform environment color tinted by base_color.
    let color = mat.base_color.rgb * BACKGROUND.rgb;

    textureStore(accum_buf, vec2<i32>(i32(px), i32(py)), vec4<f32>(color, 1.0));
}
