// shade_metallic.wgsl — perfect mirror shading kernel.
// Writes specular reflection continuation ray; multiplies path throughput by base_color.
// Composed with shade_common.wgsl at pipeline creation.

@group(1) @binding(2) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= frame_data.dim_x || py >= frame_data.dim_y { return; }

    let idx = py * frame_data.dim_x + px;
    let hit = hit_records[idx];

    if hit.t >= F32_MAX { return; }
    let mat_id = select(spheres[hit.prim_idx].back_material_id,
                        spheres[hit.prim_idx].front_material_id,
                        hit.face_forward == 1u);
    let mat = materials[mat_id];
    if mat.material_type != MAT_METALLIC { return; }

    var ray      = rays[idx];
    let normal   = interpolate_normal(hit, ray);
    let hit_pos  = hit_position(ray, hit.t);
    let refl_dir = reflect(ray.direction.xyz, normal);

    ray.origin        = vec4<f32>(offset_ray_origin(hit_pos, normal), ray.origin.w);
    ray.direction     = vec4<f32>(refl_dir, ray.direction.w);
    ray.throughput[0] *= mat.base_color.r;
    ray.throughput[1] *= mat.base_color.g;
    ray.throughput[2] *= mat.base_color.b;
    rays[idx] = ray;
}
