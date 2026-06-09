// shade_diffuse.wgsl — Lambertian shading kernel.
// Writes cosine-weighted hemisphere continuation ray; multiplies path throughput by base_color.
// NEE (shade_direct) owns direct lighting — no scratch_buf write here.
// Composed with shade_common.wgsl at pipeline creation.

// rays declared in shade_diffuse_variant_*.wgsl

fn diffuse_main(idx: u32) {
    let hit = hit_records[idx];

    if hit.t >= F32_MAX { return; }
    let mat_id = select(spheres[hit.prim_idx].back_material_id,
                        spheres[hit.prim_idx].front_material_id,
                        hit.face_forward == 1u);
    let mat = materials[mat_id];
    if mat.material_type != MAT_DIFFUSE { return; }

    var ray     = rays[idx];
    let normal  = interpolate_normal(hit, ray);
    let hit_pos = hit_position(ray, hit.t);

    var seed    = rays[idx].seed;
    let new_dir = cosine_weighted_hemisphere(normal, seed);

    ray.origin        = vec4<f32>(offset_ray_origin(hit_pos, normal), ray.origin.w);
    ray.direction     = vec4<f32>(new_dir, ray.direction.w);
    ray.throughput[0] *= mat.base_color.r;
    ray.throughput[1] *= mat.base_color.g;
    ray.throughput[2] *= mat.base_color.b;
    ray.seed  = pcg_hash(seed ^ (frame_data.bounce * FIBONACCI_HASH));
    rays[idx] = ray;
}
