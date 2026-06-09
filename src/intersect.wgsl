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
// B07a — declared, not yet used
@group(0) @binding(7) var<uniform>       frame_data     : FrameUniform;

// group(1) = per-pass resources (rays declared in variant; scratch_buf removed — background_shader owns it on frame 0)
@group(1) @binding(2) var<storage, read_write>  hit_records : array<HitRecord>;
@group(1) @binding(3) var<storage, read_write>  ray_counter : atomic<u32>;

// ── Entry point (called by intersect_variant_*.wgsl) ─────────────────────────
fn intersect_main(idx: u32) {
    // Terminated-ray early exit (sentinel written on miss or by roulette_pass).
    if rays[idx].direction.w < 0.0 {
        hit_records[idx] = HitRecord(F32_MAX, 0u, vec2<f32>(0.0), 0u, 0u, 0u, 0u);
        return;
    }

    atomicAdd(&ray_counter, 1u);

    let ray = rays[idx];

    traverse_bvh(
        ray.origin.xyz,
        ray.direction.xyz,
        ray.origin.w,
        ray.direction.w,
        idx,
    );

    // Miss: mark ray terminated so later bounces skip it.
    // Background color is written by background_shader on frame 0; subsequent frames
    // use the persisted accum_buf value for sky pixels.
//   if hit_records[idx].t >= F32_MAX {
//       rays[idx].direction.w = -1.0;
//    }
}
