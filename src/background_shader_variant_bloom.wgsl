// background_shader_variant_bloom.wgsl — escaped-ray background pass for the bloom bounce loop.
// Mirrors background_shader.wgsl but dispatches over slot-linear ray indices.
// Runs after Bloom Intersect each bounce, before Bloom Shade Diffuse.
// common_common.wgsl is prepended at pipeline creation.

@group(0) @binding(0) var<uniform> frame_data: FrameUniform;

@group(1) @binding(0) var<storage, read>       bloom_hit_records: array<HitRecord>;
@group(1) @binding(1) var<storage, read_write> bloom_scratch_buf: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> rays:              array<Ray>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= BLOOM_SLOT_CAPACITY * BLOOM_AMPLIFICATION { return; }

    if bloom_hit_records[idx].t < F32_MAX { return; }  // hit geometry
    if rays[idx].direction.w < 0.0        { return; }  // already terminated

    let tp = rays[idx].throughput;
    bloom_scratch_buf[idx] += background_color(rays[idx].direction.xyz)
                            * vec4<f32>(tp[0], tp[1], tp[2], 1.0);
    rays[idx].direction.w = -1.0;
}
