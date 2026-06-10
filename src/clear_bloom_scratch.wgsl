// clear_bloom_scratch.wgsl — frame-start kernel: zeroes bloom_scratch_buf
// before each frame's bloom bounce loop.
// Dispatched once per frame in the pre-loop encoder, after clear_pass.
// common_common.wgsl is prepended at pipeline creation.

@group(0) @binding(0) var<storage, read_write> bloom_scratch_buf: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= BLOOM_SLOT_CAPACITY * BLOOM_AMPLIFICATION { return; }
    bloom_scratch_buf[idx] = vec4<f32>(0.0);
}
