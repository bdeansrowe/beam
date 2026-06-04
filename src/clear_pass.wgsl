// clear_pass.wgsl — frame-start kernel: zeroes scratch_buf before each frame's bounce loop.
// Dispatched once per frame, before ray_gen.
// common_common.wgsl is prepended — FrameUniform (dim_x/dim_y) is in scope.

@group(0) @binding(0) var<uniform>             frame_data:  FrameUniform;
@group(1) @binding(0) var<storage, read_write> scratch_buf: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if px >= frame_data.dim_x || py >= frame_data.dim_y { return; }
    scratch_buf[py * frame_data.dim_x + px] = vec4<f32>(0.0);
}
