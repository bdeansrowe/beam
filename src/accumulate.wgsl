// accumulate.wgsl — pure additive accumulator. Adds scratch_buf into pixel_buf.accum.
// pixel_buf is never cleared during rendering — only zero-initialized at startup.
// common_common.wgsl is prepended at pipeline creation — FrameUniform is in scope.

@group(0) @binding(0) var<uniform>             frame_data:  FrameUniform;
@group(1) @binding(0) var<storage, read>       scratch_buf: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> pixel_buf:   array<PixelState>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    let idx = gid.y * frame_data.dim_x + gid.x;
    let s = scratch_buf[idx];
    pixel_buf[idx].accum += s;
    pixel_buf[idx].sq    += s * s;
}
