// resolve.wgsl — divides accum_buf by frame count and writes display_tex.
// Dispatched after accumulate, before blit. No filtering — pure average.
// common_common.wgsl is prepended at pipeline creation — FrameUniform is in scope.

@group(0) @binding(0) var<uniform>       frame_data:  FrameUniform;
@group(1) @binding(0) var<storage, read> accum_buf:   array<vec4<f32>>;
@group(1) @binding(1) var                display_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    let idx    = gid.y * frame_data.dim_x + gid.x;
    let coord  = vec2<i32>(i32(gid.x), i32(gid.y));
    let weight = 1.0 / f32(frame_data.frame + 1u);
    textureStore(display_tex, coord, accum_buf[idx] * weight);
}
