// variance_pass.wgsl — computes per-pixel variance from sq and accum fields.
// Dispatched once per frame after accumulate, before resolve.
// common_common.wgsl prepended at pipeline creation.

@group(0) @binding(0) var<uniform>             frame_data: FrameUniform;
@group(1) @binding(0) var<storage, read_write> pixel_buf:  array<PixelState>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    let idx = gid.y * frame_data.dim_x + gid.x;

    let n = f32(frame_data.frame + 1u);
    if n < 2.0 { return; }  // need at least 2 samples for meaningful variance

    let mean    = pixel_buf[idx].accum.rgb / n;
    let mean_sq = pixel_buf[idx].sq.rgb    / n;
    let var_rgb = max(mean_sq - mean * mean, vec3<f32>(0.0));

    // Scalar variance: perceptual luminance weighting
    pixel_buf[idx].variance = dot(var_rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}
