// background_shader.wgsl — per-bounce escaped-ray background pass.
// Runs in the bounce loop after each intersect dispatch, every frame.
// For geometry pixels whose rays missed all geometry (escaped rays),
// credits background_color * throughput into scratch_buf and marks
// the ray terminated.
// Keeps all background-color evaluation out of intersect.
// common_common.wgsl is prepended at pipeline creation —
// background_color, FrameUniform, and all constants are in scope.

@group(0) @binding(0) var<uniform> frame_data: FrameUniform;

@group(1) @binding(0) var<storage, read>       hit_records: array<HitRecord>;
@group(1) @binding(1) var<storage, read_write> scratch_buf: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> rays:        array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    let idx = gid.y * frame_data.dim_x + gid.x;

    // Only process escaped rays: missed geometry, not yet terminated.
    if hit_records[idx].t < F32_MAX  { return; }  // hit geometry
    if rays[idx].direction.w < 0.0   { return; }  // already terminated from prior bounce
    let tp = rays[idx].throughput;
    scratch_buf[idx] += background_color(rays[idx].direction.xyz)
                      * vec4<f32>(tp[0], tp[1], tp[2], 1.0);
    rays[idx].direction.w = -1.0;
}
