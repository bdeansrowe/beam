// accumulate.wgsl — blends scratch_tex (new sample) into ping-pong accum history.
// Dispatched last in the frame, after all shading kernels.
// common_common.wgsl is prepended at pipeline creation — FrameUniform is in scope.
//
// BG1 uses Texture bindings (not storage ReadOnly) for scratch and history to avoid
// requiring the "read-write-and-read-only-storage-textures" WebGPU feature.

@group(0) @binding(0) var<uniform> frame_data : FrameUniform;
@group(1) @binding(0) var<storage, read> scratch_buf: array<vec4<f32>>;
@group(1) @binding(1) var prev_accum : texture_2d<f32>;
@group(1) @binding(2) var curr_accum : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(curr_accum);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx        = py * dims.x + px;
    let coord      = vec2<i32>(i32(px), i32(py));
    let new_sample = scratch_buf[idx];
    let history    = textureLoad(prev_accum,  coord, 0);
    let weight     = 1.0 / f32(frame_data.frame + 1u);
    let blended    = mix(history, new_sample, weight);
    textureStore(curr_accum, coord, blended);
}
