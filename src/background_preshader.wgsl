// background_preshader.wgsl — frame-0 pre-loop background supersampling pass.
// Fires BACKGROUND_SAMPLES jittered rays per sky pixel and writes
// the averaged background color into scratch_buf.
// Runs once on frame 0, after sky_mask_init and clear, before
// ray_gen. Owns supersampled initialization of frozen sky pixels.
// common_common.wgsl is prepended at pipeline creation —
// halton2, background_color, FrameUniform, and all constants
// are in scope.

// Number of Halton samples per sky pixel. Frame-0 only — cost is
// invisible in steady-state. Increase for higher-quality background
// at no runtime cost.
const BACKGROUND_SAMPLES: u32 = 8u;
const BACKGROUND_FILTER_WIDTH: f32 = 2.4;

struct Camera {
    origin:     vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical:   vec4<f32>,
    dims:       vec2<u32>,
    _dims_pad:  vec2<u32>,
}

@group(0) @binding(0) var<uniform>             camera:      Camera;
@group(0) @binding(1) var<uniform>             frame_data:  FrameUniform;
@group(1) @binding(0) var<storage, read>       sky_mask:    array<u32>;
@group(1) @binding(1) var<storage, read_write> scratch_buf: array<vec4<f32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w  = camera.dims.x;
    let h  = camera.dims.y;
    if px >= w || py >= h { return; }

    let idx = py * w + px;
    if sky_mask[idx] != 0u { return; }

    var color_sum = vec3<f32>(0.0);
    for (var i = 0u; i < BACKGROUND_SAMPLES; i++) {
        let jitter = halton2(i) * BACKGROUND_FILTER_WIDTH;
        let u      =       (f32(px) + jitter.x) / f32(w);
        let v      = 1.0 - (f32(py) + jitter.y) / f32(h);
        let dir    = normalize(
            camera.lower_left.xyz
            + u * camera.horizontal.xyz
            + v * camera.vertical.xyz
            - camera.origin.xyz
        );
        color_sum += background_color(dir).rgb;
    }
    scratch_buf[idx] = vec4<f32>(color_sum / f32(BACKGROUND_SAMPLES), 1.0);
}
