// Mirrors CameraUniform in gpu.rs — layout must stay in sync.
struct Camera {
    origin:     vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical:   vec4<f32>,
    dims:       vec2<u32>,
    _dims_pad:  vec2<u32>,
}

@group(0) @binding(0) var<uniform>             camera         : Camera;
@group(0) @binding(1) var<uniform>             frame_data     : FrameUniform;
@group(1) @binding(0) var<storage, read>       pixel_buf      : array<PixelState>;
@group(1) @binding(1) var<storage, read_write> bloom_slot_buf : array<Ray>;

const BLOOM_FILTER_WIDTH: f32 = 1.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w  = camera.dims.x;
    let h  = camera.dims.y;
    if px >= w || py >= h { return; }

    let idx = py * w + px;
    if pixel_buf[idx].bloom_slot < 0 { return; }

    let bloom_slot = u32(pixel_buf[idx].bloom_slot);

    for (var sample_i: u32 = 0u; sample_i < 256u; sample_i = sample_i + 1u) {
        // Sub-pixel jitter via Halton. frame * 256 + sample_i extends ray_gen's
        // halton2(frame) seed so each of the 256 rays samples a distinct sub-pixel location.
        let halton_idx = frame_data.frame * 256u + sample_i + 1u;
        let jitter = vec2<f32>(halton(halton_idx, 2u), halton(halton_idx, 3u))
                   * BLOOM_FILTER_WIDTH;

        let u =       (f32(px) + jitter.x) / f32(w);
        let v = 1.0 - (f32(py) + jitter.y) / f32(h);

        let img_plane_pos = camera.lower_left.xyz
                         + u * camera.horizontal.xyz
                         + v * camera.vertical.xyz;
        let dir = normalize(img_plane_pos - camera.origin.xyz);

        // Per-ray RNG seed combining pixel index, frame, and sample_i so each ray
        // explores an independent path through the bloom bounce pipeline.
        let ray_seed = pcg_hash(pcg_hash(idx ^ frame_data.frame) ^ sample_i);

        var r: Ray;
        r.origin       = vec4<f32>(camera.origin.xyz, 1e-4);
        r.direction    = vec4<f32>(dir, 1e30);
        r.medium_depth = 0u;
        r.throughput   = array<f32, 3>(1.0, 1.0, 1.0);
        r.seed         = ray_seed;
        // medium_stack zero-initialised by var declaration; medium_depth=0 means none active.

        bloom_slot_buf[bloom_slot * 256u + sample_i] = r;
    }
}
