// Mirrors CameraUniform in gpu.rs — layout must stay in sync.
struct Camera {
    origin:     vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical:   vec4<f32>,
    dims:      vec2<u32>,   // .x=width .y=height
    _dims_pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform>             camera     : Camera;
@group(0) @binding(1) var<uniform>             frame_data : FrameUniform;
@group(1) @binding(0) var<storage, read_write> rays       : array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w  = camera.dims.x;
    let h  = camera.dims.y;
    if px >= w || py >= h { return; }

    let jitter = halton2(frame_data.frame);
    let u =       (f32(px) + jitter.x) / f32(w);
    let v = 1.0 - (f32(py) + jitter.y) / f32(h);  // flip: screen-top = world-up

    let img_plane_pos = camera.lower_left.xyz
                     + u * camera.horizontal.xyz
                     + v * camera.vertical.xyz;
    let dir = normalize(img_plane_pos - camera.origin.xyz);

    let idx = py * w + px;
    var r: Ray;
    r.origin             = vec4<f32>(camera.origin.xyz, 1e-4);
    r.direction          = vec4<f32>(dir, 1e30);
    r.medium_stack[0]    = MediumEntry(0u, 1.0);  // air
    r.medium_depth       = 1u;
    rays[idx] = r;
}

// Radical-inverse Halton sequence for sub-pixel jitter.
fn halton(i: u32, base: u32) -> f32 {
    var f = 1.0;
    var r = 0.0;
    var n = i;
    loop {
        f /= f32(base);
        r += f * f32(n % base);
        n /= base;
        if n == 0u { break; }
    }
    return r;
}

fn halton2(frame: u32) -> vec2<f32> {
    // +1 so frame 0 yields (0.5, 0.33…) rather than (0, 0)
    return vec2<f32>(halton(frame + 1u, 2u), halton(frame + 1u, 3u));
}
