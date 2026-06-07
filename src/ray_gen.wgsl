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
@group(1) @binding(1) var<storage, read>       sky_mask   : array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w  = camera.dims.x;
    let h  = camera.dims.y;
    if px >= w || py >= h { return; }

    let idx = py * w + px;
    if sky_mask[idx] == 0u { return; }

    let jitter = halton2(frame_data.frame);
    let u =       (f32(px) + jitter.x) / f32(w);
    let v = 1.0 - (f32(py) + jitter.y) / f32(h);  // flip: screen-top = world-up

    let img_plane_pos = camera.lower_left.xyz
                     + u * camera.horizontal.xyz
                     + v * camera.vertical.xyz;
    let dir = normalize(img_plane_pos - camera.origin.xyz);

    var r: Ray;
    r.origin             = vec4<f32>(camera.origin.xyz, 1e-4);
    r.direction          = vec4<f32>(dir, 1e30);
    r.medium_stack[0]    = MediumEntry(0u, 1.0);  // air
    r.medium_depth       = 1u;
    r.throughput         = array<f32, 3>(1.0, 1.0, 1.0);
    rays[idx] = r;
}

