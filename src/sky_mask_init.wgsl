// sky_mask_init.wgsl — frame-0 sky mask initialization with 2-pixel dilation.
// Runs once before the bounce loop on frame 0.
// Each pixel fires one primary ray. Hit pixels atomically push 1 to themselves
// and every pixel within a 2-pixel Chebyshev radius (5×5 neighborhood, bounds-clamped).
// Miss pixels are left as 0 — WebGPU zero-initializes storage buffers on creation.
// Composed with common_common.wgsl at pipeline creation (BvhNode, TlasInstance,
// Sphere, aabb_hit, sphere_hit, FrameUniform, and all constants are in scope).

// ── Camera — mirrors CameraUniform in gpu.rs ──────────────────────────────────
struct Camera {
    origin:    vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical:  vec4<f32>,
    dims:      vec2<u32>,
    _dims_pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform>       camera:         Camera;
@group(0) @binding(1) var<uniform>       frame_data:     FrameUniform;
@group(0) @binding(2) var<storage, read> bvh_nodes:      array<BvhNode>;
@group(0) @binding(3) var<storage, read> tlas_instances: array<TlasInstance>;
@group(0) @binding(4) var<storage, read> spheres:        array<Sphere>;

@group(1) @binding(0) var<storage, read_write> sky_mask: array<atomic<u32>>;

// ── Any-hit BVH traversal — returns true on first geometry intersection ───────
// Simplified vs intersect.wgsl: no HitRecord writes, no ray counter, early exit.
fn traverse_bvh_any_hit(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    let num_instances = arrayLength(&tlas_instances);
    for (var inst_idx = 0u; inst_idx < num_instances; inst_idx++) {
        let inst         = tlas_instances[inst_idx];
        let local_origin = (inst.transform * vec4<f32>(origin, 1.0)).xyz;
        let local_dir    = (inst.transform * vec4<f32>(dir,    0.0)).xyz;
        let local_inv    = 1.0 / local_dir;

        var stack:     array<u32, 32>;
        var stack_ptr: i32 = 0;
        stack[0]  = inst.blas_offset;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr];
            let node     = bvh_nodes[node_idx];

            if !aabb_hit(node, local_origin, local_inv, tmin, tmax) { continue; }

            if node.node_type == NODE_LEAF_SPHERE {
                let sidx = node_sphere_index(node);
                let t    = sphere_hit(spheres[sidx], local_origin, local_dir, tmin, tmax);
                if t > 0.0 { return true; }
            } else {
                let rc = node_right_child(node);
                let lc = node_left_child(node);
                if rc != INVALID_NODE && stack_ptr < 32 {
                    stack[stack_ptr] = rc;
                    stack_ptr += 1;
                }
                if lc != INVALID_NODE && stack_ptr < 32 {
                    stack[stack_ptr] = lc;
                    stack_ptr += 1;
                }
            }
        }
    }
    return false;
}

// ── Halton low-discrepancy sequence for sub-pixel jitter ─────────────────────
// Matches ray_gen.wgsl so the frame-0 ray direction is identical to ray_gen's.
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
    return vec2<f32>(halton(frame + 1u, 2u), halton(frame + 1u, 3u));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let w  = camera.dims.x;
    let h  = camera.dims.y;
    if px >= w || py >= h { return; }

    let jitter = halton2(frame_data.frame);
    let u =       (f32(px) + jitter.x) / f32(w);
    let v = 1.0 - (f32(py) + jitter.y) / f32(h);

    let img_plane_pos = camera.lower_left.xyz
                     + u * camera.horizontal.xyz
                     + v * camera.vertical.xyz;
    let dir = normalize(img_plane_pos - camera.origin.xyz);

    if traverse_bvh_any_hit(camera.origin.xyz, dir, 1e-4, 1e30) {
        // Push 1 to this pixel and all pixels within 2-pixel Chebyshev radius.
        let ix = i32(px);
        let iy = i32(py);
        let iw = i32(w);
        let ih = i32(h);
        for (var dy = -2; dy <= 2; dy++) {
            for (var dx = -2; dx <= 2; dx++) {
                let nx = ix + dx;
                let ny = iy + dy;
                if nx >= 0 && nx < iw && ny >= 0 && ny < ih {
                    atomicOr(&sky_mask[u32(ny) * w + u32(nx)], 1u);
                }
            }
        }
    }
    // Miss: no-op — buffer is WebGPU zero-initialized on creation.
}
