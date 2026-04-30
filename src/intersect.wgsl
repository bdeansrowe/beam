struct Ray {
    origin:    vec4<f32>,  // .w = tmin
    direction: vec4<f32>,  // .w = tmax
}

const SPHERE_CENTER : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const SPHERE_RADIUS : f32       = 0.5;
const BACKGROUND    : vec4<f32> = vec4<f32>(0.05, 0.05, 0.1, 1.0);

// group(0) reserved for scene-global resources (BVH, geometry) — empty until Step 5
@group(1) @binding(0) var<storage, read> rays    : array<Ray>;
@group(1) @binding(1) var                hdr_out : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx = py * dims.x + px;
    let color = shade(rays[idx].origin.xyz, rays[idx].direction.xyz);
    textureStore(hdr_out, vec2<i32>(i32(px), i32(py)), color);
}

fn shade(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    // Quadratic: |origin + t*dir - center|^2 = r^2, solved with half-b form
    let oc = origin - SPHERE_CENTER;
    let a  = dot(dir, dir);
    let h  = dot(oc, dir);   // b/2
    let c  = dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS;
    let discriminant = h * h - a * c;

    if discriminant < 0.0 { return BACKGROUND; }

    let t = (-h - sqrt(discriminant)) / a;
    if t < 1e-4 { return BACKGROUND; }   // behind camera

    let hit = origin + t * dir;
    let n   = normalize(hit - SPHERE_CENTER);
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);  // normal → 0..1 RGB
}
