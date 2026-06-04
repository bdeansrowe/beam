// shade_common.wgsl — shared declarations and utilities for all shading kernels.
// NOT a standalone compute shader — no @compute entry point.
// Composed into each shade_<variant>.wgsl via string concatenation at pipeline creation.
// common_common.wgsl is prepended before this file — all shared structs and constants
// (Ray, HitRecord, Material, Sphere, Vertex, TriangleRecord, F32_MAX, PI, MAT_*,
// BACKGROUND) are already in scope.

// ── Shade-local constants ─────────────────────────────────────────────────────
const HASH_MUL_0: u32 = 0xbf324c81u;  // PCG-derived bit-mixing constant
const HASH_MUL_1: u32 = 0x68b31f7eu;  // PCG-derived bit-mixing constant

// ── BG0 — scene resources used by shading kernels ─────────────────────────────
// Bindings 0 (bvh_nodes) and 1 (tlas_instances) are intersect-only — not declared here.
// Step 7 creates a separate shade_scene_bg0 BindGroup covering only these four slots.
@group(0) @binding(2) var<storage, read> spheres  : array<Sphere>;
@group(0) @binding(3) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(4) var<storage, read> geometry : array<TriangleRecord>;
@group(0) @binding(5) var<storage, read> materials: array<Material>;
// Step 7 — declared, not yet used by diffuse/metallic/glass kernels
@group(0) @binding(6) var<storage, read> lights:    array<LightUniform>;
@group(0) @binding(7) var<uniform>       frame_data: FrameUniform;

// ── BG1 — per-pass resources ──────────────────────────────────────────────────
@group(1) @binding(0) var<storage, read>       hit_records: array<HitRecord>;
@group(1) @binding(1) var<storage, read_write> scratch_buf: array<vec4<f32>>;
// rays: @group(1) @binding(2) — declared per-shader (read or read_write depending on material)

// ── hit_position ──────────────────────────────────────────────────────────────
fn hit_position(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin.xyz + t * ray.direction.xyz;
}

// ── interpolate_normal ────────────────────────────────────────────────────────
// Step 6b: sphere-only. Triangle barycentric interpolation added when mesh geometry arrives.
fn interpolate_normal(hit: HitRecord, ray: Ray) -> vec3<f32> {
    let pos   = hit_position(ray, hit.t);
    let out_n = normalize(pos - spheres[hit.prim_idx].center_radius.xyz);
    return select(-out_n, out_n, hit.face_forward == 1u);
}

// ── offset_ray_origin ─────────────────────────────────────────────────────────
fn offset_ray_origin(pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return pos + normal * 1e-4;
}

// ── cosine_weighted_hemisphere ────────────────────────────────────────────────
fn hash_u32(x: u32) -> u32 {
    var v  = x;
    v     ^= v >> 17u;
    v     *= HASH_MUL_0;
    v     ^= v >> 11u;
    v     *= HASH_MUL_1;
    v     ^= v >> 15u;
    return v;
}

fn cosine_weighted_hemisphere(normal: vec3<f32>, seed: u32) -> vec3<f32> {
    let r1  = f32(hash_u32(seed)      & 0x00ffffffu) / f32(0x01000000u);
    let r2  = f32(hash_u32(seed + 1u) & 0x00ffffffu) / f32(0x01000000u);
    let r   = sqrt(r1);
    let phi = 2.0 * PI * r2;
    let x   = r * cos(phi);
    let y   = r * sin(phi);
    let z   = sqrt(max(0.0, 1.0 - r1));
    let up  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.y) < 0.999);
    let tan = normalize(cross(up, normal));
    let bit = cross(normal, tan);
    return normalize(x * tan + y * bit + z * normal);
}
