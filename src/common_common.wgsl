// common_common.wgsl — shared type declarations and constants for all pipelines.
// Prepended to every pipeline source at pipeline creation time. No entry point.

struct MediumEntry {
    material_id: u32,
    ior:         f32,
}  // 8 bytes

// 80 bytes total. throughput uses array<f32,3> (align 4) not vec3<f32> (align 16)
// so that medium_depth at offset 64 + 4 bytes + throughput 12 bytes = 80 exactly.
struct Ray {
    origin:       vec4<f32>,             // .w = tmin
    direction:    vec4<f32>,             // .w = tmax
    medium_stack: array<MediumEntry, 4>, // 32 bytes — entries 0..medium_depth-1 active
    medium_depth: u32,                   // 1 = air only
    throughput:   array<f32, 3>,         // path throughput RGB; (1,1,1) at ray birth
}

struct HitRecord {
    t:            f32,
    prim_idx:     u32,
    bary_uv:      vec2<f32>,
    face_forward: u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
}  // 32 bytes

struct Material {
    base_color:    vec4<f32>,
    emission:      vec4<f32>,
    absorption:    vec4<f32>,
    material_type: u32,
    ior:           f32,
    roughness:     f32,
    _pad:          f32,
}  // 64 bytes

struct Vertex {
    position: vec4<f32>,
    normal:   vec4<f32>,
}  // 32 bytes

struct TriangleRecord {
    v0:                u32,
    v1:                u32,
    v2:                u32,
    front_material_id: u32,
    back_material_id:  u32,
    _pad0:             u32,
    _pad1:             u32,
    _pad2:             u32,
}  // 32 bytes

struct Sphere {
    center_radius:     vec4<f32>,
    front_material_id: u32,
    back_material_id:  u32,
    _pad:              vec2<u32>,
}  // 32 bytes

// ── BVH types — must mirror Rust structs in bvh.rs exactly ───────────────────
struct BvhNode {
    aabb_min_left_start:  vec4<f32>,  // .xyz=aabb_min  .w=left_child|prim_start|sphere_index (bits)
    aabb_max_right_count: vec4<f32>,  // .xyz=aabb_max  .w=right_child|prim_count|unused (bits)
    node_type:            u32,
    _r0:                  u32,
    _r1:                  u32,
    _r2:                  u32,
}  // 48 bytes

struct TlasInstance {
    transform:   mat4x4<f32>,  // 64 bytes, column-major
    blas_offset: u32,
    flags:       u32,
    _r0:         u32,
    _r1:         u32,
}  // 80 bytes

// ── Node type constants ───────────────────────────────────────────────────────
const NODE_INTERNAL:      u32 = 0u;
const NODE_LEAF_TRIANGLE: u32 = 1u;
const NODE_LEAF_SPHERE:   u32 = 2u;
const NODE_LEAF_QUARTIC:  u32 = 3u;
const INVALID_NODE:       u32 = 0xFFFFFFFFu;

// ── Named .w-field accessors ──────────────────────────────────────────────────
fn node_left_child(node:   BvhNode) -> u32 { return bitcast<u32>(node.aabb_min_left_start.w); }
fn node_prim_start(node:   BvhNode) -> u32 { return bitcast<u32>(node.aabb_min_left_start.w); }
fn node_sphere_index(node: BvhNode) -> u32 { return bitcast<u32>(node.aabb_min_left_start.w); }
fn node_right_child(node:  BvhNode) -> u32 { return bitcast<u32>(node.aabb_max_right_count.w); }
fn node_prim_count(node:   BvhNode) -> u32 { return bitcast<u32>(node.aabb_max_right_count.w); }

// ── AABB slab test ────────────────────────────────────────────────────────────
fn aabb_hit(node: BvhNode, origin: vec3<f32>, inv_dir: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    let t0 = (node.aabb_min_left_start.xyz  - origin) * inv_dir;
    let t1 = (node.aabb_max_right_count.xyz - origin) * inv_dir;
    let tenter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let texit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return tenter <= texit && texit >= tmin && tenter <= tmax;
}

// ── Analytic sphere intersection (quadratic, half-b form) ─────────────────────
// Returns near hit t >= tmin, or -1.0 on miss.
fn sphere_hit(sph: Sphere, origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> f32 {
    let oc = origin - sph.center_radius.xyz;
    let a  = dot(dir, dir);
    let h  = dot(oc, dir);
    let c  = dot(oc, oc) - sph.center_radius.w * sph.center_radius.w;
    let discriminant = h * h - a * c;
    if discriminant < 0.0 { return -1.0; }
    let sq = sqrt(discriminant);
    let t1 = (-h - sq) / a;
    if t1 >= tmin && t1 <= tmax { return t1; }
    let t2 = (-h + sq) / a;
    if t2 >= tmin && t2 <= tmax { return t2; }
    return -1.0;
}

struct LightUniform {
    position:  vec4<f32>,
    color:     vec4<f32>,
    intensity: f32,
    _pad:      vec3<f32>,
}  // 48 bytes

struct FrameUniform {
    frame:  u32,
    dim_x:  u32,
    dim_y:  u32,
    bounce: u32,
}  // 16 bytes

const F32_MAX:      f32       = bitcast<f32>(0x7f7fffffu);
const PI:           f32       = 3.14159265358979323846;
const MAT_DIFFUSE:  u32       = 0u;
const MAT_METALLIC: u32       = 1u;
const MAT_GLASS:    u32       = 2u;
const MAT_EMISSIVE: u32       = 3u;
const BACKGROUND:   vec4<f32> = vec4<f32>(0.45, 0.42, 0.38, 1.0);
