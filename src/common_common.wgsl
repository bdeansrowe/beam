// common_common.wgsl — shared type declarations and constants for all pipelines.
// Prepended to every pipeline source at pipeline creation time. No entry point.

struct MediumEntry {
    material_id: u32,
    ior:         f32,
}  // 8 bytes

// 80 bytes total. _pad uses array<u32,3> (align 4) not vec3<u32> (align 16)
// so that medium_depth at offset 64 + 4 bytes + 12 bytes pad = 80 exactly.
struct Ray {
    origin:       vec4<f32>,             // .w = tmin
    direction:    vec4<f32>,             // .w = tmax
    medium_stack: array<MediumEntry, 4>, // 32 bytes — entries 0..medium_depth-1 active
    medium_depth: u32,                   // 1 = air only
    _pad:         array<u32, 3>,
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

const F32_MAX:      f32       = bitcast<f32>(0x7f7fffffu);
const PI:           f32       = 3.14159265358979323846;
const MAT_DIFFUSE:  u32       = 0u;
const MAT_METALLIC: u32       = 1u;
const MAT_GLASS:    u32       = 2u;
const MAT_EMISSIVE: u32       = 3u;
const BACKGROUND:   vec4<f32> = vec4<f32>(0.05, 0.05, 0.1, 1.0);
