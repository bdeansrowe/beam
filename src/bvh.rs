use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MediumEntry {
    pub material_id: u32,
    pub ior:         f32,
}  // 8 bytes

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Ray {
    pub origin:       [f32; 4],         // 16 bytes
    pub direction:    [f32; 4],         // 16 bytes
    pub medium_stack: [MediumEntry; 4], // 32 bytes
    pub medium_depth: u32,              // 4 bytes
    pub throughput:   [f32; 3],         // 12 bytes → 80 bytes total; path throughput (RGB)
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MaterialType {
    Diffuse  = 0,
    Metallic = 1,
    Glass    = 2,
    Emissive = 3,
}

// SAFETY: MaterialType is #[repr(u32)] with unit variants only.
// Memory layout is identical to u32 — no padding, no uninit bytes.
unsafe impl Zeroable for MaterialType {}
unsafe impl Pod     for MaterialType {}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Material {
    pub base_color:    [f32; 4],      // .rgb=color,     .w=unused
    pub emission:      [f32; 4],      // .rgb=emission,  .w=unused
    pub absorption:    [f32; 4],      // .rgb=Beer's law coefficient, .w=unused
    pub material_type: MaterialType,  // routes to shading kernel dispatch
    pub ior:           f32,           // index of refraction (1.0 for opaque)
    pub roughness:     f32,           // reserved; 0.0=smooth (reach goal)
    pub _pad:          f32,           // alignment padding
}
// 64 bytes total. Clean multiple of 16.
// WGSL struct must mirror exactly — use vec4<f32> for [f32;4] fields.

#[allow(dead_code)]
pub const NODE_INTERNAL:      u32 = 0;
#[allow(dead_code)]
pub const NODE_LEAF_TRIANGLE: u32 = 1;
pub const NODE_LEAF_SPHERE:   u32 = 2;
#[allow(dead_code)]
pub const NODE_LEAF_QUARTIC:  u32 = 3;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BvhNode {
    pub aabb_min_left_start:  [f32; 4],  // .xyz=aabb_min, .w=left_child|prim_start|sphere_index (bits)
    pub aabb_max_right_count: [f32; 4],  // .xyz=aabb_max, .w=right_child|prim_count|unused (bits)
    pub node_type:            u32,
    pub _reserved:            [u32; 3],
}

#[allow(dead_code)]
impl BvhNode {
    pub fn left_child(&self)   -> u32 { f32::to_bits(self.aabb_min_left_start[3]) }
    pub fn prim_start(&self)   -> u32 { f32::to_bits(self.aabb_min_left_start[3]) }
    pub fn sphere_index(&self) -> u32 { f32::to_bits(self.aabb_min_left_start[3]) }
    pub fn right_child(&self)  -> u32 { f32::to_bits(self.aabb_max_right_count[3]) }
    pub fn prim_count(&self)   -> u32 { f32::to_bits(self.aabb_max_right_count[3]) }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TlasInstance {
    // Column-major, matches WGSL mat4x4<f32>. Identity is the same in both orderings.
    pub transform:   [f32; 16],
    pub blas_offset: u32,
    pub flags:       u32,
    pub _reserved:   [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Sphere {
    pub center_radius:     [f32; 4],  // .xyz=center, .w=radius
    pub front_material_id: u32,       // CCW outward-normal face (ray entering sphere)
    pub back_material_id:  u32,       // CW inward-normal face  (ray exiting sphere)
    pub _pad:              [u32; 2],  // 32 bytes total
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],  // .xyz=position, .w=0.0
    pub normal:   [f32; 4],  // .xyz=normal,   .w=0.0
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct TriangleRecord {
    pub v0:                u32,
    pub v1:                u32,
    pub v2:                u32,
    pub front_material_id: u32,
    pub back_material_id:  u32,
    pub _pad:              [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HitRecord {
    pub t:            f32,      // hit distance; f32::MAX = miss
    pub prim_idx:     u32,      // sphere buffer index (Step 6b) — triangle index later
    pub bary_uv:      [f32; 2], // barycentric u, v; unused for sphere hits
    pub face_forward: u32,      // 1 = ray hit front face, 0 = back face
    pub _pad:         [u32; 3], // 32 bytes total
}
// 32 bytes. Clean multiple of 16. WGSL struct must mirror exactly.
// Miss sentinel: t == f32::MAX (bitcast<f32>(0x7f7fffffu)).

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LightUniform {
    pub position:  [f32; 4],  // .xyz=pos, .w=unused         — offset  0
    pub color:     [f32; 4],  // .rgb=color, .w=unused        — offset 16
    pub intensity: f32,       //                               — offset 32
    pub _pad:      [f32; 7],  // WGSL vec3<f32> has align 16 — offset 36..64
}
// 64 bytes. WGSL vec3<f32> aligns to 16, placing _pad at offset 48;
// Rust [f32;7] at offset 36 covers the implicit gap + the vec3 + tail pad.

pub fn build_trivial_scene() -> (Vec<BvhNode>, Vec<TlasInstance>, Vec<Sphere>) {
    let nodes = vec![
        // Node 0: BLAS leaf — glass sphere at (0,0.5,0), radius 0.5, sphere_index=0
        BvhNode {
            aabb_min_left_start:  [-0.5, 0.0, -0.5, f32::from_bits(0)],
            aabb_max_right_count: [ 0.5,  1.0,  0.5, f32::from_bits(0)],
            node_type: NODE_LEAF_SPHERE,
            _reserved: [0; 3],
        },
        // Node 1: BLAS leaf — diffuse sphere at (0.5,-0.5,0), radius 0.5, sphere_index=1
        BvhNode {
            aabb_min_left_start:  [0.1, -1.0, -0.5, f32::from_bits(1)],
            aabb_max_right_count: [ 1.1, 0.0,  0.5, f32::from_bits(0)],
            node_type: NODE_LEAF_SPHERE,
            _reserved: [0; 3],
        },
        // Node 2: BLAS leaf — metallic sphere at (-0.5,-0.5,0), radius 0.5, sphere_index=2
        BvhNode {
            aabb_min_left_start:  [-1.1, -1.0, -0.5, f32::from_bits(2)],
            aabb_max_right_count: [ -0.1, 0.0,  0.5, f32::from_bits(0)],
            node_type: NODE_LEAF_SPHERE,
            _reserved: [0; 3],
        },
        // Node 3: BLAS leaf — air bubble inside glass sphere at (0.2,0.4,0.2), radius 0.1, sphere_index=3
        BvhNode {
            aabb_min_left_start:  [0.1, 0.3, 0.1, f32::from_bits(3)],
            aabb_max_right_count: [0.3, 0.5, 0.3, f32::from_bits(0)],
            node_type: NODE_LEAF_SPHERE,
            _reserved: [0; 3],
        },
    ];

    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0_f32,
    ];

    let instances = vec![
        TlasInstance { transform: identity, blas_offset: 0, flags: 0, _reserved: [0; 2] },
        TlasInstance { transform: identity, blas_offset: 1, flags: 0, _reserved: [0; 2] },
        TlasInstance { transform: identity, blas_offset: 2, flags: 0, _reserved: [0; 2] },
        TlasInstance { transform: identity, blas_offset: 3, flags: 0, _reserved: [0; 2] },
    ];

    let spheres = vec![
        // Sphere 0: glass sphere
        Sphere {
            center_radius:     [0.0, 0.5, 0.0, 0.5],
            front_material_id: 1,
            back_material_id:  1,
            _pad:              [0; 2],
        },
        // Sphere 1: diffuse sphere
        Sphere {
            center_radius:     [0.6, -0.5, 0.0, 0.5],
            front_material_id: 2,
            back_material_id:  2,
            _pad:              [0; 2],
        },
        // Sphere 2: metallic sphere
        Sphere {
            center_radius:     [-0.6, -0.5, 0.0, 0.5],
            front_material_id: 3,
            back_material_id:  3,
            _pad:              [0; 2],
        },
        // Sphere 2: air bubble inside glass sphere
        Sphere {
            center_radius:     [0.2, 0.4, 0.2, 0.1],
            front_material_id: 4,
            back_material_id:  4,
            _pad:              [0; 2],
        },
    ];

    (nodes, instances, spheres)
}
