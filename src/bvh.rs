use bytemuck::{Pod, Zeroable};

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
    pub center_radius: [f32; 4],  // .xyz=center, .w=radius
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

pub fn build_trivial_scene() -> (Vec<BvhNode>, Vec<TlasInstance>, Vec<Sphere>) {
    let nodes = vec![
        // Node 0: BLAS leaf — unit sphere at origin, sphere_index=0
        BvhNode {
            aabb_min_left_start:  [-0.5, -0.5, -0.5, f32::from_bits(0)],
            aabb_max_right_count: [ 0.5,  0.5,  0.5, f32::from_bits(0)],
            node_type: NODE_LEAF_SPHERE,
            _reserved: [0; 3],
        },
    ];

    let instances = vec![
        TlasInstance {
            transform: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            blas_offset: 0,
            flags:       0,
            _reserved:   [0; 2],
        },
    ];

    let spheres = vec![
        Sphere { center_radius: [0.0, 0.0, 0.0, 0.5] },
    ];

    (nodes, instances, spheres)
}
