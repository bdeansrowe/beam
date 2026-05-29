use bytemuck::{Pod, Zeroable};

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
