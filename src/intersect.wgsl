// ── Shared ray type ───────────────────────────────────────────────────────────
struct Ray {
    origin:    vec4<f32>,  // .w = tmin
    direction: vec4<f32>,  // .w = tmax
}

// ── BVH scene types — must mirror Rust structs in bvh.rs exactly ─────────────
struct BvhNode {
    aabb_min_left_start:  vec4<f32>,  // .xyz=aabb_min  .w=left_child|prim_start|sphere_index (bits)
    aabb_max_right_count: vec4<f32>,  // .xyz=aabb_max  .w=right_child|prim_count|unused (bits)
    node_type:            u32,
    _r0:                  u32,
    _r1:                  u32,
    _r2:                  u32,
}  // 48 bytes

struct TlasInstance {
    transform:   mat4x4<f32>,  // 64 bytes, column-major — world transform (object→world)
    blas_offset: u32,
    flags:       u32,
    _r0:         u32,
    _r1:         u32,
}  // 80 bytes

struct Sphere {
    center_radius: vec4<f32>,  // .xyz=center  .w=radius
}  // 16 bytes

struct Vertex {
    position: vec4<f32>,  // .xyz=position  .w=0.0
    normal:   vec4<f32>,  // .xyz=normal    .w=0.0
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

// ── Node type constants ───────────────────────────────────────────────────────
const NODE_INTERNAL:      u32 = 0u;
const NODE_LEAF_TRIANGLE: u32 = 1u;
const NODE_LEAF_SPHERE:   u32 = 2u;
const NODE_LEAF_QUARTIC:  u32 = 3u;

const INVALID_NODE: u32 = 0xFFFFFFFFu;

// ── Bindings ──────────────────────────────────────────────────────────────────
// group(0) = scene-global resources (BVH, geometry, materials)
@group(0) @binding(0) var<storage, read> bvh_nodes      : array<BvhNode>;
@group(0) @binding(1) var<storage, read> tlas_instances : array<TlasInstance>;
@group(0) @binding(2) var<storage, read> spheres        : array<Sphere>;
// Step 5.5 — declared, not yet used
@group(0) @binding(3) var<storage, read> vertices       : array<Vertex>;
@group(0) @binding(4) var<storage, read> geometry       : array<TriangleRecord>;

// group(1) = per-pass resources (ray buffer, output texture)
@group(1) @binding(0) var<storage, read> rays    : array<Ray>;
@group(1) @binding(1) var                hdr_out : texture_storage_2d<rgba16float, write>;

// ── Constants ─────────────────────────────────────────────────────────────────
const BACKGROUND : vec4<f32> = vec4<f32>(0.05, 0.05, 0.1, 1.0);

// ── Named .w-field accessors — never access .w directly in traversal code ─────
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
// Returns hit t > 0, or -1.0 on miss.
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

// ── Hit record ────────────────────────────────────────────────────────────────
struct HitRecord {
    t:      f32,
    normal: vec3<f32>,
    hit:    bool,
}

// ── BVH traversal ─────────────────────────────────────────────────────────────
// Outer loop over TLAS instances. For each, transforms ray to local space and
// runs stack-based BLAS traversal (array<u32, 32> explicit stack per CLAUDE.md).
//
// NOTE: inst.transform is object→world. Transforming world-space ray into local
// space requires the inverse. For step 5 the transform is identity, so no-op.
// Non-identity instance transforms will need the inverse stored separately.
fn traverse_bvh(origin: vec3<f32>, dir: vec3<f32>, tmin: f32, tmax: f32) -> HitRecord {
    var result: HitRecord;
    result.hit    = false;
    result.t      = tmax;
    result.normal = vec3<f32>(0.0);

    let num_instances = arrayLength(&tlas_instances);

    for (var inst_idx = 0u; inst_idx < num_instances; inst_idx++) {
        let inst = tlas_instances[inst_idx];

        // Transform ray to instance local space (identity for step 5)
        let local_origin = (inst.transform * vec4<f32>(origin, 1.0)).xyz;
        let local_dir    = (inst.transform * vec4<f32>(dir,    0.0)).xyz;
        let local_inv    = 1.0 / local_dir;

        // Stack-based BLAS traversal — 32 entries, conservative headroom for ltbl
        var stack:     array<u32, 32>;
        var stack_ptr: i32 = 0;
        stack[0]  = inst.blas_offset;
        stack_ptr = 1;

        while stack_ptr > 0 {
            stack_ptr -= 1;
            let node_idx = stack[stack_ptr];
            let node     = bvh_nodes[node_idx];

            if !aabb_hit(node, local_origin, local_inv, tmin, result.t) { continue; }

            if node.node_type == NODE_LEAF_SPHERE {
                let sidx = node_sphere_index(node);
                let t    = sphere_hit(spheres[sidx], local_origin, local_dir, tmin, result.t);
                if t > 0.0 {
                    result.hit    = true;
                    result.t      = t;
                    let hit_pos   = local_origin + t * local_dir;
                    result.normal = normalize(hit_pos - spheres[sidx].center_radius.xyz);
                }
            } else {
                // NODE_INTERNAL: push right child first so left is popped first (LIFO)
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

    return result;
}

// ── Main ──────────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(hdr_out);
    let px = gid.x;
    let py = gid.y;
    if px >= dims.x || py >= dims.y { return; }

    let idx = py * dims.x + px;
    let ray = rays[idx];

    let hit = traverse_bvh(
        ray.origin.xyz,
        ray.direction.xyz,
        ray.origin.w,
        ray.direction.w,
    );

    var color: vec4<f32>;
    if hit.hit {
        color = vec4<f32>(hit.normal * 0.5 + vec3<f32>(0.5), 1.0);
    } else {
        color = BACKGROUND;
    }
    textureStore(hdr_out, vec2<i32>(i32(px), i32(py)), color);
}
