// mesh_common.wgsl — triangle mesh binding declarations.
// NOT a standalone compute shader — no @compute entry point.
// Composed into shading kernels that may access triangle geometry
// (diffuse, metallic, glass). NOT composed into shade_direct —
// shadow-ray BVH traversal never touches vertex or triangle data.

@group(0) @binding(3) var<storage, read> vertices : array<Vertex>;
@group(0) @binding(4) var<storage, read> geometry : array<TriangleRecord>;
