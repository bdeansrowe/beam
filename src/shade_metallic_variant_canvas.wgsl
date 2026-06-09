// shade_metallic_variant_canvas.wgsl — canvas dispatch entry point for the metallic pipeline.
// Declares the rays binding and fn main; calls metallic_main(idx).
@group(1) @binding(2) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px  = gid.x;
    let py  = gid.y;
    if px >= frame_data.dim_x || py >= frame_data.dim_y { return; }
    let idx = py * frame_data.dim_x + px;
    metallic_main(idx);
}
