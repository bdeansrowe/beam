// shade_metallic_variant_bloom.wgsl — bloom dispatch entry point for the metallic pipeline.
// Declares the rays binding and fn main; calls metallic_main(idx).
@group(1) @binding(2) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= BLOOM_SLOT_CAPACITY * BLOOM_AMPLIFICATION { return; }
    metallic_main(idx);
}
