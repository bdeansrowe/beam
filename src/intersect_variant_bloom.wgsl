// intersect_variant_bloom.wgsl — bloom dispatch entry point for the intersect pipeline.
// Declares the rays binding and fn main; calls intersect_main(idx).
@group(1) @binding(0) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= BLOOM_SLOT_CAPACITY * BLOOM_AMPLIFICATION { return; }
    intersect_main(idx);
}
