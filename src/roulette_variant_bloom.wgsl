// roulette_variant_bloom.wgsl — bloom dispatch entry point for the roulette pipeline.
// Declares the rays binding and fn main; calls roulette_main(idx).
@group(1) @binding(0) var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= 4096u * 256u { return; }
    roulette_main(idx);
}
