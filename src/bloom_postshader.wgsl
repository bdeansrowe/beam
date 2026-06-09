// bloom_postshader.wgsl
// Collapses BLOOM_AMPLIFICATION bloom rays per slot into scratch_buf.
// Dispatched after bloom bounce loop, before accumulate.
// common_common.wgsl prepended at pipeline creation.

@group(0) @binding(0) var<uniform>             frame_data:        FrameUniform;
@group(1) @binding(0) var<storage, read>       bloom_index_buf:   array<u32>;
@group(1) @binding(1) var<storage, read>       bloom_scratch_buf: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> scratch_buf:       array<vec4<f32>>;

var<workgroup> wg_accum: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let slot      = wid.x;
    let sample_i  = lid.x;
    let pixel_idx = bloom_index_buf[slot];

    wg_accum[sample_i] = bloom_scratch_buf[slot * BLOOM_AMPLIFICATION + sample_i];
    workgroupBarrier();

    // Parallel reduction — sum 256 values
    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if sample_i < stride {
            wg_accum[sample_i] += wg_accum[sample_i + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if sample_i == 0u {
        scratch_buf[pixel_idx] = wg_accum[0] / f32(BLOOM_AMPLIFICATION);
    }
}
