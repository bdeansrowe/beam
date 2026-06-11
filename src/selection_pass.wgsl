// selection_pass.wgsl — promotes high-variance pixels into bloom slots.
// Dispatched after variance_pass, before ray_gen.
// common_common.wgsl prepended at pipeline creation.

const BLOOM_PROMOTION_THRESHOLD: f32 = 0.2; // claim a slot if variance exceeds this
//const BLOOM_PROMOTION_THRESHOLD: f32 = 0.3; // acne in bright areas with evict = 0.01
//const BLOOM_PROMOTION_THRESHOLD: f32 = 0.2; // keep this line for diagnotics
const BLOOM_EVICTION_THRESHOLD:  f32 = 0.05; // surrender a slot if variance drops below this
//const BLOOM_EVICTION_THRESHOLD:  f32 = 0.1; // keep this line for diagnotics

@group(0) @binding(0) var<uniform>             frame_data:    FrameUniform;
@group(1) @binding(0) var<storage, read_write> pixel_buf:     array<PixelState>;
@group(1) @binding(1) var<storage, read_write> bloom_counter: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> bloom_index_buf: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    if frame_data.frame < 1u { return; }  // no meaningful variance on frame 0
    let idx = gid.y * frame_data.dim_x + gid.x;

    let v            = pixel_buf[idx].variance;
    let was_blooming = pixel_buf[idx].bloom_slot >= 0;

    // Hysteresis decision: promote above high threshold, evict below low
    // threshold, otherwise keep prior state.
    var should_bloom: bool;
    if v >= BLOOM_PROMOTION_THRESHOLD {
        should_bloom = true;
    } else if v < BLOOM_EVICTION_THRESHOLD {
        should_bloom = false;
    } else {
        should_bloom = was_blooming;
    }

    if should_bloom {
        // Re-reserve a fresh slot every frame against the zeroed counter.
        // This keeps bloom_slot / bloom_index_buf / ray slot location mutually
        // consistent within the frame and prevents cross-pixel slot collisions.
        let slot = atomicAdd(&bloom_counter[0], 1u);
        if slot < BLOOM_SLOT_CAPACITY {
            pixel_buf[idx].bloom_slot = i32(slot);
            bloom_index_buf[slot]     = idx;
        } else {
            pixel_buf[idx].bloom_slot = -1;  // capacity exhausted this frame
        }
    } else {
        pixel_buf[idx].bloom_slot = -1;
    }
}
