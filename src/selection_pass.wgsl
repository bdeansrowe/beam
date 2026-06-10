// selection_pass.wgsl — promotes high-variance pixels into bloom slots.
// Dispatched after variance_pass, before ray_gen.
// common_common.wgsl prepended at pipeline creation.

const BLOOM_PROMOTION_THRESHOLD: f32 = 0.2; // claim a slot if variance exceeds this
const BLOOM_EVICTION_THRESHOLD:  f32 = 0.1; // surrender a slot if variance drops below this

@group(0) @binding(0) var<uniform>             frame_data:    FrameUniform;
@group(1) @binding(0) var<storage, read_write> pixel_buf:     array<PixelState>;
@group(1) @binding(1) var<storage, read_write> bloom_counter: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> bloom_index_buf: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }
    if frame_data.frame < 1u { return; }  // no meaningful variance on frame 0
    let idx = gid.y * frame_data.dim_x + gid.x;

    let v = pixel_buf[idx].variance;

    if v >= BLOOM_PROMOTION_THRESHOLD {
        // High variance — claim a slot if not already blooming
        if pixel_buf[idx].bloom_slot < 0 {
            let slot = atomicAdd(&bloom_counter[0], 1u);
            if slot < BLOOM_SLOT_CAPACITY {
                pixel_buf[idx].bloom_slot = i32(slot);
                bloom_index_buf[slot] = idx;
            }
        }
        // Already blooming — keep slot, do nothing
    } else if v < BLOOM_EVICTION_THRESHOLD {
        // Converged — evict
        pixel_buf[idx].bloom_slot = -1;
    }
    // Between thresholds — hysteresis band, keep current state
}
