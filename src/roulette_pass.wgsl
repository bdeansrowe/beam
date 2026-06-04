// roulette_pass.wgsl — Russian roulette path termination.
// common_common.wgsl is prepended; Ray, FrameUniform, hash_u32 are in scope.
// No-op for bounce < 3. From bounce 3 onward: probabilistic termination based
// on max(throughput.rgb). Survivors are importance-reweighted by 1/survival.

@group(0) @binding(0) var<uniform>       frame_data: FrameUniform;
@group(1) @binding(0) var<storage, read_write> rays: array<Ray>;

const HASH_MUL_0_R: u32 = 0xbf324c81u;
const HASH_MUL_1_R: u32 = 0x68b31f7eu;

fn hash_r(x: u32) -> u32 {
    var v  = x;
    v     ^= v >> 17u;
    v     *= HASH_MUL_0_R;
    v     ^= v >> 11u;
    v     *= HASH_MUL_1_R;
    v     ^= v >> 15u;
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.y * frame_data.dim_x + gid.x;
    if gid.x >= frame_data.dim_x || gid.y >= frame_data.dim_y { return; }

    // Skip already-terminated rays.
    if rays[idx].direction.w < 0.0 { return; }

    // No roulette on early bounces — let short paths survive unmodified.
    if frame_data.bounce < 3u { return; }

    let tp = vec3<f32>(rays[idx].throughput[0],
                       rays[idx].throughput[1],
                       rays[idx].throughput[2]);
    let survival = max(max(tp.r, tp.g), tp.b);

    // Degenerate throughput — terminate immediately.
    if survival <= 0.0 {
        rays[idx].throughput[0] = 0.0;
        rays[idx].throughput[1] = 0.0;
        rays[idx].throughput[2] = 0.0;
        rays[idx].direction.w   = -1.0;
        return;
    }

    let seed = idx ^ frame_data.frame ^ (frame_data.bounce << 16u);
    let rand = f32(hash_r(seed) & 0x00ffffffu) / f32(0x01000000u);

    if rand > survival {
        rays[idx].throughput[0] = 0.0;
        rays[idx].throughput[1] = 0.0;
        rays[idx].throughput[2] = 0.0;
        rays[idx].direction.w   = -1.0;
    } else {
        rays[idx].throughput[0] = rays[idx].throughput[0] / survival;
        rays[idx].throughput[1] = rays[idx].throughput[1] / survival;
        rays[idx].throughput[2] = rays[idx].throughput[2] / survival;
    }
}
