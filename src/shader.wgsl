// Fullscreen blit from HDR compute output to canvas.
// Clamp tonemapping placeholder — Khronos PBR Neutral added in Step 11.

@group(0) @binding(0) var hdr_tex : texture_2d<f32>;

// Oversized triangle — hardware clips it to exactly NDC [-1, 1].
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));
    let hdr   = textureLoad(hdr_tex, coord, 0);
    return vec4<f32>(clamp(hdr.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
