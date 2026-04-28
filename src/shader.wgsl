// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       color:         vec3<f32>,
};

// Hardcoded fullscreen triangle — no vertex buffer needed to start
const POSITIONS: array<vec2<f32>, 3> = array(
    vec2<f32>( 0.0,  0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>( 0.5, -0.5),
);

const COLORS: array<vec3<f32>, 3> = array(
    vec3<f32>(1.0, 0.2, 0.2),
    vec3<f32>(0.2, 1.0, 0.2),
    vec3<f32>(0.2, 0.2, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(POSITIONS[vi], 0.0, 1.0);
    out.color         = COLORS[vi];
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
