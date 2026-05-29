# beam — Brian's Extremely Amazing (rendering) Mechanism

A WebGPU wavefront path-tracer built in Rust/WASM.

## Current state

Wavefront compute pipeline rendering a normal-shaded analytic sphere via ray tracing:

1. **Ray generation** — pinhole camera, Halton sub-pixel jitter, rays written to GPU storage buffer
2. **Sphere intersection** — analytic quadratic solve against a single hardcoded sphere; hit pixels shaded by surface normal mapped to RGB
3. **HDR pipeline** — compute writes to `rgba16float` storage texture; fullscreen blit pass reads it to canvas (clamp tonemapping, Khronos PBR Neutral in a later step)

## Implementation Progress

- [x] Step 3 — Ray generation kernel
- [x] Step 4 — Analytic sphere intersection, normal shading, HDR pipeline
- [x] Step 5 — BVH scaffold
- [x] Step 5.5 — Geometry buffer format (dual-material triangles)
- [ ] Step 6 — Material system
- [ ] Step 7 — Next event estimation
- [ ] Step 8 — Sky mask
- [ ] Step 9 — Temporal accumulation
- [ ] Step 10 — Denoiser
- [ ] Step 11 — Tone mapping and bloom
- [ ] Step 12 — Ball animation
- [ ] Step 13 — Kinematic switching

## Stack

| | |
|---|---|
| Language | Rust → WASM (`wasm32-unknown-unknown`) |
| GPU | wgpu 27, WebGPU backend |
| Windowing | winit 0.30 |
| Build | wasm-pack |

## Build & run

```bash
./build.sh
# Open http://localhost:9666
```

**Prerequisites:** `rustup target add wasm32-unknown-unknown` and `cargo install wasm-pack basic-http-server`.

**Browser:** Vivaldi or Chrome 113+ required for WebGPU.

## Quick check (no browser needed)

```bash
cargo check --target wasm32-unknown-unknown
```
