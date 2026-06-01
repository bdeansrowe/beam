# beam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A WebGPU wavefront path-tracer built in Rust/WASM.

## Current state

Wavefront compute pipeline rendering a Lambertian-shaded analytic
sphere via ray tracing:

1. **Ray generation** — pinhole camera, rays written to GPU storage
   buffer
2. **BVH traversal** — analytic quadratic sphere intersection, writes
   one `HitRecord` per ray into a hit buffer; miss sentinel
   `t = f32::MAX`
3. **Diffuse shading** — reads hit buffer, evaluates Lambertian BRDF
   against a hardcoded directional light, writes to HDR texture
4. **Metallic shading** — reads hit buffer, perfect mirror stand-in
   (environment color × base_color); no bounces yet
5. **HDR pipeline** — fullscreen blit pass reads `rgba16float`
   storage texture to canvas (Khronos PBR Neutral tonemapping in a
   later step)

Current scene: one orange diffuse sphere (`base_color=[0.9,0.4,0.1]`,
`MaterialType::Diffuse`) lit from `normalize(1,1,1)`.

## Known Issues

**Intermittent blank canvas on page load** — on some loads the
sphere fails to render. Mrays will read ~115 instead of ~28.
The intermittent miss is pre-shading: the BVH traversal kernel
fires (hence the elevated Mrays) but returns no hits, writing
`F32_MAX` to all hit records. The shading kernels correctly skip
all pixels. Background is written by the traversal kernel; the
canvas is not blank from a pipeline error — the geometry simply
isn't being found on that load. Reload until the sphere appears
(usually 1–3 attempts). Root cause not identified; suspected
Dawn/Metal non-determinism on Chromium 148 / Apple Silicon. Does
not affect rendering correctness when working.

## Implementation Progress

- [x] Step 3 — Ray generation kernel
- [x] Step 4 — Analytic sphere intersection, normal shading, HDR pipeline
- [x] Step 5 — BVH scaffold
- [x] Step 5.5 — Geometry buffer format (dual-material triangles)
- [x] Step 6 — Material system buffer infrastructure
- [x] Step 6b — Shading kernel split: diffuse + metallic pipelines,
      hit record buffer, shade_common.wgsl utilities
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

**Prerequisites:** `rustup target add wasm32-unknown-unknown` and
`cargo install wasm-pack basic-http-server`.

**Browser:** Vivaldi or Chrome 113+ required for WebGPU.

## Quick check (no browser needed)

```bash
cargo check --target wasm32-unknown-unknown
```

## License

MIT — see [LICENSE](LICENSE).
