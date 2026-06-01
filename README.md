# beam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A WebGPU wavefront path-tracer built in Rust/WASM.

## Current state

Wavefront compute pipeline rendering a clear glass analytic sphere via
ray tracing:

1. **Ray generation** — pinhole camera, Halton sub-pixel jitter, rays
   written to GPU storage buffer; medium stack pre-seeded with air
   (`IOR=1.0`)
2. **BVH traversal** — analytic quadratic sphere intersection, writes
   one `HitRecord` per ray into a hit buffer; miss sentinel
   `t = f32::MAX`
3. **Diffuse shading** — reads hit buffer, evaluates Lambertian BRDF
   against a hardcoded directional light, writes to HDR texture
4. **Metallic shading** — reads hit buffer, perfect mirror stand-in
   (environment color × base_color); no bounces yet
5. **Glass shading** — reads hit buffer, evaluates full dielectric BSDF:
   Schlick Fresnel, Snell refraction, TIR detection, Russian roulette
   reflect/refract selection, Beer's law absorption, medium stack
   push/pop with write-back to ray buffer; single-bounce output is
   throughput × base_color × background
6. **HDR pipeline** — fullscreen blit pass reads `rgba16float`
   storage texture to canvas (Khronos PBR Neutral tonemapping in a
   later step)

Current scene: one clear glass sphere (`MaterialType::Glass`,
`IOR=1.5`, `base_color=[1,1,1]`, zero absorption). Air is material
index 0; glass is material index 1. Single-bounce — refracted and
reflected rays resolve to the background colour, so the sphere is
transparent against the dark background. Magenta indicates a medium
stack underflow (geometry error).

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
- [x] Step 6c — Sphere material ID: Sphere expanded to 32 bytes with
      front/back material IDs; shading kernels read material from
      sphere buffer instead of hardcoded index
- [x] Step 6d — Glass BSDF: Schlick Fresnel, Snell refraction, TIR,
      medium stack push/pop, Beer's law; sphere is now clear glass
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
