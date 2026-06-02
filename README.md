# beam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A WebGPU wavefront path-tracer built in Rust/WASM.

## Current state

Wavefront compute pipeline rendering two analytic spheres with direct
point lighting via Next Event Estimation, with progressive frame
accumulation:

1. **Ray generation** — pinhole camera, Halton sub-pixel jitter driven
   by `FrameUniform`; rays written to GPU storage buffer; medium stack
   pre-seeded with air (`IOR=1.0`)
2. **BVH traversal** — quadratic sphere intersection, writes one
   `HitRecord` per ray; miss sentinel `t = f32::MAX`; background
   written to `scratch_buf`
3. **Diffuse shading** — no-op; direct lighting handled exclusively by
   the NEE kernel
4. **Metallic shading** — perfect mirror stand-in
5. **Glass shading** — full dielectric BSDF: Schlick Fresnel, Snell
   refraction, TIR detection, Russian roulette reflect/refract,
   Beer's law absorption, medium stack push/pop
6. **Direct lighting (NEE)** — fires a shadow ray toward the point
   light; any-hit BVH traversal accumulates Beer's law transmittance
   through glass, immediately blocks on opaque hits; writes
   N·L × falloff × light_color × base_color to `scratch_buf`
7. **Accumulate** — blends `scratch_buf` (new sample) into a ping-pong
   accum pair using `mix(history, new_sample, 1/(frame+1))`; the scene
   converges progressively across frames
8. **Blit** — fullscreen blit reads the current frame's `rgba16float`
   accum texture to canvas (Khronos PBR Neutral tonemapping in a later
   step)

Current scene: clear glass sphere (`IOR=1.5`) at origin, radius 0.5;
warm tan diffuse sphere at y=−1.5, radius 1.0. Point light at (2, 4, 2),
warm white, intensity 20. Warm gray background (0.45, 0.42, 0.38).
Material indices: 0=air, 1=glass, 2=diffuse. Magenta = medium stack
underflow (geometry error).

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
- [x] Step 7 — Next event estimation: point light, shadow rays, NEE
      direct kernel; second diffuse sphere; warm background; accum_buf
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
