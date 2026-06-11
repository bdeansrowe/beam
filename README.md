# beam

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A WebGPU wavefront path-tracer built in Rust/WASM.

## Current state

Full multi-bounce wavefront path tracer rendering six analytic spheres
with indirect illumination, glass refraction, NEE direct lighting, and
per-pixel bloom convergence acceleration.
8 bounces per frame; progressive temporal accumulation.

1. **Ray generation** — pinhole camera, `FrameUniform`-seeded jitter;
   rays written to GPU storage buffer; throughput initialized to
   `(1,1,1)`; medium stack pre-seeded with air (`IOR=1.0`)
2. **Bounce loop (×8)** — each iteration updates `frame_data.bounce`
   and dispatches:
   - *BVH traversal* — quadratic sphere intersection, writes one
     `HitRecord` per ray; miss writes `F32_MAX` sentinel to `HitRecord`
     and leaves ray live for the background pass
   - *Background shader* — reads `hit_records` and ray throughput;
     for escaped rays (missed geometry, not yet terminated) writes
     `background_color × throughput` into `scratch_buf` and marks
     ray terminated; keeps background logic out of the traversal kernel
   - *Diffuse shading* — cosine-weighted hemisphere sample; writes
     continuation ray and attenuates `ray.throughput`
   - *Metallic shading* — perfect mirror reflect; attenuates throughput
   - *Glass shading* — Schlick Fresnel, Snell refraction, TIR,
     Russian roulette reflect/refract, Beer's law, medium stack push/pop
   - *Russian roulette* — from bounce 3: survival = max(throughput);
     terminates low-contribution paths; survivors rescaled by 1/survival
   - *NEE direct lighting* — shadow ray to point light; Beer's law
     transmittance through glass; throughput-weighted contribution
     added to `scratch_buf`

   On frame 0, before the bounce loop: sky mask initialization fires
   every pixel against the BVH and marks hit pixels (plus a 2-pixel
   Chebyshev dilation border) as geometry pixels; sky pixels are
   permanently frozen. The background preshader then fires 8
   Halton-jittered rays per sky pixel and writes the supersampled
   average into `scratch_buf` — sky pixels are never re-dispatched
   in subsequent frames.

3. **Bloom bounce loop (×8)** — runs immediately after the mainline
   loop on the active bloom slot set. Each slot represents one
   high-variance pixel promoted by the selection pass; 256 independent
   rays per slot trace the same scene with distinct Halton sub-pixel
   jitter and PCG-seeded paths. The per-bounce dispatches mirror the
   mainline loop: BVH traversal → background shader variant → diffuse /
   metallic / glass shading → NEE direct; contributions accumulate
   into `bloom_scratch_buf`.
4. **Bloom postshader** — for each active slot, a 256-thread workgroup
   loads the slot's rays from `bloom_scratch_buf` and reduces them to a
   mean via parallel reduction; the result is written to `scratch_buf`
   at the slot's pixel index, replacing (not adding to) the mainline
   single-ray contribution for that pixel.
5. **Accumulate** — accumulates `scratch_buf` (sum of all bounce
   contributions) into a persistent weighted-sum buffer via
   `accum += scratch`; the resolve pass divides by frame count to
   produce the display image
6. **Variance + selection** — after accumulation, a variance pass
   computes per-pixel variance from the running sum and sum-of-squares;
   the selection pass uses hysteresis thresholds to promote
   high-variance pixels into bloom slots for the next frame and evict
   converged pixels, re-reserving fresh slot indices each frame to
   prevent cross-pixel collisions.
7. **Blit** — fullscreen blit reads the current `rgba16float` accum
   texture to canvas (Khronos PBR Neutral tonemapping in a later step)

Current scene: six analytic spheres forming an egg stand-in — two
nested XZ-scaled glass spheres (mat 1, `IOR=1.5`) for the outer shell
and inner cavity (mat 4, `IOR=1.0` air bubble); a small glass sphere
(mat 1) with its own air-bubble inclusion (mat 4) inside the egg; a
diffuse sphere (mat 2) and a metallic mirror sphere (mat 3) inside the
egg — the checkerboard visible in the mirror is the background
environment map reflected. Point light at (2, 4, 2), warm white,
intensity 20. Background: procedural spherical checkerboard (royal blue
/ yellow). Medium stack depth 8. Material indices: 0=air, 1=glass
(`IOR=1.5`), 2=diffuse, 3=metallic, 4=air-bubble (`IOR=1.0`). Magenta
= medium stack underflow (geometry error).

Performance at 600×800 on Apple M2 (Chromium 148): ~15–25 FPS,
~7–12 Mrays/s depending on bloom occupancy. Bloom occupancy is
reported live in the HUD (`bloom NNN / CAPACITY`).

## Known Issues

**Intermittent blank canvas on page load** — on some loads the
sphere fails to render. Mrays will read ~115 instead of ~28.
The intermittent miss is pre-shading: the BVH traversal kernel
fires (hence the elevated Mrays) but returns no hits, writing
`F32_MAX` to all hit records. The shading kernels correctly skip
all pixels. The canvas shows the background (written by the background
passes) but no geometry — the geometry simply isn't being found on
that load. Reload until the sphere appears
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
      direct kernel; second diffuse sphere; warm background; accum_buf;
      temporal accumulation (ping-pong blend) rolled in
- [x] Step 8 — Multi-bounce wavefront loop: `Ray.throughput` in-place
      mutation, per-bounce `frame_data` updates via separate submits,
      Russian roulette termination (bounce ≥ 3), additive scratch_buf
      accumulation across 8 bounces per frame
      Also, test-scene adjust to include a metallic third sphere
- [x] Step 9 — Sky mask: frame-0 initialization with 2-pixel dilation;
      two-pass background architecture — `background_preshader.wgsl`
      (frame-0 supersampled initialization of frozen sky pixels, 8 Halton
      samples) and `background_shader.wgsl` (per-bounce escaped-ray
      throughput contribution, runs in bounce loop every frame);
      `MEDIUM_STACK_DEPTH` promoted to named constant (depth 4→8);
      `pixel_seed` canonical PCG spatial-hash RNG established in
      `shade_common.wgsl`, mixing pixel coordinates, frame, and bounce
      to break scanline correlation
- [x] Step 9.5 — Bloom convergence acceleration:
      per-pixel variance tracking, hysteresis-gated
      slot promotion, 256-ray bloom pass with
      background shader variant, postshader parallel
      reduction, bloom occupancy HUD (B12–B15)
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
