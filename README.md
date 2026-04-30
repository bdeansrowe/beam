# ltbl — Let There Be Light

A WebGPU wavefront path-traced pinball game built in Rust/WASM.
A pinball machine reimagined as a fully 3D playing field enclosed inside a glass egg, floating in space.

## Current state

Wavefront compute pipeline rendering a normal-shaded analytic sphere via ray tracing:

1. **Ray generation** — pinhole camera, Halton sub-pixel jitter, rays written to GPU storage buffer
2. **Sphere intersection** — analytic quadratic solve against a single hardcoded sphere; hit pixels shaded by surface normal mapped to RGB
3. **HDR pipeline** — compute writes to `rgba16float` storage texture; fullscreen blit pass reads it to canvas (clamp tonemapping, Khronos PBR Neutral in a later step)

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
# Open http://localhost:9000
```

**Prerequisites:** `rustup target add wasm32-unknown-unknown` and `cargo install wasm-pack basic-http-server`.

**Browser:** Vivaldi or Chrome 113+ required for WebGPU.

## Quick check (no browser needed)

```bash
cargo check --target wasm32-unknown-unknown
```
