# ltbl — Project Context

## Name
**ltbl** — "Let There Be Light"

## One-Line Description
A WebGPU ray-traced pinball game built in Rust/WASM, serving simultaneously as a personal creative project and a deliberate learning exercise in agentic development methodology using Claude Code.

---

## Current Technical State
- Working Rust/WASM/WebGPU scaffold rendering a single RGB triangle in the browser
- **Stack:** Rust, wgpu 27, winit 0.30, wasm-bindgen, wasm-pack
- **Render architecture:** Two-pass — compute pass → HDR storage texture, then tonemap blit to canvas
- **Dev server:** Simple HTTP server on port 9000
- **Dev environment:** Vivaldi (Chromium 144) on ARM64 Mac
- **Project structure:** `ltbl/` with `src/` (lib.rs, app.rs, gpu.rs, shader.wgsl) and `web/` (index.html, pkg/)

---

## Planned Rendering Architecture
- **Algorithm:** Wavefront path tracing (not megakernel) — separate compute kernels per stage
- **Acceleration:** Two-level BVH (TLAS/BLAS)
- **Color model:** RGB path tracing (not spectral) — spectral rendering is a reach goal
- **Sampling:** Variable per-pixel sample counts as a foundational architectural decision
- **Color space:** HDR linear rendering throughout
- **Pipeline order:** path trace → temporal accumulation → denoiser → tone mapping → bloom → display
- **Tonemapping:** Khronos PBR Neutral

---

## Key Technical Decisions (Already Made)
- Compute shaders (not fragment shaders) for ray tracing
- `Rc`/`RefCell` (not `Arc`/`Mutex`) — no threading primitives available in WASM
- Async GPU init via `wasm_bindgen_futures::spawn_local`
- wgpu 27 API (not earlier versions)
- Canvas size read from DOM directly on WASM to avoid 0x0 initialization issue
- Tessellated geometry for first pass (not analytic primitives) — revisit if quality demands it

---

## Game Concept
A pinball machine reimagined as a fully 3D playing field **enclosed inside a glass egg**, floating in space. Key visual and design properties:

- **The egg:** A Hügelschäffer egg (asymmetric, fatter at one end) with smoothly varying wall thickness — roughly constant through the middle, significantly thicker at both poles. Two nested surfaces (outer and inner) creating a solid dielectric shell.
- **The ball:** Chrome (highly reflective).
- **Playing field contents:** Bumpers, flippers, knockdown targets, and other pinball obstacles — reimagined into 3D space inside the egg. Most obstacles will be **transparent colored glass** so the player can track the ball's path without full occlusion.
- **Environment:** The egg floats in space against an HDR environment map — no floor, no walls. All caustics and light interplay are internal to the egg.
- **Rendering challenge:** Nested refractive surfaces, colored glass absorption (Beer's law), caustics, 8–16+ bounces for glass geometry.

---

## Methodology
Deliberately shifting from pair-programming with Claude (chat) to **full agentic development with Claude Code**. Goals:
1. Build real experience with agentic development methodology
2. Develop credible professional fluency in agentic workflows (relevant to potential return to a medtech startup considering adopting agentic development)

---

## Developer Background
Senior software engineer and creative technologist, 35+ years experience. Strong background in graphics, real-time 3D, and production VFX (Alias Research, Apple Computer, PDI/DreamWorks, Laika). Relatively new to Rust (learning it for this project). Not new to GPU programming concepts. Previously used Claude as a pair programmer to build the working scaffold, resolving numerous wgpu/winit API compatibility issues in the process.
