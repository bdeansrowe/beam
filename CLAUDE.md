# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What is ltbl?

**ltbl** ("Let There Be Light") is a WebGPU ray-traced pinball game built in Rust/WASM.
A pinball machine reimagined as a fully 3D playing field enclosed inside a glass egg,
floating in space. The player views the egg from outside. The egg is a Hügelschäffer egg
— asymmetric, fat-end-down exterior for Fabergé egg aesthetics, fat-end-up interior for
natural funnel toward drain and flippers.

This project is also a live test case for the `parley` agentic development methodology —
human-model co-authorship of design decisions before implementation.

---

## Dev Commands

```bash
# Build WASM and serve on http://localhost:9000 (requires wasm-pack + basic-http-server)
./build.sh

# Build only (no serve)
wasm-pack build --target web --out-dir web/pkg --release

# Serve only (after build)
cd web && basic-http-server --addr 127.0.0.1:9000 .
# or without basic-http-server:
cd web && python3 -m http.server 9000

# Syntax/type check without WASM (fast feedback)
cargo check
```

**Browser:** Vivaldi or Chrome 113+ required for WebGPU. Open DevTools console for GPU errors.

**Prerequisites:** `cargo install wasm-pack basic-http-server` and `rustup target add wasm32-unknown-unknown`.

---

## Session & Context Documents

The following Markdown files in the repo root are design/context documents from `parley` sessions — read them for background, not for code conventions:

- `ltbl_project_context.md` — project overview and developer background
- `ltbl_game_design_context.md` — game design details
- `ltbl_session01_counsel_layer.md` / `ltbl_session_01_checkpoint.md` — session 01 design record
- `nvidia_rtx_best_practices_for_ltbl.md` / `updated_nvidia_rtx_best_practices_for_ltbl.md` — RTX/BVH research adapted for WebGPU

---

## Current Technical State

- Working Rust/WASM/WebGPU scaffold rendering a single RGB triangle in the browser
- **Stack:** Rust, wgpu 27, winit 0.30, wasm-bindgen, wasm-pack
- **Render architecture:** Two-pass — compute pass → HDR storage texture, then tonemap
  blit to canvas
- **Dev server:** Simple HTTP server on port 9000
- **Dev environment:** Vivaldi (Chromium 144) on ARM64 Mac (M2 MacBook Air, 10-core GPU)
- **Project structure:** `src/` (lib.rs, app.rs, gpu.rs, shader.wgsl) and
  `web/` (index.html, pkg/) — this directory is the project root

---

## Planned Rendering Architecture

- **Algorithm:** Wavefront path tracing — separate compute kernels per stage, NOT megakernel
- **BVH:** Two-level software BVH (TLAS/BLAS) in WGSL compute shaders
  — WebGPU does NOT expose hardware RT units — all BVH construction and traversal is
  implemented manually in WGSL compute shaders — DXR/Vulkan RT API advice does not apply
- **Color:** RGB path tracing — spectral rendering is a reach goal only
- **Sampling:** Variable per-pixel sample counts as foundational architectural decision
- **Color space:** HDR linear rendering throughout
- **Pipeline order:** path trace → temporal accumulation → denoiser → tone mapping →
  bloom → display
- **Tonemapping:** Khronos PBR Neutral
- **Bounces:** 8 as starting value — increase only if caustic quality demands it, each
  additional bounce is a full additional compute dispatch
- **Denoiser:** Start with temporal accumulation only, add SVGF when needed

---

## Critical API Constraints — DO NOT VIOLATE

These caused problems during scaffold development and must be respected:

- **wgpu 27 API** — not earlier versions — API surface changed significantly
- **winit 0.30** — not earlier versions
- **`Rc`/`RefCell`** — NOT `Arc`/`Mutex` — no threading primitives available in WASM
- **Async GPU init** via `wasm_bindgen_futures::spawn_local`
- **Canvas size** read from DOM directly on WASM — NOT from winit — avoids 0x0
  initialization issue
- **Compile all compute pipelines at startup** — NEVER lazily during rendering —
  pipeline compilation can stall the render loop

---

## WebGPU Resource Binding Convention

- **Bind group 0** — scene-global resources: BVH buffers, geometry vertex/index buffers,
  material buffer, environment map
- **Bind group 1** — per-pass resources: ray buffers, output textures
- Minimizes bind group switching overhead between wavefront dispatches

---

## Geometry Decisions

### The Egg
- **Outer surface:** Hügelschäffer egg, fat-end-down. May eventually be analytic primitive
  (quartic intersection) — reach goal. First pass: tessellated triangle mesh.
- **Inner surface:** Hügelschäffer egg, fat-end-up (opposite orientation). Must be
  tessellated — obstacle boundaries require mesh geometry. Cannot be analytic.
- **Wall thickness:** Smoothly varying — roughly constant through middle, thickest at
  bottom where fat outer meets pointy inner bottom.

### The Ball — ANALYTIC SPHERE, NOT TESSELLATED MESH
- Represented as analytic sphere primitive (center + radius)
- Ray intersection: quadratic solve — closed form, exact, fast
- Exact normals: `normalize(hitPoint - center)` — zero tessellation error
- Critical for chrome mirror surface — normal accuracy directly determines reflection
  quality — tessellation artifacts would produce visible waviness in reflections
- Flagged as sphere primitive in BVH leaf node
- Dedicated quadratic intersection code path in traversal kernel
- This is the ONE intentional exception to triangles-over-everything rule
- Per-ray data: hit distance + primitive identifier only — no barycentric coordinates
  (sphere has none), normal computed in shading kernel not traversal kernel
- **BLAS sizing:** Measure actual BLAS size on frame 0, use that exact allocation from
  frame 1 onward — analytic sphere representation is fixed size, position never affects
  memory requirement

### All Other Geometry
- Triangle meshes in BVH
- Per-hit data passed to shading kernel: hit distance, primitive index, barycentric
  coordinates (u,v) ONLY — normal interpolation and UV lookup deferred to shading kernel
- Do NOT compute normals or material properties in traversal kernel — false economy,
  increases register pressure, hurts occupancy

---

## BVH Architecture

### Never Cull Back Faces
The glass egg and glass obstacles are refractive — rays enter from outside (front faces)
and exit from inside (back faces of the same geometry). Back-face culling would break
nested dielectric medium tracking entirely. NEVER cull back faces for any geometry in ltbl.

### Static vs Dynamic — Kinematic Switching
- **Static geometry:** egg shell, fixed obstacles at rest, flippers at rest — BVH built
  ONCE at load time, never rebuilt
- **Ball:** ALWAYS dynamic — BLAS rebuilt every frame
- **Kinematic switching for obstacles and flippers:** static by default; promoted to
  dynamic (per-frame BLAS rebuild) only when actively animating (bumper recoiling, flipper
  flipping); demoted back to static once animation completes and object returns to rest
- At any given frame, the only GUARANTEED dynamic object is the ball — animating obstacles
  are a small transient subset
- **TLAS:** Rebuilt every frame — cheap given few dynamic instances
- **Prefer TLAS rebuild over refit** — refit degrades quality over time

### Implementation Phasing for Dynamic Geometry
1. **Phase 1 (current):** Fully static scene — build BVH once, never rebuild. Validate
   all rendering before adding any animation.
2. **Phase 2:** Ball animation only — ball BLAS rebuilds per frame, everything else static.
   No kinematic switching machinery needed yet.
3. **Phase 3:** Full kinematic switching for flippers and obstacles.
Do NOT implement Phase 2 or 3 until Phase 1 rendering is correct and validated.

### Environment Map
Do NOT include HDR environment map as scene geometry in BVH. When a ray escapes the scene
without hitting geometry, sample the HDR environment map directly using ray direction.

---

## Wavefront Architecture — Key Principles

### Separate Dispatches Per Ray Type
- Primary ray generation
- BVH traversal / intersection
- Material shading (sorted by material type)
- Shadow/occlusion rays (SEPARATE from main path tracing kernel)
- Each dispatch fires exactly one category of ray — do not fold shadow ray logic into
  the main path tracing kernel

### Shadow Rays — Early Exit
In the shadow/occlusion ray kernel (next event estimation), terminate BVH traversal
immediately upon finding ANY valid intersection between shading point and light sample.
Do NOT find closest hit — any hit is sufficient to determine occlusion.
EXCEPTION: glass geometry is NOT fully occluding — shadow ray must handle partial
transmittance through glass. Terminate early once accumulated transmittance < 0.001.

### Wavefront Eliminates Live State Problem
Because each stage is a separate dispatch, there is no live state carried across kernel
boundaries. State is explicitly written to and read from wavefront ray buffers. This is
a fundamental advantage over megakernel architecture — no register spill from live state.

### Ray Records — Keep Compact
Each ray record contains ONLY what is strictly needed for the current stage:
- Origin, direction, tmin/tmax
- Packed material/medium state
- Medium stack (see below)
Do NOT store intermediate shading results in ray records. Defer all computation to the
shading kernel. Large ray records increase memory bandwidth pressure across all dispatches.

### Medium Stack
The medium stack tracks which refractive volumes a ray is currently inside — required for
correct nested dielectric rendering of the glass egg and colored glass obstacles.
- **Maximum depth: 4** (realistic maximum for ltbl geometry is 3, depth 4 is headroom)
- Each entry: material ID + IOR value (~8 bytes per entry = ~32 bytes for depth-4 stack)
- Push on front-face hit (entering volume), pop on back-face hit (exiting volume)
- Interior air is an explicit medium — the inner egg surface is its entry/exit boundary
- Do NOT increase stack depth beyond 4 — every extra entry adds to every ray record
- See `ltbl_modeling_requirements.md` for geometry requirements that keep stack correct

### Russian Roulette Termination
Implement probabilistic path termination to recover performance from low-contribution
paths. Terminate paths before maximum bounce count when throughput is very low.

### Stream Compaction
Between wavefront stages, compact active rays into contiguous buffer before dispatch.
Terminated paths (Russian roulette, escaped scene) are removed — every dispatched thread
has live work. Required once ray attrition across bounces becomes significant.

---

## Primary Ray Dispatch — Three Mechanisms

### 1. Sky Mask (implement after basic rendering works)
A large fraction of pixels (potentially 40-60%) are background pixels whose primary rays
miss the egg and hit the static HDR environment. These never need re-dispatching.

**TWO SHADERS, NOT ONE WITH A CONDITIONAL:**
- **Frame 0 shader:** fires all pixels, writes per-pixel sky mask (1 bit/pixel) flagging
  misses. Runs ONCE at scene initialization.
- **Per-frame shader:** reads sky mask, dispatches only egg-hitting pixels via stream
  compaction. No conditional branch in hot path.

Invalidate and recompute sky mask if camera moves or egg moves.

### 2. Variable Sample Counts / Foveated Sampling
- Ball region gets elevated samples — importance sampling toward ball's apparent position
- Apparent position ≠ geometric position — refraction through egg displaces it
- Use Rapier physics data (ball position + velocity) to compute approximate apparent
  position on egg surface via Snell's law
- Spread sized to ball angular subtense + refraction uncertainty; velocity scales spread
- Work queue rather than uniform dispatch for variable sample counts

### 3. Temporal Accumulation
- Converged pixels reduce per-frame sample contribution
- Background pixels (sky mask): accumulate once on frame 0, not re-sampled thereafter

---

## Temporal Accumulation and Anti-Aliasing

### Sub-Pixel Jitter — Implement From The Start
Fire each sample from a randomly chosen sub-pixel location, NOT always pixel center.
Use Halton sequence or blue noise distribution for better coverage than pure random.

### Accumulation Buffer — Implement Correctly From The Start
Store **weighted sum and sum-of-weights separately** — NOT a simple running average.
This is required for the planned Gaussian filter extension and costs nothing to do
correctly now vs retrofitting later.

Write ray results with an explicit weight parameter (initially 1.0 for uniform jitter).
Swapping to Gaussian weighted accumulation later requires only changing the weight value
and the jitter distribution function — NOT restructuring the pipeline.

### Gaussian Filter Importance Sampling (Planned Extension — Not First Pass)
Sample ray origins from Gaussian centered on pixel, sigma ~0.5-1.0 pixels.
Small percentage of rays fire from neighboring pixel space with lower weights.
Requires weighted accumulation buffer (already implemented above).
Mitchell-Netravali filter as refinement option over pure Gaussian.
Implement AFTER uniform jitter pipeline is working correctly.

---

## Materials

### Glass BSDF (Dominant Material)
- Fresnel calculation + Snell's law refraction + medium stack push/pop
- Beer's law attenuation for colored glass: `throughput *= exp(-absorption * distance)`
  Absorption coefficient is vec3 (per RGB channel). Zero for clear glass.
- At dielectric surfaces: probabilistic choice (Russian roulette) between reflection and
  refraction using Fresnel reflectance as probability — NEVER split ray into two
  Splitting rays causes exponential explosion of ray count — always sample one path
- Do NOT cull back faces — both face orientations required for correct refraction

### Chrome Ball (Mirror)
- Perfect specular mirror — delta function BRDF
- Exact normal from analytic sphere: `normalize(hitPoint - center)`
- **"Used universe" aesthetic (reach goal):** roughness/anisotropy variation as material
  parameters for scratched chrome character — no geometry changes required

### Material Sorting (Planned Optimization)
Sort rays by material type between traversal and shading kernels.
Separate shading dispatches for: glass, chrome, emissive, environment.
Reduces shading divergence — all threads in a dispatch evaluate same material.
Implement after basic rendering works, when profiling motivates it.

---

## Self-Intersection Prevention

Every spawned ray needs tmin epsilon — typically ~1e-4 in scene units.
Any intersection closer than tmin from ray origin is ignored.
This prevents floating point imprecision at ray origin from causing immediate re-hit
of the surface just left — distinct from the modeling coincident-surface problem.
Tune for ltbl's scene scale — too small: self-intersection artifacts; too large: misses
legitimate nearby geometry in thin glass obstacles.
See `ltbl_modeling_requirements.md` for related modeling constraints.

---

## GPU Buffer Management

- **Pre-allocate ALL GPU buffers at startup** — NEVER allocate or deallocate during
  render loop
- Size wavefront ray buffers for maximum expected ray count:
  width × height × max_samples_per_pixel
- Reuse scratch buffers between wavefront stages where lifetimes don't overlap
- Static geometry BVH buffers: packed tightly, never reallocated at runtime

---

## Profiling

- Instrument each wavefront stage with **WebGPU timestamp queries** early
- Use `chrome://tracing` for GPU timeline profiling
- Vivaldi DevTools Performance tab for coarse frame timing
- Pre-allocate timestamp query buffers at startup
- Performance data must be available before any optimization decisions are made
- DO NOT optimize without profiling data — measure first, optimize second

---

## Reach Goals (Do Not Implement Until Basics Work)

- Spectral rendering (wavelength-dependent IOR, dispersion, prismatic caustics)
- Analytic outer egg surface (Hügelschäffer quartic intersection)
- "Used universe" ball (roughness/anisotropy material parameters)
- Probe ray refinement for ball importance sampling
- Gaussian filter importance sampling
- Particle effects (shatter targets) — three quality tiers:
  - Tier 1: screen-space composite, depth buffer occlusion, no refraction correction
  - Tier 2: single-bounce apparent position correction via Snell's law + Beer's law tinting
  - Tier 3: full path-traced glass shard particles
  - Adaptive quality: auto-downgrade on frame budget exceeded — implement with Tier 2 only

---

## Research / Optimization Experiments (Post-Baseline, Requires Profiling Data)

1. Polar coordinate AS — spherical BVH for primary/shadow rays, bounding sphere hierarchy
2. Probe ray adaptive sampling for ball region
3. Plane-based triangle intersection vs Möller–Trumbore baseline

---

## Modeling Requirements (Summary)

See `ltbl_modeling_requirements.md` for full detail.
- No intersecting solids — explicit boundaries between all adjacent dielectric volumes
- Interior air is explicit medium with entry/exit surfaces (inner egg surface)
- Shared boundary faces carry dual material IDs (front + back) — custom asset format
- Consistent surface normal orientation — front/back face determines medium stack push/pop
- No degenerate triangles

---

## Implementation Order — Session 02 Onwards

1. Install Claude Code ← START HERE
2. Create/refine this CLAUDE.md in the repository
3. Ray generation kernel — pinhole camera, rays into storage buffer, no intersection yet
4. Analytic sphere intersection — shade by normal, first visible path-traced output
5. BVH scaffold — TLAS/BLAS structure with trivial static scene (one sphere instance)
6. Material system — diffuse first, specular, then glass BSDF
7. Next event estimation — shadow rays, direct lighting
8. Sky mask — frame 0 initialization + per-frame masked dispatch
9. Temporal accumulation — with jitter, weighted accumulation buffer from the start
10. Denoiser — temporal accumulation only first, SVGF when needed
11. Tone mapping and bloom
12. Ball animation (Phase 2) — only after Phase 1 rendering is correct and validated
13. Kinematic switching (Phase 3) — only after Phase 2 works

**Do not proceed to the next step until the current step produces correct, validated output.**
