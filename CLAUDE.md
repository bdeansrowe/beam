# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What is beam?

**beam** is a WebGPU ray-tracer built in Rust/WASM. To support a 3D pinball machine game.
A pinball machine reimagined as a fully 3D playing field enclosed inside a glass egg,
floating in space. The player views the egg from outside. The egg is a Hügelschäffer egg
— asymmetric, fat-end-down exterior for Fabergé egg aesthetics, fat-end-up interior for
natural funnel toward drain and flippers.

---

## Dev Commands

```bash
# Build WASM and serve on http://localhost:9000 (requires wasm-pack + basic-http-server)
./build.sh

# Build only (no serve)
wasm-pack build --target web --out-dir web/pkg --release

# Serve only (after build)
cd web && basic-http-server --addr 127.0.0.1:9666 .
# or without basic-http-server:
cd web && python3 -m http.server 9666

# Syntax/type check without WASM (fast feedback)
cargo check
```

**Browser:** Vivaldi or Chrome 113+ required for WebGPU. Open DevTools console for GPU errors.

**Prerequisites:** `cargo install wasm-pack basic-http-server` and `rustup target add wasm32-unknown-unknown`.

---
## Agentic Session Protocol

At the start of every implementation session, before
writing any code, present a numbered plan of all
intended steps and wait for approval before proceeding.
This allows the human supervisor to review the full
scope before any diffs land.

---
## Session & Context Documents

The following Markdown files in the repo root are design/context documents from `parley` sessions — read them for background, not for code conventions:

- `ltbl_project_context.md` — project overview and developer background
- `ltbl_game_design_context.md` — game design details
- `ltbl_session01_counsel_layer.md` / `ltbl_session_01_checkpoint.md` — session 01 design record
- `nvidia_rtx_best_practices_for_ltbl.md` / `updated_nvidia_rtx_best_practices_for_ltbl.md` — RTX/BVH research adapted for WebGPU

---

## Current Technical State

- Working Rust/WASM/WebGPU scaffold rendering a Lambertian-shaded
  analytic sphere via wavefront path tracing
- **Stack:** Rust, wgpu 27, winit 0.30, wasm-bindgen, wasm-pack
- **Render architecture:** Wavefront compute pipeline → HDR storage
  texture → blit to canvas
- **Dev server:** Simple HTTP server on port 9000
- **Dev environment:** Vivaldi (Chromium 144) on ARM64 Mac (M2 MacBook Air, 10-core GPU)
- **Project structure:** `src/` (lib.rs, app.rs, gpu.rs, shader.wgsl) and
  `web/` (index.html, pkg/) — this directory is the project root

---
## Known Issues

### Intermittent sphere miss on page load
On some page loads the sphere fails to render and the canvas is
blank. Mrays reads ~115 instead of ~28 on affected loads.

**Failure mode (as of Step 6b):** The intermittent miss is
pre-shading. The BVH traversal kernel fires (hence the elevated
Mrays) but returns no hits, writing `F32_MAX` to all hit records.
The shading kernels correctly skip all pixels — they are not the
source of the blank canvas. The traversal kernel writes BACKGROUND
for misses, so the canvas is dark blue, not undefined. The geometry
simply isn't being found on that load.

Root cause not identified after extensive investigation; suspected
Dawn/Metal non-determinism on Chromium 148 / Apple M2. Reloading
resolves it (usually 1–3 attempts). Does not affect rendering
correctness when working.

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

### Geometry Buffer Format (Step 5.5)

Two new structs in `bvh.rs`. Both 32 bytes. Both `#[repr(C)]`,
`Pod`, `Zeroable`.

```rust
pub struct Vertex {
    pub position: [f32; 4],  // .xyz = position, .w = 0.0 (pad)
    pub normal:   [f32; 4],  // .xyz = normal,   .w = 0.0 (pad)
}
// 32 bytes. [f32;4] not [f32;3] — WGSL vec3<f32> has 16-byte
// alignment; [f32;4] keeps the struct a clean multiple of 16.

pub struct TriangleRecord {
    pub v0:                u32,  // index into vertex buffer
    pub v1:                u32,
    pub v2:                u32,
    pub front_material_id: u32,
    pub back_material_id:  u32,
    pub _pad:              [u32; 3],
}
// 32 bytes. Both material IDs present on every triangle —
// required for correct medium stack push/pop on ray entry/exit
// through glass surfaces.
```

Add `vertex_buf` and `geometry_buf` to `GpuState`. Allocate both
at startup with a minimal placeholder (one zeroed Vertex, one
zeroed TriangleRecord). No rendering changes — this step is data
structure and buffer allocation only.

### Winding Order Convention — DO NOT CHANGE WITHOUT AUDITING

All triangle geometry uses **counter-clockwise (CCW) winding order**
as seen from the front face (standard WebGPU/glTF convention). The
front face is the face whose outward normal points in the direction
consistent with CCW vertex order.

`HitRecord.face_forward == 1` means the ray struck the CCW (front)
face; `== 0` means the CW (back) face.

`TriangleRecord.front_material_id` corresponds to the CCW face;
`back_material_id` corresponds to the CW face. Medium stack push/pop
logic (entering vs. exiting a volume) depends entirely on this
convention. Do NOT change winding order without auditing all
face-orientation-dependent code: traversal kernel, shading kernels,
and any geometry generation or import code.

---

## BVH Architecture

### Never Cull Back Faces
The glass egg and glass obstacles are refractive — rays enter from outside (front faces)
and exit from inside (back faces of the same geometry). Back-face culling would break
nested dielectric medium tracking entirely. NEVER cull back faces for any geometry in beam.

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

## BVH Node Structure

### BvhNode Layout (Rust + WGSL)

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BvhNode {
    aabb_min_left_start:  [f32; 4],  // .xyz=aabb_min, .w=left_child|prim_start|sphere_index
    aabb_max_right_count: [f32; 4],  // .xyz=aabb_max, .w=right_child|prim_count|unused
    node_type:            u32,       // NODE_INTERNAL|NODE_LEAF_TRIANGLE|NODE_LEAF_SPHERE|NODE_LEAF_QUARTIC
    _reserved:            [u32; 3],  // reserved for future primitive types — do not repurpose
}
// 48 bytes total. WGSL struct must mirror exactly.
```

### Node Type Constants

```rust
const NODE_INTERNAL:      u32 = 0;  // left/right are child indices into node buffer
const NODE_LEAF_TRIANGLE: u32 = 1;  // .w fields are prim_start + prim_count into geometry buffer
const NODE_LEAF_SPHERE:   u32 = 2;  // .w of first field is sphere_index into sphere buffer
const NODE_LEAF_QUARTIC:  u32 = 3;  // reserved: future analytic egg surface
```

### Accessor Methods (Rust)

Implement `.to_bits()` accessors on `BvhNode` for all `.w` field interpretations.
Named accessors are load-bearing documentation — use `left_child()`, `prim_start()`,
and `sphere_index()` as distinct methods even though they read the same bits.
Never access `.w` fields directly in traversal or shading code.

### Traversal Stack

WGSL has no recursion. Traversal uses an explicit local stack:

```wgsl
var stack: array<u32, 32>;  // node indices, 32 deep
var stack_ptr: i32 = 0;
```

32 entries is conservative headroom for beam's scene. Do not increase without profiling.

### TLAS Instance Layout

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TlasInstance {
    transform:   [f32; 16],  // mat4x4 world transform, 64 bytes
    blas_offset: u32,        // index of BLAS root node in node buffer
    flags:       u32,        // reserved for static/dynamic kinematic switching
    _reserved:   [u32; 2],   // alignment
}
// 80 bytes total.
```

**TlasInstance — inverse transform (pinned):** When non-identity instance 
transforms arrive (egg geometry, Step 5.5+), add inv_transform: [f32; 16] 
to TlasInstance (144 bytes total). The WGSL traversal kernel currently 
applies transform directly — this works only for identity. The inverse is 
required for correct ray-to-local-space transformation for non-identity 
instances. Do not add until there is actual non-identity geometry to test against.

### Two-Level Buffer Layout

- Single node buffer contains both TLAS and BLAS nodes
- TLAS nodes occupy indices 0..tlas_node_count
- Each BLAS occupies a contiguous range starting at its `blas_offset`
- Sphere primitives live in a separate sphere buffer (center + radius, fixed size)
- Triangle primitives live in the geometry buffer (defined in Step 5.5)

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
- **Maximum depth: 4** (realistic maximum for beam geometry is 3, depth 4 is headroom)
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

## Hit Record Layout (Step 6b — Shading Split)

Traversal kernel writes one `HitRecord` per ray into a hit buffer.
Shading kernels read the hit buffer; they do not participate in traversal.

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct HitRecord {
    pub t:            f32,      // hit distance; f32::MAX = miss
    pub prim_idx:     u32,      // triangle index into geometry_buf
    pub bary_uv:      [f32; 2], // barycentric u, v (w = 1-u-v)
    pub face_forward: u32,      // 1 = ray hit CCW (front) face, 0 = CW (back) face
    pub _pad:         [u32; 3], // 32 bytes total
}
// 32 bytes. Clean multiple of 16.
// Miss condition: t == f32::MAX. No special buffer entry needed.
```

Material ID is NOT stored in HitRecord. Derive it in the shading kernel:
`TriangleRecord.front_material_id` if `face_forward == 1`, else
`back_material_id`. Redundancy between HitRecord and TriangleRecord
would create a maintenance hazard — keep the hit record minimal.

`face_forward` semantics depend on the CCW winding order convention.
See **Winding Order Convention** in Geometry Decisions above.

---

## Shading Kernels (Step 6b)

### Naming Convention
One WGSL file per material type, named `shade_<variant>.wgsl`
where `<variant>` is the `MaterialType` enum variant in lowercase:

- `shade_diffuse.wgsl`
- `shade_metallic.wgsl`
- `shade_glass.wgsl` (B04 or later)
- `shade_emissive.wgsl` (when needed)

Each shader gets its own compute pipeline and dispatch. Do NOT fold
multiple material types into one shader.

### Shared Utilities — shade_common.wgsl
Common functions used across shading kernels live in `shade_common.wgsl`.
This file is NOT a standalone compute shader — it is composed into each
`shade_<variant>` pipeline at pipeline creation time via wgpu's shader
module composition. It contains no entry point (`@compute` function).

Candidates for `shade_common.wgsl`:
- `hit_position(ray, t)` — reconstruct world-space hit point
- `interpolate_normal(hit_record)` — fetch vertex normals, interpolate
  via barycentrics, normalize
- `cosine_weighted_hemisphere(normal, seed)` — diffuse sample direction
- `offset_ray_origin(pos, normal)` — apply tmin epsilon in normal
  direction to prevent self-intersection
- Any other utility called by two or more shading kernels

Do NOT put glass-specific utilities (Fresnel, Snell, medium stack ops)
in `shade_common.wgsl` until glass is implemented. Add to common only
when a second consumer exists.

### RNG — Deterministic Hash, Not PRNG

Shading kernels derive random values via `hash_u32(seed)` in
`shade_common.wgsl`, where `seed` is derived from pixel index and
bounce count. This is intentionally deterministic — the same seed
always produces the same sample direction, giving bit-exact
reproducibility across runs, platforms, and driver versions.

This property is valuable for regression testing: a test that renders
a known scene and compares pixel output will produce identical results
across runs without any special setup. Plan integration tests with
this in mind.

Do NOT replace with a clock- or thread-seeded PRNG without considering
the impact on reproducibility and regression testing. The long-term
upgrade path is blue noise textures or a Halton sequence (see
Temporal Accumulation section) — both of which can be made
deterministic by design.

---

## Primary Ray Dispatch — Three Mechanisms

### 1. Sky Mask (implement after basic rendering works)
A large fraction of pixels (potentially 40-60%) are background pixels whose primary rays
miss the egg and hit the static HDR environment. These never need re-dispatching.

**TWO SHADERS, NOT ONE WITH A CONDITIONAL:**
- **Frame 0 shader:** fires all pixels, writes per-pixel sky mask (1 bit/pixel) flagging misses. Then runs a dilation pass: any miss-pixel within SKY_MASK_DILATION_RADIUS pixels (8-connected, Chebyshev distance) of a hit-pixel is promoted to hit and will be re-dispatched every frame. This preserves correct temporal accumulation and jitter-based antialiasing along the egg's silhouette edge — without dilation, edge pixels classified as misses on frame 0 would never accumulate the hit samples needed for smooth antialiasing. SKY_MASK_DILATION_RADIUS defaults to 1; increase to 2 or 3 to trade a slightly larger dispatch footprint for more conservative silhouette protection. Runs ONCE at scene initialization.
- **Per-frame shader:** reads the dilated sky mask, dispatches only egg-hitting pixels (including the dilated border) via stream compaction. Confirmed interior-miss pixels are permanently skipped. No conditional branch in hot path.

Invalidate and recompute sky mask if camera moves or egg moves.

**REJECTED ALTERNATIVES:**

**No dilation (null option):** rejected. "Good enough" on the silhouette of the primary visual object in a beauty-first game is not good enough. One pixel of hard aliasing on the egg's edge is visible and unacceptable.
**Probabilistic mask:** fire N probe rays per edge pixel, record hit probability, threshold for skip/dispatch. More accurate but significantly more complex. Revisit only if SKY_MASK_DILATION_RADIUS=1 proves insufficient for silhouette quality at game viewing distances.

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

### Material Buffer (Step 6 — Buffer Infrastructure)

Add `material_buf` to `GpuState`. Allocate at startup with a single
zeroed placeholder entry via `upload_slice`. Add to BG0 at binding 5,
following the established pattern (vertex_buf=3, geometry_buf=4,
material_buf=5).

Add a stub `@group(0) @binding(5)` declaration to `intersect.wgsl`,
marked `// Step 6 — declared, not yet used`. The Dawn backend rejects
a pipeline whose explicit layout declares bindings the shader doesn't
declare. The shader catches up to the layout, not the other way around.

#### MaterialType Enum

```rust
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MaterialType {
    Diffuse  = 0,
    Metallic = 1,  // conductor BSDFs — chrome ball, future metallic obstacles
    Glass    = 2,
    Emissive = 3,
}

// SAFETY: MaterialType is #[repr(u32)] with unit variants only.
// Memory layout is identical to u32 — no padding, no uninit bytes.
unsafe impl bytemuck::Zeroable for MaterialType {}
unsafe impl bytemuck::Pod     for MaterialType {}
```

`Metallic` routes to the conductor/mirror shading kernel. `roughness=0.0`
produces a perfect mirror; `roughness` is reserved for the "used universe"
ball aesthetic (reach goal). Do NOT infer material type from field values
— always use the discriminant to route to shading kernels.

#### Material Struct

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Material {
    pub base_color:    [f32; 4],      // .rgb=color, .w=unused
    pub emission:      [f32; 4],      // .rgb=emission, .w=unused
    pub absorption:    [f32; 4],      // .rgb=Beer's law coeff, .w=unused
    pub material_type: MaterialType,  // routes to shading kernel dispatch
    pub ior:           f32,           // index of refraction (1.0 for opaque)
    pub roughness:     f32,           // reserved; 0.0=smooth (reach goal)
    pub _pad:          f32,           // alignment padding
}
// 64 bytes total. Clean multiple of 16.
// WGSL struct must mirror exactly — use vec4<f32> for [f32;4] fields.
```

`[f32; 4]` not `[f32; 3]` — WGSL `vec3<f32>` has 16-byte alignment;
`[f32; 4]` keeps the struct a clean multiple of 16. Same constraint
as `Vertex` and `TriangleRecord`.

**Deferred:** `metallic: f32` field for continuous metallic blending
omitted. `MaterialType::Metallic` is sufficient for the chrome ball.
Add the field when partially-metallic obstacles require it.

#### Material Sorting (Planned Optimization — Not This Step)
Sort rays by `material_type` between traversal and shading kernels.
Separate shading dispatches for: Diffuse, Metallic, Glass, Emissive.
Reduces shading divergence. Implement after basic rendering works and
profiling motivates it. The `MaterialType` enum makes the sort key
unambiguous when the time comes.

### Glass BSDF (Dominant Material)
- Fresnel calculation + Snell's law refraction + medium stack push/pop
- Beer's law attenuation for colored glass: `throughput *= exp(-absorption * distance)`
  Absorption coefficient is vec3 (per RGB channel). Zero for clear glass.
- At dielectric surfaces: probabilistic choice (Russian roulette) between reflection and
  refraction using Fresnel reflectance as probability — NEVER split ray into two
  Splitting rays causes exponential explosion of ray count — always sample one path
- Do NOT cull back faces — both face orientations required for correct refraction

### Chrome Ball (Metallic Mirror)
- `MaterialType::Metallic`, `roughness=0.0` — perfect specular mirror, delta function BRDF
- Exact normal from analytic sphere: `normalize(hitPoint - center)`
- **"Used universe" aesthetic (reach goal):** roughness/anisotropy variation as material
  parameters for scratched chrome character — no geometry changes required

---

## Self-Intersection Prevention

Every spawned ray needs tmin epsilon — typically ~1e-4 in scene units.
Any intersection closer than tmin from ray origin is ignored.
This prevents floating point imprecision at ray origin from causing immediate re-hit
of the surface just left — distinct from the modeling coincident-surface problem.
Tune for beam's scene scale — too small: self-intersection artifacts; too large: misses
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
5  BVH scaffold (TLAS/BLAS structure, sphere only, no triangles)
5.5 Geometry buffer format (dual-material triangle record definition)
6  Material system — diffuse first, specular, then glass BSDF
7. Next event estimation — shadow rays, direct lighting
8. Sky mask — frame 0 initialization + per-frame masked dispatch
9. Temporal accumulation — with jitter, weighted accumulation buffer from the start
10. Denoiser — temporal accumulation only first, SVGF when needed
11. Tone mapping and bloom
12. Ball animation (Phase 2) — only after Phase 1 rendering is correct and validated
13. Kinematic switching (Phase 3) — only after Phase 2 works

**Step 5.5 — Geometry buffer format:** Define the dual-material triangle 
record struct before implementing the material system. Each triangle record 
carries front_material_id and back_material_id. The BVH leaf node references 
a range of records in this buffer by index. No rendering changes — this step 
is data structure definition only.

**Do not proceed to the next step until the current step produces correct, validated output.**

---

## Documentation

Update README.md to reflect current state when completing each
numbered implementation step.
