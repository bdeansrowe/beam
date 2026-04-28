# NVIDIA RTX Best Practices — Annotated for ltbl

Source: https://developer.nvidia.com/blog/rtx-best-practices/ (2019 GDC presentation)

**Important architectural note:** This article was written for the DXR (DirectX Raytracing) and
Vulkan RT pipeline APIs, which expose dedicated hardware ray tracing units via specialized shader
stages (Ray Generation, Closest Hit, Any Hit, Miss shaders) and API-managed acceleration
structures. ltbl uses WebGPU compute shaders for all ray tracing work. Hardware RT units are NOT
exposed through WebGPU. All BVH construction and traversal is implemented manually in WGSL
compute shaders. API-specific advice does not apply, but the underlying principles often do.

---

## Section 1 — Acceleration Structure Management

### 1.1 General Practices

---

**ORIGINAL:** Move AS management (build/update) to an async compute queue.

**STATUS:** Does not apply as written — WebGPU does not expose async compute queues or the
ability to overlap BVH build with other GPU work at the API level.

**PRINCIPLE THAT APPLIES:** BVH construction is expensive. In ltbl, BVH builds should be
minimized — build once for static geometry at load time, only rebuild/refit dynamic geometry
(the pinball and any moving obstacles) per frame. Do not rebuild the full scene BVH every frame.

---

**ORIGINAL:** Build the TLAS rather than Update.

**STATUS:** Does not apply as written — ltbl manages its own two-level BVH (TLAS/BLAS) in
compute shaders rather than using API-managed acceleration structures.

**PRINCIPLE THAT APPLIES:** For ltbl's software BVH, prefer full TLAS rebuild over incremental
refit when instance transforms change significantly. Refit degrades BVH quality over time.
Since ltbl has very few dynamic instances (primarily the ball), the cost of rebuilding the TLAS
each frame is low and preferred over quality degradation from repeated refits.

---

**ORIGINAL:** Don't include the skybox/skysphere in your TLAS. Implement sky shading in the
Miss Shader instead.

**STATUS:** Gray area — ltbl has no Miss Shader (that's a DXR concept), but the principle is
directly applicable.

**REWRITTEN FOR ltbl:** Do not include the HDR environment map as scene geometry in the BVH.
When a ray escapes the scene without hitting any geometry (misses everything), sample the HDR
environment map directly in the path tracing kernel using the ray direction. This avoids
unnecessary BVH traversal cost for rays that will never hit scene geometry.

---

**ORIGINAL:** Implement a single barrier between BLAS and TLAS build.

**STATUS:** Does not apply as written — this is DXR API-specific synchronization advice.

**PRINCIPLE THAT APPLIES:** In WebGPU compute shader BVH construction, ensure BLAS build
dispatches are fully complete (via appropriate pipeline barriers / storage buffer barriers)
before the TLAS build dispatch begins, since TLAS construction reads BLAS data. Use the minimum
number of barriers necessary — don't over-synchronize.

---

### 1.2 Bottom-Level Acceleration Structures (BLAS)

---

**ORIGINAL:** Use triangles over AABBs. RTX GPUs excel in accelerating traversal of AS created
from triangle geometry.

**STATUS:** Directly applicable in principle, with context.

**FOR ltbl:** Almost all scene geometry should be represented as triangle meshes in the BVH,
not as axis-aligned bounding box primitives. Triangle intersection is faster and more precise.
This applies even though ltbl uses software BVH traversal — the same efficiency principle holds.
The Hügelschäffer egg outer and inner surfaces and all glass obstacles should be tessellated to
triangle meshes for BVH purposes.

**EXCEPTION — the chrome pinball:** The ball is represented as an analytic sphere primitive
rather than a tessellated mesh. A sphere has a closed-form ray intersection (quadratic solve)
that produces geometrically exact normals — computed as normalize(hitPoint - center) — with
zero tessellation error. For a chrome mirror surface where the entire visual is reflections,
normal accuracy matters more than almost anywhere else in the scene, and tessellation artifacts
in normals would produce visible waviness in reflections. The ball is flagged as a sphere
primitive in its BVH leaf node, with a dedicated quadratic intersection code path in the
traversal kernel. This is the one intentional exception to the triangles-over-AABBs principle
in ltbl — chosen because the visual payoff justifies the additional traversal kernel complexity,
and because a quadratic solve is still fast. Note: importance sampling concentrates more rays
toward the ball than elsewhere in the scene, making normal quality on the ball higher-value
than on any other object.

---

**ORIGINAL:** Mark geometry as OPAQUE whenever possible.

**STATUS:** Does not apply as written — OPAQUE is a DXR/Vulkan RT flag that bypasses the
Any Hit shader stage, which ltbl does not use.

**PRINCIPLE THAT APPLIES:** In ltbl's software BVH traversal, implement a fast path for opaque
geometry that skips any transparency or alpha testing logic. For the chrome ball and any fully
opaque geometry, avoid executing glass/dielectric material code during traversal. Reserve the
more expensive nested dielectric medium tracking logic for geometry flagged as refractive.

---

**ORIGINAL:** Group geometry into BLAS/instances following spatial locality. Do not throw
everything with the same material into the same BLAS regardless of spatial position.

**STATUS:** Directly applicable in principle.

**FOR ltbl:** When organizing the software BVH, group geometry by spatial proximity, not by
material type. The glass egg (outer surface, inner surface) should form one BLAS. The chrome
ball is one BLAS. Glass obstacles near each other spatially should be grouped. Do not create a
single "all glass geometry" BLAS regardless of where obstacles are located in the egg.

---

**ORIGINAL:** Know when to update versus rebuild. Continually updating a BLAS degrades its
quality as a spatial data structure.

**STATUS:** Directly applicable in principle.

**FOR ltbl:** The pinball is the primary dynamic object. Its BLAS should be rebuilt (not
incrementally updated) each frame since it moves continuously and potentially wildly. Static
geometry (egg shell, fixed obstacles, flippers in rest position) should be built once. Flippers
follow the same kinematic switching pattern as obstacles — static at rest, promoted to dynamic
during a flip, demoted back to static when the flip completes and the flipper returns to rest.

**KINEMATIC SWITCHING:** Obstacles (bumpers, knockdown targets, etc.) use a static/dynamic
promotion pattern. By default all obstacles live as static BLAS instances in the TLAS —
built once, compacted, no per-frame cost. When the ball contacts an obstacle and triggers
an animation (e.g. a bumper recoiling), that obstacle is promoted to dynamic — its BLAS
is rebuilt per-frame for the duration of the animation. Once the animation completes and
the obstacle returns to rest, it is demoted back to static with a final one-time rebuild
at the rest position. At any given frame, the only guaranteed dynamic object is the ball
— animating obstacles are a small transient subset of total scene geometry. This keeps
per-frame BVH rebuild cost minimal regardless of scene complexity.

---

**ORIGINAL:** Use compaction with all static geometry.

**STATUS:** Does not apply as written — compaction is a DXR API memory optimization.

**PRINCIPLE THAT APPLIES:** For ltbl's software BVH, build static geometry BVH nodes once and
do not allocate scratch/temporary buffers for them beyond initial construction. Static BVH data
should be packed tightly in GPU buffers with no wasted space, and those buffers should not be
reallocated at runtime.

**BALL BLAS SIZING OPTIMIZATION:** The ball's BLAS is rebuilt every frame but its geometry
never changes — it is an analytic sphere primitive with a fixed representation (center +
radius). The BVH node structure for a single analytic primitive is constant in size regardless
of the ball's position. On frame 0, build the ball's BLAS, measure its actual size, and record
that value. From frame 1 onward, allocate exactly that recorded size for every rebuild — no
worst-case sizing, no wasted memory, no measurement overhead. The ball moving does not change
how much memory its BVH requires.

**SELF-INTERSECTION / TMIN EPSILON:** Every ray spawned from a surface hit point has its
origin mathematically on the surface. Due to floating point imprecision, the origin may be
infinitesimally on the wrong side, causing the next intersection test to immediately re-hit
the same surface — corrupting the medium stack and producing incorrect refraction. This is
distinct from the coincident-surface modeling problem (handled by modeling discipline) and
exists even with perfectly clean geometry.

Solution: every ray is given a small minimum distance `tmin` — typically ~1e-4 in scene
units — and any intersection closer than `tmin` from the ray origin is ignored. This prevents
self-intersection without affecting legitimate nearby geometry. An alternative is to offset
the ray origin slightly along the surface normal at spawn time (into the new medium for
transmission, away from the surface for reflection).

The tmin value requires tuning for ltbl's scene scale. Too small and self-intersection
persists; too large and the renderer misses legitimate nearby geometry (e.g. a ray spawned
inside a thin glass obstacle that exits immediately). Start with 1e-4 and adjust based on
observed artifacts. See `ltbl_modeling_requirements.md` Requirement 6 for the related
coincident-surface modeling constraint.

---

### 1.3 Build Flags

**STATUS:** Entire section does not apply — these are DXR/Vulkan RT API build flags
(PREFER_FAST_TRACE, PREFER_FAST_BUILD, ALLOW_UPDATE, ALLOW_COMPACTION, MINIMIZE_MEMORY) with
no WebGPU equivalents. The underlying trade-offs (build quality vs. build speed, memory vs.
performance) are real and worth understanding conceptually, but there are no flags to set.

---

## Section 2 — Ray Tracing

### 2.1 Pipeline Management

---

**ORIGINAL:** Avoid State Object creation on the critical path. PSO compilation can take
20ms–300ms.

**STATUS:** Does not apply as written — DXR/Vulkan RT pipeline state objects (PSOs) have no
direct WebGPU equivalent in this context.

**PRINCIPLE THAT APPLIES:** Compile all WebGPU compute shader pipelines (ray generation,
intersection/traversal, shading kernels, denoising, tone mapping) at startup, not lazily during
rendering. Pipeline compilation in WebGPU can stall the render loop if triggered at runtime.

---

**ORIGINAL:** Consider using more than one ray tracing pipeline for different ray types
(shadow rays vs reflection rays) to improve scheduling efficiency.

**STATUS:** Does not apply as written — separate DXR pipelines are not a WebGPU concept.

**PRINCIPLE THAT APPLIES (IMPORTANT FOR WAVEFRONT ARCHITECTURE):** ltbl uses wavefront path
tracing with separate compute dispatches per stage. This is the WebGPU equivalent of separate
pipelines per ray type. Shadow/occlusion rays (for next event estimation) should be a separate
compute kernel from the main path tracing kernel, since they are simpler (early termination on
any hit, no material evaluation needed) and benefit from separate dispatch to improve GPU
occupancy. Do not fold shadow ray logic into the main path tracing kernel.

---

**ORIGINAL:** Set payload and attribute sizes to the minimum possible. Large payloads increase
register pressure and spill to memory.

**STATUS:** Gray area — DXR ray payloads have no direct equivalent, but the principle maps
directly to ltbl's wavefront ray buffer design.

**REWRITTEN FOR ltbl:** Keep per-ray data in the wavefront ray buffers as compact as possible.
Each ray record should contain only what is strictly needed for the current stage: origin,
direction, tmin/tmax, and a packed material/medium state. Avoid storing intermediate shading
results in ray records if they can be recomputed. Large ray records increase memory bandwidth
pressure across all wavefront dispatches. Pack ray data using appropriate types (f16 where
precision allows, packed normals, etc.).

**MEDIUM STACK:** The per-ray medium stack — which tracks which refractive volumes the ray
is currently inside, required for correct nested dielectric rendering of the glass egg and
glass obstacles — is part of the ray record and directly affects its size. For ltbl's geometry,
the maximum realistic nesting depth is 3 (e.g. egg wall + embedded obstacle + interior air
as an explicit medium). A stack depth of 3 with one entry of headroom gives a maximum depth
of 4. Each stack entry holds a material ID and IOR value — approximately 8 bytes per entry,
so 32 bytes for a depth-4 stack. This is a fixed, known cost that should be accounted for
in ray record sizing. Do not increase maximum stack depth beyond what ltbl's geometry requires
— every extra entry adds to every ray record across the entire wavefront buffer.

**MODELING DEPENDENCY:** Correct medium stack behavior depends on specific geometry modeling
discipline. The renderer assumes explicit boundary surfaces between all adjacent or embedded
dielectric volumes — no intersecting solids, no implicit medium transitions. See
`ltbl_modeling_requirements.md` for the full set of renderer-derived modeling constraints.

---

**ORIGINAL:** Set max trace recursion depth to the minimum possible.

**STATUS:** Does not apply as written — recursion depth is a DXR pipeline parameter. ltbl's
wavefront architecture is iterative, not recursive.

**PRINCIPLE THAT APPLIES (CRITICAL FOR ltbl):** ltbl targets 8–16 bounces for glass geometry.
The wavefront loop should enforce a hard maximum bounce count per path. For the glass egg with
colored glass obstacles, 8 bounces is a reasonable starting value — increase only if caustic
quality demands it. Each additional bounce is a full additional compute dispatch. Profile before
increasing. Russian roulette path termination should be implemented to probabilistically
terminate low-contribution paths before the maximum bounce count, recovering performance for
paths that exit the egg early or hit the chrome ball.

---

### 2.2 Shaders

---

**ORIGINAL:** Keep the ray payload small. Payload size translates to register count and
affects occupancy. Pack payload data like a GBuffer.

**STATUS:** See rewritten version above (under 2.1 payload/attribute sizes). Same principle,
same application to wavefront ray buffers.

---

**ORIGINAL:** Keep attribute count low for custom intersection shaders.

**STATUS:** Does not apply as written — custom intersection shader attributes are a DXR
concept. ltbl performs triangle intersection in compute shaders.

**PRINCIPLE THAT APPLIES:** In ltbl's traversal kernel, minimize the data computed and passed
forward per intersection. Defer normal interpolation, UV lookup, and material property
evaluation to the shading kernel. Do not compute shading data inside the traversal kernel.

The data passed forward differs by primitive type:

**Triangle intersections:** hit distance, primitive index, barycentric coordinates (u, v).
Normal interpolation and UV lookup are deferred to the shading kernel, which reconstructs
them from the primitive index and barycentric coordinates plus the geometry buffers.

**Analytic sphere intersection (the chrome ball):** hit distance, and a flag identifying
the sphere primitive. The exact surface normal is trivially computed in the shading kernel
as normalize(hitPoint - sphereCenter) — there is no need to pass the normal forward from
the traversal kernel, and no barycentric coordinates exist for a sphere. The hit point
itself can be reconstructed from ray origin + ray direction * hit distance, so only the
hit distance and primitive identifier need to be passed forward.

The sphere case is actually cleaner than the triangle case — less data, simpler shading
kernel reconstruction. This is one of the advantages of the analytic sphere representation.

**"USED UNIVERSE" AESTHETIC (reach goal):** A perfect mirror sphere is visually correct but
aesthetically sterile. Real pinballs accumulate micro-scratches, worn patches, and subtle
surface character from repeated impacts — the visual language of an object that has been
somewhere and done something. This aesthetic depth is worth adding as a reach goal once the
basic chrome mirror is working.

Implementation: roughness and anisotropy variation baked into the ball's material parameters
rather than its geometry — the analytic sphere stays analytic. A roughness map in sphere
surface coordinates (procedural or texture-based) varies the specular roughness across the
ball's surface. Where roughness is zero: perfect mirror. Where slightly non-zero: glossy
highlight with spread reading as scratched or worn chrome. Directional scratch character
comes from an anisotropic BSDF — reflections elongated along scratch directions rather than
perfectly circular. The visual payoff is significant since the player watches the ball
constantly and the added character will read clearly. The architecture supports this as a
material parameter enrichment requiring no geometry or intersection changes.

---

**ORIGINAL:** Use RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH for shadow and AO rays.

**STATUS:** Does not apply as written — this is a DXR ray flag.

**REWRITTEN FOR ltbl:** In the shadow/occlusion ray kernel (used for next event estimation),
terminate BVH traversal immediately upon finding any valid intersection between the shading
point and the light sample. Do not find the closest hit — any hit is sufficient to determine
occlusion. This is a critical optimization: implement early-exit traversal for shadow rays
rather than finding the nearest intersection. Note that for ltbl's glass geometry, a shadow ray
hitting a glass obstacle is NOT fully occluded — the glass transmits some light. The shadow
ray kernel must handle partial transmittance through glass, but can still terminate early once
accumulated transmittance falls below a threshold (e.g. < 0.001).

---

**ORIGINAL:** Avoid live state across TraceRay calls to prevent register spills.

**STATUS:** Does not apply as written — TraceRay is a DXR construct. ltbl's wavefront
architecture eliminates this problem by design.

**NOTE:** This is one area where the wavefront architecture is inherently superior to the
megakernel or recursive approach. Because each stage is a separate dispatch, there is no live
state carried across kernel boundaries — state is explicitly written to and read from the
wavefront ray buffers. This is a fundamental advantage of the wavefront approach.

---

**ORIGINAL:** Avoid too many TraceRay calls in a single shader.

**STATUS:** Does not apply as written — same reason as above.

**NOTE:** Again, the wavefront architecture handles this by construction. Each compute dispatch
fires exactly one category of ray. Shadow rays are a separate dispatch from primary rays.

---

**ORIGINAL:** Try to execute TraceRay calls unconditionally.

**STATUS:** Does not apply as written.

**PRINCIPLE THAT APPLIES:** In ltbl's compute kernels, minimize conditional branching around
BVH traversal calls. Where possible, all threads in a workgroup should execute the traversal
path. Use data-driven early termination (e.g. checking accumulated throughput against a
threshold) rather than complex conditional logic to skip traversal for inactive paths. Inactive
paths (terminated by Russian roulette or having escaped the scene) should be handled by
compaction between wavefront stages rather than conditional no-ops within a kernel.

---

**ORIGINAL:** Use RAY_FLAG_CULL_BACK_FACING_TRIANGLES judiciously. Unlike rasterization,
backface culling in ray tracing is usually not a performance optimization.

**STATUS:** Does not apply as written.

**PRINCIPLE THAT APPLIES (IMPORTANT FOR ltbl):** Do NOT cull back-facing triangles during BVH
traversal in ltbl. The glass egg and glass obstacles are refractive — rays will enter from the
outside (hitting front faces) and exit from the inside (hitting back faces of the same geometry).
Both faces are required for correct dielectric refraction. Back-face culling would break the
nested dielectric medium tracking entirely.

---

**ORIGINAL:** Ensure each ray-gen thread produces a ray. Unused threads harm scheduling.

**STATUS:** Does not apply as written — Ray Generation shaders are a DXR concept.

**PRINCIPLE THAT APPLIES:** In ltbl's primary ray generation compute kernel, ensure all
dispatched threads are productive — the dispatch size should match the active pixel count
exactly, with no threads that immediately exit due to having no work. ltbl implements this
through three complementary mechanisms described below.

**ltbl-SPECIFIC DISPATCH ARCHITECTURE — THREE MECHANISMS:**

**1. Sky mask (background pixel elimination):**
A significant fraction of pixels — potentially 40-60% depending on framing — are background
pixels whose primary rays miss the egg entirely and simply sample the static HDR environment
map. These pixels never change (absent camera movement or egg nudging) and do not need to be
re-dispatched every frame.

Implementation: two separate primary ray shaders, not one with a conditional.
- **Frame 0 shader** — fires all pixels, writes a per-pixel sky mask (one bit per pixel)
  flagging misses. Runs once at scene initialization.
- **Per-frame shader** — reads the sky mask, dispatches only egg-hitting pixels via stream
  compaction. Background pixels are excluded entirely. No conditional branch in the hot path.

The sky mask must be invalidated and recomputed if camera position changes or the egg moves
(nudge feature, planned as a reach goal).

**2. Variable sample count / foveated sampling:**
Not all egg-hitting pixels need the same number of samples per frame. Two areas receive
elevated sample counts:
- **Ball region:** The chrome ball is the player's primary focus. Importance sampling
  concentrates additional rays toward the ball's apparent position through the refractive
  egg surface. Naive importance sampling toward the ball's geometric position fails because
  refraction through the egg displaces the ball's apparent position — rays fired at the
  geometric position are bent away from the ball by the egg wall. The solution uses Rapier
  physics data (ball position + velocity) to compute the approximate apparent position on
  the egg surface, then fires importance rays toward that position with a spread sized to
  the ball's angular subtense plus refraction uncertainty margin. Ball velocity scales the
  spread — faster ball, wider spread. A two-pass probe ray refinement is a planned
  optimization: a small number of cheap intersection-only probe rays refine the spread
  estimate before the full importance sample budget is fired. Probe rays do full refraction
  computation but no shading — approximately 10-20% of the cost of a full path tracing ray.
  Implement the simpler Rapier-position-based approach first; add probe ray refinement if
  profiling shows the ball region has noticeably higher variance than the rest of the scene.
- **Active gameplay regions:** Areas with recent ball contact, active bumpers, or flipper
  activity may warrant elevated samples during those events.

Variable sample counts are handled via a work queue rather than a uniform dispatch — pixels
with N samples contribute N entries to the work queue, ensuring all dispatched threads have
productive work regardless of per-pixel sample variation.

**3. Temporal accumulation:**
Pixels that have already converged — typically static background regions of the egg interior
with no recent ball activity — can reduce their per-frame sample contribution and rely on
temporal accumulation to maintain quality. The accumulation buffer blends current frame
samples with history, amortizing sample cost across frames. This interacts with the sky mask:
background pixels accumulate once (frame 0) and are not re-sampled; interior pixels accumulate
progressively across frames with sample counts modulated by activity.

**IMPLEMENTATION ORDER:**
1. First pass: uniform dispatch of all egg-hitting pixels, fixed sample count — establish
   correct rendering before any of these optimizations
2. Add sky mask — two shaders, frame 0 initialization, per-frame masked dispatch
3. Add temporal accumulation
4. Add variable sample counts and foveated sampling toward ball region
5. Add probe ray refinement for ball importance sampling

---

**ORIGINAL:** Keep any-hit shaders minimalistic — they execute many times per TraceRay and
run at highest register pressure.

**STATUS:** Does not apply as written — Any Hit shaders are a DXR stage.

**PRINCIPLE THAT APPLIES:** In ltbl's BVH traversal kernel, any per-candidate-intersection
test that runs before accepting a hit (equivalent to any-hit logic) should be kept minimal.
For glass geometry alpha testing (if any transparency cutout is used), the per-candidate test
should be a single texture lookup and comparison — nothing more. Expensive material evaluation
belongs in the shading kernel after the closest hit is confirmed.

---

**ORIGINAL:** Shading divergence — start with straightforward shading implementation, then
address divergence. Consider simplified shaders for RT, reduced texture resolution, deferred
lighting.

**STATUS:** Directly applicable — shading divergence is a fundamental GPU concern regardless
of API.

**FOR ltbl:** Wavefront path tracing partially mitigates shading divergence by sorting rays by
material type before shading dispatches — all threads in a dispatch evaluate the same material
class. However, within a material class there will still be parameter variation. Start with
the simplest correct material implementation. Profile for divergence before optimizing.
For the glass BSDF (the dominant material in ltbl), the Fresnel calculation and refraction
direction are deterministic given the ray direction and normal — divergence comes primarily
from the medium stack state (which medium the ray is currently inside). Keep medium stack
depth small (maximum 4 nested dielectrics should be sufficient for ltbl's geometry).

**IMPORTANT FOR ltbl:** Manual sorting/binning of shading work by material type is a planned
wavefront optimization. Implement a material ID sort between the traversal kernel and the
shading kernel. Glass geometry, chrome (mirror), emissive, and any diffuse elements should be
shaded in separate dispatches or at minimum sorted within a dispatch.

---

### 2.3 Resources

---

**ORIGINAL:** Use global root signature / global resource bindings for scene-global resources.

**STATUS:** Does not apply as written — root signatures are a DXR/D3D12 concept.

**PRINCIPLE THAT APPLIES:** In WebGPU compute shaders, bind scene-global resources (BVH
buffers, geometry vertex/index buffers, material buffer, environment map) in bind group 0.
Per-pass resources (ray buffers, output textures) go in bind group 1. This matches WebGPU
best practices and minimizes bind group switching overhead between wavefront dispatches.

---

**ORIGINAL:** Avoid resource temporaries — they cause code duplication and inefficiency.

**STATUS:** Directly applicable to WGSL compute shader authoring.

**FOR ltbl:** In WGSL shaders, avoid conditional resource selection patterns that cause the
compiler to duplicate sampling operations. Use array indexing to select materials and textures
dynamically (e.g. `materials[hit.material_id]`) rather than conditional assignments.

---

**ORIGINAL:** Prefer StructuredBuffer over ByteAddressBuffer for aligned raw data.

**STATUS:** Does not apply as written — these are HLSL/DXR buffer types.

**PRINCIPLE THAT APPLIES:** In WebGPU/WGSL, use typed storage buffers with explicit struct
layouts for all ray data, BVH nodes, and geometry data. Avoid raw byte-addressed buffer
patterns. Define explicit WGSL structs for ray records, BVH nodes, triangle data, and material
parameters. This enables the compiler to generate efficient vectorized loads.

---

## Section 3 — Denoisers

---

**ORIGINAL:** Use the NVIDIA RTX Denoiser SDK.

**STATUS:** Does not apply — the NVIDIA RTX Denoiser SDK is a native library unavailable in
WebGPU/WASM.

**FOR ltbl:** A denoiser is planned in the pipeline (path trace → temporal accumulation →
denoiser → tone mapping → bloom → display). For WebGPU, the denoiser must be implemented as
a compute shader. Practical options in order of complexity:
1. **Temporal accumulation only** (simplest — blend current frame with history, effective for
   static or slow-moving scenes)
2. **SVGF (Spatiotemporal Variance-Guided Filtering)** — the standard academic/indie path
   tracer denoiser, implementable in compute shaders, good quality
3. **A-SVGF** — improved version of SVGF with better temporal stability

Start with temporal accumulation. Add SVGF if noise remains unacceptable at target sample
counts.

---

## Section 4 — Memory Management

---

**ORIGINAL:** Various DXR/D3D12 memory management advice (Command Allocators, QueryVideoMemory,
resource heaps, etc.)

**STATUS:** Entire section does not apply — all DXR/D3D12 specific. WebGPU manages memory
at a higher abstraction level.

**PRINCIPLE THAT APPLIES:** Pre-allocate all GPU buffers at startup. Do not allocate or
deallocate GPU buffers during the render loop. Size wavefront ray buffers for the maximum
expected ray count (width × height × max_samples_per_pixel). Reuse scratch buffers between
wavefront stages where buffer lifetimes don't overlap.

---

## Section 5 — Profiling and Debugging

---

**ORIGINAL:** Use NVIDIA Nsight Graphics and Nsight Systems for profiling.

**STATUS:** Does not apply directly — Nsight tools profile native GPU applications, not
WebGPU/WASM running in a browser.

**FOR ltbl:** Primary profiling tools for WebGPU in Chromium (Vivaldi):
- **chrome://tracing** — GPU timeline profiling including compute dispatch timing
- **WebGPU timestamp queries** — if enabled via the timestamp-query feature, can measure
  individual compute dispatch durations from within the application
- **Vivaldi DevTools Performance tab** — coarse frame timing

Instrument each wavefront stage with timestamp queries early so performance data is available
as the renderer grows in complexity.

---

## Summary: Key Principles for ltbl

These are the highest-priority takeaways that directly govern ltbl implementation:

1. **Almost all scene geometry as triangles** in the BVH — no AABB primitives. Exception:
   the chrome ball is an analytic sphere primitive with a dedicated quadratic intersection
   code path — exact normals, no tessellation error, critical for chrome mirror quality.
2. **Never cull back faces** — refractive geometry requires both face orientations
3. **Separate dispatch for shadow/occlusion rays** with early-exit traversal
4. **Keep ray records compact** — pack aggressively, defer computation to shading stage.
   Medium stack (max depth 4) is part of the ray record — approximately 32 bytes fixed cost.
5. **Static geometry BVH built once** — only rebuild dynamic geometry (ball, active flippers)
   per frame. Use kinematic switching for obstacles: obstacles live as static BLAS instances
   in the TLAS by default; promote to dynamic (per-frame rebuild) only when actively animating
   (e.g. a bumper recoiling from ball contact); demote back to static once animation completes
   and the obstacle returns to rest. At any given frame, the only guaranteed dynamic object is
   the ball — animating obstacles are a small transient subset of total scene geometry.
6. **Prefer TLAS rebuild over refit** for the few dynamic instances
7. **Environment map sampled on miss** — not included in BVH
8. **Medium stack depth ≤ 4** for nested dielectric tracking
9. **Compile all pipelines at startup** — never on the render critical path
10. **Bind group 0 for scene globals, bind group 1 for per-pass resources**
11. **Russian roulette termination** to recover performance from low-contribution paths
12. **SVGF denoiser** as target — start with temporal accumulation, add SVGF when needed
13. **Pre-allocate all GPU buffers** at startup — no runtime allocation in render loop.
    Ball BLAS exception: measure actual size on first frame, use that exact allocation for
    all subsequent per-frame rebuilds — geometry count is fixed so size never changes.
14. **Instrument with timestamp queries early** — visibility into per-stage performance from
    the beginning
15. **tmin epsilon on all spawned rays** — offset minimum intersection distance (~1e-4 in
    scene units) to prevent self-intersection from floating point imprecision at ray origins.
    Tune for ltbl's scene scale — too small causes self-intersection artifacts, too large
    misses legitimate nearby geometry in thin glass obstacles.

---

## Pipeline Steps Outside NVIDIA Article Scope

The following pipeline stages are part of ltbl's planned render pipeline but are not covered
by the NVIDIA RTX Best Practices article, which focuses on ray tracing and acceleration
structure concerns. They are noted here for completeness.

**Full pipeline order:** path trace → temporal accumulation → denoiser → tone mapping →
bloom → display

**Temporal accumulation:** Blends current frame samples with history frames to amortize
sample cost across frames. Interacts with the sky mask — background pixels accumulate once
on frame 0 and are not re-sampled. Interior pixels accumulate progressively with sample
counts modulated by activity level.

**Sub-pixel jitter (implement from the start):** Each sample's ray origin should be offset
by a randomly chosen sub-pixel location within the pixel bounds rather than always firing
from the pixel center. This gives temporal accumulation genuine new spatial information each
frame rather than repeatedly averaging the same point, and produces anti-aliasing as a natural
byproduct of accumulation. Use a Halton sequence or blue noise distribution for jitter offsets
rather than purely random values — better coverage of the pixel area over time with less
clustering.

**Implementation note for future extension:** The uniform jitter implementation should be
structured to be amenable to Gaussian filter importance sampling as a planned upgrade. Specifically:
- Sample the jitter offset from a distribution (initially uniform within pixel bounds) rather
  than hardcoding uniform sampling — swapping the distribution to Gaussian requires changing
  one function, not restructuring the pipeline
- The accumulation buffer should store weighted sum and sum-of-weights per pixel separately
  (not a simple running average) — this is required for Gaussian weighted accumulation and
  costs nothing to implement correctly from the start
- Ray results should be written to the accumulation buffer with an explicit sample weight
  parameter (initially 1.0 for uniform jitter) — the Gaussian extension just changes this
  weight to the Gaussian evaluation at the sample offset

**Gaussian filter importance sampling (planned extension):** Sample ray origins from a
Gaussian distribution centered on the pixel with sigma ~0.5-1.0 pixels, allowing a small
percentage of rays to fire from neighboring pixel space. Samples carry weights proportional
to the Gaussian value at their offset. Rays landing outside the firing pixel's bounds
contribute to the neighboring pixel's weighted accumulation buffer. This bakes reconstruction
filtering directly into the sampling distribution — producing anti-aliasing on curved surfaces,
refractive boundaries, and specular highlights (all prominent in ltbl) without a post-process
filter pass. The Mitchell-Netravali filter (a Gaussian with a slight negative lobe for
subjective sharpness) is worth considering over a pure Gaussian as a refinement.

Implement uniform jitter first to establish baseline. Upgrade to Gaussian filter importance
sampling once the accumulation pipeline is working correctly.

**Tone mapping:** Khronos PBR Neutral tone mapper. Converts HDR linear rendering output
to display-ready values. Implemented as a post-process compute or fragment shader after
denoising.

**Bloom:** A post-process screen-space effect applied after tone mapping. Adds the
characteristic glow around bright emissive elements. Implemented as a separable Gaussian
blur or similar on the bright regions of the tone-mapped image. Not part of the ray tracing
pipeline — purely a screen-space composite step.

**Display:** Final blit to the canvas. In the current two-pass architecture this is already
implemented as a tonemap blit — this stage will expand to include bloom compositing as that
feature is added.

**Note on particle effects (reach goal):** The shatter target particle effect is composited
into the frame as a screen-space overlay after the ray traced image is produced — not ray
traced. This is a reach goal with three quality tiers, togglable in game preferences, allowing
players with more capable hardware to opt into higher fidelity particle rendering.

**Tier 1 — Standard (all hardware):** Screen-space composite at the particle's geometric
screen position. Depth buffer used for occlusion — particles behind geometry are correctly
hidden. No refraction correction. Fast, works everywhere. The explosion is brief and energetic
enough that refractive inaccuracy is not visually disqualifying.

**Tier 2 — Enhanced (moderate hardware):** Single-bounce apparent position correction. For
each particle, one ray is traced from the camera through the egg surface to the particle's
world-space position, computing the correct refracted apparent screen position via Snell's law.
The particle sprite is composited at the corrected position rather than its geometric screen
position — particles appear in the right place as seen through the glass. Beer's law
attenuation along the refracted path optionally applied for color tinting through the egg wall.
Cost is approximately 5-10% of full ray tracing — one refraction computation per particle
pixel, no material evaluation, no further bounces. Reuses egg surface intersection and
refraction infrastructure already built for the path tracer.

**Tier 3 — Ultra (beefy hardware, reach goal of the reach goal):** Full path-traced glass
shard particles with correct caustics, internal reflections, and glass BSDF evaluation.
Particles added to the BVH, intersected by rays, shaded with multiple bounces. Expensive
and stunning. Viable on discrete GPU hardware. Probably not on M2 Air.

**Toggle architecture:** Exposed in game preferences as a quality setting. The three tiers
produce visibly distinct results that justify the distinction — Standard looks fine, Enhanced
looks noticeably correct, Ultra looks like magic. Players with enthusiast hardware get a
meaningfully better visual experience as a reward for their investment.

**Implementation note:** Tier 2 reuses core path tracer components — egg surface intersection
and Snell's law refraction — in a stripped-down particle refraction pass. No new fundamental
machinery required. Tier 3 requires particles to be BVH residents, which is the primary
architectural addition.

**Adaptive quality (implement when Tier 2 is implemented, not before):** An automatic
downgrade system monitors frame time continuously. If frame budget is being exceeded at the
moment a shatter event triggers, the system falls back one tier for the duration of the effect,
then restores the player's preferred tier. Implementation: sample the last N frame times at
effect trigger time; if average exceeds threshold, drop one tier for this effect. The decision
is made once per effect event and held for its duration — not adjusted mid-explosion.

Player control model: manual preferred tier in settings as the primary control; automatic
downgrade as an optional safety net (can be disabled by players who prefer consistent quality
over consistent frame rate). A brief on-screen indicator when the system has downgraded keeps
the player informed rather than confused about why an effect looked different.

Requires frame timing infrastructure (timestamp queries) which will be built anyway for
performance monitoring — the adaptive logic is a small addition once that infrastructure exists.
No value in implementing before Tier 2 exists, since there is nothing to adapt between.
