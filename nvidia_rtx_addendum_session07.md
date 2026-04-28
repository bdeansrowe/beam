# NVIDIA RTX Best Practices for ltbl — Session 07 Addendum

This document captures all amendments made to `nvidia_rtx_best_practices_for_ltbl.md`
during parley Session 07. Apply these changes to the existing document.

---

## Amendment 1 — Section 1.2, "Use triangles over AABBs"

**After the existing FOR ltbl paragraph, ADD:**

```
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
```

---

## Amendment 2 — Section 1.2, "Know when to update versus rebuild"

**Replace the existing FOR ltbl paragraph with:**

```
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
```

---

## Amendment 3 — Section 1.2, "Use compaction with all static geometry"

**After the existing PRINCIPLE THAT APPLIES paragraph, ADD:**

```
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
self-intersection without affecting legitimate nearby geometry.

The tmin value requires tuning for ltbl's scene scale. Too small and self-intersection
persists; too large and the renderer misses legitimate nearby geometry (e.g. a ray spawned
inside a thin glass obstacle that exits immediately). Start with 1e-4 and adjust based on
observed artifacts. See `ltbl_modeling_requirements.md` Requirement 6 for the related
coincident-surface modeling constraint.
```

---

## Amendment 4 — Section 2.2, "Keep attribute count low"

**Replace the existing PRINCIPLE THAT APPLIES paragraph with:**

```
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
```

---

## Amendment 5 — Section 2.2, "Ensure each ray-gen thread produces a ray"

**Replace the existing ltbl-SPECIFIC DISPATCH ARCHITECTURE section with:**

```
**ltbl-SPECIFIC DISPATCH ARCHITECTURE — THREE MECHANISMS:**

**1. Sky mask (background pixel elimination):**
A significant fraction of pixels — potentially 40-60% depending on framing — are background
pixels whose primary rays miss the egg entirely and simply sample the static HDR environment
map. These pixels never change (absent camera movement or egg nudging) and do not need to be
re-dispatched every frame.

Implementation: TWO separate primary ray shaders, not one with a conditional.
- **Frame 0 shader** — fires all pixels, writes a per-pixel sky mask (one bit per pixel)
  flagging misses. Runs once at scene initialization.
- **Per-frame shader** — reads the sky mask, dispatches only egg-hitting pixels via stream
  compaction. Background pixels are excluded entirely. No conditional branch in the hot path.

Why two shaders rather than one with a conditional: the frame 0 shader is logically part of
scene initialization (grouped with BVH construction and pipeline compilation). The per-frame
shader is part of the render loop. Keeping them separate reflects that logical distinction
and eliminates a uniform branch that would otherwise evaluate on every thread of every frame
for the lifetime of the session.

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
```

---

## Amendment 6 — Summary Section, Point 1

**Replace summary point 1 with:**

```
1. **Almost all scene geometry as triangles** in the BVH — no AABB primitives. Exception:
   the chrome ball is an analytic sphere primitive with a dedicated quadratic intersection
   code path — exact normals, no tessellation error, critical for chrome mirror quality.
   Do NOT tessellate the ball; the analytic representation is architecturally intentional.
```

---

## Amendment 7 — Summary Section, ADD Points 13 and 15

**After existing point 12, ADD:**

```
13. **Pre-allocate all GPU buffers** at startup — no runtime allocation in render loop.
    Ball BLAS exception: measure actual size on frame 0, use that exact allocation for
    all subsequent per-frame rebuilds — analytic sphere representation is fixed size,
    position never affects memory requirement.
14. **Instrument with timestamp queries early** — visibility into per-stage performance
    from the beginning
15. **tmin epsilon on all spawned rays** — offset minimum intersection distance (~1e-4 in
    scene units) to prevent self-intersection from floating point imprecision at ray origins.
    Tune for ltbl's scene scale — too small causes self-intersection artifacts, too large
    misses legitimate nearby geometry in thin glass obstacles.
```

---

## Amendment 8 — Pipeline Steps Section, Sub-pixel Jitter

**After the existing temporal accumulation paragraph, ADD:**

```
**Sub-pixel jitter (implement from the start):** Each sample's ray origin should be offset
by a randomly chosen sub-pixel location within the pixel bounds rather than always firing
from the pixel center. This gives temporal accumulation genuine new spatial information each
frame rather than repeatedly averaging the same point, and produces anti-aliasing as a natural
byproduct of accumulation. Use a Halton sequence or blue noise distribution for jitter offsets
rather than purely random values — better coverage of the pixel area over time with less
clustering.

**Implementation note for future extension:** The uniform jitter implementation should be
structured to be amenable to Gaussian filter importance sampling as a planned upgrade.
Specifically:
- Sample the jitter offset from a distribution function (initially uniform within pixel
  bounds) rather than hardcoding uniform sampling — swapping the distribution to Gaussian
  requires changing one function, not restructuring the pipeline
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
```

---

## Amendment 9 — Pipeline Steps Section, Particle Effects

**After the existing Display paragraph, ADD the entire particle effects section:**

```
**Note on particle effects (reach goal):** The shatter target particle effect is composited
into the frame as a screen-space overlay after the ray traced image is produced — not ray
traced. This is a reach goal with three quality tiers, togglable in game preferences, allowing
players with more capable hardware to opt into higher fidelity particle rendering.

**Tier 1 — Standard (all hardware manufactured after ~1985 with a functioning GPU):**
Screen-space composite at the particle's geometric screen position. Depth buffer used for
occlusion — particles behind geometry are correctly hidden. No refraction correction. Fast,
works everywhere. The explosion is brief and energetic enough that refractive inaccuracy is
not visually disqualifying.

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
```

---

## Amendment 10 — ADD Research / Optimization Experiments Section

**At the end of the document, ADD:**

```
---

## Research / Optimization Experiments (Post-Baseline, Requires Profiling Data)

These are ideas to explore once a working baseline renderer exists and profiling data is
available. Do not implement any of these before the baseline is established and measured.

**1. Polar Coordinate Acceleration Structure**

Recast ray space in polar coordinates centered on the camera. Geometry is organized by radial
shells (r_min, r_max from camera origin). Primary rays from a pinhole camera monotonically
increase in radius, so traversal becomes an ordered sequence of shell tests with no backtracking.

For ltbl specifically: the scene has natural radial structure (outer egg at one radius, inner
egg at larger radius, obstacles and ball at interior radii). Bounding spheres are the ideal
primitive for this representation — a sphere transforms cleanly between arbitrary polar origins
because its shape doesn't change, only its angular extent and radial interval. For ltbl's small
scene (~20-50 bounding spheres), the per-bounce-origin transformation of all spheres is trivially
parallel and likely cheap.

Limitation: breaks down for bounce rays, whose origin is no longer the camera. Proposed hybrid:
polar AS for primary rays and shadow rays (camera-origin and light-origin, the highest-volume
categories); conventional BVH fallback for bounce rays.

Status: genuine research direction, not an optimization. Only relevant after baseline profiling
shows primary ray traversal as a significant cost.

---

**2. Probe Ray Adaptive Sampling for Ball Region**

Two-pass importance sampling for the ball's apparent position through the refractive egg:

- Pass 1: fire a small number of cheap probe rays (intersection-test only, no shading, ~10-20%
  of full path tracing ray cost) toward the predicted apparent position of the ball
- Use the hit/miss pattern from probe rays to refine the spread estimate
- Pass 2: fire the full importance sample budget into the refined spread

Predicting apparent position: Rapier physics provides ball world-space position and velocity
each frame. Compute approximate apparent position on the egg surface via Snell's law. Use
velocity to scale spread (faster ball = wider spread). Probe rays validate and refine the
estimate before the expensive pass fires.

Status: implement only if profiling shows the ball region has noticeably higher variance than
the rest of the scene, indicating the fixed-spread importance sampling is either too wide
(wasting samples) or too narrow (missing the ball).

---

**3. Plane-Based Triangle Intersection**

Represent each triangle as the intersection of four planes: one support plane (the plane
containing the triangle) and three edge planes (each containing one edge, perpendicular to
the support plane). Ray-triangle intersection becomes: intersect ray with support plane, then
test the intersection point against the three edge planes with sign tests and early exit on
first failure.

**The miss-heavy argument for this approach:** In a well-built BVH, rays miss far more
triangles than they hit. By the time a ray reaches a leaf node and tests individual triangles,
the BVH has already culled the vast majority of the scene. Even at leaf nodes, hit rates are
typically in single-digit percentages. The early-exit on the first failed edge plane test
fires on misses — which is precisely the common case. Möller–Trumbore has no equivalent
early exit for the miss case.

**The counterargument:** Möller–Trumbore produces barycentric coordinates as a free byproduct
of the intersection test — needed for normal interpolation and UV lookup in the shading kernel.
The plane-based approach requires a subsequent barycentric computation on hits, partially
offsetting the savings from early exit on misses. Storage cost is also higher: 4 planes × 4
floats = 64 bytes per triangle vs ~48 bytes for indexed vertex representation.

**Net assessment:** The miss-heavy nature of BVH traversal makes the plane-based approach more
theoretically interesting than the storage cost alone would suggest. Whether the early-exit
savings on misses outweighs the barycentric recomputation cost on hits plus the storage overhead
is an empirical question specific to ltbl's geometry and BVH quality.

Status: implement only if profiling shows triangle intersection (not BVH traversal) as the
bottleneck. BVH traversal is far more commonly the bottleneck — measure before experimenting.
```

---

*End of Session 07 addendum. Apply all amendments to the existing
`nvidia_rtx_best_practices_for_ltbl.md` to produce the complete updated document.*
