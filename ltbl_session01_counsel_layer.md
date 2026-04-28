# ltbl — Session 01 Counsel Layer
## What CLAUDE.md Doesn't Tell You (And Why It Matters)

This document captures the reasoning, rejected alternatives, provenance, and constraint
rationale from Session 01 that didn't make it into CLAUDE.md or the NVIDIA annotations
document. It is the difference between "what was decided" and "why it was decided" —
the counsel layer that allows an agentic model to generalize decisions correctly to cases
not explicitly covered, and to resist accidentally undoing correct decisions while "helping."

---

## On the Architecture as a Whole

### Why Wavefront, Not Megakernel

**Decision:** Wavefront path tracing with separate compute kernels per stage.

**The reasoning chain that produced this:**
- WebGPU does not expose hardware RT units — all BVH traversal is software in WGSL
- Without hardware RT, the megakernel advantage (hardware manages ray state) doesn't apply
- Wavefront's core benefit — reducing shader divergence by keeping all threads in a dispatch doing similar work — applies equally or more so when traversal is software
- A secondary benefit emerged during design: wavefront eliminates the "live state across TraceRay" problem entirely by design. Each dispatch is stateless relative to previous dispatches; state is explicitly written to and read from ray buffers. This is a fundamental architectural advantage, not just a performance optimization.

**What was explicitly rejected:**
- Megakernel architecture: rejected because it carries live state across bounce iterations, causing register spill and occupancy problems that get worse with each additional bounce. For a glass-heavy scene targeting 8+ bounces, this is unacceptable.
- Fragment shader ray tracing: not discussed as a serious option — compute shaders are the correct choice for this workload.

**The constraint that makes this load-bearing:** ltbl targets 8+ bounces for glass geometry. The wavefront architecture's per-stage dispatch means each additional bounce is one additional compute dispatch, not an exponentially more complex shader. This is why the bounce count is a tunable parameter rather than an architectural constraint.

---

### Why RGB, Not Spectral (Yet)

**Decision:** RGB path tracing for first implementation; spectral as reach goal.

**The full reasoning:**
- Spectral rendering (tracking wavelength distributions rather than RGB triples) produces physically correct dispersion — the prismatic color fringing at glass edges — that RGB cannot reproduce
- For ltbl's glass egg specifically, spectral rendering would produce noticeably better caustics and colored fringes that are directly relevant to the visual appeal
- However: spectral rendering requires carrying N wavelength samples per ray (typically 4-8 with hero wavelength sampling) vs 3 RGB values, with all material evaluations extended to spectral. Render cost increases 2-3x.
- The decision: start with RGB because it is the correct starting point regardless, establish a working baseline, and add spectral specifically for the glass BSDF if/when the visual quality of caustics demands it.

**What spectral would unlock that RGB cannot:**
- Wavelength-dependent IOR → prismatic color separation through glass
- Physically correct colored caustics
- Iridescence (not planned for ltbl but theoretically possible)

**Why "fast enough" is the right question, not "is it faster":** This framing emerged explicitly during the analytic sphere discussion and applies throughout — a slightly slower technique that produces visibly better results on the geometry the player watches most is always worth considering. The question is never absolute speed but whether it meets the frame budget.

---

## On the Ball

### Why Analytic Sphere

**Decision:** Ball is analytic sphere primitive, not tessellated mesh. This is the ONE intentional exception to triangles-over-everything.

**The full reasoning chain:**
1. Chrome mirror surface is uniquely sensitive to normal accuracy — the entire visual is reflections, and the reflection direction is determined entirely by the surface normal (delta function BRDF)
2. Tessellated spheres have faceted normals that require interpolation — even with smooth normal interpolation, there is approximation error
3. Analytic sphere gives exact normals via `normalize(hitPoint - center)` — zero error by construction
4. A small normal error on a diffuse surface produces a small shading error. A small normal error on a mirror produces a visibly wrong reflection direction. The stakes are highest on the chrome ball.
5. Importance sampling concentrates more rays toward the ball than anywhere else in the scene — normal quality on the ball is higher-value than anywhere else precisely because it gets the most rays.
6. Quadratic solve for ray-sphere intersection is fast and closed-form — the cost of the separate code path is low.
7. WebGPU compute shaders have no hardware penalty for analytic primitives (unlike DXR where analytic primitives go through the Any Hit path, which is slower). We're writing the intersection code ourselves.

**What was explicitly rejected:**
- Tessellated sphere with high polygon count: rejected because the visual payoff of analytic normals on a chrome mirror justifies the implementation complexity, and the cost of the quadratic solve is negligible.
- The naive "just move the nearest control point" approach (from the developer's NURBS work): mentioned as exactly the wrong instinct — it produces the wrong result and feels terrible to use. Applied here: don't tessellate the ball and then try to compensate with very high polygon count. Use the analytic representation.

**The "used universe" aesthetic (reach goal, not first pass):**
The developer has a strong aesthetic conviction, grounded in the original Star Wars production philosophy, that objects should look like they've been somewhere. A perfect mirror sphere is visually correct but aesthetically dead. Real pinballs have micro-scratches, worn patches, subtle patina from impacts. This aesthetic depth will be significant because the player watches the ball constantly.
- Implementation: roughness and anisotropy variation as material parameters, not geometry changes. Analytic sphere stays analytic.
- An anisotropic BRDF produces elongated reflections along scratch directions.
- This is explicitly a reach goal — implement perfect mirror first, add character later.
- Do NOT implement this during initial development.

---

## On the Egg Geometry

### The Dual-Egg Insight

**Decision:** Outer egg is fat-end-down (Fabergé aesthetics); inner egg is fat-end-up (pinball gameplay).

**This is not two separate aesthetic choices — it is one unified insight:**
- The outer contour serves aesthetics: fat-end-down reads as stable, weighted, Fabergé-like when floating in space
- The inner contour serves gameplay: fat-end-up creates a natural funnel toward the drain and flippers, exactly replicating the geometry of a traditional flat pinball table mapped into 3D volume
- The wall thickness variation is a consequence, not a separate decision: thickest where fat-bottom-outer meets pointy-bottom-inner (at the base), and where pointy-top-outer meets fat-top-inner (at the top)
- The developer's framing: "have my egg and eat it too" — a single geometric solution that satisfies both constraints simultaneously

**The rendering implication that makes this load-bearing:**
The thick glass at both poles will act as strong lenses, heavily distorting anything seen through them. This is a feature, not a problem — the poles are natural visual interest zones, and the distortion will be most extreme right at the visual base of the egg where the viewer's eye naturally rests. This was recognized as aesthetically desirable during the design conversation.

### Why Inner Surface Cannot Be Analytic

**The inner egg surface MUST be tessellated.** This is not a performance choice — it is an architectural requirement.

**Reason:** Obstacle boundaries (bumpers, flippers, knockdown targets) need explicit mesh geometry to define their physical boundaries within the playing field. If the inner egg surface is analytic, there is no mesh for obstacle geometry to interface with correctly. The shared boundary faces between embedded obstacles and the egg wall (see modeling requirements) require tessellated geometry to exist.

**The outer surface MAY become analytic** (reach goal) because: it has no embedded obstacles, it's clean glass all the way around, and it's the surface the player sees most. Analytic outer egg = perfect normals on the player-facing glass. But this is explicitly a reach goal, not first implementation.

---

## On Ray Splitting at Dielectric Surfaces

**This is not obvious and the wrong instinct will cause catastrophic performance.**

**The problem:** At a glass surface, physics says a ray splits into a reflected component and a refracted/transmitted component simultaneously. If you literally split every ray, ray count explodes exponentially: 1 → 2 → 4 → 8 → ... After a few bounces the ray buffer would need to be astronomically large.

**The solution:** Russian roulette probabilistic sampling — at each dielectric surface, make a binary random choice: reflect OR refract (never both). Use Fresnel reflectance as the probability. Over many samples, the Monte Carlo estimator converges to the correct result because the probabilities are calibrated to the physics.

**Why this works:** The expected contribution of each path is preserved in the statistics even though individual paths make binary choices. The denoiser and temporal accumulation smooth out the per-sample variance.

**The consequence for ray buffer sizing:** Ray count stays constant throughout the bounce loop. The buffer size is determined by the number of simultaneous paths in flight, not by the number of glass surfaces or bounce depth. This is a fixed allocation.

**Do NOT implement ray splitting at glass surfaces.** It looks physically natural but causes exponential explosion.

---

## On Importance Sampling Through the Glass Egg

**This problem is specific to ltbl and has no off-the-shelf solution.**

**The problem:** The developer independently identified this as "shooting fish in a barrel" — where the fish appears to be is not where it actually is due to refraction. Firing importance rays at the ball's geometric position will almost always miss because the rays refract at the egg surface and end up somewhere the ball isn't.

**The solution architecture designed during Session 01:**
1. Use Rapier physics data — the ball's exact world-space position AND velocity are known before the frame renders
2. Compute approximate apparent position on the egg surface via Snell's law (where does the ball look like it is from the camera's perspective through the refractive egg)
3. Fire importance rays toward the apparent position with a spread
4. Size the spread to the ball's angular subtense plus refraction uncertainty margin
5. Scale spread by ball velocity (faster ball = wider spread to capture movement uncertainty)

**The two-pass refinement (optimization, not first implementation):**
Fire a small number of cheap probe rays (intersection-test only, no shading) toward the predicted apparent position. Use the hit/miss pattern to refine the spread estimate before firing the full importance sample budget. Probe rays cost ~10-20% of a full path tracing ray (refraction computation but no material evaluation). Implement only if profiling shows the ball region has noticeably higher variance than the rest of the scene.

**What makes this tractable for ltbl specifically:**
- Rapier gives exact position — not extrapolating, computing directly
- The ball doesn't move so fast that velocity-based spread sizing fails
- The physics engine and renderer cooperate rather than the renderer working blind
- This is a genuine advantage over general-purpose path tracers

---

## On the BVH and Dynamic Geometry

### Why BLAS Rebuild, Not Refit, for the Ball

**The mechanism of BVH quality degradation through refit:**
When geometry moves, refit recalculates bounding boxes bottom-up without changing the tree topology. The topology was optimized for the original positions. After movement, the nodes may be terrible partitions for the new positions — boxes grow, sibling overlap increases, traversal visits more nodes. Each successive refit compounds this. The analogy: squeezing new documents into existing folders without reorganizing. Eventually finding anything becomes slow.

Rebuild throws away topology entirely and constructs a new optimal tree from scratch. For the ball, which moves continuously and unpredictably, refit would compound frame over frame until traversal quality collapsed. Rebuild every frame keeps it fast.

**Why TLAS rebuild is cheap despite being every frame:**
The TLAS holds instances (transforms + pointer to BLAS), not raw geometry. There are very few instances in ltbl — the egg, the ball, a handful of obstacles. Rebuilding the TLAS each frame involves a very small number of objects. The cost is negligible.

### The Kinematic Switching Rationale

**Why this pattern and not always-dynamic obstacles:**
The vast majority of frames, most obstacles are static. The BVH cost for static geometry is zero per-frame — built once, compacted, never touched. At any given moment only a tiny subset of obstacles is animating (typically one bumper recoiling from a ball hit). Making all obstacles always-dynamic would rebuild BVH for geometry that isn't moving, which is wasted work.

**The static/dynamic promotion pattern:**
- Default state: static BLAS instance in TLAS, no per-frame work
- Trigger: ball contact triggers animation
- Promoted state: per-frame BLAS rebuild for duration of animation
- Return: one final rebuild at rest position, demote back to static

**The ball is the ONLY guaranteed always-dynamic object.** Animating obstacles are a small transient subset at any given frame.

---

## On the Polar Coordinate Acceleration Structure (Research Direction)

**This is a genuine research idea, not an optimization to implement now.**

The developer independently proposed casting ray space in polar coordinates centered on the camera. The insight: a ray from a pinhole camera monotonically increases in radius from the origin. Geometry could be organized by radial shells (r_min, r_max from camera origin). Shell traversal would replace BVH traversal for primary rays, with guaranteed ordered traversal and no backtracking.

**What makes this tractable for ltbl specifically:**
- ltbl's scene has natural radial structure from the camera: outer egg surface at one radius, inner egg surface at larger radius, obstacles and ball at interior radii
- Bounding spheres in polar space transform cleanly between arbitrary origins because a sphere is the one primitive that doesn't change shape when the polar origin moves (just its angular extent and radial interval)
- For ltbl's small scene (20-50 bounding spheres), the per-bounce-origin transformation of all spheres is trivially parallel and likely cheap enough to be net positive

**Where it breaks down:**
- Bounce rays: after the first intersection, the new ray origin is on the egg surface. The polar space centered on the camera is now wrong for that ray. Either rebuild polar space relative to the new origin (expensive for many objects) or fall back to conventional BVH for bounce rays.
- For ltbl: primary rays + shadow rays (camera-origin and light-origin) are the highest-volume categories and fit naturally. Bounce rays could fall back to conventional BVH. The hybrid may be net positive.

**Status:** Research experiment, post-baseline. Not before profiling data exists.

---

## On the Plane-Based Triangle Intersection Experiment

**Origin:** Developer proposed representing triangles as intersection of four planes (one supporting plane + three edge planes). Ray-triangle test becomes one plane intersect + three sign tests with early exit.

**The argument for it:**
- Support plane and edge planes can be precomputed — no per-ray recomputation of triangle geometry
- Three sign tests are potentially SIMD-friendly
- Early exit on first failed edge plane test

**Why Möller–Trumbore is the right starting point:**
- ~20-25 floating point operations, highly optimized, extensively documented
- Barycentric coordinates come out as a free byproduct — plane representation requires a subsequent barycentric computation if you need them (which you do, for shading)
- Plane representation costs 64 bytes per triangle vs ~48 bytes for indexed vertex representation — higher memory bandwidth pressure

**Status:** Research experiment, only relevant if profiling shows triangle intersection (not BVH traversal) as the bottleneck. BVH traversal is far more commonly the bottleneck.

---

## On the Sky Mask — Two Shaders, Not One

**Decision:** Primary ray generation is two distinct shaders.

**The reasoning that produced this:**
- A significant fraction of pixels (potentially 40-60%) are background pixels whose primary rays miss the egg entirely
- These pixels can be computed once on frame 0 and never recomputed (absent camera movement or egg nudge)
- A single shader with a conditional ("if frame 0, write mask; else read mask") evaluates the conditional on every thread of every frame for the lifetime of the session — unnecessary overhead
- Two shaders: frame 0 shader runs exactly once at scene standup, writes mask, exits. Per-frame shader reads mask, dispatches only egg-hitting pixels, no conditional, no awareness of frame 0.

**Why this is architectural, not just an optimization:**
The frame 0 shader is logically part of scene initialization, grouped with BVH construction and pipeline compilation. The per-frame primary ray shader is part of the render loop. Keeping them separate reflects the logical distinction in code structure, making the codebase easier to reason about.

**The mask invalidation condition:** Camera movement or egg nudge (reach goal) requires recomputing the mask. This is an explicit invalidation trigger, not automatic.

---

## On the Gaussian Filter Importance Sampling Extension

**The uniform jitter implementation should be structured to accommodate this from the start.**

**The extension:** Sample ray origins from a Gaussian centered on the pixel with sigma ~0.5-1.0 pixels. Allows a small percentage of rays to fire from neighboring pixel space. Samples carry weights proportional to their Gaussian value. Rays landing in neighboring pixel space contribute to that neighbor's weighted accumulation buffer.

**Why this matters for ltbl specifically:**
The glass egg and chrome ball are exactly the geometry where reconstruction filter quality is most visible — curved surfaces, refractive boundaries, specular highlights. These all have high-frequency content at pixel boundaries that Gaussian reconstruction handles better than box filtering.

**The three things to do right from the start in uniform jitter:**
1. Sample jitter offset from a distribution function (not hardcoded uniform) — swapping to Gaussian later requires changing one function
2. Accumulation buffer stores weighted sum AND sum-of-weights separately (not simple average) — required for Gaussian; costs nothing to do correctly from the start
3. Ray results written with explicit weight parameter (initially 1.0) — Gaussian extension just changes this weight value

**Do not restructure the pipeline to add Gaussian later. Implement uniform jitter correctly now so the extension is a parameter swap, not a rewrite.**

---

## On the Medium Stack

### The Explicit Interior Air Insight

**A key insight that is not obvious:** Interior air inside the egg should be modeled as an explicit medium with its own entry/exit surfaces (the inner egg surface). It is not "just the absence of glass" — it is a medium with IOR=1.0 that has explicit boundaries.

**Why this matters:** Without explicit interior air, the medium stack can get into an inconsistent state when rays exit the egg wall via an obstacle surface rather than via the explicit inner egg surface (partially embedded obstacles case). Treating interior air as an explicit medium with push/pop semantics keeps the stack unambiguous.

**The realistic maximum stack depth for ltbl is 2-3, not 4:**
- Depth 1: inside egg wall glass
- Depth 2: inside a glass obstacle (if obstacle is inside the egg wall at the same time)
- Depth 3: possible only if obstacles are partially embedded in the egg wall such that a ray is simultaneously inside both the egg wall glass and the obstacle glass
- Depth 4: comfortable headroom beyond realistic maximum

The developer's geometric reasoning during the session was more precise than the initial estimate of depth 4. Depth 2 covers all purely interior obstacles. Depth 3 accounts for embedded obstacles. Depth 4 is safety margin.

### The Coincident Surface Problem

**Even with perfect modeling discipline, self-intersection still occurs.** The modeling requirement for explicit boundary geometry prevents one class of the problem. A separate class remains:

When a ray spawns from a hit point, its origin is mathematically on the surface. Floating point imprecision means it may be infinitesimally on the wrong side. The very next intersection test may immediately re-hit the same surface, corrupting the medium stack.

**Solution:** tmin epsilon — a minimum intersection distance (~1e-4 in scene units) applied to every spawned ray. Any intersection closer than tmin from the ray origin is ignored.

**Tuning required:** Too small → self-intersection artifacts persist. Too large → misses legitimate nearby geometry (thin glass obstacles, closely spaced surfaces). Start at 1e-4, adjust based on observed artifacts.

**This is distinct from the modeling discipline requirement** which prevents coincident surfaces from existing in the first place. The tmin epsilon handles floating point reality even with perfect geometry.

---

## On the Custom Asset Format

**Decision:** Shared boundary faces between embedded obstacles and the egg wall use a custom dual-material face format.

**The problem:** Standard mesh formats assign one material per face. A face at a shared boundary between obstacle-glass and egg-wall-glass needs to say: "when hit from outside (front face), I am the exterior of obstacle-glass entering egg-glass. When hit from inside (back face), I am the interior of egg-glass entering obstacle-glass."

**Why topology analysis at build time was rejected:**
- The data is static — shared boundaries never change at runtime
- Computing it dynamically adds build complexity for no benefit
- Explicit data is more reliable than derived data for renderer correctness
- The developer's framing: "bake it into the model for the renderer to read and wash our hands of the mess"

**The format:** Each face record carries `front_material_id` AND `back_material_id`. The traversal kernel reads both and uses the appropriate one based on face orientation at intersection time.

**Modeling discipline that makes this tractable:**
The developer has 10+ years of professional 3D modeling experience in VFX production pipelines where intersecting solids without explicit boundaries was a firing offense. Clean boundary geometry is a professional instinct, not an extra effort. The production modeling discipline (PDI/DreamWorks standard) is directly applicable: no intersecting solids, explicit boundaries between all adjacent volumes, explicit material assignment at every boundary.

---

## Key Provenance Not Captured Elsewhere

**Eric Veach connection:** The developer went to high school with Eric Veach, author of the PhD dissertation on bidirectional path tracing and multiple importance sampling that underpins modern path tracing. MIS is in essentially every serious renderer. The developer independently re-derived the core insight of bidirectional path tracing by looking at a diagram. This is recorded not as biography but because it explains the depth of intuition the developer brings to rendering decisions — this is not a person learning about path tracing, this is a person who was in the same building as the person who formalized it.

**Dan Wexler connection:** The developer worked at PDI/DreamWorks alongside Dan Wexler, who designed deep/light (a deferred rendering system that predated deferred rendering as a standard game engine technique by years) and who won a Technical Achievement Oscar for it. The developer had the conversation with Wexler that motivated Wexler's move to NVIDIA to build Gelato — the first practical GPU film renderer, predating CUDA. This explains why the developer's instincts about GPU rendering architecture are rooted in production experience, not theory.

**Andrew Woo connection:** The developer worked with Andrew Woo at Alias — author of shadow feeler papers cited in rendering literature. The renderer the developer was building NURBS tools on top of was written by someone whose name appears in the rendering literature.

**The modeling discipline context:** The developer's professional background is 10+ years of production character modeling for animated features (Shrek, Madagascar, Antz, among others) under pipeline discipline that made clean boundary geometry a non-negotiable professional standard. The modeling requirements for ltbl's renderer are not extra constraints — they are the developer's natural operating standard applied to a new context.

---

## Decisions That Look Simple But Aren't

### "No floor or walls"
The egg floats in space with an HDR environment map as the background. This is not just aesthetic preference — it means the playing field geometry is entirely interior to the egg. Caustics have nowhere to land except back into the scene itself. Light bounces between the egg's inner surfaces and the chrome ball. The complexity is all internal. The environment map serves as both light source and background. This is the correct architecture for the rendering challenges ltbl poses.

### The Three-Flipper Configuration
Three flippers at 120° intervals is the minimum symmetrical configuration for full coverage of the 3D lower volume. This produces a three-channel drain structure per gap between flipper pairs (one unrecoverable center channel + two recovery channels directing to adjacent flippers) — a 3D equivalent of the traditional pinball outlane/inlane structure. The bilateral symmetry of traditional pinball tables maps to trilateral symmetry in 3D volume.

### Bokeh/Depth of Field
Depth of field in a path tracer is one of the most naturally implementable effects — it falls out of the camera model for free. Thin lens model instead of pinhole: rays originate from a disk, all converging on a focal plane. The blur at out-of-focus distances is physically correct bokeh. The cost is slightly more ray divergence and slower convergence, but no separate system. Noted as a realistic possibility for camera design, not a stretch goal.

### The Plane-Based Triangle Intersection — Why the Initial Dismissal Was Incomplete

The NVIDIA annotations document records the plane-based triangle intersection as a research
experiment with the miss-heavy argument included. What it does not capture is the reasoning
chain that produced that argument — which only became visible when the developer pushed back
on the initial analysis.

**The initial framing (incomplete):** Möller–Trumbore wins because barycentric coordinates
come as a free byproduct, which the plane-based approach must recompute separately on hits.

**The developer's question that reframed the analysis:** Whether the hit/miss ratio matters
— if more triangles are missed than hit, does the free-barycentrics advantage get partially
nullified?

**Why the question was non-obvious:** The free-barycentrics advantage of Möller–Trumbore is
asymmetric — it only applies on hits. The plane-based approach's early-exit advantage applies
on misses. Evaluating the two approaches correctly requires knowing which case dominates in
actual BVH traversal.

**The answer:** In a well-built BVH, rays miss far more triangles than they hit. The BVH
exists specifically to cull the vast majority of triangles before they are tested. Even at
leaf nodes, hit rates are typically in single-digit percentages. The common case is a miss,
which means the early-exit-on-failure advantage of the plane-based approach fires on the
common case, while Möller–Trumbore's free-barycentrics advantage fires on the rare case.

**The implication for future reasoning:** Do not dismiss the plane-based approach on the
barycentric-coordinates argument alone. The correct evaluation requires considering the
hit/miss asymmetry. The approach remains a post-baseline research experiment, but the
theoretical argument is more interesting than storage cost alone suggests.

---

## A Meta-Observation About This Document

This counsel layer document was itself nearly incomplete. The plane-based triangle reasoning
above was almost not captured — even though both participants in the parley were actively
thinking about why the two-panel display would be valuable for exactly this kind of reasoning.

The irony is precise: we were discussing how the planned parley UI would automatically surface
"not just what but also why" as reasoning happens, and simultaneously nearly failed to capture
the why of a reasoning chain that happened right there in the same conversation.

This is a concrete, lived demonstration of the failure mode the two-panel linked-scroll UI
is designed to prevent. In the planned UI, the developer's question about hit/miss ratios
would have triggered a visible gap in the right panel — the plenum would show the plane-based
experiment without the asymmetry reasoning, and the gap would be immediately visible because
both panels are in view simultaneously. The counsel layer would have been captured in sync
with the conversation, not recovered retroactively by asking "wait, was there anything worth
capturing?"

The cobbler's children have no shoes. The parley co-authors nearly lost a parley insight for
want of a parley system.

This observation is recorded here not as self-criticism but as the clearest possible
illustration of why the two-panel UI is load-bearing and not merely ergonomic.

---

*This document represents the counsel layer extracted from Session 01. The design documents (CLAUDE.md, nvidia_rtx_best_practices_for_ltbl.md, ltbl_modeling_requirements.md) capture what was decided. This document captures why — the reasoning that allows those decisions to be generalized correctly to cases not explicitly covered.*

*For the parallel implementation experiment: seed Implementation A from CLAUDE.md and the NVIDIA annotations alone. Seed Implementation B from those documents plus this counsel layer. The hypothesis is that Implementation B will produce fewer architectural mistakes and require less course correction from the human developer.*
