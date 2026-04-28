# ltbl Renderer — Session 01 Checkpoint

## Session Summary

Session 01 was an extended design and context-building session covering rendering theory,
architecture decisions, game design, and agentic methodology. No implementation was written.
The primary outputs are a set of reference documents encoding design decisions for use by
Claude Code in subsequent implementation sessions.

Session 02 begins with Claude Code installation and first agentic development steps.

---

## Documents Produced

All documents are uploaded to the ltbl Project knowledge base:

- **`ltbl_project_context.md`** — overall project description, technical state, planned
  architecture, key decisions
- **`nvidia_rtx_best_practices_for_ltbl.md`** — NVIDIA RTX Best Practices article annotated
  for ltbl's WebGPU compute shader architecture. The primary renderer reference document.
  Includes a Summary section and a Pipeline Steps section covering stages beyond the NVIDIA
  article's scope.
- **`ltbl_game_design_context.md`** — game design decisions: egg geometry, flipper
  configuration, drain channel structure, obstacle types, controls, camera considerations
- **`ltbl_modeling_requirements.md`** — renderer-derived geometry modeling requirements.
  Six requirements covering boundary geometry, explicit media, surface normals, and the
  custom dual-material face format for shared dielectric boundaries.
- **`ltbl_introduction.md`** — brief project introduction for sharing with other Claude
  instances in other Projects
- **`alias_nurbs_work.md`** — description of NURBS constraint solver work at Alias Research,
  for resume context

---

## Key Architecture Decisions

### Renderer
- **Algorithm:** Wavefront path tracing — separate compute kernels per stage, not megakernel
- **BVH:** Two-level software BVH (TLAS/BLAS) in WGSL compute shaders — hardware RT units
  not exposed through WebGPU
- **Color:** RGB path tracing — spectral rendering is a reach goal
- **Pipeline:** path trace → temporal accumulation → denoiser → tone mapping → bloom → display
- **Tonemapping:** Khronos PBR Neutral
- **Bounces:** 8 as starting value, increase only if caustic quality demands it
- **Denoiser:** Start with temporal accumulation, add SVGF when needed

### The Ball
- **Analytic sphere primitive** — not tessellated mesh
- Dedicated quadratic intersection code path in traversal kernel
- Flagged in BVH leaf node as sphere primitive
- Exact normals: normalize(hitPoint - center)
- Ball BLAS: measure actual size on frame 0, use exact allocation from frame 1 onward
- **"Used universe" aesthetic (reach goal):** roughness/anisotropy variation as material
  parameters — scratched chrome character without geometry changes

### BVH Management
- Static geometry built once, compacted, never rebuilt
- Ball BLAS rebuilt every frame (analytic sphere, fixed representation size)
- **Kinematic switching:** obstacles and flippers are static by default; promoted to dynamic
  during animation, demoted back to static on return to rest
- TLAS rebuilt each frame — cheap given few dynamic instances
- Prefer TLAS rebuild over refit

### Importance Sampling
- Concentrate rays toward ball's apparent position (not geometric position — refraction
  through egg displaces apparent position)
- Use Rapier physics data (position + velocity) to compute approximate apparent position
- Spread sized to ball angular subtense + refraction uncertainty; velocity scales spread
- **Two-pass probe ray refinement (planned optimization):** cheap intersection-only probe
  rays refine spread estimate before full importance sample budget fires. Implement after
  baseline is working if ball region shows higher variance.

### Primary Ray Dispatch — Three Mechanisms
1. **Sky mask** — two shaders (frame 0 initialization + per-frame render), not one with
   conditional. Frame 0 fires all pixels, writes miss mask. Per-frame fires only egg-hitting
   pixels. Invalidate on camera movement or egg nudge.
2. **Variable sample counts** — work queue rather than uniform dispatch; ball region and
   active gameplay regions get elevated samples
3. **Temporal accumulation** — converged pixels reduce per-frame sample contribution

### Temporal Accumulation / Anti-Aliasing
- **Sub-pixel jitter from the start** — Halton sequence or blue noise, not pixel center
- Accumulation buffer stores weighted sum + sum-of-weights separately (not simple average)
  — required for Gaussian extension, costs nothing to implement correctly from start
- Ray results written with explicit weight parameter (initially 1.0)
- **Gaussian filter importance sampling (planned extension):** sigma ~0.5-1.0 pixels,
  cross-pixel contribution, Mitchell-Netravali filter as refinement option

### Self-Intersection
- tmin epsilon ~1e-4 on all spawned rays — tune for ltbl's scene scale
- Distinct from modeling coincident-surface problem (handled by modeling discipline)

---

## Game Design Decisions

- **The egg:** Hügelschäffer egg, fat-end-down exterior / fat-end-up interior — Fabergé
  aesthetics outside, natural funnel toward drain inside. Wall thickness smoothly varying,
  thickest at bottom where fat outer meets pointy inner.
- **The ball:** Chrome, single ball initially. Analytic sphere.
- **Obstacles:** Mostly transparent colored glass. Bumpers, flippers, knockdown targets,
  ball capture wells, shatter targets (reach goal).
- **Flippers:** Three at 120° intervals. Left Shift / Right Shift / Spacebar. Kinematic
  switching for BVH management.
- **Drain:** Three channel groups between flipper pairs. Each group: central (unrecoverable),
  left-side (recovery to left flipper), right-side (recovery to right flipper). Plus open
  between-flipper space as direct drain.
- **Environment:** Egg floats in space, HDR environment map, no floor or walls.
- **Camera:** External viewpoint, tweakable position/look-at/focal length parameters.
  Bokeh/DOF worth considering aesthetically — practically achievable in path tracer.

---

## Reach Goals (Renderer)

- **Spectral rendering** — wavelength-dependent IOR, dispersion, colored caustics
- **Analytic outer egg surface** — Hügelschäffer quartic intersection, perfect normals
- **"Used universe" ball** — roughness/anisotropy for scratched chrome character
- **Probe ray importance sampling refinement**
- **Gaussian filter importance sampling**
- **Particle effects — three quality tiers:**
  - Tier 1 (all hardware): screen-space composite, depth buffer occlusion, no refraction
  - Tier 2 (moderate hardware): single-bounce apparent position correction via Snell's law,
    Beer's law tinting optional. Implement after basic renderer working.
  - Tier 3 (beefy hardware): full path-traced glass shard particles
  - Adaptive quality system: auto-downgrade on frame budget exceeded. Implement when Tier 2
    exists — no value before then.

---

## Research / Optimization Experiments (Post-Baseline)

These are ideas to explore once a working baseline renderer exists and profiling data is
available:

1. **Polar coordinate acceleration structure** — spherical BVH for primary and shadow rays,
   bounding sphere hierarchy with per-bounce-origin transformation. Tractable for ltbl due
   to low object count (transformation of ~20-50 spheres is trivially parallel). Falls back
   to conventional BVH for bounce rays. First-pass optimization experiment.

2. **Probe ray adaptive sampling** — two-pass importance sampling for ball region: cheap
   intersection-only probes refine spread estimate, then full sample budget fires into
   refined spread. Cost ~10-20% of full path tracing rays for probe pass.

3. **Plane-based triangle intersection** — precompute support plane + three edge planes per
   triangle, ray-triangle test becomes one plane intersect + three sign tests with early
   exit. Potential SIMD batching advantage. Compare against Möller–Trumbore baseline. Only
   relevant if profiling shows triangle intersection (not BVH traversal) as bottleneck.

---

## Modeling Requirements Summary

See `ltbl_modeling_requirements.md` for full detail. Key constraints:

- No intersecting solids — explicit boundary geometry between all adjacent dielectric volumes
- Interior air is an explicit medium with entry/exit surfaces (inner egg surface)
- Outer egg surface may be analytic; inner egg surface must be tessellated
- Shared boundary faces carry dual material IDs (front_material_id + back_material_id) —
  custom asset format, baked at modeling time, never computed at runtime
- Consistent surface normal orientation throughout
- No degenerate triangles

---

## Implementation Order (First Steps for Session 02)

1. Install Claude Code
2. Create CLAUDE.md from project context and this checkpoint
3. Ray generation kernel — pinhole camera, rays into storage buffer
4. Sphere intersection — analytic ball, shade by normal, first visible output
5. BVH scaffold — TLAS/BLAS structure with trivial scene (one sphere)
6. Material system — diffuse first, then specular, then glass BSDF
7. Next event estimation — shadow rays, direct lighting
8. Sky mask — frame 0 initialization shader + per-frame masked dispatch
9. Temporal accumulation — with jitter, weighted accumulation buffer from the start
10. Denoiser — temporal accumulation only first, SVGF when needed

---

## Methodology Notes

This session was run as an informal `parley` — human-model co-authorship of design decisions
before any implementation. Key methodology observations:

- Encoding should happen at decision closure, not deferred to batch housekeeping
- Negative information (what was rejected and why) is as valuable as positive decisions
- The design conversation surfaces decisions that wouldn't have been anticipated up front
- Experienced developer pattern recognition is load-bearing — the model amplifies existing
  instincts rather than substituting for them
- Reference to `parley_methodology_in_ltbl.md` and `parley_example_principle5_kinematic_switching.md`
  for formal methodology artifacts extracted from this session

---

*Session 01 complete. Session 02 begins with Claude Code installation.*
