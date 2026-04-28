# # ltbl — Game Design Context

## Concept

A pinball machine reimagined as a fully 3D playing field enclosed inside a glass egg, floating
in space. The player views the egg from outside, like a snow globe. All gameplay takes place
inside the egg.

---

## The Egg

### Outer Surface
- Shape: **Hügelschäffer egg** — asymmetric, fat end down (like a Fabergé egg)
- Fat end down gives the object visual gravitas and aesthetic stability from the player's
  external viewpoint
- Wall thickness varies smoothly — roughly constant through the middle, significantly thicker
  at both poles

### Inner Surface
- Shape: **Hügelschäffer egg, fat end UP** — oriented opposite to the outer surface
- This inversion means the inner playing volume is wide at the top and narrows toward the
  bottom, creating a natural funnel toward the drain and flippers
- The result is a playing field that is the volumetric equivalent of a traditional flat
  pinball table — wide open play space above, natural gravity toward the drain below

### The Egg-Within-An-Egg Solution
The outer and inner surfaces are oriented oppositely, solving two competing design goals
simultaneously:
- **Exterior aesthetic:** fat-end-down Fabergé egg shape
- **Interior gameplay:** fat-end-up funnel toward drain, equivalent to traditional pinball
  table geometry

Wall thickness is asymmetric as a result — thickest where the fat bottom of the outer egg
meets the pointy bottom of the inner egg.

---

## The Ball

- **Chrome** — highly reflective
- Single ball (initially)

---

## Playing Field Contents

Most obstacles are **transparent colored glass** so the player can track the ball's path
through the full depth of the scene without full occlusion. Obstacle types include:

- Bumpers
- Flippers (see below)
- Knockdown targets
- Ball capture wells — can trigger interesting behaviours such as holding the ball while the
  player launches a second ball into play, with some subsequent action freeing the trapped ball
  and giving the player two balls to track simultaneously
- Other standard pinball obstacle types — reimagined for 3D space

The volume directly above the flippers and drain is kept **clear of obstacles**, following
traditional pinball convention. This serves both readability (player can see the danger zone
clearly) and tension (the "oh no" moment when the ball heads for the drain is unambiguous).

---

## Flippers

### Configuration
- **Three flippers** at **120-degree intervals** around the lower interior of the egg
- Minimum symmetrical configuration for full coverage of the lower volume
- Creates trilateral symmetry matching the egg's circular cross-section

### Orientation (from player's viewpoint)
- **Left flipper** — to the player's left
- **Right flipper** — to the player's right
- **Z-flipper** — directly at the back of the playing field as seen by the player

### Controls (default keyboard mapping)
- **Left Shift** — left flipper
- **Right Shift** — right flipper
- **Spacebar** — z-flipper
- Ambidextrous design — usable by left- and right-handed players equally
- Configurable remapping planned as a follow-on feature

---

## Drain Channels

On a traditional pinball table there are two distinct loss paths:
1\. **Between the flippers** — ball passes straight down the middle, direct drain
2\. **Outlane channels** — far left and right channels that route the ball behind the flipper
   and to the drain, bypassing the flipper entirely

Both loss paths have 3D equivalents in ltbl:

### Between the flippers (3D equivalent)
The open space directly between all three flippers at their tips — ball passes straight down
through the center and into the drain. No channel involved. Equivalent to going cleanly between
both flippers on a traditional table.

### Outlane channel structure (per 120-degree gap)
Between each pair of adjacent flippers there is a channel structure, giving three channel
groups total. Each channel group contains three channels:

1\. **Central channel** — outlane equivalent. Routes ball behind the adjacent flippers and to
   the drain, bypassing both. Unrecoverable loss. The "go directly to jail" channel.
2\. **Left-side channel** — directs ball to the left-adjacent flipper. Recovery possible.
3\. **Right-side channel** — directs ball to the right-adjacent flipper. Recovery possible.

This maps the traditional pinball outlane/inlane distinction onto the 3D trilateral geometry.
Skilled players can work the side channels for recovery; the central channel and the open
between-flipper space are always losses.

---

## Environment

- Egg floats in **space** — no floor, no walls, no ground plane
- **HDR environment map** provides the background and all external lighting
- All caustics and light interplay are internal to the egg
- No external geometry to distract from the egg as the sole visual object

---

## Camera

- Player viewpoint is **external to the egg**
- Slightly elevated, slightly off-center from the z-flipper axis is the expected starting
  point, but camera position is explicitly a **"build it and look" design decision**
- Camera position should be exposed as easily tweakable parameters early in the prototype
  (position + look-at point as simple floats) to allow rapid iteration without code changes
- **Focal length** should be included as a tweakable parameter — the difference between a
  long telephoto (compressed, intimate, egg fills the frame) and a wide angle (expansive,
  dramatic, egg floating in vast space) is a significant aesthetic variable worth exploring
  early
- **Bokeh / depth of field** is worth considering as an aesthetic tool — selectively focusing
  on the interior play action while softening the egg shell, or vice versa, could be
  compelling. Whether this is practically achievable in the real-time renderer is a separate
  question (to be addressed in the renderer design conversation); whether it would add
  aesthetic value or just complexity is the question for this conversation to explore when
  the prototype is far enough along to judge

---

## Implementation Path

Game design and renderer design are **mostly orthogonal** and can be developed on separate
tracks:

1\. **Gameplay prototype:** Build using **Rapier** (physics) + **Three.js** (rendering) —
   fast iteration on game mechanics, flipper feel, obstacle layout, drain dynamics, controls
2\. **Renderer development:** Build `ltbl` WebGPU path tracer independently
3\. **Integration:** Replace Three.js rendering with `ltbl` renderer once both tracks are
   sufficiently advanced

This allows gameplay to be validated and tuned without waiting for the renderer, and the
renderer to be developed without needing working gameplay.