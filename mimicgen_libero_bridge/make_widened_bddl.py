"""Writes a copy of the bowl/plate bddl with BOTH the bowl's and the plate's
init regions widened toward the table's real edge / the robot's reach limit,
so MimicGen's generation-time env.reset() samples placements that actually
stretch the arm, not just "somewhat wider than the native 5cm boxes."

Layout (kitchen_table-relative coords), grounded in two direct measurements
against the live MuJoCo sim (not guessed from bddl regions, which only
describe where objects are PLACED, not the table's or the robot's own
physical limits):

  1. Table top geom (body "table", geom "table_collision"): center (0,0,0.875),
     half-size [0.5, 0.6, 0.025] -> real surface extent x in [-0.5, 0.5],
     y in [-0.6, 0.6]. The table's SHORT edges are the x=+-0.5 ends; the LONG
     edges are the y=+-0.6 sides.

  2. Robot base ("robot0_base") sits at x=-0.66, y=0 -- i.e. just behind the
     table's x=-0.5 edge, not centered on the table at all. Probed reach
     empirically (closed-loop position control, fixed downward orientation,
     150-step convergence budget, pos_tol=1.5cm) over a grid spanning the
     whole table surface at a fixed height (z=0.97, just above the resting
     bowl height ~0.898+object radius). Key finding -- reach is HIGHLY
     ANISOTROPIC, not a simple radius from the base:
       - Toward the table's FAR short edge (+x, away from the robot): reach
         collapses fast. x=+0.15 at y~0 converges; x=+0.25 at y~0 already
         fails (final pos error ~0.10m, nowhere close). The far edge at
         x=+0.5 (a straight-line distance of ~1.16m from the base) is not
         remotely reachable with this controller.
       - Along the table's LONG edges (+-y, lateral sweep): reach is excellent
         as long as x stays moderate/negative (close to the robot's own side
         of the table) -- e.g. x=-0.45 or x=-0.25 converge cleanly all the
         way out to y=+-0.55, right at the table's real y-edge (0.6).
     Net: "near the table edge, max stretch" for THIS robot mostly means
     pushing along Y toward the table's long-edge limits, not along X toward
     the far short edge (which saturates around x~0.15-0.20 regardless of y).
     This is a real, non-obvious asymmetry -- not an assumption.

  3. Cabinet fixture's real geom footprint (measured directly, same as the
     prior iteration): x in [-0.12, 0.12], y in [-0.525, -0.191]. Still the
     binding constraint on how far negative either region's y can go near
     x~0.

  - bowl native region: (-0.025 -0.025 0.025 0.025) -- a ~5cm x 5cm point.
  - plate native region: (-0.025 0.225 0.025 0.275) -- a ~5cm x 5cm point.
  - prior iteration widened: bowl (-0.15 -0.12 0.15 0.05) 30x17cm,
    plate (-0.15 0.10 0.15 0.35) 30x25cm. Both still well short of either
    the x reach ceiling (~0.15-0.20) or the y edge (0.6).
  - this iteration's widened bowl: (-0.20 -0.12 0.20 0.12) -- 40cm x 24cm.
    x pushed from +-0.15 to +-0.20, straddling the measured x=0.15-converges /
    x=0.25-fails boundary (deliberately -- some corner attempts near x=0.20
    are expected to fail, that's the point). y kept at the same safe -0.12
    lower bound (0.07m clearance from the cabinet's -0.191 edge, unchanged
    from last iteration -- no reason to re-risk that already-debugged
    collision), extended symmetrically to +0.12 on the other side.
  - this iteration's widened plate: (-0.20 0.17 0.20 0.50) -- 40cm x 33cm.
    x matches the bowl's new +-0.20. y pushed from [0.10,0.35] to [0.17,0.50]
    -- this is the real "stretch toward the table edge" move, reaching for
    the table's actual y=0.6 long edge (0.50 is comfortably inside the
    measured-converging region at moderate x, e.g. x=-0.05..0.05, while still
    meaningfully closer to the true edge than before). y_min=0.17 keeps a
    0.05 gap from the bowl's new y_max=0.12, so the two sampled boxes can
    never overlap and cause a spawn-interpenetration collision (the same
    failure mode already hit once in the very first iteration).

  NEXT ITERATION (this edit): the above iteration's y=0.50 for the plate
  overshoots BOTH constraints at once, confirmed two ways:
    - agentview camera frustum, measured directly via cam_xpos/cam_xmat/fovy
      and raycast to the z=0.9 table plane: visible y at the x values used is
      only ~0.33-0.46, not 0.50-0.6. One of the actual generated demos (plate
      at x=-0.137, y=0.492) has the plate completely outside the rendered
      frame -- a wasted "success" for vision-based training.
    - empirical OSC reach probe (grid of xy targets, closed-loop P-control,
      150 steps, measuring final steady-state xy error): reliable reach
      (err<5cm) tops out at y~=0.30 across the x range actually used, not
      0.5-0.6 -- y=0.4 already fails (err~0.12) almost everywhere tested.
      Reach, not the camera, is the binding constraint on y.
  Meanwhile the PREVIOUS x range (+-0.20) wasted half its budget on the +x
  side, which the same reach probe shows collapses around x~=0.10-0.15
  (at x=0.15, nothing converges, even at y=0). The -x side has no such
  problem -- x=-0.3 reaches cleanly with full y range, and the live
  generation run already placed successful bowls as far as x=-0.36.
  So this edit re-shapes the box rather than reusing its area: y pulled back
  in to a reach+camera-safe <=0.30-ish ceiling, and the freed-up budget
  reallocated to x, pushed out to -0.40 (close to the table's real x=-0.5
  edge) on the side that actually supports it, with x_max capped at +0.08
  to stay inside the measured reach ceiling instead of +0.20.
"""
import re

SRC_BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
OUT_BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"

BOWL_RANGE = "(-0.45 -0.12 0.10 0.35)"
PLATE_RANGE = "(-0.45 -0.12 0.10 0.35)"

with open(SRC_BDDL) as fh:
    text = fh.read()


def substitute(text, region_name, new_range):
    pattern = re.compile(
        r"(\(" + re.escape(region_name) + r"\s*\(:target kitchen_table\)\s*\(:ranges \(\s*)"
        r"\([^)]*\)",
        re.S,
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + new_range, text)
    assert n == 1, f"expected exactly 1 substitution for {region_name}, got {n}"
    return new_text


new_text = substitute(text, "akita_black_bowl_init_region", BOWL_RANGE)
new_text = substitute(new_text, "plate_init_region", PLATE_RANGE)

with open(OUT_BDDL, "w") as fh:
    fh.write(new_text)

print(f"wrote {OUT_BDDL}")
for region_name in ("akita_black_bowl_init_region", "plate_init_region"):
    idx = new_text.index(region_name)
    print(f"--- {region_name} ---")
    print(new_text[idx - 5 : idx + 200])
