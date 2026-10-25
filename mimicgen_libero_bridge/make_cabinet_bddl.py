"""Writes 4 copies of the SCENE1 bddl, one per cabinet distance-to-edge
state, each with ONLY wooden_cabinet_init_region's declared range edited
(bowl/plate stay at their native bddl declaration -- those are repositioned
at runtime via direct qpos poke after reset, independent of their bddl
range; see run_scene1_4factor.py).

Why edit the bddl at all instead of the runtime sampler-mutation trick used
for bowl/plate: VERIFIED empirically (see run_scene1_4factor.py's module
docstring) that env.env.placement_initializer is a FRESH object on every
single env.reset() call -- confirmed via id() across 5 resets, all
different -- rebuilt from the bddl file's declared regions each time. So
any sampler mutation made before calling reset() is always discarded before
it can take effect; it never actually worked, not here nor (as it turns out)
in the already-deployed 31-task batch's widen_object_region (that one just
didn't matter in practice, since bowl/plate's resulting positions get
overwritten by direct qpos poke anyway regardless of what the sampler drew).
The cabinet has no free joint to poke post-reset, so for it, the bddl's
declared region is the ONLY lever -- meaning cabinet state must be fixed
per ENV CONSTRUCTION (one bddl/env per state), not per attempt.

Representative boxes, each verified against dist_to_edge(x,y) =
min(x+0.5, 0.5-x, y+0.6, 0.6-y) (real table extent x in [-0.5,0.5],
y in [-0.6,0.6]) staying inside the target CABINET_EDGE_BINS range for
EVERY corner of the box (not just the center):
  near   (0.14-0.20m): x in [-0.36,-0.30], y in [-0.42,-0.29]
  medium (0.20-0.26m): x in [-0.30,-0.24], y in [-0.42,-0.29]
  far    (0.26-0.32m): x in [-0.15, 0.15], y in [-0.33,-0.29]
  absent: parked at (2.0, 2.0), far outside the table/camera entirely.
All stay within the safe band (y <= -0.29) that keeps the cabinet's real
24x33cm footprint clear of the bowl/plate region (y >= -0.12).
"""
import re

SRC_BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
OUT_DIR = "/mgbridge"

CABINET_RANGES = {
    "near": "(-0.36 -0.42 -0.30 -0.29)",
    "medium": "(-0.30 -0.42 -0.24 -0.29)",
    "far": "(-0.15 -0.33 0.15 -0.29)",
    "absent": "(1.99 1.99 2.01 2.01)",  # exact point (2,2,2,2) crashes MuJoCo: zero-size init-region geom
}


def substitute(text, region_name, new_range):
    pattern = re.compile(
        r"(\(" + re.escape(region_name) + r"\s*\(:target kitchen_table\)\s*\(:ranges \(\s*)"
        r"\([^)]*\)",
        re.S,
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + new_range, text)
    assert n == 1, f"expected exactly 1 substitution for {region_name}, got {n}"
    return new_text


def write_all():
    with open(SRC_BDDL) as fh:
        text = fh.read()
    paths = {}
    for state, rng in CABINET_RANGES.items():
        new_text = substitute(text, "wooden_cabinet_init_region", rng)
        out_path = f"{OUT_DIR}/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_cab_{state}.bddl"
        with open(out_path, "w") as fh:
            fh.write(new_text)
        paths[state] = out_path
        print(f"wrote {out_path} (cabinet range {rng})")
    return paths


if __name__ == "__main__":
    write_all()
