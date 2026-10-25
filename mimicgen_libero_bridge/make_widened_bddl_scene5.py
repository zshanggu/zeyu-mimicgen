"""Widen bowl/plate regions for KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.

Measured directly (check_scene5_geometry.py): cabinet real footprint
x in [-0.127,0.115], y in [0.036,0.522] -- on the POSITIVE y side here (SCENE1's
cabinet was on the negative side), so bowl/plate must stay at y <= 0 near x~0.
Native (undilated) positions cluster at bowl~(0.03,-0.05), plate~(-0.05,-0.25),
ketchup~(-0.10,-0.10) -- i.e. this scene's working area is the NEGATIVE-y half
of the table, mirrored from SCENE1.

Shared region for bowl AND plate (same box, rely on env_interface.py's
reject-and-reroll for no-overlap, same approach as SCENE1's merged-region fix):
60cm x 35cm, centered on the native cluster, capped at y=0.00 for a clean
3.6cm margin below the cabinet's y_min=0.036. x widened to +-0.30 (SCENE1's own
widening found good reach out to about +-0.40; kept more conservative here
since this is the first run on a new scene -- can push further once this one's
measured success rate confirms it's safe).
"""
import re

SRC_BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate.bddl"
OUT_BDDL = "/mgbridge/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate_widened.bddl"

BOWL_RANGE = "(-0.18 -0.28 0.18 -0.02)"
PLATE_RANGE = "(-0.18 -0.28 0.18 -0.02)"

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
