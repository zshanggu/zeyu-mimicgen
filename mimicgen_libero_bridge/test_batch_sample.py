"""Quick validation: run the generic pipeline on 3 tasks spanning both
categories (free-free vs region-target) and 3 different scene arenas, at a
small per-task target, before committing to the full 31-task x 100 batch."""
import os
os.environ["MG_TARGET_SUCCESSES"] = "8"
os.environ["MG_MAX_ATTEMPTS"] = "30"

import sys
sys.path.insert(0, "/mgbridge")
import run_batch as RB

SAMPLE = [
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",         # free-free, living room
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",      # region-target, kitchen
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",  # region-target, study
]

out_dir = "/mgbridge/out/batch_test"


def log(msg):
    print(msg, flush=True)


for t in SAMPLE:
    try:
        RB.run_one_task(t, out_dir, log)
    except Exception as e:
        import traceback
        print(f"FAILED {t}: {e}")
        traceback.print_exc()
