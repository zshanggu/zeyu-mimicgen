import sys
import numpy as np
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
from env_interface import LiberoEnvForDataGen, BOWL_NAME, PLATE_NAME, BOWL_PLATE_DIST_BINS

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
raw_env.seed(42)
env = LiberoEnvForDataGen(raw_env)

for bin_name in ["near", "medium", "far"]:
    env.target_bin = bin_name
    dists = []
    for _ in range(15):
        env.reset()
        sim = raw_env.env.sim
        bowl_xy = sim.data.get_body_xpos(raw_env.env.objects_dict[BOWL_NAME].root_body)[:2]
        plate_xy = sim.data.get_body_xpos(raw_env.env.objects_dict[PLATE_NAME].root_body)[:2]
        dists.append(np.linalg.norm(bowl_xy - plate_xy))
    lo, hi = BOWL_PLATE_DIST_BINS[bin_name]
    dists = np.array(dists)
    in_range = ((dists >= lo) & (dists <= hi)).sum()
    print(f"{bin_name} [{lo},{hi}]: dists={dists.round(3)} in_range={in_range}/{len(dists)}")

raw_env.close()
