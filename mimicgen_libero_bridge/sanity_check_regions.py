import sys
sys.path.insert(0, "/mgbridge")
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(999)

NUM_SETTLE_STEPS = 30
dummy = [0.0] * 6 + [-1.0]

for trial in range(20):
    env.reset()
    for _ in range(NUM_SETTLE_STEPS):
        env.step(dummy)
    bowl_pos = env.env.sim.data.body_xpos[env.env.sim.model.body_name2id(env.env.objects_dict["akita_black_bowl_1"].root_body)].copy()
    plate_pos = env.env.sim.data.body_xpos[env.env.sim.model.body_name2id(env.env.objects_dict["plate_1"].root_body)].copy()
    dist_xy = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])
    print(f"trial {trial}: bowl={bowl_pos.round(3)} plate={plate_pos.round(3)} xy_dist={dist_xy:.3f}")
env.close()
