"""Place a single marker-like object (reuse bowl) at controlled xy via direct
sim state poke (bypass the sampler) and see where it falls out of the wrist
frame at reset pose, to find the TRUE empirical visible-y cutoff."""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
from env_interface import BOWL_NAME

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
env.seed(1)
env.reset()
sim = env.env.sim
bowl_obj = env.env.objects_dict[BOWL_NAME]
bowl_joint = bowl_obj.joints[0] if hasattr(bowl_obj, "joints") else None
qpos_addr = sim.model.get_joint_qpos_addr(bowl_joint)
start = qpos_addr[0]

import os
outdir = "/mgbridge/wrist_check/sweep"
os.makedirs(outdir, exist_ok=True)

for y in [-0.3,-0.2,-0.1,0.0,0.1,0.15,0.2,0.25,0.3,0.35]:
    qpos = sim.data.qpos.copy()
    qpos[start:start+2] = [-0.10, y]
    qpos[start+2] = 0.95
    sim.data.qpos[:] = qpos
    sim.forward()
    obs = env.env._get_observations()
    img = obs["robot0_eye_in_hand_image"]
    Image.fromarray(img).save(f"{outdir}/y_{y:+.2f}.png")

for x in [-0.45,-0.35,-0.25,-0.15,-0.05,0.0,0.05,0.10]:
    qpos = sim.data.qpos.copy()
    qpos[start:start+2] = [x, 0.0]
    qpos[start+2] = 0.95
    sim.data.qpos[:] = qpos
    sim.forward()
    obs = env.env._get_observations()
    img = obs["robot0_eye_in_hand_image"]
    Image.fromarray(img).save(f"{outdir}/x_{x:+.2f}.png")

env.close()
print("done")
