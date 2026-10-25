"""Verify LiberoEnvForDataGen.reset()'s new reject-and-reroll actually keeps
bowl/plate apart, and sample wrist-cam frame-0 visibility across the enlarged
shared region."""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

from libero.libero.envs import OffScreenRenderEnv
from env_interface import LiberoEnvForDataGen, BOWL_NAME, PLATE_NAME

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
raw_env.seed(99)
env = LiberoEnvForDataGen(raw_env)

outdir = "/mgbridge/wrist_check"
import os
os.makedirs(outdir, exist_ok=True)

dists = []
for i in range(20):
    env.reset()
    sim = raw_env.env.sim
    bowl_xy = sim.data.get_body_xpos(raw_env.env.objects_dict[BOWL_NAME].root_body)[:2].copy()
    plate_xy = sim.data.get_body_xpos(raw_env.env.objects_dict[PLATE_NAME].root_body)[:2].copy()
    d = np.linalg.norm(bowl_xy - plate_xy)
    dists.append(d)
    obs = raw_env.env._get_observations()
    wrist = obs["robot0_eye_in_hand_image"]
    agent = obs["agentview_image"]
    Image.fromarray(wrist).save(f"{outdir}/wrist_{i:02d}_b{bowl_xy[0]:.2f}_{bowl_xy[1]:.2f}_p{plate_xy[0]:.2f}_{plate_xy[1]:.2f}.png")
    Image.fromarray(agent).save(f"{outdir}/agent_{i:02d}.png")

print("min dist", min(dists), "mean", np.mean(dists))
print("all dists >= 0.16:", all(d >= 0.16 - 1e-6 for d in dists))
raw_env.close()
