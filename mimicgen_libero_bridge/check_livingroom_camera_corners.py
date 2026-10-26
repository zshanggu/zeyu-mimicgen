"""Empirical visual check: place butter_1 and basket_1 at the 4 corners of the
proposed widened placement box (x in [-0.22,0.22], y in [-0.24,0.24]) and
render agentview, to confirm nothing is clipped out of frame -- same method
as check_camera_offset_bound.py's empirical corner test for the kitchen
scenes, since LIVING_ROOM_SCENE2 has a different camera/table geometry and
nothing here should be assumed."""
import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=1000)
env.seed(0)
env.reset()
e = env.env
sim = e.sim


def set_xy(name, xy):
    obj = e.objects_dict[name]
    joint = obj.joints[0]
    qs, _ = sim.model.get_joint_qpos_addr(joint)
    vs, ve = sim.model.get_joint_qvel_addr(joint)
    sim.data.qpos[qs:qs + 2] = xy
    sim.data.qvel[vs:ve] = 0.0
    sim.forward()


outdir = "/mgbridge/out/livingroom_corner_check"
os.makedirs(outdir, exist_ok=True)

corners = {
    "bottom_left": (-0.22, -0.24),
    "bottom_right": (0.22, -0.24),
    "top_left": (-0.22, 0.24),
    "top_right": (0.22, 0.24),
    "center": (0.0, 0.0),
}

for name, (bx, by) in corners.items():
    # put butter near this corner, basket at a different corner (diagonal) so both show
    set_xy("butter_1", (bx, by))
    set_xy("basket_1", (-bx * 0.6, -by * 0.6))
    for _ in range(10):
        e.step([0.0] * 6 + [-1.0])
    obs = e._get_observations()
    Image.fromarray(obs["agentview_image"]).save(f"{outdir}/{name}.png")
    print(f"{name}: butter={bx,by} basket={-bx*0.6,-by*0.6} saved")

env.close()
