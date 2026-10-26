"""Cleaner corner check: park every object except butter_1 far off-camera
(below the table), then move ONLY butter to each candidate-region corner one
at a time and render, so there's no clutter/ambiguity about which pixel blob
is which object."""
import sys
import os
from PIL import Image

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=1000)
env.seed(0)
env.reset()
e = env.env
sim = e.sim


def set_xy(name, xy, z=None):
    obj = e.objects_dict[name]
    joint = obj.joints[0]
    qs, _ = sim.model.get_joint_qpos_addr(joint)
    vs, ve = sim.model.get_joint_qvel_addr(joint)
    sim.data.qpos[qs:qs + 2] = xy
    if z is not None:
        sim.data.qpos[qs + 2] = z
    sim.data.qvel[vs:ve] = 0.0
    sim.forward()


others = ["basket_1", "milk_1", "cream_cheese_1", "tomato_sauce_1", "orange_juice_1", "alphabet_soup_1", "ketchup_1"]
for n in others:
    set_xy(n, (2.0, 2.0), z=0.1)  # parked far away, low

outdir = "/mgbridge/out/livingroom_corner_check"
os.makedirs(outdir, exist_ok=True)

corners = {
    "c_bl": (-0.22, -0.24), "c_br": (0.22, -0.24),
    "c_tl": (-0.22, 0.24), "c_tr": (0.22, 0.24),
    "c_mid_top": (0.0, 0.24), "c_mid_bottom": (0.0, -0.24),
    "c_left_mid": (-0.22, 0.0), "c_right_mid": (0.22, 0.0),
}

for name, (bx, by) in corners.items():
    set_xy("butter_1", (bx, by))
    for _ in range(10):
        e.step([0.0] * 6 + [-1.0])
    obs = e._get_observations()
    Image.fromarray(obs["agentview_image"]).save(f"{outdir}/{name}.png")
    print(f"{name}: butter={bx,by} saved")

env.close()
