"""Diagnose why the finger-contact grasp signal never fires for butter_1
during the native demo_0 replay -- enumerate ALL contacts involving any
gripper-related body at each state, not just the hardcoded
_FINGER_BODY_NAMES, to find what body actually registers contact with the
butter during the real grasp."""
import sys
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
SRC = "/mgbridge/_scratch_prepared_source_LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.hdf5"

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(0)
env.reset()
sim = env.env.sim

f = h5py.File(SRC, "r")
states = f["data/demo_0/states"][()]
f.close()

butter_body = env.env.objects_dict["butter_1"].root_body
print("butter_body:", butter_body)

all_gripper_bodies = set()
for i in range(sim.model.nbody):
    n = sim.model.body_id2name(i)
    if n and "gripper" in n.lower():
        all_gripper_bodies.add(n)
print("all gripper-related bodies:", sorted(all_gripper_bodies))

contact_bodies_over_time = set()
min_gripper_butter_dist = 1e9
for t, st in enumerate(states):
    env.regenerate_obs_from_state(st)
    sim = env.env.sim
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1])
        b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2])
        if b1 == butter_body or b2 == butter_body:
            other = b2 if b1 == butter_body else b1
            contact_bodies_over_time.add(other)
    butter_pos = sim.data.get_body_xpos(butter_body)
    grip_pos = sim.data.get_site_xpos("gripper0_grip_site")
    d = np.linalg.norm(butter_pos - grip_pos)
    min_gripper_butter_dist = min(min_gripper_butter_dist, d)

print("bodies ever in contact with butter during replay:", sorted(contact_bodies_over_time))
print("min gripper-site to butter distance over whole demo:", min_gripper_butter_dist)
env.close()
