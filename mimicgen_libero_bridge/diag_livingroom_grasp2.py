"""Deeper diag: per-gripper-body closest approach to butter over the whole
demo, plus full contact list (ALL pairs, not just butter-involving ones) at
the single timestep where the gripper is closest to butter -- to see what the
gripper is actually touching at its closest approach, and whether the grasp
happens via a different body/geom than assumed."""
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
gripper_bodies = ["gripper0_eef", "gripper0_leftfinger", "gripper0_rightfinger",
                   "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip"]

best_t, best_d = None, 1e9
per_body_min = {b: 1e9 for b in gripper_bodies}
butter_z_at_best = None
for t, st in enumerate(states):
    env.regenerate_obs_from_state(st)
    sim = env.env.sim
    butter_pos = sim.data.get_body_xpos(butter_body)
    for b in gripper_bodies:
        try:
            p = sim.data.get_body_xpos(b)
        except Exception:
            continue
        d = np.linalg.norm(p - butter_pos)
        per_body_min[b] = min(per_body_min[b], d)
        if d < best_d:
            best_d, best_t = d, t
            butter_z_at_best = butter_pos[2]

print("per-gripper-body min distance to butter over whole demo:")
for b, d in per_body_min.items():
    print(f"  {b}: {d:.4f}")
print(f"overall closest: t={best_t}, d={best_d:.4f}, butter_z={butter_z_at_best:.4f}")

# full contact list at the closest-approach timestep
env.regenerate_obs_from_state(states[best_t])
sim = env.env.sim
print(f"\nALL contacts at t={best_t} (ncon={sim.data.ncon}):")
for i in range(sim.data.ncon):
    c = sim.data.contact[i]
    b1 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom1])
    b2 = sim.model.body_id2name(sim.model.geom_bodyid[c.geom2])
    print(f"  {b1} <-> {b2}")

# also check butter's z trajectory to see if it's ever lifted off the table at all
print("\nbutter z over time (every 10th state):")
zs = []
for t, st in enumerate(states):
    env.regenerate_obs_from_state(st)
    sim = env.env.sim
    zs.append(sim.data.get_body_xpos(butter_body)[2])
zs = np.array(zs)
print("min/max/first/last butter z:", zs.min(), zs.max(), zs[0], zs[-1])
print("argmax z (peak lift) at t=", zs.argmax())

env.close()
