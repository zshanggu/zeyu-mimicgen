"""Step 1: sanity-check that we can replay a source LIBERO demo frame-by-frame
and read back object/eef poses that line up with what's in the HDF5.
Run inside the viewer container (has robosuite/libero/mujoco)."""
import sys
import pathlib
import numpy as np
import h5py

sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
SRC_HDF5 = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"

from libero.libero.envs import OffScreenRenderEnv

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
env.seed(0)
env.reset()

f = h5py.File(SRC_HDF5, "r")
d0 = f["data/demo_0"]
states = d0["states"][:]
actions = d0["actions"][:]
ee_pos_hdf5 = d0["obs/ee_pos"][:]
print("states shape", states.shape, "actions shape", actions.shape)

bowl_body = env.env.objects_dict["akita_black_bowl_1"].root_body
plate_body = env.env.objects_dict["plate_1"].root_body
print("bowl_body", bowl_body, "plate_body", plate_body)

eef_site = env.env.robots[0].controller.eef_name
print("eef_site", eef_site)


def body_pose(name):
    bid = env.sim.model.body_name2id(name)
    return env.sim.data.body_xpos[bid].copy(), env.sim.data.body_xmat[bid].reshape(3, 3).copy()


def site_pos(name):
    sid = env.sim.model.site_name2id(name)
    return env.sim.data.site_xpos[sid].copy()


for t in [0, 10, 50, len(states) - 1]:
    env.regenerate_obs_from_state(states[t])
    bowl_pos, _ = body_pose(bowl_body)
    plate_pos, _ = body_pose(plate_body)
    eef_pos = site_pos(eef_site)
    print(f"t={t:4d} bowl={bowl_pos.round(3)} plate={plate_pos.round(3)} "
          f"eef(query)={eef_pos.round(3)} eef(hdf5 obs)={ee_pos_hdf5[t].round(3)} "
          f"diff={np.abs(eef_pos - ee_pos_hdf5[t]).max():.5f}")

print("success at final state:", env.env._check_success())
f.close()
print("PLUMBING CHECK DONE")
