"""Use gripper closedness (not distance) to find the real grasp window, then
check grip_site-to-butter distance specifically during that window -- avoids
being confounded by post-release carry/drop phases."""
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
actions = f["data/demo_0/actions"][()]
f.close()

butter_body = "butter_1_main"
print(f"n states={len(states)}, n actions={len(actions)}")
print("action gripper dim (last col) over time (every 5th):")
print(actions[::5, -1])

zs, dists, grip_cmd = [], [], []
for t, st in enumerate(states):
    env.regenerate_obs_from_state(st)
    sim = env.env.sim
    butter_pos = sim.data.get_body_xpos(butter_body)
    grip_pos = sim.data.get_site_xpos("gripper0_grip_site")
    zs.append(butter_pos[2])
    dists.append(np.linalg.norm(butter_pos - grip_pos))

zs = np.array(zs)
dists = np.array(dists)
act_grip = actions[:, -1] if len(actions) == len(states) else actions[:len(states), -1]

print("\nt, butter_z, grip_dist, action_grip_cmd (every 5th row):")
for t in range(0, len(states), 5):
    ag = act_grip[t] if t < len(act_grip) else float("nan")
    print(f"  t={t:3d}  z={zs[t]:.4f}  dist={dists[t]:.4f}  grip_cmd={ag:+.2f}")

print(f"\nargmin dist: t={dists.argmin()} dist={dists.min():.4f} z={zs[dists.argmin()]:.4f}")
print(f"argmax z: t={zs.argmax()} z={zs.max():.4f} dist_there={dists[zs.argmax()]:.4f}")
env.close()
