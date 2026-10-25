"""Empirical reach probe: drive the OSC-controlled eef straight toward a grid of
xy targets at table height and see how close it actually gets (steady-state
error), rather than assuming an analytic reach radius."""
import sys
import numpy as np
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=1000)
env.seed(1)
env.reset()
sim = env.env.sim

base_pos = sim.data.get_body_xpos("robot0_base").copy()
print("robot base pos", base_pos)

TARGET_Z = 0.97  # above table, matches typical approach height
targets = []
for x in np.arange(-0.45, 0.35, 0.075):
    for y in np.arange(-0.5, 0.55, 0.1):
        targets.append((round(x, 3), round(y, 3)))

results = []
for (tx, ty) in targets:
    env.reset()
    sim = env.env.sim
    for step in range(150):
        eef_pos = sim.data.get_site_xpos("gripper0_grip_site").copy()
        err = np.array([tx, ty, TARGET_Z]) - eef_pos
        delta = np.clip(err * 4.0, -0.15, 0.15)
        action = np.zeros(7)
        action[:3] = delta
        env.step(action)
    sim = env.env.sim
    final_eef = sim.data.get_site_xpos("gripper0_grip_site").copy()
    reach_err = np.linalg.norm(final_eef[:2] - np.array([tx, ty]))
    results.append((tx, ty, final_eef[0], final_eef[1], reach_err))

env.close()
print(f"{'tx':>7} {'ty':>7} {'fx':>7} {'fy':>7} {'xy_err':>7}")
for tx, ty, fx, fy, e in results:
    flag = "  <-- OUT OF REACH" if e > 0.05 else ""
    print(f"{tx:7.3f} {ty:7.3f} {fx:7.3f} {fy:7.3f} {e:7.3f}{flag}")
