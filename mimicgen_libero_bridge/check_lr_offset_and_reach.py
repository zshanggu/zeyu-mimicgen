"""Check (a) agentview visibility after applying the kitchen-style camera
offset (X+0.5, Z+0.4) to the living-room camera, and (b) reach feasibility
at the basket's actual working height (~0.68-0.70, hover/release height, not
the 0.56 table-height used for the earlier butter reach probe) for the
specific target point the user wants covered: X=0.1759, Y=-0.38."""
import sys
import numpy as np

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
CAMERA_X_OFFSET = 0.5
CAMERA_Z_OFFSET = 0.4
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256


def camera_y_bound(cam_pos, cam_mat, fovy, x, table_z):
    m = 1 - 2 * CAMERA_MARGIN_PX / CAMERA_IMG_SIZE

    def hit(u, v):
        t = np.tan(np.radians(fovy) / 2)
        d = np.array([u * t, v * t, -1.0]); d /= np.linalg.norm(d)
        d = cam_mat @ d
        tt = (table_z - cam_pos[2]) / d[2]
        return cam_pos + tt * d

    lo, hi = -m, m
    for _ in range(40):
        mid = (lo + hi) / 2
        if hit(0, mid)[0] > x:
            lo = mid
        else:
            hi = mid
    v = (lo + hi) / 2
    return hit(m, v)[1]


env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=1000)
env.seed(0)
env.reset()
sim = env.env.sim
cid = sim.model.camera_name2id("agentview")
sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + np.array([CAMERA_X_OFFSET, 0.0, CAMERA_Z_OFFSET])
sim.forward()
cam_pos = sim.data.cam_xpos[cid].copy()
cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
fovy = float(sim.model.cam_fovy[cid])
print("offset cam_pos:", cam_pos, "fovy:", fovy)

for x in [-0.30, -0.225, -0.15, -0.05, 0.0, 0.1, 0.1759, 0.2, 0.25, 0.3]:
    yb = camera_y_bound(cam_pos, cam_mat, fovy, x, table_z=0.445)
    print(f"x={x:+.4f} table_z=0.445 y_bound=+-{yb:.4f}")

print()
print("--- reach probe at release/hover height z=0.68-0.70 for target region ---")
base_pos = sim.data.get_body_xpos("robot0_base").copy()
print("robot base pos:", base_pos)

targets = [(0.1759, -0.38), (0.15, -0.35), (0.1, -0.30), (0.0, -0.30), (-0.1, -0.30),
           (0.1759, -0.30), (0.1759, -0.40), (0.2, -0.38), (0.1759, -0.45)]
TARGET_Z = 0.70
for (tx, ty) in targets:
    env.reset()
    sim = env.env.sim
    cid = sim.model.camera_name2id("agentview")
    sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + np.array([CAMERA_X_OFFSET, 0.0, CAMERA_Z_OFFSET])
    sim.forward()
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
    flag = "  <-- OUT OF REACH" if reach_err > 0.05 else ""
    print(f"tx={tx:+.4f} ty={ty:+.4f} -> final=({final_eef[0]:+.4f},{final_eef[1]:+.4f}) err={reach_err:.4f}{flag}")

env.close()
