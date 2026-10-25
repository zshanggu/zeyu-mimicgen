"""What does the wrist (eye-in-hand) camera see at the RESET pose, before any
robot motion -- same raycast-to-table-plane method already used for agentview.
The wrist cam is attached to the gripper, so unlike agentview its footprint
is centered wherever the arm's home pose points, not over the whole table."""
import sys
import numpy as np
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
env.seed(7)
env.reset()
sim = env.env.sim

print("all camera names:", [sim.model.camera_id2name(i) for i in range(sim.model.ncam)])

cam_name = "robot0_eye_in_hand"
cam_id = sim.model.camera_name2id(cam_name)
cam_pos = sim.data.cam_xpos[cam_id].copy()
cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
fovy = sim.model.cam_fovy[cam_id]
print("wrist camera pos", cam_pos, "fovy", fovy)

h_px, w_px = 256, 256
table_z = 0.90

def pixel_ray(u, v):
    aspect = w_px / h_px
    tan_half_fovy = np.tan(np.radians(fovy) / 2)
    dir_cam = np.array([u * tan_half_fovy * aspect, v * tan_half_fovy, -1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    return cam_mat @ dir_cam

corners = {"top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1), "center": (0, 0)}
for name, (u, v) in corners.items():
    d = pixel_ray(u, v)
    if abs(d[2]) < 1e-6:
        print(f"{name}: ray parallel to table, no intersection")
        continue
    t = (table_z - cam_pos[2]) / d[2]
    if t < 0:
        print(f"{name}: ray points away from table (t={t:.3f})")
        continue
    p = cam_pos + t * d
    print(f"{name}: hits z={table_z} at xy=({p[0]:.3f},{p[1]:.3f})  (ray dist {t:.3f})")

env.close()
