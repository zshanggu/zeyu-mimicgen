"""Two things, measured directly instead of assumed:
1. What ground-plane (table, z~0.9) xy extent is actually visible in the
   agentview camera's frame? (so we don't place objects where they vanish,
   as happened at plate y=0.49 in iteration 3)
2. What's the robot's empirical max horizontal reach via the OSC controller,
   probed by driving straight toward target xy and seeing where it stalls?
"""
import sys
import numpy as np
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
env.seed(7)
env.reset()
sim = env.env.sim

# ---- 1. camera ground-plane visible extent ----
cam_name = "agentview"
cam_id = sim.model.camera_name2id(cam_name)
cam_pos = sim.data.cam_xpos[cam_id].copy()
cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
fovy = sim.model.cam_fovy[cam_id]
print("camera pos", cam_pos, "fovy", fovy)

# mujoco camera looks down its -z axis in cam frame
h_px, w_px = 256, 256
table_z = 0.90  # approx tabletop height used elsewhere in this bridge

def pixel_ray(u, v):
    # u,v in [-1,1], standard mujoco convention: cam -z forward, +y up, +x right
    aspect = w_px / h_px
    tan_half_fovy = np.tan(np.radians(fovy) / 2)
    dir_cam = np.array([u * tan_half_fovy * aspect, v * tan_half_fovy, -1.0])
    dir_cam /= np.linalg.norm(dir_cam)
    dir_world = cam_mat @ dir_cam
    return dir_world

corners = {"top_left": (-1, 1), "top_right": (1, 1), "bottom_left": (-1, -1), "bottom_right": (1, -1), "center": (0, 0)}
pts = {}
for name, (u, v) in corners.items():
    d = pixel_ray(u, v)
    if abs(d[2]) < 1e-6:
        continue
    t = (table_z - cam_pos[2]) / d[2]
    p = cam_pos + t * d
    pts[name] = p
    print(f"{name}: ray hits z={table_z} plane at xy=({p[0]:.3f},{p[1]:.3f})")

env.close()
