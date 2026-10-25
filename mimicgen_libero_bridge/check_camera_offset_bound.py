"""With the agentview camera offset by (CAMERA_X_OFFSET, 0, CAMERA_Z_OFFSET),
recompute the real visible-y-bound(x) the same way as before, and check it
against the user's proposed Y=+-0.55 extremes -- both analytically (raycast)
and empirically (actual render at the extreme corners)."""
import sys
import numpy as np

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
import os

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
CAMERA_X_OFFSET = 0.5
CAMERA_Z_OFFSET = 0.4
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256


def camera_y_bound(cam_pos, cam_mat, fovy, x, table_z=0.90):
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


env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=10)
env.seed(0)
env.reset()
e = env.env
sim = e.sim
cid = sim.model.camera_name2id("agentview")
sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + np.array([CAMERA_X_OFFSET, 0.0, CAMERA_Z_OFFSET])
sim.forward()

cam_pos = sim.data.cam_xpos[cid].copy()
cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
fovy = float(sim.model.cam_fovy[cid])
print("new cam_pos", cam_pos, "fovy", fovy)

for x in [0.13, 0.10, 0.0, -0.1, -0.2, -0.3, -0.45]:
    print(f"x={x:+.3f}: y_bound=+-{camera_y_bound(cam_pos, cam_mat, fovy, x):.4f}")

# empirical: place bowl at extreme corners and render
outdir = "/mgbridge/out/camoffset_check"
os.makedirs(outdir, exist_ok=True)
bowl_obj = e.objects_dict["akita_black_bowl_1"]
plate_obj = e.objects_dict["plate_1"]


def set_xy(obj, xy):
    joint = obj.joints[0]
    qs, _ = sim.model.get_joint_qpos_addr(joint)
    vs, ve = sim.model.get_joint_qvel_addr(joint)
    sim.data.qpos[qs:qs + 2] = xy
    sim.data.qvel[vs:ve] = 0.0
    sim.forward()


tests = {
    "y_pos_055": ([-0.15, 0.55], [-0.15, 0.30]),
    "y_neg_040": ([-0.15, -0.40], [-0.15, -0.15]),
    "x013_y055": ([0.13, 0.55], [-0.10, 0.30]),
    "x013_y000": ([0.13, 0.0], [-0.10, 0.0]),
}
for name, (bowl_xy, plate_xy) in tests.items():
    set_xy(bowl_obj, bowl_xy)
    set_xy(plate_obj, plate_xy)
    e._post_process()
    e._update_observables(force=True)
    obs = e._get_observations()
    Image.fromarray(obs["agentview_image"]).save(f"{outdir}/{name}.png")
    print(f"{name}: bowl={bowl_xy} saved")

env.close()
