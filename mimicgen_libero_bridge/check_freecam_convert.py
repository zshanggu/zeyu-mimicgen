"""Convert MuJoCo free-camera params (azimuth/elevation/distance/lookat) to
a fixed camera pos+orientation, using MuJoCo's OWN camera math (mjv_updateCamera
via an actual scene update) rather than hand-derived trig, to avoid a sign
error. Then apply that pose to the agentview camera and render to confirm."""
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"

AZIMUTH = 180.30
ELEVATION = -29.30
DISTANCE = 1.6095
LOOKAT = np.array([0.0012, 0.0281, 0.8620])

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=10)
env.seed(0)
env.reset()
sim = env.env.sim
model = sim.model._model  # underlying mujoco.MjModel
data = sim.data._data     # underlying mujoco.MjData

cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, cam)
cam.azimuth = AZIMUTH
cam.elevation = ELEVATION
cam.distance = DISTANCE
cam.lookat[:] = LOOKAT

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
scn = mujoco.MjvScene(model, maxgeom=1000)
pert = mujoco.MjvPerturb()
mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

# scn.camera[1] is unpopulated for mono (non-stereo) rendering -- use camera[0] directly.
cam_pos = np.array(scn.camera[0].pos)
forward = np.array(scn.camera[0].forward)
up = np.array(scn.camera[0].up)
right = np.cross(forward, up)
right /= np.linalg.norm(right)
up2 = np.cross(right, forward)
up2 /= np.linalg.norm(up2)
# mujoco camera local frame: x=right, y=up, z=-forward (camera looks down -z)
cam_mat = np.stack([right, up2, -forward], axis=1)

print("computed cam_pos:", cam_pos)
print("computed forward:", forward)
print("computed cam_mat:\n", cam_mat)

cid = sim.model.camera_name2id("agentview")
sim.model.cam_pos[cid] = cam_pos
quat = np.zeros(4)
mujoco.mju_mat2Quat(quat, cam_mat.flatten())
sim.model.cam_quat[cid] = quat
sim.forward()

print("applied cam_xpos:", sim.data.cam_xpos[cid])
print("applied cam_xmat:", sim.data.cam_xmat[cid])

obs = env.env._get_observations()
Image.fromarray(obs["agentview_image"]).save("/mgbridge/out/freecam_check.png")
print("saved")
env.close()
