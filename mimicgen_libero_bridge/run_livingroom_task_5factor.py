"""Living-room pick-and-place-into-container tasks (e.g. LIVING_ROOM_SCENE2_
pick_up_the_butter_and_put_it_in_the_basket): same balancing/always-randomize-
distractors logic as run_kitchen_task_6factor_v2.py, adapted for a scene with
NO cabinet-like fixture (fixtures_dict is empty here -- just a bare
living_room_table) and a goal of the form (And (In moving target_contain_
region)) instead of (And (On moving target)).

So this is the kitchen pipeline's 6 factors minus the cabinet appear/near/
medium/far dimension -- 5 factors: (dist, movingX, movingY, targetX, targetY)
-- with the mandatory per-episode distractor randomization (6 other grocery
objects here, e.g. milk/cream_cheese/tomato_sauce/orange_juice/alphabet_soup/
ketchup for the butter task) carried over unchanged from v2, since
randomize_distractors/write_episode are already fully generic (object-name-
agnostic, driven by objects_dict - all_objects - {moving,target}).

Arena geometry (robot reach + camera frame) was measured fresh for this
scene via check_livingroom_reach.py (empirical closed-loop OSC reach probe)
and check_livingroom_camera_corners2.py / check_livingroom_xbound.py
(empirical + analytic agentview frame-margin check) -- NOT reused from the
kitchen scenes, which have a different table, different robot-base offset,
and a different camera pose/orientation entirely. Findings:
  - robot0_base at x=-0.51 (vs kitchen's ~-0.66); reach is good (xy err<2cm)
    for x in [-0.225,0.225], y in [-0.25,0.25], but degrades sharply outside
    that box on every side (not just the "far edge" the kitchen table had) --
    e.g. x=-0.30 even develops a near-centerline failure band (y in
    [-0.05,0.05]) not seen at x=-0.225, likely a self-collision/kinematic
    quirk specific to this reach, not a simple radius -- so the SAFE box is
    kept inside the clean region, not pushed to the last point that still
    sort-of worked.
  - agentview camera (pos [0.607,0,0.96], looking back at -x,-z toward the
    robot) clips the +x (toward-camera) direction well before reach does:
    empirically x=0.22 already clips the top edge even at y=0, confirmed
    analytically via the same raycast-to-object-height method as the kitchen
    scenes' camera_y_bound (here evaluated at butter's resting height
    z=0.44, since x maps mostly to the vertical image axis and y to the
    horizontal one for this camera orientation -- opposite axis pairing from
    the kitchen cameras, worth noting since it's not an assumption carried
    over, it's how THIS camera's rotation matrix projects). x=0.18 is the
    last clean sample (v=-0.818, margin +-0.844); x=0.15 used as ceiling for
    a safety margin.
  - net region: X in [-0.225, 0.15] (asymmetric -- camera caps +x well
    before reach would), Y in [-0.24, 0.24] (reach-limited, symmetric; the
    camera's own y-bound at this x range is 0.22-0.34, looser than reach, so
    it's kept only as a secondary safety-net check via _camera_safe, same
    architecture as the kitchen pipeline).
"""
import sys
import os
import re
import json
import shutil
import numpy as np
import h5py

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

import distractor_lib as DL

BDDL_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90"
SRC_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_ori"
SCRATCH_SRC = "/mgbridge/_scratch_prepared_source_{task}.hdf5"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.44  # object resting height (butter), used for the camera raycast margin check

CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

# Same camera-offset technique as the kitchen SCENE1 v2 pivot: push the
# agentview camera back/up so it sees more of the table, which directly
# relaxes the camera-visibility constraint on usable placement area.
# Reused verbatim at the user's request rather than re-derived from scratch;
# re-verified for THIS camera's different pose/orientation via
# check_lr_offset_and_reach.py (y_bound at the offset camera is ±0.44 to
# ±0.59 across x in [-0.3,0.3] at table height -- comfortably looser than
# the reach constraint everywhere, so it's a pure safety-net check here, not
# the binding one).
# Camera now set via explicit MuJoCo free-camera params (azimuth/elevation/
# distance/lookat), not a cam_pos offset -- user-specified values, presumably
# read directly off the LiLo-VLA viewer's own camera controls. Converted to a
# fixed pos+quat using MuJoCo's OWN camera math (mjv_updateScene -> read back
# scn.camera[0].pos/forward/up), not hand-derived trig, to avoid a sign error
# (see check_freecam_convert.py). The result is a CONSTANT pose (none of
# azimuth/elevation/distance/lookat depend on sim state), computed once at
# import time and applied verbatim every reset.
FREECAM_AZIMUTH = 180.30
FREECAM_ELEVATION = -29.30
FREECAM_DISTANCE = 1.6095
FREECAM_LOOKAT = np.array([0.0012, 0.0281, 0.8620])


def _compute_freecam_pose():
    import mujoco
    from libero.libero.envs import OffScreenRenderEnv
    probe = OffScreenRenderEnv(
        bddl_file_name=os.path.join(
            BDDL_DIR, "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"),
        camera_heights=64, camera_widths=64, horizon=10)
    probe.seed(0)
    probe.reset()
    model = probe.env.sim.model._model
    data = probe.env.sim.data._data

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation, cam.distance = FREECAM_AZIMUTH, FREECAM_ELEVATION, FREECAM_DISTANCE
    cam.lookat[:] = FREECAM_LOOKAT

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scn = mujoco.MjvScene(model, maxgeom=1000)
    pert = mujoco.MjvPerturb()
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)

    cam_pos = np.array(scn.camera[0].pos)
    forward = np.array(scn.camera[0].forward)
    up = np.array(scn.camera[0].up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    cam_mat = np.stack([right, up, -forward], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, cam_mat.flatten())
    probe.close()
    return cam_pos, quat


FREECAM_POS, FREECAM_QUAT = _compute_freecam_pose()
# Explicit camera-position nudge requested on top of the free-camera pose
# above: 0.1 closer in X, 0.1 lower in Z. Translates the already-computed
# position only -- orientation (FREECAM_QUAT) is left unchanged, since
# "move the camera" means shift its location, not re-aim it.
FREECAM_POS = FREECAM_POS + np.array([-0.1, 0.0, -0.2])  # cumulative: -0.1 X, -0.2 Z total

# MOVING (butter) placement region: widened to match TARGET's full reach
# envelope per explicit instruction -- both butter and basket should range
# all the way to the edge of the workspace, not stay in a conservative
# interior safety margin. Grasp success is expected to drop near the edges
# (confirmed: grasp_descent_did_not_converge becomes more frequent there),
# which is the intended tradeoff -- the point is to see/collect the edge
# behavior, not to avoid it.
REGION_X_LO, REGION_X_HI = -0.225, 0.19
REGION_Y_LO, REGION_Y_HI = -0.40, 0.40

# TARGET (basket) placement region: the basket only needs the eef to get
# close enough to release over it, not a precision grasp, so it can safely
# use a much wider envelope. Re-measured via check_lr_offset_and_reach.py
# with a stronger controller (gain=8 vs the original gain=4 reach probe --
# the original probe's "out of reach" calls at y<=-0.35 turned out to be a
# slow-convergence false negative, not a real kinematic limit: the exact
# point below converged to <3mm error once given enough gain/steps). Full
# grid (release height z=0.70, gain=8, 400 steps):
#   x<=0.1:    y in [-0.55,0.55] all converge cleanly
#   x=0.1759:  y in [-0.38,0.38] clean, +-0.45 marginal (~2cm off), +-0.55 fails
#   x=0.25:    y in [-0.25,0.25] clean
#   x=0.35:    unreachable at any y
# i.e. a trapezoid, not a rectangle -- target_y_bound(x) below is a
# conservative piecewise-linear fit to that grid (small margin inside the
# measured-OK region at every breakpoint), used as an extra rejection filter
# alongside the TARGET_* box so sampling never proposes a point outside the
# real reach envelope. TARGET_REGION_X_HI=0.19 and the y_bound at x=0.1759
# (0.40) together guarantee the user's specified point (0.1759,-0.38) is
# covered with margin.
TARGET_REGION_X_LO, TARGET_REGION_X_HI = -0.225, 0.19
TARGET_REGION_Y_LO, TARGET_REGION_Y_HI = -0.40, 0.40

_TARGET_Y_BOUND_PTS = [(-0.30, 0.55), (0.00, 0.50), (0.10, 0.45), (0.1759, 0.40), (0.25, 0.22), (0.30, 0.0)]
_target_y_bound_xs = [p[0] for p in _TARGET_Y_BOUND_PTS]
_target_y_bound_ys = [p[1] for p in _TARGET_Y_BOUND_PTS]


def target_y_bound(x):
    return float(np.interp(x, _target_y_bound_xs, _target_y_bound_ys))


LEVELS = ["near", "medium", "far"]
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]

_x_edges = np.linspace(REGION_X_LO, REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_y_edges = np.linspace(REGION_Y_LO, REGION_Y_HI, 4)
Y_BINS = {lvl: (float(_y_edges[i]), float(_y_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}

_tx_edges = np.linspace(TARGET_REGION_X_LO, TARGET_REGION_X_HI, 4)
TARGET_X_BINS = {lvl: (float(_tx_edges[i]), float(_tx_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_ty_edges = np.linspace(TARGET_REGION_Y_LO, TARGET_REGION_Y_HI, 4)
TARGET_Y_BINS = {lvl: (float(_ty_edges[i]), float(_ty_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}

# region diagonal ~0.61m (0.375 x 0.48); bins sized as roughly 20/33/49-74%
# fractions of that, same relative shape as the kitchen BP_BINS.
DIST_BINS = {"near": (0.12, 0.20), "medium": (0.20, 0.30), "far": (0.30, 0.45)}

PRIMARY_COMBOS = [(d, mx, tx) for d in LEVELS for mx in X_LEVELS for tx in X_LEVELS]
SECONDARY_COMBOS = [(my, ty) for my in Y_LEVELS for ty in Y_LEVELS]

TARGET_TOTAL = 100
MAX_ATTEMPTS = 500
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3


def parse_task_goal(bddl_path):
    text = open(bddl_path).read()
    m = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n\)", text, re.S)
    goal_norm = re.sub(r"\s+", " ", m.group(1).strip())
    m2 = re.fullmatch(r"\(And \((On|In) ([\w_]+) ([\w_]+)\)\)", goal_norm)
    if not m2:
        raise ValueError(f"goal doesn't match (And (On|In a b)): {goal_norm!r}")
    _, moving, target_raw = m2.groups()
    target = re.sub(r"_contain_region$", "", target_raw)
    return moving, target


def camera_y_bound(cam_pos, cam_mat, fovy, x, table_z=TABLE_Z):
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


def label_for(primary, secondary):
    d, mx, tx = primary
    my, ty = secondary
    return f"dist-{d}_movX-{mx}_movY-{my}_tgtX-{tx}_tgtY-{ty}"


class MG_LiberoBowlPlateGeneric(RobosuiteInterface):
    INTERFACE_TYPE = "libero"
    _FINGER_BODY_NAMES = ("gripper0_leftfinger", "gripper0_rightfinger",
                           "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")

    def __init__(self, env, moving, target):
        super().__init__(env=env)
        self.moving = moving
        self.target = target

    def get_object_poses(self):
        moving_body = self.env.env.objects_dict[self.moving].root_body
        target_body = self.env.env.objects_dict[self.target].root_body
        return dict(bowl=self.get_object_pose(obj_name=moving_body, obj_type="body"),
                    plate=self.get_object_pose(obj_name=target_body, obj_type="body"))

    def _grasped_object_name(self):
        sim = self.env.sim
        finger_ids = set()
        for name in self._FINGER_BODY_NAMES:
            try:
                finger_ids.add(sim.model.body_name2id(name))
            except Exception:
                continue
        if not finger_ids:
            return None
        root_to_name = {obj.root_body: name for name, obj in self.env.env.objects_dict.items()}
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            b1, b2 = sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2]
            for fb, ob in ((b1, b2), (b2, b1)):
                if fb in finger_ids:
                    n = sim.model.body_id2name(ob)
                    if n in root_to_name:
                        return root_to_name[n]
        return None

    def set_last_gripper_cmd(self, cmd):
        """prepare_source() calls this right before get_datagen_info(action=...)
        for each source-demo timestep. Grasp detection here uses the demo's OWN
        recorded gripper actuation command (close = cmd>0), not finger-geom
        contact -- confirmed via diag_livingroom_grasp*.py that butter_1 (and
        likely other small LIVING_ROOM grocery items) never registers ANY
        finger-pad contact with the object even while being rigidly carried
        (grip_site-to-object distance stays constant to <0.2mm through the
        whole lift/carry/arc, i.e. clearly grasped, but sim.data.ncon never
        includes a finger<->object pair at any replayed state -- this LIBERO
        scene's demos appear to hold small objects via something other than
        frictional finger-pad contact, e.g. a scripted/kinematic attachment
        during collection that qpos-replay reproduces positionally but not as
        a real contact pair). The action's own open/close command is exactly
        the signal the original demonstrator used to decide "grasp now," so
        it's the correct generic proxy whether or not contact happens to
        register."""
        self._last_gripper_cmd = float(cmd)

    def get_subtask_term_signals(self):
        cmd = getattr(self, "_last_gripper_cmd", None)
        if cmd is not None:
            return dict(grasp=int(cmd > 0))
        return dict(grasp=int(self._grasped_object_name() == self.moving))


class LiberoEnv5FactorGeneric:
    def __init__(self, libero_env, moving, target):
        self.base_env = libero_env
        self.moving = moving
        self.target = target
        self.target_primary = None
        self.target_secondary = None
        self.rng = np.random.default_rng()
        self._cam_params = None

    def _read_camera_params(self):
        sim = self.base_env.env.sim
        cid = sim.model.camera_name2id("agentview")
        # cam_pos/cam_quat reset to native every hard env.reset()/
        # regenerate_obs_from_state, so the free-camera pose must be
        # reapplied every time this is called.
        sim.model.cam_pos[cid] = FREECAM_POS
        sim.model.cam_quat[cid] = FREECAM_QUAT
        sim.forward()
        self._cam_params = (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                             float(sim.model.cam_fovy[cid]))

    def _camera_safe(self, xy):
        cam_pos, cam_mat, fovy = self._cam_params
        return abs(xy[1]) <= camera_y_bound(cam_pos, cam_mat, fovy, xy[0])

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _sample_box(self, x_bin, y_bin):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = Y_BINS[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = self.rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            if abs(cand[1]) <= target_y_bound(cand[0]) and self._camera_safe(cand):
                return cand
        return None

    def _sample_target_xy(self, moving_xy, d_lo, d_hi, x_bin, y_bin):
        x_lo, x_hi = TARGET_X_BINS[x_bin]
        y_lo, y_hi = TARGET_Y_BINS[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(d_lo, d_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = moving_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if (x_lo <= cand[0] <= x_hi and y_lo <= cand[1] <= y_hi
                    and abs(cand[1]) <= target_y_bound(cand[0])):
                return cand
        return None

    def reset(self):
        # Occasionally a freshly-placed object overlaps something badly enough
        # that the settle steps' contact solver flings it meters away (a real
        # MuJoCo "explosion", not a sampling bug -- confirmed via
        # diag_scripted_grasp.py: butter ended up at [-6.6,5.3] post-settle,
        # ~0.3 units/step away from the FAR_AWAY distractor-parking formula's
        # ballpark, i.e. a genuine contact-force blowup, not a region-sampling
        # error). A full base_env.reset() (clean physics state, not just a
        # re-sampled xy) before each retry is needed since the exploded state
        # can leave nonzero velocities that would corrupt a retry in place.
        d_bin, moving_x_bin, target_x_bin = self.target_primary
        moving_y_bin, target_y_bin = self.target_secondary
        d_lo, d_hi = DIST_BINS[d_bin]
        sanity_lo = np.array([REGION_X_LO, REGION_Y_LO]) - 0.2
        sanity_hi = np.array([REGION_X_HI, REGION_Y_HI]) + 0.2

        for _ in range(MAX_OUTER_RETRIES):
            ret = self.base_env.reset()
            self._read_camera_params()
            moving_xy = self._sample_box(moving_x_bin, moving_y_bin)
            if moving_xy is None:
                continue
            target_xy = self._sample_target_xy(moving_xy, d_lo, d_hi, target_x_bin, target_y_bin)
            if target_xy is None:
                continue
            self._set_xy(self.moving, moving_xy)
            self._set_xy(self.target, target_xy)
            for _ in range(NUM_SETTLE_STEPS):
                ret = self.base_env.step(_DUMMY_ACTION)
            sim = self.base_env.env.sim
            final_moving = sim.data.get_body_xpos(self.base_env.env.objects_dict[self.moving].root_body)[:2]
            final_target = sim.data.get_body_xpos(self.base_env.env.objects_dict[self.target].root_body)[:2]
            if (np.all(final_moving >= sanity_lo) and np.all(final_moving <= sanity_hi)
                    and np.all(final_target >= sanity_lo) and np.all(final_target <= sanity_hi)):
                return ret
        raise RuntimeError(f"could not jointly satisfy {self.target_primary}/{self.target_secondary} "
                            f"(with a sane post-settle position) in {MAX_OUTER_RETRIES} outer tries")

    def get_state(self):
        return {"states": self.base_env.get_sim_state()}

    def reset_to(self, state):
        ret = self.base_env.regenerate_obs_from_state(state["states"])
        self._read_camera_params()
        return ret

    def get_observation(self):
        raw = self.base_env.env._get_observations()
        obs = dict(raw)
        if "agentview_image" in raw:
            obs["agentview_rgb"] = raw["agentview_image"]
        if "robot0_eye_in_hand_image" in raw:
            obs["eye_in_hand_rgb"] = raw["robot0_eye_in_hand_image"]
        if "robot0_eef_pos" in raw:
            obs["ee_pos"] = raw["robot0_eef_pos"]
        if "robot0_eef_quat" in raw:
            obs["ee_ori"] = raw["robot0_eef_quat"]
        if "robot0_gripper_qpos" in raw:
            obs["gripper_states"] = raw["robot0_gripper_qpos"]
        if "robot0_joint_pos" in raw:
            obs["joint_states"] = raw["robot0_joint_pos"]
        ee_pos, ee_ori = obs.get("ee_pos"), obs.get("ee_ori")
        if ee_pos is not None and ee_ori is not None:
            obs["ee_states"] = np.concatenate([ee_pos, ee_ori])
        return obs

    def step(self, action):
        return self.base_env.step(action)

    def is_success(self):
        return {"task": bool(self.base_env.env._check_success())}

    def render(self, *a, **k):
        raise NotImplementedError

    def serialize(self):
        return {"env_name": "LIBERO_generic_5factor_livingroom", "type": "libero"}


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def prepare_source(bddl_path, src_hdf5, moving, target, scratch_path):
    from libero.libero.envs import OffScreenRenderEnv
    import robomimic.utils.tensor_utils as TensorUtils
    from tqdm import tqdm

    shutil.copy(src_hdf5, scratch_path)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    env.seed(0)
    env.reset()
    env_interface = MG_LiberoBowlPlateGeneric(env=env, moving=moving, target=target)

    f = h5py.File(scratch_path, "a")
    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for ep in tqdm(demo_keys):
        ep_grp = f[f"data/{ep}"]
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]
        all_infos = []
        for t in range(len(states)):
            env.regenerate_obs_from_state(states[t])
            env_interface.set_last_gripper_cmd(actions[t][-1])
            all_infos.append(env_interface.get_datagen_info(action=actions[t]).to_dict())
        merged = TensorUtils.list_of_flat_dict_to_dict_of_list(all_infos)
        for k in merged:
            if k in ["object_poses", "subtask_term_signals"]:
                merged[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(merged[k])
                for k2 in merged[k]:
                    merged[k][k2] = np.array(merged[k][k2])
            else:
                merged[k] = np.array(merged[k])
        if "datagen_info" in ep_grp:
            del ep_grp["datagen_info"]
        for k in merged:
            if k in ["object_poses", "subtask_term_signals"]:
                for k2 in merged[k]:
                    ep_grp.create_dataset(f"datagen_info/{k}/{k2}", data=merged[k][k2])
            else:
                ep_grp.create_dataset(f"datagen_info/{k}", data=merged[k])
        ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_LiberoBowlPlateGeneric"
        ep_grp["datagen_info"].attrs["env_interface_type"] = "libero"
    f.close()
    env.close()


def write_episode(path, rec, attrs):
    with h5py.File(path, "w") as out_f:
        g = out_f.create_group("data").create_group("demo_0")
        o = g.create_group("obs")
        ee = np.stack(rec["ee_states"], 0)
        o.create_dataset("gripper_states", data=np.stack(rec["gripper_states"], 0))
        o.create_dataset("joint_states", data=np.stack(rec["joint_states"], 0))
        o.create_dataset("ee_states", data=ee)
        o.create_dataset("ee_pos", data=ee[:, :3])
        o.create_dataset("ee_ori", data=ee[:, 3:])
        for key, frames in (("agentview_rgb", rec["agent"]), ("eye_in_hand_rgb", rec["wrist"])):
            arr = np.stack(frames, 0)
            o.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:], compression="gzip", compression_opts=1)
        for obj_name, poses in rec["object_poses"].items():
            o.create_dataset(f"{obj_name}_pos", data=np.stack([p[0] for p in poses], 0))
            o.create_dataset(f"{obj_name}_quat", data=np.stack([p[1] for p in poses], 0))
        g.create_dataset("actions", data=np.stack(rec["actions"]))
        g.create_dataset("states", data=np.stack(rec["states"]))
        n_ = len(rec["actions"])
        dones = np.zeros(n_, np.uint8); dones[-1] = 1
        g.create_dataset("dones", data=dones)
        for k, v in attrs.items():
            g.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)


def randomize_distractors(env, raw_env, traj, task_object_names, seed):
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()
    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, task_object_names)
    scene.measure_radii()
    if not scene.distractors:
        return None, "no_distractors_in_this_task"

    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
                               p_remove=0.3, table_z=TABLE_Z, vis=vis, xy_lo=(-0.5, -0.65), xy_hi=(0.5, 0.65))
    if result is None:
        return None, "no_valid_layout"
    st, removed, moved = result

    pre = {n: scene.sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy() for n in task_object_names}
    env.reset_to({"states": st})
    for _ in range(10):
        env.base_env.step(_DUMMY_ACTION)
    sim = raw_env.env.sim
    for n in task_object_names:
        post = sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy()
        if np.linalg.norm(post - pre[n]) > 0.01:
            return None, f"task_object_drift({n})"

    all_object_names = list(raw_env.env.objects_dict.keys())
    actions, states, ee_states, gripper_states, joint_states, agent, wrist = ([] for _ in range(7))
    object_poses = {n: [] for n in all_object_names}
    for action in traj["actions"]:
        states.append(sim.get_state().flatten())
        actions.append(action)
        o = env.get_observation()
        gripper_states.append(o["gripper_states"])
        joint_states.append(o["joint_states"])
        ee_states.append(np.hstack((o["ee_pos"], T.quat2axisangle(o["ee_ori"]))))
        agent.append(o["agentview_rgb"])
        wrist.append(o["eye_in_hand_rgb"])
        for n in all_object_names:
            body = raw_env.env.objects_dict[n].root_body
            object_poses[n].append((sim.data.get_body_xpos(body).copy(), sim.data.get_body_xquat(body).copy()))
        env.base_env.step(action.tolist() if hasattr(action, "tolist") else list(action))

    if not env.is_success()["task"]:
        return None, "replay_failed"
    rec = dict(actions=actions, states=states, ee_states=ee_states, gripper_states=gripper_states,
               joint_states=joint_states, agent=agent, wrist=wrist, object_poses=object_poses)
    return rec, dict(removed=removed, moved=moved, seed=int(seed))


def main():
    task_name = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/mgbridge/out"
    label = sys.argv[3] if len(sys.argv) > 3 else task_name

    bddl_path = os.path.join(BDDL_DIR, f"{task_name}.bddl")
    src_hdf5 = os.path.join(SRC_DIR, f"{task_name}_demo.hdf5")
    assert os.path.exists(bddl_path), bddl_path
    assert os.path.exists(src_hdf5), src_hdf5

    moving, target = parse_task_goal(bddl_path)
    print(f"task={task_name} moving={moving} target={target}")

    scratch_path = SCRATCH_SRC.format(task=task_name)
    print("preparing source dataset...", flush=True)
    prepare_source(bddl_path, src_hdf5, moving, target, scratch_path)

    from libero.libero.envs import OffScreenRenderEnv

    probe_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
    probe_env.seed(0); probe_env.reset()
    all_objects = set(probe_env.env.objects_dict.keys())
    probe_env.close()
    task_object_names = {moving, target}
    print(f"objects_dict={all_objects}, task_object_names={task_object_names}, "
          f"distractors={all_objects - task_object_names}")

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=scratch_path, filter_key=None, start=None, n=None)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    rng_combo = np.random.default_rng(42)
    per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}
    per_primary_secondary = {}
    succ_records = []
    total_attempts = 0
    distractor_discards = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    raw_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    env = LiberoEnv5FactorGeneric(raw_env, moving, target)
    env_interface = MG_LiberoBowlPlateGeneric(env=env.base_env, moving=moving, target=target)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

    while len(succ_records) < TARGET_TOTAL and total_attempts < MAX_ATTEMPTS:
        primary = pick_least_used(PRIMARY_COMBOS, lambda p: (per_primary[p][1], per_primary[p][0]), rng_combo)
        for sc in SECONDARY_COMBOS:
            per_primary_secondary.setdefault((primary, sc), [0, 0])
        secondary = pick_least_used(SECONDARY_COMBOS, lambda sc: tuple(per_primary_secondary[(primary, sc)]), rng_combo)
        env.target_primary = primary
        env.target_secondary = secondary
        per_primary[primary][0] += 1
        per_primary_secondary[(primary, secondary)][0] += 1
        try:
            traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                            transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                            render=False, video_writer=None)
        except Exception as e:  # noqa: BLE001
            total_attempts += 1
            if total_attempts % 20 == 0:
                print(f"[attempt {total_attempts}] exception for {primary}/{secondary}: {type(e).__name__}: {e}", flush=True)
            continue
        total_attempts += 1
        if not traj["success"]:
            if total_attempts % 10 == 0:
                print(f"[attempt {total_attempts}] {label_for(primary, secondary)} success=False "
                      f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
            continue

        rec = info = None
        for retry in range(N_DISTRACTOR_RETRIES):
            rec, info = randomize_distractors(env, raw_env, traj, task_object_names, seed=8_000_000 + total_attempts * 10 + retry)
            if rec is not None:
                break
        if rec is None:
            distractor_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary, secondary)} "
                  f"bowl/plate succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
            continue

        per_primary[primary][1] += 1
        per_primary_secondary[(primary, secondary)][1] += 1
        idx = len(succ_records)
        succ_records.append((primary, secondary, info))
        write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                      attrs=dict(dist_bin=primary[0], moving_x_bin=primary[1], target_x_bin=primary[2],
                                 moving_y_bin=secondary[0], target_y_bin=secondary[1],
                                 removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
        print(f"[attempt {total_attempts}] {label_for(primary, secondary)} success=True "
              f"distractors: removed={info['removed']} moved={list(info['moved'].keys())} "
              f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)

    raw_env.close()
    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"({distractor_discards} discarded after bowl/plate succeeded but distractor randomization failed) ===")
    for p in PRIMARY_COMBOS:
        a, s = per_primary[p]
        print(f"  dist-{p[0]}_movX-{p[1]}_tgtX-{p[2]}: {s}/{a}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": task_name})
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = total_attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_records)
        data_grp.attrs["moving_object"] = moving
        data_grp.attrs["target_object"] = target
        data_grp.attrs["distractors_always_randomized"] = True
        data_grp.attrs["stratification"] = json.dumps({
            "dist_bins_m": DIST_BINS, "x_bins_m": X_BINS, "y_bins_m": Y_BINS,
        })
        for i, (primary, secondary, info) in enumerate(succ_records):
            group_name = f"demo_{i:03d}_{label_for(primary, secondary)}"
            with h5py.File(os.path.join(tmp_dir, f"{i:04d}.hdf5"), "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=group_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(scratch_path)
    print(f"wrote {out_path}: {len(succ_records)} episodes, every one with randomized distractors")


if __name__ == "__main__":
    main()
