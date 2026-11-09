"""Living-room "move object to a region defined relative to the plate" task,
for LIVING_ROOM_SCENE6 (e.g. LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_
the_right_of_the_plate): a derivative of run_livingroom1_wide.py's 5-factor
design, adapted for a goal shape where the target is a FIXED table-attached
site rather than a free object -- (And (On moving site)).

v2 (this version): per explicit user request, plate_1 is now ALSO randomized
across the full widened bounds (X in [-0.35,0.00], Y in [-0.45,0.45], same as
every other living-room task this session), and the success-check region
(the site `living_room_table_plate_right_region`) is made to MOVE WITH the
plate rather than stay at its native fixed table coordinates.

Confirmed via direct probing (not assumed): the site's parent body is
`living_room_table`, whose body_xpos/body_xmat are the identity transform
([0,0,0] / I), so the site's local `site_pos` offset IS its world position
directly -- shifting `site_pos` by the same (new_plate_xy - native_plate_xy)
delta applied to plate_1 moves the site in lockstep, confirmed empirically
(shifting site_pos by an arbitrary test delta moved site_xpos by the exact
same amount). This is now architecturally a standard 5-factor design again
(dist, movingX, movingY, targetX, targetY) where target=plate_1 (an object),
with one addition: every place `target`'s position is set, the site is
re-derived from it via the same fixed local offset captured at the start of
that reset (site's native xy minus plate's native xy for that reset, ~(0,
0.10) matching plate_right_region sitting just right of plate_init_region).

Two gotchas carried over from this session's established patterns:
  - `site_pos` is a MODEL parameter, not per-step qpos/qvel state, so a plain
    `env.reset_to()` does NOT restore a shifted site after randomize_
    distractors' own internal `raw_env.reset()` (which rebuilds the model,
    resetting the site back to its native position) -- same class of bug
    fixed for SCENE3's stove fixture and SCENE2/5's cabinet fixture this
    session: capture `site_pos` before that internal reset, refresh the
    `sim` reference after (reset() builds a NEW MjSim), restore it, forward().
  - plate_1 is excluded from the mandatory distractor-randomization pass
    (added to task_object_names) since its position is now deliberately
    controlled by the 5-factor sampler, not left to chance like a real
    distractor -- same role "target" plays in every other object-target
    script this session.
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
SCRATCH_SRC = "/mgbridge/_scratch_prepared_source_{task}_{label}.hdf5"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.44

CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

FREECAM_AZIMUTH = 180.30
FREECAM_ELEVATION = -29.30
FREECAM_DISTANCE = 1.6095
FREECAM_LOOKAT = np.array([0.0012, 0.0281, 0.8620])


def _compute_freecam_pose(bddl_path):
    import mujoco
    from libero.libero.envs import OffScreenRenderEnv
    probe = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
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


_DEFAULT_TASK_FOR_CAMERA = "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate"
FREECAM_POS, FREECAM_QUAT = _compute_freecam_pose(
    os.path.join(BDDL_DIR, f"{_DEFAULT_TASK_FOR_CAMERA}.bddl"))
FREECAM_POS = FREECAM_POS + np.array([-0.1, 0.0, -0.2])

REGION_X_LO, REGION_X_HI = -0.35, 0.00
REGION_Y_LO, REGION_Y_HI = -0.45, 0.45
# plate_1 shares the identical box -- same bounds for moving and target.
TARGET_REGION_X_LO, TARGET_REGION_X_HI = REGION_X_LO, REGION_X_HI
TARGET_REGION_Y_LO, TARGET_REGION_Y_HI = REGION_Y_LO, REGION_Y_HI

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

TARGET_X_BINS = X_BINS
TARGET_Y_BINS = Y_BINS

DIST_BINS = {"near": (0.12, 0.20), "medium": (0.20, 0.30), "far": (0.30, 0.45)}

PRIMARY_COMBOS = [(d, mx, tx) for d in LEVELS for mx in X_LEVELS for tx in X_LEVELS]
SECONDARY_COMBOS = [(my, ty) for my in Y_LEVELS for ty in Y_LEVELS]

TARGET_TOTAL = int(os.environ.get("MG_TARGET_TOTAL", "100"))
MAX_ATTEMPTS = int(os.environ.get("MG_MAX_ATTEMPTS", "500"))
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3
PER_COMBO_GIVEUP_ATTEMPTS = 60


def parse_task_goal(bddl_path):
    """Returns (moving, target_name) for a goal of the form (And (On a b)),
    where b may be a free object OR a fixed site name -- caller determines
    which by checking the live env's objects_dict/object_sites_dict."""
    text = open(bddl_path).read()
    m = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n\)", text, re.S)
    goal_norm = re.sub(r"\s+", " ", m.group(1).strip())
    m2 = re.fullmatch(r"\(And \(On ([\w_]+) ([\w_]+)\)\)", goal_norm)
    if not m2:
        raise ValueError(f"goal doesn't match (And (On a b)): {goal_norm!r}")
    return m2.group(1), m2.group(2)


def find_other_plate(bddl_path, target):
    """Name of a `plate`-typed object other than `target` (target here is a
    site name, so every declared plate object qualifies as 'other')."""
    text = open(bddl_path).read()
    m = re.search(r"\(:objects(.*?)\(:obj_of_interest", text, re.S)
    if not m:
        return None
    plate_names = []
    for line in m.group(1).splitlines():
        line = line.strip()
        pm = re.match(r"((?:[\w_]+\s+)*[\w_]+)\s*-\s*plate\b", line)
        if pm:
            plate_names.extend(pm.group(1).split())
    others = [p for p in plate_names if p != target]
    return others[0] if others else None


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


class MG_LiberoGenericSite(RobosuiteInterface):
    """Grasp detection uses the demo's own recorded gripper command (not
    finger-contact), same proven reason as the other living-room scripts
    (small grocery items don't register finger-pad contact even while
    held). Subtask retargeting for the 'plate' subtask uses the SITE's pose
    (where the pudding must actually end up), not plate_1's own pose."""
    INTERFACE_TYPE = "libero"

    def __init__(self, env, moving, target_site):
        super().__init__(env=env)
        self.moving = moving
        self.target_site = target_site

    def get_object_poses(self):
        moving_body = self.env.env.objects_dict[self.moving].root_body
        poses = dict(bowl=self.get_object_pose(obj_name=moving_body, obj_type="body"))
        poses["plate"] = self.get_object_pose(obj_name=self.target_site, obj_type="site")
        return poses

    def set_last_gripper_cmd(self, cmd):
        self._last_gripper_cmd = float(cmd)

    def get_subtask_term_signals(self):
        cmd = getattr(self, "_last_gripper_cmd", None)
        if cmd is not None:
            return dict(grasp=int(cmd > 0))
        return dict(grasp=0)


class LiberoEnvSitePlateGeneric:
    """Standard 5-factor env (dist, movingX, movingY, targetX, targetY) where
    moving=chocolate_pudding_1, target=plate_1 -- plus the site-follows-plate
    coupling: whenever plate_1's xy is set, the paired site is re-derived
    from it via the fixed local offset captured fresh each reset (site's own
    native xy minus plate's native xy for that reset)."""

    def __init__(self, libero_env, moving, target_plate, target_site):
        self.base_env = libero_env
        self.moving = moving
        self.target_plate = target_plate
        self.target_site = target_site
        self.target_primary = None
        self.target_secondary = None
        self.rng = np.random.default_rng()
        self._cam_params = None
        # The native (pre-teleport) plate yaw for whichever reset() call most
        # recently returned successfully -- the post-generation check reuses
        # this as the "undisturbed" reference, since the robot's actual pick
        # -and-place motion (not just our own placement settle) can also spin
        # the plate (confirmed on demo 41 -- a 'medium' distance combo, so
        # not even a too-close initial placement).
        self.last_native_plate_yaw = None

    def _read_camera_params(self):
        sim = self.base_env.env.sim
        cid = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[cid] = FREECAM_POS
        sim.model.cam_quat[cid] = FREECAM_QUAT
        sim.forward()
        self._cam_params = (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                             float(sim.model.cam_fovy[cid]))

    def _camera_safe(self, xy):
        cam_pos, cam_mat, fovy = self._cam_params
        return abs(xy[1]) <= camera_y_bound(cam_pos, cam_mat, fovy, xy[0])

    def _obj_xy(self, name):
        sim = self.base_env.env.sim
        return sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[:2].copy()

    def _obj_z(self, name):
        sim = self.base_env.env.sim
        return float(sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[2])

    def _is_upright(self, name, thresh=0.8):
        """Reject objects that tipped/flipped over during physics settle.
        quat is MuJoCo (w,x,y,z); for a unit quaternion the z-component of
        the rotated up-vector is 1 - 2*(x^2+y^2) -- +1 upright, -1 upside
        down. Confirmed empirically: a tipped plate (demo 83) measured -1.0
        here vs. ~1.0 for every normal upright demo."""
        sim = self.base_env.env.sim
        quat = sim.data.get_body_xquat(self.base_env.env.objects_dict[name].root_body)
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return up_z > thresh

    def _obj_yaw(self, name):
        """Yaw (radians) via the standard ZYX-Euler extraction from a MuJoCo
        (w,x,y,z) quaternion -- reduces to 2*atan2(z,w) for a pure z-rotation
        (the common case here since objects stay upright), but this form is
        correct for small roll/pitch too."""
        sim = self.base_env.env.sim
        w, x, y, z = sim.data.get_body_xquat(self.base_env.env.objects_dict[name].root_body)
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _shift_site(self, new_plate_xy, native_plate_xy, native_site_pos):
        sim = self.base_env.env.sim
        sid = sim.model.site_name2id(self.target_site)
        sim.model.site_pos[sid][:2] = native_site_pos[:2] + (new_plate_xy - native_plate_xy)
        sim.forward()

    def _sample_box(self, x_bin, y_bin, x_bins=X_BINS, y_bins=Y_BINS):
        x_lo, x_hi = x_bins[x_bin]
        y_lo, y_hi = y_bins[y_bin]
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
        d_bin, moving_x_bin, target_x_bin = self.target_primary
        moving_y_bin, target_y_bin = self.target_secondary
        d_lo, d_hi = DIST_BINS[d_bin]
        sanity_lo = np.array([REGION_X_LO, REGION_Y_LO]) - 0.2
        sanity_hi = np.array([REGION_X_HI, REGION_Y_HI]) + 0.2

        for _ in range(MAX_OUTER_RETRIES):
            ret = self.base_env.reset()
            self._read_camera_params()
            sim = self.base_env.env.sim
            sid = sim.model.site_name2id(self.target_site)
            native_site_pos = sim.model.site_pos[sid].copy()
            native_plate_xy = self._obj_xy(self.target_plate)
            native_plate_yaw = self._obj_yaw(self.target_plate)

            moving_xy = self._sample_box(moving_x_bin, moving_y_bin)
            if moving_xy is None:
                continue
            target_xy = self._sample_target_xy(moving_xy, d_lo, d_hi, target_x_bin, target_y_bin)
            if target_xy is None:
                continue
            self._set_xy(self.moving, moving_xy)
            self._set_xy(self.target_plate, target_xy)
            self._shift_site(target_xy, native_plate_xy, native_site_pos)
            for _ in range(NUM_SETTLE_STEPS):
                ret = self.base_env.step(_DUMMY_ACTION)
            final_moving = self._obj_xy(self.moving)
            final_target = self._obj_xy(self.target_plate)
            if not (np.all(final_moving >= sanity_lo) and np.all(final_moving <= sanity_hi)
                    and np.all(final_target >= sanity_lo) and np.all(final_target <= sanity_hi)):
                continue
            if not (self._is_upright(self.moving) and self._is_upright(self.target_plate)):
                continue
            # Cheap early filter only -- reject placements where the initial
            # settle ALONE (two freshly-teleported objects landing close) has
            # already spun the plate past a generous margin. This is not
            # sufficient by itself: the robot's actual pick-and-place motion
            # can also spin the plate (confirmed on demo 41, a 'medium'
            # distance combo with no initial-settle collision at all), so the
            # authoritative check is the post-generation one in main() using
            # last_native_plate_yaw captured below.
            yaw_drift = abs((self._obj_yaw(self.target_plate) - native_plate_yaw + np.pi) % (2 * np.pi) - np.pi)
            if yaw_drift > np.radians(25.0):
                continue
            if self.is_success()["task"]:
                # initial placement already satisfies the goal before any
                # manipulation happens (demo 87's bug) -- discard and retry.
                continue
            self.last_native_plate_yaw = native_plate_yaw
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
        return {"env_name": "LIBERO_generic_site_plate_livingroom", "type": "libero"}


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def prepare_source(bddl_path, src_hdf5, moving, target_site, scratch_path):
    from libero.libero.envs import OffScreenRenderEnv
    import robomimic.utils.tensor_utils as TensorUtils
    from tqdm import tqdm

    shutil.copy(src_hdf5, scratch_path)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    env.seed(0)
    env.reset()
    env_interface = MG_LiberoGenericSite(env=env, moving=moving, target_site=target_site)

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
        ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_LiberoGenericSite"
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


def randomize_distractors(env, raw_env, traj, task_object_names, seed, target_site):
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    # Capture the (deliberately shifted) site position BEFORE raw_env.reset()
    # rebuilds the model and resets it back to native -- same pattern as the
    # SCENE3 stove-fixture / SCENE2-5 cabinet-fixture restore fix this
    # session. Must refresh the `sim` reference after reset() (it's a new
    # MjSim object).
    sim_before = raw_env.env.sim
    sid_before = sim_before.model.site_name2id(target_site)
    saved_site_pos = sim_before.model.site_pos[sid_before].copy()

    raw_env.seed(seed)
    raw_env.reset()
    sim_after = raw_env.env.sim
    sid_after = sim_after.model.site_name2id(target_site)
    sim_after.model.site_pos[sid_after] = saved_site_pos
    sim_after.forward()

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


def _geom_ids_for(sim, name):
    m = sim.model
    bodies = [(i, m.body_id2name(i) or "") for i in range(m.nbody)]
    ids = {i for i, bn in bodies if bn.startswith(name + "_") or bn == name}
    return set(g for g in range(m.ngeom) if m.geom_bodyid[g] in ids)


def check_final_geometry(raw_env, rec, moving, target_plate, native_plate_yaw,
                          moving_geoms, plate_geoms, rest_z, z_margin=0.010,
                          yaw_thresh_deg=25.0):
    """Authoritative post-generation check -- the robot's actual pick-and-
    place motion (not just our own pre-manipulation placement settle) can
    spin the plate or drop the pudding onto it. Two independent signals for
    "pudding resting on the plate rim instead of beside it on the table"
    (confirmed on demos 47/61/81): (a) an actual MuJoCo contact between the
    pudding's and plate's collision geoms at the final pose (demo 47 showed
    this directly), and (b) the pudding's final z elevated above its own
    isolated, no-plate-nearby resting height -- calibrated once per run in
    main() (0.44984, essentially zero variance across the whole wide
    region), NOT a per-episode "native" reference (that one carries a
    spurious ~3cm pre-settle spawn-height bias, confirmed by direct probe,
    and not a hardcoded magic constant either). A naive combined-footprint-
    radius-sum distance check was tried and discarded: the task's correct
    "to the right of the plate" placement routinely lands AT or even
    slightly inside that sum (the two objects are meant to sit adjacent),
    so the sum is not a meaningful "overlap" boundary here -- measured
    empirically to reject ~75% of otherwise-correct demos.

    Separately, the plate-yaw-spin check (demo 41 -- plate yaw drifted ~96
    deg from its own native value despite no overlap/contact at all) is
    unchanged, using last_native_plate_yaw."""
    final_moving_pos = rec["object_poses"][moving][-1][0]
    final_moving_quat = rec["object_poses"][moving][-1][1]
    final_plate_pos = rec["object_poses"][target_plate][-1][0]
    final_plate_quat = rec["object_poses"][target_plate][-1][1]

    sim = raw_env.env.sim
    e = raw_env.env

    def _set_pose(name, pos, quat):
        joint = e.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        sim.data.qpos[qs:qs + 3] = pos
        sim.data.qpos[qs + 3:qs + 7] = quat

    _set_pose(moving, final_moving_pos, final_moving_quat)
    _set_pose(target_plate, final_plate_pos, final_plate_quat)
    sim.forward()  # recompute contacts at this pose only -- no time integration

    ncon = 0
    for c in range(sim.data.ncon):
        con = sim.data.contact[c]
        if (con.geom1 in moving_geoms and con.geom2 in plate_geoms) or \
                (con.geom2 in moving_geoms and con.geom1 in plate_geoms):
            ncon += 1
    if ncon > 0:
        return False, f"pudding_touches_plate(ncon={ncon})"

    z = float(final_moving_pos[2])
    if z > rest_z + z_margin:
        return False, f"pudding_elevated(z={z:.4f}>{rest_z + z_margin:.4f})"

    def up_z(quat):
        w, x, y, z = quat
        return 1.0 - 2.0 * (x * x + y * y)

    # The reset()-time upright check only guards the pre-manipulation
    # settle; the robot's actual pick-and-place motion can also tip the
    # plate without a full flip or a yaw spin (confirmed post-hoc on a
    # production demo: up_z=0.74, a ~44 deg tilt, yaw-drift only ~-2 deg --
    # neither of the other checks would have caught it). Same threshold as
    # the reset()-time _is_upright.
    for name, quat in ((moving, final_moving_quat), (target_plate, final_plate_quat)):
        if up_z(quat) <= 0.8:
            return False, f"not_upright({name},up_z={up_z(quat):.2f})"

    def yaw_of(quat):
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    yaw_drift = abs((yaw_of(final_plate_quat) - native_plate_yaw + np.pi) % (2 * np.pi) - np.pi)
    if yaw_drift > np.radians(yaw_thresh_deg):
        return False, f"plate_spun(yaw_drift_deg={np.degrees(yaw_drift):.1f})"
    return True, None


def main():
    task_name = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/mgbridge/out"
    label = sys.argv[3] if len(sys.argv) > 3 else task_name

    bddl_path = os.path.join(BDDL_DIR, f"{task_name}.bddl")
    src_hdf5 = os.path.join(SRC_DIR, f"{task_name}_demo.hdf5")
    assert os.path.exists(bddl_path), bddl_path
    assert os.path.exists(src_hdf5), src_hdf5

    moving, target_site = parse_task_goal(bddl_path)

    from libero.libero.envs import OffScreenRenderEnv

    probe_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
    probe_env.seed(0); probe_env.reset()
    e = probe_env.env
    all_objects = set(e.objects_dict.keys())
    assert hasattr(e, "object_sites_dict") and target_site in e.object_sites_dict, \
        f"target {target_site!r} is not a known site"
    probe_env.close()

    target_plate = find_other_plate(bddl_path, target_site)
    assert target_plate is not None, "this task needs a plate object to physically move"
    print(f"task={task_name} moving={moving} target_site={target_site} target_plate={target_plate} "
          f"TARGET_TOTAL={TARGET_TOTAL} MAX_ATTEMPTS={MAX_ATTEMPTS}")

    scratch_path = SCRATCH_SRC.format(task=task_name, label=label)
    print("preparing source dataset...", flush=True)
    prepare_source(bddl_path, src_hdf5, moving, target_site, scratch_path)

    task_object_names = {moving, target_plate}
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

    def primary_key(p):
        attempts, successes = per_primary[p][0], per_primary[p][1]
        given_up = attempts >= PER_COMBO_GIVEUP_ATTEMPTS and successes == 0
        return (given_up, successes, attempts)

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    raw_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    env = LiberoEnvSitePlateGeneric(raw_env, moving, target_plate, target_site)
    env_interface = MG_LiberoGenericSite(env=env.base_env, moving=moving, target_site=target_site)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

    # One-time calibration for the post-generation geometry check: the
    # moving object's own isolated resting z (no plate nearby) in THIS wide
    # region, and the collision-geom id sets used for contact detection.
    # Confirmed by direct probe: this isolated value is essentially
    # constant (std ~2e-9) across the whole wide region -- a per-episode
    # "native" (pre-teleport) reading is NOT usable here, it carries a
    # spurious ~3cm pre-settle spawn-height bias relative to this.
    raw_env.reset()
    moving_geoms = _geom_ids_for(raw_env.env.sim, moving)
    plate_geoms = _geom_ids_for(raw_env.env.sim, target_plate)
    far_corner = np.array([REGION_X_LO, REGION_Y_LO])
    calib_xy = np.array([(REGION_X_LO + REGION_X_HI) / 2, (REGION_Y_LO + REGION_Y_HI) / 2])
    env._set_xy(target_plate, far_corner)
    env._set_xy(moving, calib_xy)
    for _ in range(NUM_SETTLE_STEPS):
        raw_env.step(_DUMMY_ACTION)
    rest_z = env._obj_z(moving)
    print(f"moving_geoms={moving_geoms} plate_geoms={plate_geoms} calibrated_rest_z={rest_z:.5f}", flush=True)
    geometry_discards = 0

    while len(succ_records) < TARGET_TOTAL and total_attempts < MAX_ATTEMPTS:
        primary = pick_least_used(PRIMARY_COMBOS, primary_key, rng_combo)
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
            rec, info = randomize_distractors(env, raw_env, traj, task_object_names,
                                               seed=8_000_000 + total_attempts * 10 + retry, target_site=target_site)
            if rec is not None:
                break
        if rec is None:
            distractor_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary, secondary)} "
                  f"succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
            continue

        geo_ok, geo_reason = check_final_geometry(raw_env, rec, moving, target_plate, env.last_native_plate_yaw,
                                                   moving_geoms, plate_geoms, rest_z)
        if not geo_ok:
            geometry_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary, secondary)} "
                  f"succeeded but final geometry check failed ({geo_reason}) -- discarded", flush=True)
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
          f"({distractor_discards} discarded after succeeded but distractor randomization failed, "
          f"{geometry_discards} discarded after final geometry check) ===")
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
        data_grp.attrs["target_plate_object"] = target_plate
        data_grp.attrs["target_site"] = target_site
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
