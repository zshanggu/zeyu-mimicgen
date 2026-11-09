"""KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle,
adapted from run_kitchen_task_6factor_v2.py (the generic bowl/plate 6-factor
script already used for KITCHEN_SCENE1/SCENE2 bowl-on-plate tasks). The goal
here is `(And (On akita_black_bowl_1 akita_black_bowl_2))` -- front bowl
stacked on the middle bowl -- which matches the SAME `(And (On a b))` shape
`parse_task_goal` already expects, and both bowls are plain pokeable objects
(no fixture-site retargeting needed, unlike SCENE3's stove), so the existing
MG_LiberoBowlPlateGeneric/LiberoEnv6FactorGeneric classes are reused
unchanged -- only the region/camera constants below and one bugfix differ
from the precedent script.

Region widened per explicit instruction, reusing the exact bounds from the
SCENE3 (moka pot/stove) session: moving-bowl region X in [-0.35, 0.00], Y in
[-0.45, 0.45]. The old script's asymmetric cabinet-present/absent Y split
is dropped -- a single Y range was requested here, and the clearance logic
below (ported from the SCENE3 fix) makes the asymmetry unnecessary: it
directly rejects candidates that interpenetrate the cabinet or another
object's native spot, rather than relying on a coarse "stay further from
Y=-0.45 when the cabinet might be present" heuristic.

Camera offset set to (0, 0): confirmed by direct rendering (session probe)
that this scene's native agentview camera already frames the full counter
(bowls, plate, cabinet) correctly, unlike the old script's (0.5, 0.4) which
was tuned for the narrower original bowl/plate region. `_camera_safe`
still rejects any out-of-frame candidate regardless of the offset chosen,
so this is a safety net either way, not a leap of faith.

Bugfix ported from the SCENE3 session: `randomize_distractors`'s own
`raw_env.reset()` call silently re-randomizes ANY BDDL-region-placed
fixture's position (confirmed root cause of a dataset-wide bug there --
`env.reset_to()` only restores qpos/qvel, not a fixture's model-baked body
position). `wooden_cabinet_1` here is exactly such a fixture, so the same
capture-before/restore-after poke is applied -- otherwise the `cabinet_state`
label (near/medium/far/absent) would stop reliably describing where the
cabinet actually ends up in the saved episode.
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
TABLE_Z = 0.90

CAMERA_X_OFFSET = 0.0
CAMERA_Z_OFFSET = 0.0
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

# Widened bounds, reused verbatim from the SCENE3 (moka pot/stove) session.
REGION_X_LO, REGION_X_HI = -0.35, 0.00
Y_MIN, Y_MAX = -0.45, 0.45

CABINET_FIXTURE = "wooden_cabinet_1"
CABINET_RANGES = {
    "near": "(-0.36 -0.42 -0.30 -0.29)",
    "medium": "(-0.30 -0.42 -0.24 -0.29)",
    "far": "(-0.15 -0.33 0.15 -0.29)",
    "absent": "(1.99 1.99 2.01 2.01)",
}
CABINET_EDGE_BINS = {"near": (0.14, 0.20), "medium": (0.20, 0.26), "far": (0.26, 0.32)}

LEVELS = ["near", "medium", "far"]
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]
CABINET_LEVELS = ["near", "medium", "far", "absent"]

_x_edges = np.linspace(REGION_X_LO, REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_y_edges = np.linspace(Y_MIN, Y_MAX, 4)
Y_BINS = {lvl: (float(_y_edges[i]), float(_y_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}


def y_bins_for(cabinet_state):
    return Y_BINS


BP_BINS = {"near": (0.16, 0.26), "medium": (0.26, 0.38), "far": (0.38, 0.55)}
PRIMARY_COMBOS = [(bp, bx, px) for bp in LEVELS for bx in X_LEVELS for px in X_LEVELS]
SECONDARY_COMBOS = [(by, py) for by in Y_LEVELS for py in Y_LEVELS]

TARGET_TOTAL = int(os.environ.get("MG_TARGET_TOTAL", "100"))
MAX_ATTEMPTS = int(os.environ.get("MG_MAX_ATTEMPTS", "500"))
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3

# Same clearance-safety mechanism built for SCENE3's stove-knob/frypan
# collision bug: reject a candidate XY that interpenetrates the cabinet
# fixture's own geoms, or another object's CURRENT (native, not-yet-
# distractor-randomized) position.
BOWL_FOOTPRINT_RADIUS = 0.07  # akita_black_bowl's own rim radius + settle slop
FIXTURE_CLEARANCE_MARGIN = 0.03
OBJECT_CLEARANCE_RADIUS = 0.12


def parse_task_goal(bddl_path):
    text = open(bddl_path).read()
    m = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n\)", text, re.S)
    goal_norm = re.sub(r"\s+", " ", m.group(1).strip())
    m2 = re.fullmatch(r"\(And \(On ([\w_]+) ([\w_]+)\)\)", goal_norm)
    if not m2:
        raise ValueError(f"goal doesn't match (And (On a b)): {goal_norm!r}")
    return m2.group(1), m2.group(2)


def make_cabinet_bddl(src_bddl, out_dir, task_name):
    with open(src_bddl) as fh:
        text = fh.read()
    paths = {}
    for state, rng in CABINET_RANGES.items():
        pattern = re.compile(
            r"(\(wooden_cabinet_init_region\s*\(:target kitchen_table\)\s*\(:ranges \(\s*)\([^)]*\)", re.S)
        new_text, n = pattern.subn(lambda m: m.group(1) + rng, text)
        assert n == 1, f"expected exactly 1 cabinet region substitution, got {n}"
        out_path = f"{out_dir}/{task_name}_cab_{state}.bddl"
        with open(out_path, "w") as fh:
            fh.write(new_text)
        paths[state] = out_path
    return paths


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


def label_for(primary, secondary, cab_state):
    bp, bx, px = primary
    by, py = secondary
    return f"bp-{bp}_bowlX-{bx}_bowlY-{by}_targetX-{px}_targetY-{py}_cab-{cab_state}"


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

    def get_subtask_term_signals(self):
        return dict(grasp=int(self._grasped_object_name() == self.moving))


class LiberoEnv6FactorGeneric:
    def __init__(self, libero_env, moving, target, cabinet_state):
        self.base_env = libero_env
        self.moving = moving
        self.target = target
        self.cabinet_state = cabinet_state
        self.y_bins = y_bins_for(cabinet_state)
        self.target_primary = None
        self.target_secondary = None
        self.last_placed_xy = {}  # {moving/target: xy} -- true commanded placement, see check_placement_integrity
        self.rng = np.random.default_rng()
        self._cam_params = None

    def _apply_camera_offset(self):
        sim = self.base_env.env.sim
        cid = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + np.array([CAMERA_X_OFFSET, 0.0, CAMERA_Z_OFFSET])
        sim.forward()
        self._cam_params = (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                             float(sim.model.cam_fovy[cid]))

    def _camera_safe(self, xy):
        cam_pos, cam_mat, fovy = self._cam_params
        return abs(xy[1]) <= camera_y_bound(cam_pos, cam_mat, fovy, xy[0])

    def _clearance_safe(self, xy, exclude=()):
        sim = self.base_env.env.sim
        xy = np.asarray(xy)
        for i in range(sim.model.ngeom):
            body_id = sim.model.geom_bodyid[i]
            bname = sim.model.body_id2name(body_id) or ""
            if CABINET_FIXTURE in bname:
                gpos = sim.data.geom_xpos[i][:2]
                gsize = sim.model.geom_size[i]
                req = (float(np.max(gsize[:2])) if gsize.size >= 2 else 0.02) \
                    + BOWL_FOOTPRINT_RADIUS + FIXTURE_CLEARANCE_MARGIN
                if np.linalg.norm(xy - gpos) < req:
                    return False
        for name, obj in self.base_env.env.objects_dict.items():
            if name in exclude:
                continue
            opos = sim.data.body_xpos[sim.model.body_name2id(obj.root_body)][:2]
            if np.linalg.norm(xy - opos) < OBJECT_CLEARANCE_RADIUS:
                return False
        return True

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
        y_lo, y_hi = self.y_bins[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = self.rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            if self._camera_safe(cand) and self._clearance_safe(cand):
                return cand
        return None

    def _sample_target_xy(self, bowl_xy, bp_lo, bp_hi, x_bin, y_bin):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = self.y_bins[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(bp_lo, bp_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = bowl_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if x_lo <= cand[0] <= x_hi and y_lo <= cand[1] <= y_hi and self._camera_safe(cand) \
                    and self._clearance_safe(cand, exclude={self.moving}):
                return cand
        return None

    def reset(self):
        ret = self.base_env.reset()
        self._apply_camera_offset()
        bp_bin, bowl_x_bin, target_x_bin = self.target_primary
        bowl_y_bin, target_y_bin = self.target_secondary
        bp_lo, bp_hi = BP_BINS[bp_bin]

        placed = False
        for _ in range(MAX_OUTER_RETRIES):
            bowl_xy = self._sample_box(bowl_x_bin, bowl_y_bin)
            if bowl_xy is None:
                continue
            target_xy = self._sample_target_xy(bowl_xy, bp_lo, bp_hi, target_x_bin, target_y_bin)
            if target_xy is None:
                continue
            self._set_xy(self.moving, bowl_xy)
            self._set_xy(self.target, target_xy)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"could not jointly satisfy {self.target_primary}/{self.target_secondary} in {MAX_OUTER_RETRIES} outer tries")

        self.last_placed_xy = {self.moving: np.array(bowl_xy), self.target: np.array(target_xy)}
        for _ in range(NUM_SETTLE_STEPS):
            ret = self.base_env.step(_DUMMY_ACTION)
        return ret

    def get_state(self):
        return {"states": self.base_env.get_sim_state()}

    def reset_to(self, state):
        ret = self.base_env.regenerate_obs_from_state(state["states"])
        self._apply_camera_offset()
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
        return {"env_name": "LIBERO_generic_6factor_wide", "type": "libero"}


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
    """Always-randomize pass -- see run_kitchen_task_6factor_v2.py's original
    docstring for the rationale. Ported fix: `wooden_cabinet_1` is a BDDL
    fixture, so the `raw_env.reset()` below (needed to re-seed for the
    layout search) silently re-randomizes its position same as SCENE3's
    stove did -- captured/restored around that call so `cabinet_state`
    keeps describing where the cabinet actually ends up."""
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    _cab_sim = raw_env.env.sim
    _cab_body_id = _cab_sim.model.body_name2id(f"{CABINET_FIXTURE}_main")
    _cab_pos = _cab_sim.model.body_pos[_cab_body_id].copy()
    _cab_quat = _cab_sim.model.body_quat[_cab_body_id].copy()

    raw_env.seed(seed)
    raw_env.reset()
    _cab_sim = raw_env.env.sim  # reset() rebuilds the model -- refresh the reference
    _cab_body_id = _cab_sim.model.body_name2id(f"{CABINET_FIXTURE}_main")
    _cab_sim.model.body_pos[_cab_body_id] = _cab_pos
    _cab_sim.model.body_quat[_cab_body_id] = _cab_quat
    _cab_sim.forward()
    env.reset_to({"states": initial_state})

    pre = {n: raw_env.env.sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy() for n in task_object_names}
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
    print(f"task={task_name} moving={moving} target={target} TARGET_TOTAL={TARGET_TOTAL} MAX_ATTEMPTS={MAX_ATTEMPTS}")

    cabinet_bddl_paths = make_cabinet_bddl(bddl_path, "/mgbridge", task_name)

    scratch_path = SCRATCH_SRC.format(task=task_name, label=label)
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
    succ_records = []  # (primary, secondary, cab_state, distractor_info)
    total_attempts = 0
    distractor_discards = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    _base = TARGET_TOTAL // len(CABINET_LEVELS)
    _rem = TARGET_TOTAL % len(CABINET_LEVELS)
    pass_targets = {lvl: _base + (1 if i < _rem else 0) for i, lvl in enumerate(CABINET_LEVELS)}
    pass_max_attempts = max(1, MAX_ATTEMPTS // len(CABINET_LEVELS))

    for cab_state in CABINET_LEVELS:
        pass_target = pass_targets[cab_state]
        print(f"\n--- cabinet state: {cab_state} (bddl {cabinet_bddl_paths[cab_state]}, target {pass_target}, budget {pass_max_attempts}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=cabinet_bddl_paths[cab_state], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnv6FactorGeneric(raw_env, moving, target, cab_state)
        env_interface = MG_LiberoBowlPlateGeneric(env=env.base_env, moving=moving, target=target)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

        # Fresh per cabinet_state, with the same starvation-proof give-up
        # rule built for SCENE3 -- a primary combo that's geometrically
        # infeasible for THIS cabinet_state would otherwise permanently
        # outrank every combo that's already succeeded once.
        per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}
        per_primary_secondary = {}
        GIVEUP_ATTEMPTS = 60

        def primary_key(p):
            a, s = per_primary[p]
            given_up = a >= GIVEUP_ATTEMPTS and s == 0
            return (given_up, s, a)

        pass_successes = 0
        pass_attempts = 0
        while pass_successes < pass_target and pass_attempts < pass_max_attempts:
            primary = pick_least_used(PRIMARY_COMBOS, primary_key, rng_combo)
            for sc in SECONDARY_COMBOS:
                per_primary_secondary.setdefault((primary, sc), [0, 0])
            secondary = pick_least_used(SECONDARY_COMBOS, lambda sc: tuple(per_primary_secondary[(primary, sc)]), rng_combo)
            env.target_primary = primary
            env.target_secondary = secondary
            per_primary[primary][0] += 1
            per_primary_secondary[(primary, secondary)][0] += 1
            pass_attempts += 1
            label_str = f"cab={cab_state} attempt {pass_attempts}/{pass_max_attempts} {label_for(primary, secondary, cab_state)}"
            try:
                traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                                render=False, video_writer=None)
            except Exception as e:  # noqa: BLE001
                total_attempts += 1
                print(f"[{label_str}] exception: {type(e).__name__}: {e}", flush=True)
                continue
            total_attempts += 1
            if not traj["success"]:
                print(f"[{label_str}] success=False", flush=True)
                continue

            rec = info = None
            for retry in range(N_DISTRACTOR_RETRIES):
                rec, info = randomize_distractors(env, raw_env, traj, task_object_names, seed=8_000_000 + total_attempts * 10 + retry)
                if rec is not None:
                    break
            if rec is None:
                distractor_discards += 1
                print(f"[{label_str}] bowl/target succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
                continue

            per_primary[primary][1] += 1
            per_primary_secondary[(primary, secondary)][1] += 1
            pass_successes += 1
            idx = len(succ_records)
            succ_records.append((primary, secondary, cab_state, info))
            write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                          attrs=dict(bowl_target_dist_bin=primary[0], bowl_x_bin=primary[1], target_x_bin=primary[2],
                                     bowl_y_bin=secondary[0], target_y_bin=secondary[1], cabinet_state=cab_state,
                                     removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
            print(f"[{label_str}] success=True distractors: removed={info['removed']} moved={list(info['moved'].keys())} "
                  f"(pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL})", flush=True)

        raw_env.close()
        print(f"--- cabinet state {cab_state} done: {pass_successes}/{pass_attempts} ---", flush=True)

    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"({distractor_discards} discarded after bowl/target succeeded but distractor randomization failed) ===")
    from collections import Counter
    print("cabinet state distribution:", dict(Counter(r[2] for r in succ_records)))

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
            "bowl_target_distance_bins_m": BP_BINS, "x_bins_m": X_BINS, "y_bins_m": Y_BINS,
            "cabinet_edge_distance_bins_m": CABINET_EDGE_BINS,
            "camera_offset": {"x": CAMERA_X_OFFSET, "z": CAMERA_Z_OFFSET},
        })
        for i, (primary, secondary, cab_state, info) in enumerate(succ_records):
            group_name = f"demo_{i:03d}_{label_for(primary, secondary, cab_state)}"
            with h5py.File(os.path.join(tmp_dir, f"{i:04d}.hdf5"), "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=group_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(scratch_path)
    print(f"wrote {out_path}: {len(succ_records)} episodes, every one with randomized distractors")


if __name__ == "__main__":
    main()
