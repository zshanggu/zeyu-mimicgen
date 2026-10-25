"""v2 of run_kitchen_task_6factor.py: distractor randomization is no longer
an optional bonus pass alongside an un-randomized "standard" episode -- it's
now mandatory for every output episode.

Why: the previous "standard" pass placed moving/target (e.g. bowl_3/plate)
via the 6-factor scheme but left every OTHER object (bowl_1, bowl_2) sitting
wherever LIBERO's own native sampler put them, completely unaware of where
moving/target ended up. Confirmed in the deployed SCENE2 data (user report):
demo_009 has the target bowl visibly penetrating a distractor bowl, because
nothing ever checked moving/target's final position against the OTHER
bowls' native positions. distractor_lib.sample_layout already rejects
placements overlapping task objects -- the bug was simply that the
"standard" episodes never ran it at all. Also per explicit instruction: the
near/medium/far/absent "appear or not, how close" principle the cabinet
already had is a general distractor concept, not cabinet-specific -- it
should apply to every distractor (cabinet AND the two bowls here), and
distractors should always be randomized, not occasionally.

So: every successful (moving, target) placement now ALWAYS gets a
distractor pass (randomize/remove every other object in objects_dict) before
being accepted as output. If the distractor layout can't be found or its
replay breaks the task, RETRY with new seeds a few times; if it still can't
be made to work, the whole attempt is discarded (not kept un-randomized) and
a fresh (moving, target) placement is attempted instead -- consistent with
"always randomize", never falls back to a penetration-risk episode.

Single output group (`data/`, no more separate `data_distractor/`): every
episode already has its distractor layout baked in, recorded in its own
`removed`/`moved`/`seed` attrs.
"""
import sys
import os
import re
import glob
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
TABLE_Z = 0.90

CAMERA_X_OFFSET = 0.5
CAMERA_Z_OFFSET = 0.4
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

REGION_X_LO, REGION_X_HI = -0.45, 0.13
Y_MAX = 0.48
Y_MIN_CABINET_PRESENT = -0.12
Y_MIN_CABINET_ABSENT = -0.48

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


def y_bins_for(cabinet_state):
    y_lo = Y_MIN_CABINET_ABSENT if cabinet_state == "absent" else Y_MIN_CABINET_PRESENT
    edges = np.linspace(y_lo, Y_MAX, 4)
    return {lvl: (float(edges[i]), float(edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}


BP_BINS = {"near": (0.16, 0.26), "medium": (0.26, 0.38), "far": (0.38, 0.55)}
PRIMARY_COMBOS = [(bp, bx, px) for bp in LEVELS for bx in X_LEVELS for px in X_LEVELS]
SECONDARY_COMBOS = [(by, py) for by in Y_LEVELS for py in Y_LEVELS]

TARGET_TOTAL = 16
MAX_ATTEMPTS = 120  # small test
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3  # per accepted (moving,target) placement, before discarding the whole attempt


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
    return f"bp-{bp}_bowlX-{bx}_bowlY-{by}_plateX-{px}_plateY-{py}_cab-{cab_state}"


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
            if self._camera_safe(cand):
                return cand
        return None

    def _sample_plate_xy(self, bowl_xy, bp_lo, bp_hi, x_bin, y_bin):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = self.y_bins[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(bp_lo, bp_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = bowl_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if x_lo <= cand[0] <= x_hi and y_lo <= cand[1] <= y_hi and self._camera_safe(cand):
                return cand
        return None

    def reset(self):
        ret = self.base_env.reset()
        self._apply_camera_offset()
        bp_bin, bowl_x_bin, plate_x_bin = self.target_primary
        bowl_y_bin, plate_y_bin = self.target_secondary
        bp_lo, bp_hi = BP_BINS[bp_bin]

        placed = False
        for _ in range(MAX_OUTER_RETRIES):
            bowl_xy = self._sample_box(bowl_x_bin, bowl_y_bin)
            if bowl_xy is None:
                continue
            plate_xy = self._sample_plate_xy(bowl_xy, bp_lo, bp_hi, plate_x_bin, plate_y_bin)
            if plate_xy is None:
                continue
            self._set_xy(self.moving, bowl_xy)
            self._set_xy(self.target, plate_xy)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"could not jointly satisfy {self.target_primary}/{self.target_secondary} in {MAX_OUTER_RETRIES} outer tries")

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
        return {"env_name": "LIBERO_generic_6factor_v2", "type": "libero"}


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
        # per-object pose, kept for verification (QA scripts check inter-object distance for
        # penetration) and harmless/potentially useful as privileged state for training
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
    """Always-randomize pass: moves/removes every object NOT in
    task_object_names, replays the SAME (moving,target) trajectory with that
    layout, and returns the resulting episode only if the task still
    succeeds with no task-object drift (i.e. the distractor layout didn't
    collide with moving/target -- this is exactly what prevents the
    penetration bug, applied to EVERY episode now, not an optional extra)."""
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

    cabinet_bddl_paths = make_cabinet_bddl(bddl_path, "/mgbridge", task_name)

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
    succ_records = []  # (primary, secondary, cab_state, distractor_info)
    total_attempts = 0
    distractor_discards = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    pass_target = TARGET_TOTAL // len(CABINET_LEVELS)
    pass_max_attempts = MAX_ATTEMPTS // len(CABINET_LEVELS)

    for cab_state in CABINET_LEVELS:
        print(f"\n--- cabinet state: {cab_state} (bddl {cabinet_bddl_paths[cab_state]}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=cabinet_bddl_paths[cab_state], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnv6FactorGeneric(raw_env, moving, target, cab_state)
        env_interface = MG_LiberoBowlPlateGeneric(env=env.base_env, moving=moving, target=target)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

        pass_successes = 0
        pass_attempts = 0
        while pass_successes < pass_target and pass_attempts < pass_max_attempts:
            primary = pick_least_used(PRIMARY_COMBOS, lambda p: (per_primary[p][1], per_primary[p][0]), rng_combo)
            for sc in SECONDARY_COMBOS:
                per_primary_secondary.setdefault((primary, cab_state, sc), [0, 0])
            secondary = pick_least_used(SECONDARY_COMBOS,
                                         lambda sc: tuple(per_primary_secondary[(primary, cab_state, sc)]), rng_combo)
            env.target_primary = primary
            env.target_secondary = secondary
            per_primary[primary][0] += 1
            per_primary_secondary[(primary, cab_state, secondary)][0] += 1
            try:
                traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                                render=False, video_writer=None)
            except Exception as e:  # noqa: BLE001
                pass_attempts += 1
                total_attempts += 1
                if pass_attempts % 20 == 0:
                    print(f"[cab={cab_state} {pass_attempts}] exception for {primary}/{secondary}: {type(e).__name__}: {e}", flush=True)
                continue
            pass_attempts += 1
            total_attempts += 1
            if not traj["success"]:
                if pass_attempts % 10 == 0:
                    print(f"[cab={cab_state} attempt {pass_attempts}] {label_for(primary, secondary, cab_state)} success=False "
                          f"(pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
                continue

            # ALWAYS randomize distractors now -- retry a few seeds, discard (not fall back) if none work
            rec = info = None
            for retry in range(N_DISTRACTOR_RETRIES):
                rec, info = randomize_distractors(env, raw_env, traj, task_object_names, seed=8_000_000 + total_attempts * 10 + retry)
                if rec is not None:
                    break
            if rec is None:
                distractor_discards += 1
                print(f"[cab={cab_state} attempt {pass_attempts}] {label_for(primary, secondary, cab_state)} "
                      f"bowl/plate succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
                continue

            per_primary[primary][1] += 1
            per_primary_secondary[(primary, cab_state, secondary)][1] += 1
            pass_successes += 1
            idx = len(succ_records)
            succ_records.append((primary, secondary, cab_state, info))
            write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                          attrs=dict(bowl_plate_bin=primary[0], bowl_x_bin=primary[1], plate_x_bin=primary[2],
                                     bowl_y_bin=secondary[0], plate_y_bin=secondary[1], cabinet_state=cab_state,
                                     removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
            print(f"[cab={cab_state} attempt {pass_attempts}] {label_for(primary, secondary, cab_state)} success=True "
                  f"distractors: removed={info['removed']} moved={list(info['moved'].keys())} "
                  f"(pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL})", flush=True)

        raw_env.close()
        print(f"--- cabinet state {cab_state} done: {pass_successes}/{pass_attempts} ---", flush=True)

    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"({distractor_discards} discarded after bowl/plate succeeded but distractor randomization failed) ===")
    for p in PRIMARY_COMBOS:
        a, s = per_primary[p]
        print(f"  bp-{p[0]}_bowlX-{p[1]}_plateX-{p[2]}: {s}/{a}")
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
            "bowl_plate_distance_bins_m": BP_BINS, "x_bins_m": X_BINS,
            "y_bins_cabinet_present_m": y_bins_for("near"), "y_bins_cabinet_absent_m": y_bins_for("absent"),
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
