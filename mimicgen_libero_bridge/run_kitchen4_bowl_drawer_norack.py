"""KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet, wine-
rack-removed variant of run_kitchen4_bowl_drawer_wide.py.

Two changes relative to the wide-bounds version:
  1. wine_rack_1 (fixture) is removed from the scene entirely -- new bddl
     bddl_wooden_cabinet/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_norack.bddl
     drops the wine_rack fixture, its init region, and the wine_rack-targeted
     top_region block. wine_bottle_1 is left in (its init region targets
     kitchen_table directly, at a completely different X/Y than the rack's
     own region, so it was never physically resting on the rack -- removing
     the rack doesn't orphan it).
  2. The cabinet's own bddl init region is widened from Y in [0.00, 0.45]
     (the wide-bounds version's half-range, chosen there to keep the cabinet
     on the "positive Y" side) to the FULL Y in [-0.45, 0.45] -- same full
     range as the bowl -- so the stratified sampler (unchanged from the wide
     version) can actually land the cabinet at the far Y- end for some
     resets and the far Y+ end for others, including the specific
     bowl-far-Y-minus / cabinet-far-Y-plus configuration this variant exists
     to cover. No code changes are needed to realize that: cab_xy is already
     read fresh from wherever the cabinet actually landed every reset(), and
     the bowl's distance/footprint-avoidance sampling in
     LiberoEnvBowlDrawerWide._sample_bowl_xy already works relative to that
     live cab_xy regardless of which side it's on.

NATIVE_BDDL_PATH is unchanged from the wide-bounds version (still includes
wine_rack_1/wine_bottle_1 at their stock positions) -- it's used ONLY to
replay the ORIGINAL source demo's recorded (world-frame) states for
prepare_source(), which must match the scene the source hdf5 was actually
recorded in, not the modified generation scene. See run_kitchen4_bowl_drawer_
wide.py's own docstring/comments for the full reasoning (prepare_source
reference-transform corruption, body_pos save/restore around
randomize_distractors' internal reset()) -- all of that is reused verbatim
here since both bugs are orthogonal to the rack-removal/Y-widening changes.

TARGET_TOTAL/MAX_ATTEMPTS are environment-overridable (MG_TARGET_TOTAL/
MG_MAX_ATTEMPTS) for a quick smoke test before any full production run.
"""
import sys
import os
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

TASK_NAME = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet"
BDDL_PATH = ("/data/zeyu/PHD_LAB/Amazon_Project/libero_rerender_script/"
             "bddl_wooden_cabinet/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_norack.bddl")
NATIVE_BDDL_PATH = ("/data/zeyu/PHD_LAB/Amazon_Project/libero_rerender_script/"
                     "bddl_wooden_cabinet/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_native.bddl")
SRC_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_ori"
SRC_HDF5 = os.path.join(SRC_DIR, f"{TASK_NAME}_demo.hdf5")
SCRATCH_SRC = "/mgbridge/_scratch_prepared_source_{task}_{label}.hdf5"

MOVING = "akita_black_bowl_1"
TARGET_FIXTURE = "wooden_cabinet_1"
TARGET_SITE = "wooden_cabinet_1_bottom_region"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]

REGION_X_LO, REGION_X_HI = -0.35, 0.00
REGION_Y_LO, REGION_Y_HI = -0.45, 0.45

CABINET_FOOTPRINT_MARGIN = 0.06  # keep the bowl this far clear of the cabinet's REAL (measured) footprint

LEVELS = ["near", "medium", "far", "extreme"]
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]

_x_edges = np.linspace(REGION_X_LO, REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_y_edges = np.linspace(REGION_Y_LO, REGION_Y_HI, 4)
Y_BINS = {lvl: (float(_y_edges[i]), float(_y_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}

# "extreme" covers the cabinet-at-one-Y-extreme / bowl-at-the-opposite-Y-
# extreme configuration (e.g. cabinet ~(-0.10, 0.45), bowl ~(-0.15, -0.42) --
# a real example the user asked about -- whose distance is ~0.88m, well
# past the "far" bin's 0.55 ceiling. Upper bound 0.95 covers the full
# REGION_X/Y bounding box's theoretical max separation (~0.97m corner to
# corner) without reaching into clearly-infeasible territory.
DIST_BINS = {"near": (0.12, 0.22), "medium": (0.22, 0.34), "far": (0.34, 0.55), "extreme": (0.55, 0.95)}

PRIMARY_COMBOS = [(d, bx, by) for d in LEVELS for bx in X_LEVELS for by in Y_LEVELS]

# Optional: restrict generation to a subset of distance levels (comma-
# separated, e.g. "extreme" or "near,far") -- used to top up/replace just
# one distance bin's demos in an already-existing dataset without
# regenerating everything.
_DIST_FILTER = os.environ.get("MG_DIST_FILTER")
if _DIST_FILTER:
    _dist_filter_set = set(_DIST_FILTER.split(","))
    PRIMARY_COMBOS = [c for c in PRIMARY_COMBOS if c[0] in _dist_filter_set]

TARGET_TOTAL = int(os.environ.get("MG_TARGET_TOTAL", "100"))
MAX_ATTEMPTS = int(os.environ.get("MG_MAX_ATTEMPTS", "500"))
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3
PER_COMBO_GIVEUP_ATTEMPTS = 60

_FINGER_BODY_NAMES = ("gripper0_leftfinger", "gripper0_rightfinger",
                      "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")


def _grasped_object_name(env):
    sim = env.sim
    finger_ids = set()
    for name in _FINGER_BODY_NAMES:
        try:
            finger_ids.add(sim.model.body_name2id(name))
        except Exception:
            continue
    if not finger_ids:
        return None
    root_to_name = {obj.root_body: name for name, obj in env.env.objects_dict.items()}
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        b1, b2 = sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2]
        for fb, ob in ((b1, b2), (b2, b1)):
            if fb in finger_ids:
                n = sim.model.body_id2name(ob)
                if n in root_to_name:
                    return root_to_name[n]
    return None


def label_for(primary):
    d, bx, by = primary
    return f"dist-{d}_bowlX-{bx}_bowlY-{by}"


class MG_LiberoBowlDrawer(RobosuiteInterface):
    INTERFACE_TYPE = "libero"

    def __init__(self, env, moving, target_site):
        super().__init__(env=env)
        self.moving = moving
        self.target_site = target_site

    def get_object_poses(self):
        moving_body = self.env.env.objects_dict[self.moving].root_body
        return dict(bowl=self.get_object_pose(obj_name=moving_body, obj_type="body"),
                    plate=self.get_object_pose(obj_name=self.target_site, obj_type="site"))

    def get_subtask_term_signals(self):
        return dict(grasp=int(_grasped_object_name(self.env) == self.moving))


def _measure_cabinet_footprint(sim, fixture_name):
    """Real (live geom AABB) XY footprint of the cabinet fixture, including
    whatever its open bottom drawer currently sticks out to -- NOT the
    nominal bddl init-region box, which is only a few cm wide and would
    badly under-estimate how much table space the cabinet actually
    occupies once its drawer is open (see generic_pipeline.py's
    _measure_real_footprints, same technique)."""
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    found = False
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        name = sim.model.body_id2name(body_id) or ""
        if not (name == fixture_name or name.startswith(fixture_name + "_")):
            continue
        gpos = sim.data.geom_xpos[i]
        gsize = sim.model.geom_size[i]
        sx = gsize[0] if len(gsize) > 0 else 0.03
        sy = gsize[1] if len(gsize) > 1 else 0.03
        lo = np.minimum(lo, [gpos[0] - sx, gpos[1] - sy])
        hi = np.maximum(hi, [gpos[0] + sx, gpos[1] + sy])
        found = True
    assert found, f"no geoms found for fixture {fixture_name!r}"
    return lo, hi


class LiberoEnvBowlDrawerWide:
    def __init__(self, libero_env, moving, target_site, target_fixture):
        self.base_env = libero_env
        self.moving = moving
        self.target_site = target_site
        self.target_fixture = target_fixture
        self.target_primary = None
        self.rng = np.random.default_rng()

    def _obj_xy(self, name):
        sim = self.base_env.env.sim
        return sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[:2].copy()

    def _site_xy(self, name):
        sim = self.base_env.env.sim
        sid = sim.model.site_name2id(name)
        return sim.data.site_xpos[sid][:2].copy()

    def _is_upright(self, name, thresh=0.8):
        sim = self.base_env.env.sim
        quat = sim.data.get_body_xquat(self.base_env.env.objects_dict[name].root_body)
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return up_z > thresh

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _sample_bowl_xy(self, x_bin, y_bin, d_lo, d_hi, cab_xy, fp_lo, fp_hi):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = Y_BINS[y_bin]
        margin = CABINET_FOOTPRINT_MARGIN
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = self.rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            d = np.linalg.norm(cand - cab_xy)
            if not (d_lo <= d <= d_hi):
                continue
            if (fp_lo[0] - margin <= cand[0] <= fp_hi[0] + margin
                    and fp_lo[1] - margin <= cand[1] <= fp_hi[1] + margin):
                continue  # inside (or too close to) the cabinet's real footprint
            return cand
        return None

    def reset(self):
        d_bin, bowl_x_bin, bowl_y_bin = self.target_primary
        d_lo, d_hi = DIST_BINS[d_bin]
        sanity_lo = np.array([REGION_X_LO, REGION_Y_LO]) - 0.2
        sanity_hi = np.array([REGION_X_HI, REGION_Y_HI]) + 0.2

        for _ in range(MAX_OUTER_RETRIES):
            ret = self.base_env.reset()  # draws a FRESH cabinet position from the widened bddl range
            sim = self.base_env.env.sim
            cab_xy = self._obj_xy(self.target_fixture) if self.target_fixture in self.base_env.env.objects_dict \
                else self._site_xy(self.target_site)
            fp_lo, fp_hi = _measure_cabinet_footprint(sim, self.target_fixture)

            bowl_xy = self._sample_bowl_xy(bowl_x_bin, bowl_y_bin, d_lo, d_hi, cab_xy, fp_lo, fp_hi)
            if bowl_xy is None:
                continue
            self._set_xy(self.moving, bowl_xy)
            for _ in range(NUM_SETTLE_STEPS):
                ret = self.base_env.step(_DUMMY_ACTION)
            final_bowl = self._obj_xy(self.moving)
            if not (np.all(final_bowl >= sanity_lo) and np.all(final_bowl <= sanity_hi)):
                continue
            if not self._is_upright(self.moving):
                continue
            if self.is_success()["task"]:
                continue  # bowl already landed in the drawer region before any manipulation
            return ret
        raise RuntimeError(f"could not place bowl for {self.target_primary} in {MAX_OUTER_RETRIES} outer tries")

    def get_state(self):
        return {"states": self.base_env.get_sim_state()}

    def reset_to(self, state):
        return self.base_env.regenerate_obs_from_state(state["states"])

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
        return {"env_name": "LIBERO_bowl_drawer_norack", "type": "libero"}


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
    env_interface = MG_LiberoBowlDrawer(env=env, moving=moving, target_site=target_site)

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
        ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_LiberoBowlDrawer"
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

    # wooden_cabinet_1 has no free joint, so its position is a MODEL param
    # (body_pos), not per-step qpos/qvel state -- a plain env.reset_to()
    # below does NOT restore it after raw_env.reset() rebuilds the model
    # (which draws a FRESH random cabinet position from the widened bddl).
    # Must refresh the `sim` reference after reset() (it's a new MjSim
    # object).
    sim_before = raw_env.env.sim
    cab_body_id_before = sim_before.model.body_name2id(f"{env.target_fixture}_main")
    saved_cab_body_pos = sim_before.model.body_pos[cab_body_id_before].copy()

    raw_env.seed(seed)
    raw_env.reset()
    sim_after = raw_env.env.sim
    cab_body_id_after = sim_after.model.body_name2id(f"{env.target_fixture}_main")
    sim_after.model.body_pos[cab_body_id_after] = saved_cab_body_pos
    sim_after.forward()

    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, task_object_names)
    scene.measure_radii()
    if not scene.distractors:
        return None, "no_distractors_in_this_task"

    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
                               p_remove=0.3, table_z=0.44, vis=vis, xy_lo=(-0.5, -0.65), xy_hi=(0.5, 0.65))
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


def check_final_upright(rec, moving, thresh=0.8):
    """The reset()-time upright check only guards the pre-manipulation
    settle; the robot's actual pick-and-place motion could still tip the
    bowl over while lowering it into the drawer -- this checks the FINAL
    recorded pose instead."""
    final_quat = rec["object_poses"][moving][-1][1]
    w, x, y, z = final_quat
    up_z = 1.0 - 2.0 * (x * x + y * y)
    if up_z <= thresh:
        return False, f"not_upright(up_z={up_z:.2f})"
    return True, None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/mgbridge/out"
    label = sys.argv[2] if len(sys.argv) > 2 else TASK_NAME

    assert os.path.exists(BDDL_PATH), BDDL_PATH
    assert os.path.exists(NATIVE_BDDL_PATH), NATIVE_BDDL_PATH
    assert os.path.exists(SRC_HDF5), SRC_HDF5

    print(f"task={TASK_NAME} moving={MOVING} target_site={TARGET_SITE} target_fixture={TARGET_FIXTURE} "
          f"TARGET_TOTAL={TARGET_TOTAL} MAX_ATTEMPTS={MAX_ATTEMPTS}", flush=True)

    scratch_path = SCRATCH_SRC.format(task=TASK_NAME, label=label)
    print("preparing source dataset...", flush=True)
    prepare_source(NATIVE_BDDL_PATH, SRC_HDF5, MOVING, TARGET_SITE, scratch_path)

    from libero.libero.envs import OffScreenRenderEnv
    task_object_names = {MOVING}

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=scratch_path, filter_key=None, start=None, n=None)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    rng_combo = np.random.default_rng(42)
    per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}
    succ_records = []
    total_attempts = 0
    distractor_discards = 0
    geometry_discards = 0

    def primary_key(p):
        attempts, successes = per_primary[p][0], per_primary[p][1]
        given_up = attempts >= PER_COMBO_GIVEUP_ATTEMPTS and successes == 0
        return (given_up, successes, attempts)

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    raw_env = OffScreenRenderEnv(bddl_file_name=BDDL_PATH, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    env = LiberoEnvBowlDrawerWide(raw_env, MOVING, TARGET_SITE, TARGET_FIXTURE)
    env_interface = MG_LiberoBowlDrawer(env=env.base_env, moving=MOVING, target_site=TARGET_SITE)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

    while len(succ_records) < TARGET_TOTAL and total_attempts < MAX_ATTEMPTS:
        primary = pick_least_used(PRIMARY_COMBOS, primary_key, rng_combo)
        env.target_primary = primary
        per_primary[primary][0] += 1
        try:
            traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                            transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                            render=False, video_writer=None)
        except Exception as e:  # noqa: BLE001
            total_attempts += 1
            if total_attempts % 20 == 0:
                print(f"[attempt {total_attempts}] exception for {primary}: {type(e).__name__}: {e}", flush=True)
            continue
        total_attempts += 1
        if not traj["success"]:
            if total_attempts % 10 == 0:
                print(f"[attempt {total_attempts}] {label_for(primary)} success=False "
                      f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
            continue

        rec = info = None
        for retry in range(N_DISTRACTOR_RETRIES):
            rec, info = randomize_distractors(env, raw_env, traj, task_object_names,
                                               seed=8_000_000 + total_attempts * 10 + retry)
            if rec is not None:
                break
        if rec is None:
            distractor_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary)} "
                  f"succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
            continue

        geo_ok, geo_reason = check_final_upright(rec, MOVING)
        if not geo_ok:
            geometry_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary)} "
                  f"succeeded but final geometry check failed ({geo_reason}) -- discarded", flush=True)
            continue

        per_primary[primary][1] += 1
        idx = len(succ_records)
        succ_records.append((primary, info))
        write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                      attrs=dict(dist_bin=primary[0], bowl_x_bin=primary[1], bowl_y_bin=primary[2],
                                 removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
        print(f"[attempt {total_attempts}] {label_for(primary)} success=True "
              f"distractors: removed={info['removed']} moved={list(info['moved'].keys())} "
              f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)

    raw_env.close()
    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"({distractor_discards} discarded after succeeded but distractor randomization failed, "
          f"{geometry_discards} discarded after final geometry check) ===")
    for p in PRIMARY_COMBOS:
        a, s = per_primary[p]
        print(f"  dist-{p[0]}_bowlX-{p[1]}_bowlY-{p[2]}: {s}/{a}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": TASK_NAME})
        data_grp.attrs["mimicgen_bddl_path"] = BDDL_PATH
        data_grp.attrs["mimicgen_num_attempts"] = total_attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_records)
        data_grp.attrs["moving_object"] = MOVING
        data_grp.attrs["target_fixture"] = TARGET_FIXTURE
        data_grp.attrs["target_site"] = TARGET_SITE
        data_grp.attrs["distractors_always_randomized"] = True
        data_grp.attrs["wine_rack_removed"] = True
        data_grp.attrs["stratification"] = json.dumps({
            "dist_bins_m": DIST_BINS, "x_bins_m": X_BINS, "y_bins_m": Y_BINS,
        })
        for i, (primary, info) in enumerate(succ_records):
            group_name = f"demo_{i:03d}_{label_for(primary)}"
            with h5py.File(os.path.join(tmp_dir, f"{i:04d}.hdf5"), "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=group_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(scratch_path)
    print(f"wrote {out_path}: {len(succ_records)} episodes, every one with randomized distractors")


if __name__ == "__main__":
    main()
