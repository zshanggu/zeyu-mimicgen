"""KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet, wide-
bounds version, using the "wooden_cabinet" asset swap already established for
this task family (see libero_rerender_script/bddl_wooden_cabinet/ and the two
already-shipped KITCHEN_SCENE4_close_the_bottom_drawer... _wooden_cabinet_demo
files under libero90_openvla_256_wooden_cabinet/) and the wide X/Y bounds
already used for every living-room "_wide"/"_site_wide" task this session:
  bowl (moving):   X in [-0.35, 0.00], Y in [-0.45, 0.45]
  cabinet (fixed): X in [-0.35, 0.00], Y in [ 0.00, 0.45]
both ranges baked directly into a new bddl,
bddl_wooden_cabinet/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl
(wooden_cabinet_init_region / akita_black_bowl_init_region widened; wine_rack/
wine_bottle left at their native tiny regions).

Architecturally different from every other "_wide" script this session in one
important way: the TARGET here (wooden_cabinet_1_bottom_region) is a site
whose :target is the cabinet FIXTURE itself, not the table -- confirmed by
reading the bddl (`(bottom_region (:target wooden_cabinet_1))`) -- so unlike
LIVING_ROOM_SCENE6's plate-region site (which sits on the TABLE body and had
to be manually re-shifted via site_pos every time plate_1 moved), this site
is a child of the cabinet's own body and simply moves rigidly with it via
ordinary forward kinematics. No `_shift_site` analog is needed here at all.

The flip side (also confirmed by make_cabinet_bddl.py's own prior finding
this session): wooden_cabinet_1 is a FIXTURE with no free joint, so unlike
the bowl there is no qpos poke available to place it at runtime -- its
position can only be controlled through the bddl's declared init region, and
is drawn completely fresh (LIBERO's own native UniformRandomSampler) on
every single base_env.reset() call, since LIBERO rebuilds its placement
sampler from the bddl on every reset. So this script does not stratify the
cabinet into X/Y bins the way other "_wide" scripts stratify their target
object -- it can't be steered per-attempt, only widened once at the bddl
level. What IS stratified (3 factors: dist-from-cabinet, bowl X bin, bowl Y
bin) is the bowl, exactly like every other moving object in this session,
via direct qpos poke after the cabinet's fresh native position is read.

Because the bowl's wide Y range [-0.45,0.45] overlaps the cabinet's Y range
[0,0.45], a real collision/overlap-avoidance check is required (unlike the
pudding/plate script, where plate and pudding share the box but MimicGen's
own subtask sequencing plus a simple min-distance check was enough) --
here it's enforced by measuring the cabinet's REAL (not nominal-bddl) live
geom AABB footprint each reset (same technique as generic_pipeline.py's
_measure_real_footprints, generalized there specifically because "the
cabinet's real footprint extends well beyond its declared init region",
e.g. the open bottom drawer itself sticks out further than the cabinet
body) and rejecting any candidate bowl XY that falls inside it plus a
safety margin.

Grasp detection reuses the finger-contact-based `_grasped_object_name`
check (not the living-room scripts' gripper-command heuristic) -- the
akita_black_bowl is a wide, rigid object that registers real finger-pad
contact reliably, same assumption already used for every other bowl task
in this bridge (run_kitchen_task_6factor_v2.py's MG_LiberoBowlPlateGeneric).

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
             "bddl_wooden_cabinet/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.bddl")
# Used ONLY for prepare_source(): the source hdf5's recorded states are
# world-frame qpos captured with the cabinet at this NATIVE (tiny, unwidened)
# position. Replaying them into an env built from the WIDENED bddl instead
# would silently re-sample the cabinet to a random wide-region position
# (LIBERO draws a fresh placement every env.reset(), and the cabinet has no
# joint for us to override it with afterward like the bowl) -- confirmed
# empirically via diag_kitchen4b.py: even the native, genuinely-successful
# source demo's OWN recorded trajectory registered _check_success()=False
# when replayed that way, because the bowl-to-drawer-site relative transform
# captured into datagen_info ends up measured against the WRONG cabinet
# position, corrupting every subsequent retargeted placement regardless of
# where that episode's own cabinet lands (root cause of the kitchen4 smoke
# test's first run going 0/100+ -- every attempt failed, not just far ones).
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

LEVELS = ["near", "medium", "far"]
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]

_x_edges = np.linspace(REGION_X_LO, REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_y_edges = np.linspace(REGION_Y_LO, REGION_Y_HI, 4)
Y_BINS = {lvl: (float(_y_edges[i]), float(_y_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}

DIST_BINS = {"near": (0.12, 0.22), "medium": (0.22, 0.34), "far": (0.34, 0.55)}

PRIMARY_COMBOS = [(d, bx, by) for d in LEVELS for bx in X_LEVELS for by in Y_LEVELS]

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
        return {"env_name": "LIBERO_bowl_drawer_wide", "type": "libero"}


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
    # Same class of bug already fixed for SCENE2/5's cabinet fixture via
    # site_pos this session; here it's body_pos since the whole fixture
    # moves, not just a site offset. Must refresh the `sim` reference after
    # reset() (it's a new MjSim object).
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
    bowl over while lowering it into the drawer (same class of bug as
    demo_030 in the LIVING_ROOM_SCENE6 pudding/plate task this session) --
    this checks the FINAL recorded pose instead."""
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
