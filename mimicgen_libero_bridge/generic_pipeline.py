"""Generalization of the SCENE1/SCENE5 bowl-plate work to arbitrary LIBERO-90
"(On moving_obj target)" tasks, where target is either another free movable
object (e.g. plate) or a fixture/table region (e.g. "wooden_cabinet_1_top_side").

Key simplification vs. the SCENE1/SCENE5 approach: instead of writing a
widened COPY of each task's bddl file (regex text substitution, one file per
task, fragile), this mutates the already-built env's placement sampler
directly at runtime. Every LIBERO kitchen/living-room/study task's per-object
sampler is a Libero100TableRegionSampler named f"table-middle-{obj_name}",
stored in env.env.placement_initializer.samplers (confirmed by reading
robosuite's SequentialCompositeSampler and LIBERO's own region sampler
classes) -- so widening a region is just:
    env.env.placement_initializer.samplers[f"table-middle-{name}"].x_ranges = [[lo, hi]]
No bddl file, no regex, works identically for all 90 tasks.

Carries forward every gotcha found building the SCENE1/5 pipeline:
  - settle steps after reset (object pose is ~7cm off immediately after
    reset, before physics settles it onto the table) -- NUM_SETTLE_STEPS.
  - a plain regenerate_obs_from_state does NOT reproduce a rollout; a full
    env.reset() must run first to clear controller/interpolator state left
    over from the just-finished generate() call (diag_replay_determinism.py).
  - LIBERO's own region sampler does NOT reliably prevent two independently-
    sampled objects from overlapping; explicit reject-and-reroll (or direct
    placement) is needed when two free objects share overlapping ranges.
  - real fixture footprints must be measured from the live geom AABBs, not
    trusted from the bddl region definition (the cabinet's real footprint
    extends well beyond its declared init region).
  - distractor_lib.Scene/sample_layout already generalizes cleanly: pass it
    whatever set of object names are "task-related" and it treats everything
    else in objects_dict as a distractor automatically.
"""
import re
import numpy as np

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

import distractor_lib as DL

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.90
MIN_OBJ_OBJ_DIST = 0.16
MAX_RESET_RETRIES = 40
MAX_PLACEMENT_TRIES = 80
REGION_MARGIN = 0.12          # how much to widen native per-object regions by, each side
FIXTURE_SAFETY_MARGIN = 0.04  # keep this far clear of any fixture's REAL (measured) footprint
TABLE_EDGE_MARGIN = 0.05

DIST_BINS = {
    "near": (0.16, 0.26),
    "medium": (0.26, 0.38),
    "far": (0.38, 0.55),
}
BIN_CYCLE = ["near", "medium", "far"]

_FINGER_BODY_NAMES = (
    "gripper0_leftfinger",
    "gripper0_rightfinger",
    "gripper0_finger_joint1_tip",
    "gripper0_finger_joint2_tip",
)


def _grasped_object_name(env):
    sim = env.sim
    finger_body_ids = set()
    for name in _FINGER_BODY_NAMES:
        try:
            finger_body_ids.add(sim.model.body_name2id(name))
        except Exception:
            continue
    if not finger_body_ids:
        return None
    root_body_to_name = {obj.root_body: name for name, obj in env.env.objects_dict.items()}
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        body1 = sim.model.geom_bodyid[contact.geom1]
        body2 = sim.model.geom_bodyid[contact.geom2]
        for finger_body, other_body in ((body1, body2), (body2, body1)):
            if finger_body in finger_body_ids:
                other_name = sim.model.body_id2name(other_body)
                if other_name in root_body_to_name:
                    return root_body_to_name[other_name]
    return None


# --------------------------------------------------------------------------- task config
def parse_task_goal(bddl_path):
    """Returns (moving_obj, target_kind, target_name) for a bddl whose goal is
    exactly (And (On moving_obj target)). Raises if the goal doesn't match."""
    text = open(bddl_path).read()
    m = re.search(r"\(:goal\s*(.*?)\n\s*\)\s*\n\)", text, re.S)
    goal_norm = re.sub(r"\s+", " ", m.group(1).strip())
    m2 = re.fullmatch(r"\(And \(On ([\w_]+) ([\w_]+)\)\)", goal_norm)
    if not m2:
        raise ValueError(f"goal doesn't match (And (On a b)): {goal_norm!r}")
    return m2.group(1), m2.group(2)


def build_task_config(env):
    """env: a freshly-constructed (not yet reset) OffScreenRenderEnv. Determines
    ground truth (not regex-guessed) whether the target is a free object or a
    site/region, directly from the live env."""
    e = env.env
    moving, target = parse_task_goal(e.bddl_file_name)
    assert moving in e.objects_dict, f"moving object {moving!r} not in objects_dict"
    if target in e.objects_dict:
        target_kind = "object"
    elif hasattr(e, "object_sites_dict") and target in e.object_sites_dict:
        target_kind = "site"
    else:
        raise ValueError(f"target {target!r} is neither a free object nor a known site")
    task_object_names = {moving}
    if target_kind == "object":
        task_object_names.add(target)
    return dict(moving=moving, target=target, target_kind=target_kind, task_object_names=task_object_names)


# --------------------------------------------------------------------------- env_interface
class MG_LiberoGeneric(RobosuiteInterface):
    INTERFACE_TYPE = "libero"

    def __init__(self, env, moving, target, target_kind):
        super().__init__(env=env)
        self.moving = moving
        self.target = target
        self.target_kind = target_kind

    def get_object_poses(self):
        moving_body = self.env.env.objects_dict[self.moving].root_body
        poses = dict(moving=self.get_object_pose(obj_name=moving_body, obj_type="body"))
        if self.target_kind == "object":
            target_body = self.env.env.objects_dict[self.target].root_body
            poses["target"] = self.get_object_pose(obj_name=target_body, obj_type="body")
        else:
            poses["target"] = self.get_object_pose(obj_name=self.target, obj_type="site")
        return poses

    def get_subtask_term_signals(self):
        grasped = _grasped_object_name(self.env)
        return dict(grasp=int(grasped == self.moving))


# --------------------------------------------------------------------------- auto region widening
def _measure_real_footprints(sim):
    """geom-AABB footprints for 'table' and every fixture/object body, keyed by
    the keyword found in the geom's body name (same technique validated in
    check_table_cabinet_xy.py / check_scene5_geometry.py, generalized)."""
    footprints = {}
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        name = (sim.model.body_id2name(body_id) or "").lower()
        if not name:
            continue
        gpos = sim.data.geom_xpos[i]
        gsize = sim.model.geom_size[i]
        sx = gsize[0] if len(gsize) > 0 else 0.03
        sy = gsize[1] if len(gsize) > 1 else 0.03
        box = (gpos[0] - sx, gpos[0] + sx, gpos[1] - sy, gpos[1] + sy)
        key = name.rsplit("_main", 1)[0].rsplit("_g", 1)[0]
        lo = footprints.get(key)
        if lo is None:
            footprints[key] = list(box)
        else:
            lo[0], lo[2] = min(lo[0], box[0]), min(lo[2], box[2])
            lo[1], lo[3] = max(lo[1], box[1]), max(lo[3], box[3])
    return footprints


def _find_table_sampler(e, obj_name):
    """The sampler for an object placed directly on the plain table region is
    keyed by the BARE object name -- NOT "table-middle-{name}" as
    Libero100TableRegionSampler.__init__ would suggest: it computes that
    string into a local `name` variable but then (a LIBERO bug) passes the
    ORIGINAL object_name positionally to its parent's `name` param instead,
    so the string never takes effect. Confirmed empirically
    (diag_sampler_lookup.py): samplers dict keys are exactly the object
    names, e.g. "red_coffee_mug_1", "akita_black_bowl_1".

    Some objects are instead placed via a SiteRegionRandomSampler relative to
    ANOTHER object/fixture's own sub-region (bddl_base_domain.py's "on
    sites" path, e.g. "on top of a tray"); that sampler's x_ranges/y_ranges
    are relative offsets from the reference site, not absolute table
    coordinates, so widening them the same way would be wrong. Only return a
    sampler here when it's the plain absolute-coordinate kind (has x_ranges/
    y_ranges and isn't conditional); otherwise return None and the caller
    skips widening for that object (keeps its native variance instead of
    crashing)."""
    sampler = e.placement_initializer.samplers.get(obj_name)
    if sampler is not None and hasattr(sampler, "x_ranges") and hasattr(sampler, "y_ranges"):
        return sampler
    return None


def widen_object_region(env, obj_name, other_native_boxes=()):
    """Mutates env's placement sampler for obj_name to a modestly widened,
    fixture-safe box. Returns (lo_xy, hi_xy) actually applied (falls back to
    the object's current live position as a zero-size box if this object
    isn't on the plain table-middle sampler -- see _find_table_sampler).
    Must be called BEFORE any env.reset() (the sampler instance persists
    across resets once created, so this only needs to run once per env)."""
    e = env.env
    sampler = _find_table_sampler(e, obj_name)
    if sampler is None:
        xy = e.sim.data.get_body_xpos(e.objects_dict[obj_name].root_body)[:2].copy()
        return xy.copy(), xy.copy()
    native_x = list(sampler.x_ranges[0])
    native_y = list(sampler.y_ranges[0])

    sim = e.sim
    footprints = _measure_real_footprints(sim)
    table_box = footprints.get("table") or footprints.get("kitchen_table") or footprints.get("living_room_table") or footprints.get("study_table")

    lo_x, hi_x = native_x[0] - REGION_MARGIN, native_x[1] + REGION_MARGIN
    lo_y, hi_y = native_y[0] - REGION_MARGIN, native_y[1] + REGION_MARGIN
    for (ox_lo, ox_hi, oy_lo, oy_hi) in other_native_boxes:
        lo_x, hi_x = min(lo_x, ox_lo - REGION_MARGIN), max(hi_x, ox_hi + REGION_MARGIN)
        lo_y, hi_y = min(lo_y, oy_lo - REGION_MARGIN), max(hi_y, oy_hi + REGION_MARGIN)

    if table_box is not None:
        tx_lo, tx_hi, ty_lo, ty_hi = table_box
        lo_x = max(lo_x, tx_lo + TABLE_EDGE_MARGIN)
        hi_x = min(hi_x, tx_hi - TABLE_EDGE_MARGIN)
        lo_y = max(lo_y, ty_lo + TABLE_EDGE_MARGIN)
        hi_y = min(hi_y, ty_hi - TABLE_EDGE_MARGIN)

    # shrink away from any fixture whose real footprint would overlap the box
    obj_fixture_names = set(e.fixtures_dict.keys())
    for fname in obj_fixture_names:
        fp = footprints.get(fname.lower())
        if fp is None:
            continue
        fx_lo, fx_hi, fy_lo, fy_hi = fp
        fx_lo -= FIXTURE_SAFETY_MARGIN; fx_hi += FIXTURE_SAFETY_MARGIN
        fy_lo -= FIXTURE_SAFETY_MARGIN; fy_hi += FIXTURE_SAFETY_MARGIN
        overlap_x = min(hi_x, fx_hi) - max(lo_x, fx_lo)
        overlap_y = min(hi_y, fy_hi) - max(lo_y, fy_lo)
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        # shrink from whichever side costs less area
        if overlap_x < overlap_y:
            if (fx_lo + fx_hi) / 2 > (lo_x + hi_x) / 2:
                hi_x = min(hi_x, fx_lo)
            else:
                lo_x = max(lo_x, fx_hi)
        else:
            if (fy_lo + fy_hi) / 2 > (lo_y + hi_y) / 2:
                hi_y = min(hi_y, fy_lo)
            else:
                lo_y = max(lo_y, fy_hi)

    if hi_x - lo_x < 0.04 or hi_y - lo_y < 0.04:
        # widening collapsed the box (tight native region boxed in by fixtures) -- fall back to native
        lo_x, hi_x, lo_y, hi_y = native_x[0], native_x[1], native_y[0], native_y[1]

    sampler.x_ranges = [[lo_x, hi_x]]
    sampler.y_ranges = [[lo_y, hi_y]]
    return np.array([lo_x, lo_y]), np.array([hi_x, hi_y])


# --------------------------------------------------------------------------- env wrapper
class LiberoEnvForDataGenGeneric:
    def __init__(self, libero_env, moving, target, target_kind, moving_lo, moving_hi, target_lo=None, target_hi=None):
        self.base_env = libero_env
        self.moving = moving
        self.target = target
        self.target_kind = target_kind  # "object" or "site"
        self.moving_lo, self.moving_hi = moving_lo, moving_hi
        self.target_lo, self.target_hi = target_lo, target_hi
        self.target_bin = None
        self.rng = np.random.default_rng()

    def _obj_xy(self, name):
        sim = self.base_env.env.sim
        return sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[:2].copy()

    def _target_xy(self):
        if self.target_kind == "object":
            return self._obj_xy(self.target)
        sim = self.base_env.env.sim
        sid = sim.model.site_name2id(self.target)
        return sim.data.site_xpos[sid][:2].copy()

    def _place_object_at_distance(self, name, anchor_xy, lo, hi, lo_bound, hi_bound):
        sim = self.base_env.env.sim
        obj = self.base_env.env.objects_dict[name]
        joint = obj.joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(lo, hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = anchor_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if not (lo_bound[0] <= cand[0] <= hi_bound[0] and lo_bound[1] <= cand[1] <= hi_bound[1]):
                continue
            sim.data.qpos[qs:qs + 2] = cand
            sim.data.qvel[vs:ve] = 0.0
            sim.forward()
            return True
        return False

    def reset(self):
        ret = self.base_env.reset()
        bin_range = DIST_BINS.get(self.target_bin) if self.target_bin else None
        for attempt in range(MAX_RESET_RETRIES):
            moving_xy = self._obj_xy(self.moving)
            target_xy = self._target_xy()
            dist = np.linalg.norm(moving_xy - target_xy)
            lo, hi = bin_range if bin_range else (MIN_OBJ_OBJ_DIST, 1e9)
            if lo <= dist <= hi:
                break
            if self.target_kind == "object":
                ok = self._place_object_at_distance(self.target, moving_xy, lo, hi, self.target_lo, self.target_hi)
            else:
                ok = self._place_object_at_distance(self.moving, target_xy, lo, hi, self.moving_lo, self.moving_hi)
            if ok:
                break
            ret = self.base_env.reset()
        for _ in range(NUM_SETTLE_STEPS):
            ret = self.base_env.step(_DUMMY_ACTION)
        return ret

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

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def serialize(self):
        return {"env_name": "LIBERO_GENERIC", "type": "libero"}


# --------------------------------------------------------------------------- distractor variant (shared)
def try_distractor_variant(env, raw_env, traj, task_object_names, seed, p_remove=0.3, max_drift=0.01, settle_steps=10):
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()  # REQUIRED: clears controller state left by generate()'s rollout -- see module docstring
    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, task_object_names)
    scene.measure_radii()
    if not scene.distractors:
        return None, "no_distractors"

    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
                               p_remove=p_remove, table_z=TABLE_Z, vis=vis,
                               xy_lo=(-0.5, -0.65), xy_hi=(0.5, 0.65))
    if result is None:
        return None, "no_valid_layout"
    st, removed, moved = result

    pre = {n: scene.sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy() for n in task_object_names}
    env.reset_to({"states": st})
    for _ in range(settle_steps):
        env.base_env.step(_DUMMY_ACTION)
    sim = raw_env.env.sim
    for n in task_object_names:
        post = sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy()
        if np.linalg.norm(post - pre[n]) > max_drift:
            return None, f"task_object_drift({n})"

    actions, states, ee_states, gripper_states, joint_states, agent, wrist = ([] for _ in range(7))
    for action in traj["actions"]:
        states.append(sim.get_state().flatten())
        actions.append(action)
        o = env.get_observation()
        gripper_states.append(o["gripper_states"])
        joint_states.append(o["joint_states"])
        ee_states.append(np.hstack((o["ee_pos"], T.quat2axisangle(o["ee_ori"]))))
        agent.append(o["agentview_rgb"])
        wrist.append(o["eye_in_hand_rgb"])
        env.base_env.step(action.tolist() if hasattr(action, "tolist") else list(action))

    if not env.is_success()["task"]:
        return None, "replay_failed"

    rec = dict(actions=actions, states=states, ee_states=ee_states, gripper_states=gripper_states,
               joint_states=joint_states, agent=agent, wrist=wrist)
    return rec, dict(removed=removed, moved=moved, seed=int(seed))
