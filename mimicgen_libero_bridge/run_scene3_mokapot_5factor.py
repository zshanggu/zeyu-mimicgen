"""KITCHEN_SCENE3_put_the_moka_pot_on_the_stove, following the same established
pattern as run_scene1_6factor_v2.py / run_kitchen_task_6factor_v2.py (stratified
position randomization + mandatory per-episode distractor randomization), with
two task-specific adaptations:

1. `flat_stove_1` is a BDDL *fixture* (confirmed empirically: it's in
   `env.env.fixtures_dict`, not `objects_dict`, and its only joint is
   `flat_stove_1_button` -- there is no free joint to poke its position post-
   reset). So unlike bowl/plate (both plain pokeable objects in SCENE1/2), the
   stove's position is randomized the SAME way SCENE1's cabinet was: generate
   a few BDDL variants with different `flat_stove_init_region` ranges and
   reload a fresh env per variant (`STOVE_STATES`/`make_stove_bddl`, mirroring
   `make_cabinet_bddl`). Calibrated empirically (see session notes): the
   region's (y1,y2) maps ~linearly to world Y with a tiny deterministic
   offset at seed=0; X is kept at a fixed narrow band for all three states
   (the native BDDL region is already narrow in X -- the stove sits against
   the counter wall and only really varies along Y in practice).
2. The actual goal predicate is `(On moka_pot_1 flat_stove_1_cook_region)` --
   a SITE on the stove fixture, offset ~0.15m in +X from the stove body's own
   origin. MimicGen's placement retargeting must aim at that site, not the
   stove body's origin (reporting the body pose instead would retarget the
   whole trajectory ~15cm off target) -- see `MG_LiberoMokaOnStove.get_object_poses`.

Orientation factor (standing vs "laydown", rotating the pot 90deg about its
local Y axis -- see `LAYDOWN_QUAT`/`MOKA_CROSS_SECTION_HALF_EXTENT` below)
was tried and DROPPED per explicit instruction after review: across two
5-demo smoke batches (~53 combined attempts that reached actual trajectory
generation), laydown had a 0% success rate while standing succeeded at the
normal ~50-60% rate seen elsewhere in this bridge -- consistent with this
project's own "butter" precedent (`run_livingroom_butter_scripted.py`'s
docstring): the source demos were almost certainly collected with the pot
standing, so MimicGen's nearest-neighbor retargeting never transfers to a
90deg-rotated start pose. The pose-poke machinery (`LAYDOWN_QUAT` etc.) is
left in place, unused, as a record of what was tried -- a scripted
P-controller fallback (the butter task's own escape hatch) is the natural
next step if laydown data is wanted later, not a retry of this approach.

Position bounds were widened twice per explicit instruction, past what's
comfortably within the robot's reach envelope (Franka Panda, ~0.85m nominal
max, base at world [-0.66,0,0.912]) -- final bounds moka_x in [-0.35,0.00],
moka_y/stove_y in [-0.45,0.45] (see MOKA_REGION_*/STOVE_STATES below). This
is a deliberate coverage/diversity tradeoff, confirmed via review batches to
cost roughly 3.5x the attempts-per-success of the original, reach-conservative
bounds (yield dropped from ~13% to ~3.7%) -- NOT a bug, expected and accepted.

FORCED_CONFIGS guarantees two specific example configurations (given
explicitly, not left to chance under the stratified sampling) are present in
the output: moka_pot at (-0.2499,0.0504) with the stove's cook_region at
(-0.02,0.0), and moka_pot at (-0.2426,0.3004) with cook_region at
(-0.02,0.306) -- each gets its own narrow single-point stove BDDL variant
(`make_forced_stove_bddl`, same region-substitution mechanism as the
stratified `STOVE_STATES`, just a 0.02m box instead of a 0.3m band) and a
direct (non-sampled) moka_pot placement, generated BEFORE the stratified
pass so they count toward (and don't get crowded out of) the TARGET_TOTAL
budget.

Factors (4, standing-only): moka-stove distance bin, moka X bin, moka Y bin,
stove Y-position state -- plus the 2 forced exact configurations above.
  moka_stove_dist_bin in {near,medium,far}  -- distance moka_pot <-> cook_region
  moka_x_bin in {bottom,center,top}         -- moka_pot's own absolute X band
  moka_y_bin in {left,center,right}         -- moka_pot's own absolute Y band
  stove_state in {near,mid,far}             -- which BDDL variant (stove's Y position)
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

TASK_NAME = "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"
MOVING = "moka_pot_1"
TARGET_FIXTURE = "flat_stove_1"
TARGET_SITE = "flat_stove_1_cook_region"

BDDL_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90"
SRC_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_ori"
SCRATCH_SRC = "/mgbridge/_scratch_prepared_source_{task}_{label}.hdf5"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.90

# Unlike SCENE1/SCENE2 (whose native agentview camera was too tight for their
# widened bowl/plate region and needed a push-back offset), SCENE3's native
# camera already frames the moka_pot/stove counter area well -- confirmed by
# rendering it directly (session probe). Applying the SCENE1-tuned offset here
# pushed the camera into the cabinet overhang instead. So this is 0/0: the
# camera is left exactly where the arena puts it. _camera_safe/camera_y_bound
# below are camera-pose-agnostic raycasts, not SCENE1-specific -- they still
# correctly reject out-of-frame candidates for whatever the actual camera pose
# is, which is why they're kept rather than removed.
CAMERA_X_OFFSET = 0.0
CAMERA_Z_OFFSET = 0.0
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

# Per explicit instruction (widened past the previous, reach-conservative
# box): X in [-0.35, 0.00], Y in [-0.45, 0.45] -- note this pushes the
# far corners (X near 0.00 combined with |Y| near 0.45) to ~0.80m from the
# robot base, FARTHER than the ~0.71m "far" stove point already flagged as
# borderline (Panda's nominal max reach is ~0.85m) -- expect lower yield at
# those extreme corners, not higher; this is a deliberate diversity/coverage
# tradeoff, not a reach fix.
MOKA_REGION_X_LO, MOKA_REGION_X_HI = -0.35, 0.00
MOKA_REGION_Y_LO, MOKA_REGION_Y_HI = -0.45, 0.45

# Stove BDDL region variants -- (x1,x2) fixed for all states (calibrated so
# cook_region's world X lands near -0.02; the stove is a wall fixture, kept
# at its physically-sensible narrow X band rather than widened to the new
# bound). (y1,y2) now spans the full new Y bound [-0.45,0.45] in three equal
# ~0.3m-wide bands (previously bracketed just [0.0, 0.306]) -- the "far"
# band's upper edge (Y=0.45) is the point already flagged as pushing past
# ~0.71m from the robot base.
STOVE_X_RANGE = (-0.19, -0.17)
STOVE_STATES = {
    "near": (-0.45, -0.15),
    "mid": (-0.15, 0.15),
    "far": (0.15, 0.45),
}
STOVE_LEVELS = ["near", "mid", "far"]

DIST_BINS = {"near": (0.10, 0.20), "medium": (0.20, 0.32), "far": (0.32, 0.45)}
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]

_x_edges = np.linspace(MOKA_REGION_X_LO, MOKA_REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}
_y_edges = np.linspace(MOKA_REGION_Y_LO, MOKA_REGION_Y_HI, 4)
Y_BINS = {lvl: (float(_y_edges[i]), float(_y_edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}

LEVELS = ["near", "medium", "far"]
PRIMARY_COMBOS = [(d, x, y) for d in LEVELS for x in X_LEVELS for y in Y_LEVELS]

TARGET_TOTAL = int(os.environ.get("MG_TARGET_TOTAL", "100"))
MAX_ATTEMPTS = int(os.environ.get("MG_MAX_ATTEMPTS", "500"))
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100
N_DISTRACTOR_RETRIES = 3

# pick_least_used's fairness key sorts purely on (successes, attempts), so a
# primary combo that's geometrically INFEASIBLE for a given stove_state (no
# candidate XY in its bin box is simultaneously camera-safe, clearance-safe,
# and within the required distance band -- confirmed to happen for several
# combos specifically in the "far" stove state) permanently stays at
# successes=0 and therefore permanently outranks every combo that has
# already succeeded at least once -- the sampler gets stuck retrying only
# the infeasible combos forever once every feasible one has its first hit
# (confirmed empirically: a stove="far" pass spent 100% of ~5000 attempts on
# exactly 7 combos after its first 2 successes). PER_COMBO_GIVEUP_ATTEMPTS
# pushes a combo to the back of the priority order once it's burned this
# many attempts with zero successes, freeing the sampler to revisit combos
# that have already proven they CAN succeed (so pass_target, a plain
# success COUNT, isn't capped at the number of distinct feasible combos).
PER_COMBO_GIVEUP_ATTEMPTS = 60

# Tried and dropped (see module docstring) -- kept only as a record of what
# was tested, not used by the active (standing-only) codepath below.
LAYDOWN_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
MOKA_CROSS_SECTION_HALF_EXTENT = 0.025  # geom115 size[0], see probe notes
LAYDOWN_DROP_BUFFER = 0.02

# Two exact example configurations to guarantee in the output (not left to
# the stratified sampler's chance) -- see module docstring. stove_xy is the
# cook_region SITE's target world xy; make_forced_stove_bddl below converts
# that to the fixture body's region (subtracting the ~0.15m site offset).
#
# forced_far's moka_xy is NUDGED 0.055m in -X from the originally-given
# (-0.2426, 0.3004): probed directly (session notes) that the stove's own
# knob/button assembly (`flat_stove_1_button` geoms) sits at world
# x~[-0.21,-0.12] y~[0.27,0.36] when the stove is at its far-band position --
# the original point was interpenetrating those geoms by ~0.05m, causing an
# explosive MuJoCo contact-resolution fling (pot ends up flung ~0.5m and
# tumbled within 60 physics-settle steps, BEFORE generate() is even called --
# this is deterministic, not something more attempts could ever fix). The
# nudged point (-0.2976, 0.3004) is the first along -X that clears every
# stove geom by the same margin _clearance_safe enforces below, verified
# stable through 30 settle steps (quat stays ~identity, pos drifts <0.001m).
FORCED_CONFIGS = [
    {"tag": "forced_near", "moka_xy": (-0.2499, 0.0504), "stove_cook_xy": (-0.0200, 0.0000)},
    {"tag": "forced_far", "moka_xy": (-0.2976, 0.3004), "stove_cook_xy": (-0.0200, 0.3060)},
]
FORCED_MAX_ATTEMPTS_EACH = 100
FORCED_REGION_HALF_WIDTH = 0.01  # narrow box around the exact target, same convention as the native bddl's own 0.02m-wide region

# Guards against a different failure mode than the distractor-drift check:
# MimicGen's nearest-neighbor retargeting selects/interpolates from source
# demos that only ever saw moka_pot near its tiny native spawn point -- for a
# placement far from there (which the widened bounds make common), the
# retargeted APPROACH can clip the pot (narrow ~5cm base, ~13cm tall -- easily
# tipped) before the grasp even completes, and the trajectory can still go on
# to "succeed" from the pot's new, disturbed pose. Caught here, right after
# generate() returns and before distractor randomization even starts, by
# comparing against the TRUE commanded placement (env.last_placed_xy) rather
# than a later re-derived baseline (that's what the distractor-drift check
# alone missed -- it only ever saw the already-disturbed state).
PLACEMENT_DRIFT_TOLERANCE = 0.03

# Guards against a THIRD failure mode (distinct from both the camera-safety
# check and the drift check above): the moka candidate XY itself directly
# interpenetrating either the stove fixture's own geometry (its knob/button
# assembly protrudes well past the cook_region site -- see FORCED_CONFIGS
# comment) or another object's un-randomized native spawn position (e.g.
# chefmate_8_frypan_1's native spot at ~(-0.05,-0.24) falls inside the
# moka_x=top/moka_y=left bin box). Either causes the exact same explosive
# settle-phase physics fling diagnosed for forced_far, just stochastically
# for stratified sampling -- confirmed empirically to be most of the
# stove="far" pass's ~0% yield, since the stove's own knobs move into the
# far Y-band along with it. Checked in _clearance_safe, used by both the
# stratified sampler (reject-and-resample, same pattern as _camera_safe) and
# as a sanity check on forced placements (should trivially pass post-nudge).
# Calibrated directly against the two known forced-config outcomes (not
# guessed): forced_near (already empirically succeeds) measures 0.048m of
# true clearance against the nearest stove-knob geom surface; the original,
# broken forced_far measures only 0.029m (i.e. genuinely interpenetrating).
# MOKA_FOOTPRINT_RADIUS + STOVE_CLEARANCE_MARGIN = 0.04m sits between those
# two measurements, so it rejects the known-bad case without also rejecting
# the known-good one.
MOKA_FOOTPRINT_RADIUS = 0.035  # moka_pot's own base footprint (geom115 half-extent ~0.025m + settle slop)
STOVE_CLEARANCE_MARGIN = 0.005
OBJECT_CLEARANCE_RADIUS = 0.18  # flat XY-distance radius vs other objects' native spawn (covers the frypan's handle, which extends ~0.19m from its root body)


def make_stove_bddl(src_bddl, out_dir, task_name):
    with open(src_bddl) as fh:
        text = fh.read()
    paths = {}
    x1, x2 = STOVE_X_RANGE
    pattern = re.compile(
        r"(\(flat_stove_init_region\s*\(:target kitchen_table\)\s*\(:ranges \(\s*)\([^)]*\)", re.S)
    for state, (y1, y2) in STOVE_STATES.items():
        rng = f"({x1} {y1} {x2} {y2})"
        new_text, n = pattern.subn(lambda m: m.group(1) + rng, text)
        assert n == 1, f"expected exactly 1 stove region substitution, got {n}"
        out_path = f"{out_dir}/{task_name}_stove_{state}.bddl"
        with open(out_path, "w") as fh:
            fh.write(new_text)
        paths[state] = out_path
    return paths


def make_forced_stove_bddl(src_bddl, out_dir, task_name):
    """One narrow single-point BDDL variant per FORCED_CONFIGS entry, same
    region-substitution mechanism as make_stove_bddl but centered on the
    exact target cook_region xy (converted to the fixture body's region by
    subtracting the site's ~0.15m +X offset -- see module docstring)."""
    with open(src_bddl) as fh:
        text = fh.read()
    pattern = re.compile(
        r"(\(flat_stove_init_region\s*\(:target kitchen_table\)\s*\(:ranges \(\s*)\([^)]*\)", re.S)
    paths = {}
    for cfg in FORCED_CONFIGS:
        cook_x, cook_y = cfg["stove_cook_xy"]
        body_x = cook_x - 0.15
        h = FORCED_REGION_HALF_WIDTH
        rng = f"({body_x - h} {cook_y - h} {body_x + h} {cook_y + h})"
        new_text, n = pattern.subn(lambda m: m.group(1) + rng, text)
        assert n == 1, f"expected exactly 1 stove region substitution, got {n}"
        out_path = f"{out_dir}/{task_name}_stove_{cfg['tag']}.bddl"
        with open(out_path, "w") as fh:
            fh.write(new_text)
        paths[cfg["tag"]] = out_path
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


def label_for(primary, stove_state):
    d, x, y = primary
    return f"dist-{d}_mokaX-{x}_mokaY-{y}_stove-{stove_state}"


class MG_LiberoMokaOnStove(RobosuiteInterface):
    INTERFACE_TYPE = "libero"
    _FINGER_BODY_NAMES = ("gripper0_leftfinger", "gripper0_rightfinger",
                           "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip")

    def get_object_poses(self):
        moving_body = self.env.env.objects_dict[MOVING].root_body
        return dict(
            moka=self.get_object_pose(obj_name=moving_body, obj_type="body"),
            # the goal is (On moka_pot flat_stove_1_cook_region) -- a SITE, not
            # the stove fixture's own body origin (offset ~0.15m away). Retarget
            # the "place" subtask at the site the goal actually checks.
            stove=self.get_object_pose(obj_name=TARGET_SITE, obj_type="site"),
        )

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
        return dict(grasp=int(self._grasped_object_name() == MOVING))


class LiberoEnvScene3:
    def __init__(self, libero_env, stove_state):
        self.base_env = libero_env
        self.stove_state = stove_state
        self.target_primary = None    # (dist_bin, moka_x_bin, moka_y_bin) -- unused when forced_moka_xy is set
        self.forced_moka_xy = None    # set to an exact (x,y) to bypass sampling -- see FORCED_CONFIGS
        self.last_placed_xy = None    # the TRUE commanded placement from the last reset() -- see check_placement_integrity
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

    def _clearance_safe(self, xy):
        sim = self.base_env.env.sim
        xy = np.asarray(xy)
        for i in range(sim.model.ngeom):
            body_id = sim.model.geom_bodyid[i]
            bname = sim.model.body_id2name(body_id) or ""
            if TARGET_FIXTURE in bname:
                gpos = sim.data.geom_xpos[i][:2]
                gsize = sim.model.geom_size[i]
                req = (float(np.max(gsize[:2])) if gsize.size >= 2 else 0.02) \
                    + MOKA_FOOTPRINT_RADIUS + STOVE_CLEARANCE_MARGIN
                if np.linalg.norm(xy - gpos) < req:
                    return False
        for name, obj in self.base_env.env.objects_dict.items():
            if name == MOVING:
                continue
            opos = sim.data.body_xpos[sim.model.body_name2id(obj.root_body)][:2]
            if np.linalg.norm(xy - opos) < OBJECT_CLEARANCE_RADIUS:
                return False
        return True

    def _stove_site_xy(self):
        sim = self.base_env.env.sim
        sid = sim.model.site_name2id(TARGET_SITE)
        return sim.data.site_xpos[sid][:2].copy()

    def _set_moka_pose(self, xy):
        # standing only (laydown dropped -- see module docstring): only override
        # XY, same convention as bowl/plate elsewhere in this bridge -- z/quat
        # are left exactly as the native reset settled them.
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[MOVING].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _sample_moka_xy(self, x_bin, y_bin, dist_lo, dist_hi, stove_xy):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = Y_BINS[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(dist_lo, dist_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = stove_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if x_lo <= cand[0] <= x_hi and y_lo <= cand[1] <= y_hi and self._camera_safe(cand) \
                    and self._clearance_safe(cand):
                return cand
        return None

    def reset(self):
        ret = self.base_env.reset()
        self._apply_camera_offset()

        if self.forced_moka_xy is not None:
            # exact placement, no sampling -- still camera-safety-checked for
            # sanity, but these are real observed points, expected to pass trivially.
            moka_xy = np.array(self.forced_moka_xy)
            if not self._camera_safe(moka_xy):
                raise RuntimeError(f"forced_moka_xy {self.forced_moka_xy} failed the camera-safety check")
            if not self._clearance_safe(moka_xy):
                raise RuntimeError(f"forced_moka_xy {self.forced_moka_xy} failed the clearance check "
                                    f"(interpenetrates the stove or another object)")
            self._set_moka_pose(moka_xy)
        else:
            dist_bin, moka_x_bin, moka_y_bin = self.target_primary
            dist_lo, dist_hi = DIST_BINS[dist_bin]
            stove_xy = self._stove_site_xy()

            placed = False
            for _ in range(MAX_OUTER_RETRIES):
                moka_xy = self._sample_moka_xy(moka_x_bin, moka_y_bin, dist_lo, dist_hi, stove_xy)
                if moka_xy is None:
                    continue
                self._set_moka_pose(moka_xy)
                placed = True
                break
            if not placed:
                raise RuntimeError(f"could not satisfy {self.target_primary}/stove-{self.stove_state} in {MAX_OUTER_RETRIES} outer tries")

        self.last_placed_xy = np.array(moka_xy)  # the TRUE commanded placement -- see check_placement_integrity
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
        return {"env_name": "LIBERO_scene3_mokapot_5factor", "type": "libero"}


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="moka", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="stove", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def prepare_source(bddl_path, src_hdf5, scratch_path):
    from libero.libero.envs import OffScreenRenderEnv
    import robomimic.utils.tensor_utils as TensorUtils
    from tqdm import tqdm

    shutil.copy(src_hdf5, scratch_path)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    env.seed(0)
    env.reset()
    env_interface = MG_LiberoMokaOnStove(env=env)

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
        ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_LiberoMokaOnStove"
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

    # flat_stove_1 is a BDDL fixture (see module docstring) -- its world
    # position is baked into the MJCF model at `raw_env.reset()` time via
    # region sampling, NOT stored in qpos. The `raw_env.reset()` below (used
    # to re-seed for the distractor layout search) rebuilds the model and
    # therefore silently re-randomizes the stove to a brand-new position;
    # `env.reset_to(initial_state)` only restores dynamic qpos/qvel state, so
    # it does NOT undo this. Left unfixed, every episode's stove ends up at
    # an uncontrolled position uncorrelated with the one actually used for
    # distance-bin sampling and retargeting -- confirmed empirically (session
    # audit: 36/100 saved demos had a moka-to-stove distance inconsistent
    # with their labeled bin, occasionally landing the stove almost exactly
    # where the moka pot already was). Worse, the final success check also
    # runs against this wrong position. Fix: capture the stove's true pose
    # from `raw_env`'s CURRENT (correct, untouched-since-generate()) sim
    # before the reset, then poke it back into the rebuilt model afterward.
    _stove_sim = raw_env.env.sim
    _stove_body_id = _stove_sim.model.body_name2id(f"{TARGET_FIXTURE}_main")
    _stove_pos = _stove_sim.model.body_pos[_stove_body_id].copy()
    _stove_quat = _stove_sim.model.body_quat[_stove_body_id].copy()

    raw_env.seed(seed)
    raw_env.reset()
    _stove_sim = raw_env.env.sim  # reset() rebuilds the model -- refresh the reference
    _stove_body_id = _stove_sim.model.body_name2id(f"{TARGET_FIXTURE}_main")
    _stove_sim.model.body_pos[_stove_body_id] = _stove_pos
    _stove_sim.model.body_quat[_stove_body_id] = _stove_quat
    _stove_sim.forward()
    env.reset_to({"states": initial_state})
    # Measure the TRUE pre-search baseline right here, before DL.sample_layout
    # runs -- it was previously measured AFTER sample_layout returned, which
    # reads whatever state sample_layout's own internal candidate search left
    # raw_env's live sim in (not necessarily `st`, and not necessarily the
    # true original placement). That let a distractor's search/settle process
    # silently knock over a task object (confirmed: moka_pot placed far from
    # its native spawn, per the wide position bounds) without the drift check
    # ever catching it, since both `pre` and `post` ended up reflecting the
    # SAME already-disturbed state. Measuring here instead, right after the
    # state is known-correct (fresh from `initial_state`), makes the check
    # compare against the real original position.
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

    import os as _os
    _debug = _os.environ.get("MG_DEBUG_DRIFT") == "1"
    if _debug:
        print(f"    [DEBUG] pre={pre}", flush=True)
    env.reset_to({"states": st})
    for _ in range(10):
        env.base_env.step(_DUMMY_ACTION)
    sim = raw_env.env.sim
    for n in task_object_names:
        post = sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy()
        if _debug:
            print(f"    [DEBUG] post[{n}]={post} drift={np.linalg.norm(post - pre[n]):.4f}", flush=True)
        if np.linalg.norm(post - pre[n]) > 0.01:
            return None, f"task_object_drift({n})"

    all_object_names = list(raw_env.env.objects_dict.keys())
    actions, states, ee_states, gripper_states, joint_states, agent, wrist = ([] for _ in range(7))
    object_poses = {n: [] for n in all_object_names}
    if _debug:
        for n in task_object_names:
            body = raw_env.env.objects_dict[n].root_body
            print(f"    [DEBUG] right before replay loop, {n} pos={sim.data.get_body_xpos(body)}", flush=True)
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


def _count_existing_by_stove_state(existing_path):
    """For --topup: count demos already in `existing_path`, keyed by their
    recorded `stove_state` attr. Forced-config demos (stove_state starting
    with "forced_") are counted separately and not charged against the
    near/mid/far stratified quota."""
    counts = {"near": 0, "mid": 0, "far": 0}
    forced_tags = set()
    with h5py.File(existing_path, "r") as f:
        for name in f["data"].keys():
            st = f[f"data/{name}"].attrs.get("stove_state")
            if st is None:
                continue
            if isinstance(st, bytes):
                st = st.decode()
            if st.startswith("forced_"):
                forced_tags.add(st)
            elif st in counts:
                counts[st] += 1
    return counts, forced_tags


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/mgbridge/out"
    label = sys.argv[2] if len(sys.argv) > 2 else "scene3_mokapot_5factor"
    topup_path = None
    if "--topup" in sys.argv:
        topup_path = sys.argv[sys.argv.index("--topup") + 1]
        assert os.path.exists(topup_path), topup_path
        print(f"--topup mode: continuing from {topup_path}", flush=True)

    bddl_path = os.path.join(BDDL_DIR, f"{TASK_NAME}.bddl")
    src_hdf5 = os.path.join(SRC_DIR, f"{TASK_NAME}_demo.hdf5")
    assert os.path.exists(bddl_path), bddl_path
    assert os.path.exists(src_hdf5), src_hdf5
    print(f"task={TASK_NAME} moving={MOVING} target_site={TARGET_SITE} "
          f"TARGET_TOTAL={TARGET_TOTAL} MAX_ATTEMPTS={MAX_ATTEMPTS}")

    stove_bddl_paths = make_stove_bddl(bddl_path, "/mgbridge", TASK_NAME)
    forced_bddl_paths = make_forced_stove_bddl(bddl_path, "/mgbridge", TASK_NAME)

    scratch_path = SCRATCH_SRC.format(task=TASK_NAME, label=label)
    print("preparing source dataset...", flush=True)
    prepare_source(bddl_path, src_hdf5, scratch_path)

    from libero.libero.envs import OffScreenRenderEnv

    probe_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
    probe_env.seed(0); probe_env.reset()
    all_objects = set(probe_env.env.objects_dict.keys())
    probe_env.close()
    task_object_names = {MOVING}  # flat_stove_1 is a fixture, not in objects_dict/distractor-movable set
    print(f"objects_dict={all_objects}, task_object_names={task_object_names}, "
          f"distractors={all_objects - task_object_names}")

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=scratch_path, filter_key=None, start=None, n=None)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    rng_combo = np.random.default_rng(42)
    succ_records = []  # (primary, stove_state, info) -- primary is ("forced","forced","forced") for forced entries
    total_attempts = 0
    distractor_discards = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    def try_once(env, raw_env, env_interface, data_generator, stove_state, label_str):
        nonlocal total_attempts, distractor_discards
        try:
            traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                            transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                            render=False, video_writer=None)
        except Exception as e:  # noqa: BLE001
            total_attempts += 1
            print(f"[{label_str}] exception: {type(e).__name__}: {e}", flush=True)
            return None
        total_attempts += 1
        if not traj["success"]:
            print(f"[{label_str}] success=False", flush=True)
            return None

        # Check for an approach-phase knockover (see PLACEMENT_DRIFT_TOLERANCE)
        # BEFORE touching distractors -- uses the exact state generate() left
        # behind, same one randomize_distractors would otherwise inherit unchecked.
        env.reset_to({"states": traj["initial_state"]["states"]})
        actual_xy = raw_env.env.sim.data.get_body_xpos(raw_env.env.objects_dict[MOVING].root_body)[:2].copy()
        placement_drift = float(np.linalg.norm(actual_xy - env.last_placed_xy))
        if placement_drift > PLACEMENT_DRIFT_TOLERANCE:
            print(f"[{label_str}] moka_pot drifted {placement_drift:.3f}m from its commanded placement during the "
                  f"retargeted approach (likely knocked over) -- discarded", flush=True)
            return None

        rec = info = None
        for retry in range(N_DISTRACTOR_RETRIES):
            rec, info = randomize_distractors(env, raw_env, traj, task_object_names, seed=8_000_000 + total_attempts * 10 + retry)
            if rec is not None:
                break
        if rec is None:
            distractor_discards += 1
            print(f"[{label_str}] moka/stove succeeded but distractor randomization failed ({info}) after {N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
            return None
        print(f"[{label_str}] success=True distractors: removed={info['removed']} moved={list(info['moved'].keys())}", flush=True)
        return rec, info

    existing_counts = {"near": 0, "mid": 0, "far": 0}
    existing_forced_tags = set()
    if topup_path:
        existing_counts, existing_forced_tags = _count_existing_by_stove_state(topup_path)
        print(f"--topup: existing counts {existing_counts}, existing forced tags {existing_forced_tags}", flush=True)

    # --- forced exact configurations first, so they're guaranteed a slot in TARGET_TOTAL ---
    # In --topup mode, a tag already present in existing_forced_tags is
    # skipped (already carried forward from the existing file). forced_far
    # used to be unconditionally skipped here too (it couldn't be generated
    # cleanly even at FORCED_MAX_ATTEMPTS_EACH=100), but that was a
    # deterministic stove-knob interpenetration bug, now fixed (nudged
    # coordinate + _clearance_safe -- see FORCED_CONFIGS/PLACEMENT_DRIFT_TOLERANCE
    # comments), so it's attempted again like any other missing tag.
    for cfg in FORCED_CONFIGS:
        tag = cfg["tag"]
        if topup_path and tag in existing_forced_tags:
            print(f"\n--- forced config: {tag} already present -- skipping ---", flush=True)
            continue
        print(f"\n--- forced config: {tag} (moka={cfg['moka_xy']} stove_cook={cfg['stove_cook_xy']}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=forced_bddl_paths[tag], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnvScene3(raw_env, tag)
        env.forced_moka_xy = cfg["moka_xy"]
        env_interface = MG_LiberoMokaOnStove(env=env.base_env)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

        found = False
        for attempt in range(1, FORCED_MAX_ATTEMPTS_EACH + 1):
            result = try_once(env, raw_env, env_interface, data_generator, tag, f"forced={tag} attempt {attempt}/{FORCED_MAX_ATTEMPTS_EACH}")
            if result is not None:
                rec, info = result
                idx = len(succ_records)
                primary = ("forced", "forced", "forced")
                succ_records.append((primary, tag, info))
                write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                              attrs=dict(moka_stove_dist_bin="forced", moka_x_bin="forced", moka_y_bin="forced",
                                         stove_state=tag, moka_xy=list(cfg["moka_xy"]), stove_cook_xy=list(cfg["stove_cook_xy"]),
                                         removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
                found = True
                break
        if not found:
            print(f"!!! forced config {tag} did NOT succeed after {FORCED_MAX_ATTEMPTS_EACH} attempts -- skipping it, continuing with the rest", flush=True)
        raw_env.close()

    # --- stratified near/mid/far pass fills the remaining TARGET_TOTAL budget ---
    # In --topup mode, each state's target is computed against what's already
    # in the source file (existing_counts), so this run only adds the shortfall.
    if topup_path:
        # Integer division alone silently drops the remainder (e.g. 98 // 3 ==
        # 32 for all three states, even though 2 more are needed to reach
        # 100) -- same bug/fix as the non-topup branch below, distribute the
        # remainder to the first _rem states instead of dropping it.
        _total_for_split = max(0, TARGET_TOTAL - len(existing_forced_tags))
        _base = _total_for_split // len(STOVE_LEVELS)
        _rem = _total_for_split % len(STOVE_LEVELS)
        per_state_targets = {lvl: _base + (1 if i < _rem else 0) for i, lvl in enumerate(STOVE_LEVELS)}
        pass_targets = {lvl: max(0, per_state_targets[lvl] - existing_counts[lvl]) for lvl in STOVE_LEVELS}
    else:
        remaining = max(0, TARGET_TOTAL - len(succ_records))
        _base = remaining // len(STOVE_LEVELS)
        _rem = remaining % len(STOVE_LEVELS)
        pass_targets = {lvl: _base + (1 if i < _rem else 0) for i, lvl in enumerate(STOVE_LEVELS)}
    # Per-state attempt budgets -- MG_MAX_ATTEMPTS_<STATE> overrides let a
    # --topup run throw a much bigger budget at specifically the state(s)
    # that are short (e.g. "far", which showed ~0% yield in the first pass)
    # without also forcing that same huge budget on states that already
    # yield reasonably (near/mid).
    default_pass_max = max(1, MAX_ATTEMPTS // len(STOVE_LEVELS))
    pass_max_attempts_by_state = {
        lvl: int(os.environ.get(f"MG_MAX_ATTEMPTS_{lvl.upper()}", default_pass_max)) for lvl in STOVE_LEVELS
    }

    for stove_state in STOVE_LEVELS:
        pass_target = pass_targets[stove_state]
        pass_max_attempts = pass_max_attempts_by_state[stove_state]
        if pass_target <= 0:
            print(f"\n--- stove state: {stove_state} already has enough ({existing_counts[stove_state]}) -- skipping ---", flush=True)
            continue
        print(f"\n--- stove state: {stove_state} (bddl {stove_bddl_paths[stove_state]}, target +{pass_target}, budget {pass_max_attempts}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=stove_bddl_paths[stove_state], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnvScene3(raw_env, stove_state)
        env_interface = MG_LiberoMokaOnStove(env=env.base_env)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=scratch_path, demo_keys=all_demos)

        # Fresh per stove_state: feasibility is state-specific (geometry
        # differs a lot between near/mid/far stove placements), so carrying
        # successes/attempts over from a previous state's pass would wrongly
        # deprioritize combos that are actually easy in THIS state.
        per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}

        def primary_key(p):
            attempts, successes = per_primary[p][0], per_primary[p][1]
            given_up = attempts >= PER_COMBO_GIVEUP_ATTEMPTS and successes == 0
            return (given_up, successes, attempts)

        pass_successes = 0
        pass_attempts = 0
        while pass_successes < pass_target and pass_attempts < pass_max_attempts and len(succ_records) < TARGET_TOTAL:
            primary = pick_least_used(PRIMARY_COMBOS, primary_key, rng_combo)
            env.target_primary = primary
            per_primary[primary][0] += 1
            pass_attempts += 1
            label_str = f"stove={stove_state} attempt {pass_attempts}/{pass_max_attempts} {label_for(primary, stove_state)}"
            result = try_once(env, raw_env, env_interface, data_generator, stove_state, label_str)
            if result is None:
                continue
            rec, info = result
            per_primary[primary][1] += 1
            pass_successes += 1
            idx = len(succ_records)
            succ_records.append((primary, stove_state, info))
            write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), rec,
                          attrs=dict(moka_stove_dist_bin=primary[0], moka_x_bin=primary[1], moka_y_bin=primary[2],
                                     stove_state=stove_state,
                                     removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
            print(f"  -> pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL}", flush=True)

        raw_env.close()
        print(f"--- stove state {stove_state} done: {pass_successes}/{pass_attempts} ---", flush=True)

    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"({distractor_discards} discarded after moka/stove succeeded but distractor randomization failed) ===")
    from collections import Counter
    print("stove state distribution:", dict(Counter(r[1] for r in succ_records)))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    # --topup writes to a side file first, then atomically replaces out_path at
    # the end -- out_path and topup_path are typically the SAME file (resuming
    # the production file in place), so it can't be opened in "w" mode while
    # still being read from.
    write_path = out_path + ".topup_tmp" if topup_path else out_path
    with h5py.File(write_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": TASK_NAME})
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = total_attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_records) + sum(existing_counts.values()) + len(existing_forced_tags)
        data_grp.attrs["moving_object"] = MOVING
        data_grp.attrs["target_site"] = TARGET_SITE
        data_grp.attrs["distractors_always_randomized"] = True
        data_grp.attrs["stratification"] = json.dumps({
            "moka_stove_distance_bins_m": DIST_BINS, "moka_x_bins_m": X_BINS, "moka_y_bins_m": Y_BINS,
            "stove_x_range_m": STOVE_X_RANGE, "stove_y_states_m": STOVE_STATES,
            "camera_offset": {"x": CAMERA_X_OFFSET, "z": CAMERA_Z_OFFSET},
            "forced_configs": FORCED_CONFIGS,
        })
        i = 0
        if topup_path:
            with h5py.File(topup_path, "r") as old_f:
                for name in old_f["data"].keys():
                    new_name = f"demo_{i:03d}_" + "_".join(name.split("_")[2:])
                    old_f.copy(f"data/{name}", data_grp, name=new_name)
                    i += 1
        for idx_in_tmp, (primary, stove_state, info) in enumerate(succ_records):
            group_name = f"demo_{i:03d}_{label_for(primary, stove_state)}"
            with h5py.File(os.path.join(tmp_dir, f"{idx_in_tmp:04d}.hdf5"), "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=group_name)
            i += 1

    if topup_path:
        shutil.move(write_path, out_path)
        print(f"--topup: merged {sum(existing_counts.values()) + len(existing_forced_tags)} existing + "
              f"{len(succ_records)} new = {i} total demos into {out_path}", flush=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(scratch_path)
    print(f"wrote {out_path}: {len(succ_records)} episodes, every one with randomized distractors")


if __name__ == "__main__":
    main()
