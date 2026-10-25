"""MimicGen env_interface + minimal EnvBase-duck-type wrapper for LIBERO's
OffScreenRenderEnv, for the KITCHEN_SCENE1_put_the_black_bowl_on_the_plate task.

Why a custom wrapper instead of robomimic's EnvUtils/EnvRobosuite: that path
goes through robosuite.make(env_name, ...), which requires the env be a
registered, named robosuite env class. LIBERO environments are instead built
directly from a bddl file via libero.libero.envs.OffScreenRenderEnv, which
isn't registered that way. mimicgen.datagen.data_generator.DataGenerator and
mimicgen.datagen.waypoint.WaypointTrajectory.execute only ever call a small
duck-typed subset of the robomimic EnvBase API on `env`
(reset/get_state/reset_to/get_observation/step/is_success/render), so this
file implements exactly that subset directly against LIBERO, with no
robomimic env dependency at all.
"""
import numpy as np

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

BOWL_NAME = "akita_black_bowl_1"
PLATE_NAME = "plate_1"

# Same convention as zeyu-Pi0.5Viewer/backend/sim_worker.py's do_reset_and_warmup:
# a freshly reset LIBERO env drops objects from their placement height and they
# need a few physics steps to actually settle onto the table before any pose
# is trustworthy. Skipping this made the bowl's read-off pose ~7cm higher than
# its true resting height, so the first generation pass retargeted the whole
# grasp trajectory at a stale height and the gripper closed on empty air.
NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Bowl and plate now sample from the SAME region (so their relative left/right
# arrangement actually varies episode to episode, instead of being pinned by
# disjoint y-bands). LIBERO's region sampler is supposed to reject overlapping
# placements (ensure_valid_placement=True) but measured empirically over 60
# resets, the true min bowl-plate center distance was 0.053m -- well inside
# the ~0.14m (bowl r~0.05 + plate r~0.09) needed to not interpenetrate. So the
# built-in check isn't actually enforced across these two independently-built
# samplers; reject-and-reroll here instead.
_MIN_BOWL_PLATE_DIST = 0.16
_MAX_RESET_RETRIES = 30

# Distance stratification: left to pure independent-uniform sampling over the
# shared bowl/plate region, bowl-plate separation is NOT uniform -- measured
# empirically over 76 episodes, it clustered at 0.2-0.4m (median 0.32m) and
# NEVER exceeded 0.53m, even though the region's own diagonal allows up to
# 0.72m. That's the expected shape for two independent uniform points in a
# box (a large separation needs both draws to land in near-opposite corners
# at once -- rare by chance), not a bug -- but it means "far apart" cases are
# real but under-represented if coverage, not just occurrence, is the goal.
# Fix: instead of relying on chance, sample bowl normally, then DIRECTLY place
# plate at a target distance (drawn from a chosen bin) and random direction
# from the bowl, clipped to stay inside the same shared region. This replaces
# rejection sampling (which would need very many retries to hit the rare far
# tail) with direct construction.
BOWL_PLATE_DIST_BINS = {
    "near": (0.16, 0.26),
    "medium": (0.26, 0.40),
    "far": (0.40, 0.60),
}
# Must match the shared bowl/plate bddl region (make_widened_bddl.py's
# BOWL_RANGE/PLATE_RANGE for KITCHEN_SCENE1): x in [-0.45,0.10], y in [-0.12,0.35].
REGION_XY_LO = np.array([-0.45, -0.12])
REGION_XY_HI = np.array([0.10, 0.35])
_MAX_PLACEMENT_TRIES = 60

# Any-finger-touching grasp check (robosuite's own _check_grasp requires BOTH
# fingers simultaneously and is known to under-report for this robot model --
# see zeyu-Pi0.5Viewer/backend/sim_worker.py's _grasped_object_name, which
# this mirrors). A missed grasp transition would make
# mimicgen.utils.file_utils.parse_source_dataset's subtask-boundary detection
# raise ("No valid subtask termination found"), so under-reporting here is a
# hard blocker, not just an accuracy nit.
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


class MG_LiberoBowlOnPlate(RobosuiteInterface):
    """
    object_ref names used in the task spec: "bowl" (subtask 1, grasp) and
    "plate" (subtask 2, final placement).
    """
    INTERFACE_TYPE = "libero"

    def get_object_poses(self):
        bowl_body = self.env.env.objects_dict[BOWL_NAME].root_body
        plate_body = self.env.env.objects_dict[PLATE_NAME].root_body
        return dict(
            bowl=self.get_object_pose(obj_name=bowl_body, obj_type="body"),
            plate=self.get_object_pose(obj_name=plate_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        grasped = _grasped_object_name(self.env)
        return dict(grasp=int(grasped == BOWL_NAME))


class LiberoEnvForDataGen:
    """Minimal robomimic-EnvBase-duck-type wrapper around a LIBERO
    OffScreenRenderEnv, exposing just what mimicgen.datagen.data_generator /
    mimicgen.datagen.waypoint actually call: reset, get_state, reset_to,
    get_observation, step, is_success, render.
    """

    def __init__(self, libero_env):
        # same object the env_interface above reads poses/contacts from --
        # mimicgen requires stepping and pose-reading to hit the same live sim.
        self.base_env = libero_env
        # None (default) = old behavior, just enforce the no-overlap minimum.
        # Set to "near"/"medium"/"far" before calling reset() (e.g. by the
        # generation driver, cycling per attempt) to target that bin instead.
        self.target_bin = None
        self.rng = np.random.default_rng()

    def _xy(self, name):
        sim = self.base_env.env.sim
        return sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[:2].copy()

    def _place_plate_at_distance(self, bowl_xy, lo, hi):
        """Directly set plate's xy at a distance in [lo,hi] from bowl_xy and a
        random direction, clipped to stay inside REGION_XY_LO/HI. Returns True
        if a valid candidate was found and applied, False otherwise (caller
        should reroll the bowl position and retry)."""
        sim = self.base_env.env.sim
        plate_obj = self.base_env.env.objects_dict[PLATE_NAME]
        joint = plate_obj.joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        for _ in range(_MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(lo, hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = bowl_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if not (REGION_XY_LO[0] <= cand[0] <= REGION_XY_HI[0] and REGION_XY_LO[1] <= cand[1] <= REGION_XY_HI[1]):
                continue
            sim.data.qpos[qs:qs + 2] = cand
            sim.data.qvel[vs:ve] = 0.0
            sim.forward()
            return True
        return False

    def reset(self):
        ret = self.base_env.reset()
        bin_range = BOWL_PLATE_DIST_BINS.get(self.target_bin) if self.target_bin else None
        for attempt in range(_MAX_RESET_RETRIES):
            bowl_xy = self._xy(BOWL_NAME)
            plate_xy = self._xy(PLATE_NAME)
            dist = np.linalg.norm(bowl_xy - plate_xy)
            if bin_range is None:
                if dist >= _MIN_BOWL_PLATE_DIST:
                    break
            else:
                lo, hi = bin_range
                if lo <= dist <= hi:
                    break
                if self._place_plate_at_distance(bowl_xy, lo, hi):
                    break
            ret = self.base_env.reset()
        else:
            raise RuntimeError(
                f"could not find a valid bowl/plate placement (target_bin={self.target_bin}) in {_MAX_RESET_RETRIES} tries"
            )
        for _ in range(NUM_SETTLE_STEPS):
            ret = self.base_env.step(_DUMMY_ACTION)
        return ret

    def get_state(self):
        return {"states": self.base_env.get_sim_state()}

    def reset_to(self, state):
        return self.base_env.regenerate_obs_from_state(state["states"])

    def get_observation(self):
        raw = self.base_env.env._get_observations()
        # Minimal remap toward the source dataset's own obs schema
        # (agentview_rgb/eye_in_hand_rgb/ee_pos/ee_ori/ee_states/gripper_states/
        # joint_states) -- best-effort for this quick test; not pixel-verified
        # against the source hdf5's own flip/encoding convention.
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
        ee_pos = obs.get("ee_pos")
        ee_ori = obs.get("ee_ori")
        if ee_pos is not None and ee_ori is not None:
            obs["ee_states"] = np.concatenate([ee_pos, ee_ori])
        return obs

    def step(self, action):
        return self.base_env.step(action)

    def is_success(self):
        return {"task": bool(self.base_env.env._check_success())}

    def render(self, *args, **kwargs):
        raise NotImplementedError("rendering not used in this quick test (render=False, video_writer=None)")

    def serialize(self):
        """mimicgen.utils.file_utils.write_demo_to_hdf5 stores this under the
        episode's env_args attr for bookkeeping -- not used by anything else here."""
        return {"env_name": "LIBERO_KITCHEN_SCENE1_put_the_black_bowl_on_the_plate", "type": "libero"}
