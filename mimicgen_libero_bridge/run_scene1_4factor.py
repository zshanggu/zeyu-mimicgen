"""Extends run_scene1_3factor.py with a 4th axis: the wooden_cabinet_1
fixture's distance to the desk edge (near/medium/far), or absent entirely.

The cabinet can't be repositioned the way bowl/plate are (direct qpos poke
after reset): it has NO free joint -- `fixtures_dict['wooden_cabinet_1'].joints`
only returns its three drawer/shelf SLIDE joints (confirmed: get_joint_qpos_addr
on cab.joints[0] raised, since it's a 1-DOF slide joint, not 6-DOF free).

First attempt was to mutate env.env.placement_initializer.samplers
['wooden_cabinet_1_sampler'].x_ranges/y_ranges before each reset() -- same
trick generic_pipeline.py uses for movable-object regions. Measured and
disproved that this does anything at all here: id(env.env.placement_initializer)
is DIFFERENT on every single env.reset() call (checked across 5 resets --
all distinct Python objects), i.e. it's rebuilt FRESH from the bddl file's
declared regions every reset, discarding anything mutated beforehand. This
turns out to be harmless for bowl/plate (their sampler mutation in
generic_pipeline.py was ALSO silently a no-op the whole time -- it just
never mattered, since their position gets overwritten by a direct qpos poke
after reset regardless of what the sampler drew). The cabinet has no qpos to
poke, so for it the bddl's declared region is the ONLY real lever -- meaning
cabinet state must be fixed per ENV CONSTRUCTION (one bddl + one env per
state, via make_cabinet_bddl.py), not mutated per attempt.

Measured directly (not assumed): cabinet real footprint half-size
x=0.121, y=0.167 (full ~24x33cm), native center (-0.009,-0.308). To stay
clear of the bowl/plate region (y >= -0.12) it must stay in a narrow band:
x in [-0.36,0.36], y in [-0.42,-0.29]. Within that band, achievable
distance-to-table-edge is ~0.14-0.31m (narrower than bowl/plate's
0.05-0.40, since the cabinet is much bulkier and more confined).
make_cabinet_bddl.py writes 4 bddl variants, one per state, each verified
corner-by-corner to land in its target CABINET_EDGE_BINS range. "absent"
parks it at (2.0, 2.0), outside the table/camera entirely.

Architecture: outer loop over the 4 cabinet states, one env built per
state (4 constructions total, each a few seconds -- cheap). Within each
state's pass, the SAME global per-27-combo counters drive the existing
least-covered-first rule, so the 27 (bp, bowl_edge, plate_edge) combos stay
balanced in aggregate across all 4 passes (per the explicit instruction:
keep ~100 total, prioritize full coverage of the 27 over the cabinet axis).
Each pass targets TARGET_TOTAL/4 successes.
"""
import sys
import os
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

from env_interface import MG_LiberoBowlOnPlate, BOWL_NAME, PLATE_NAME
import distractor_lib as DL

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.90

REGION_XY_LO = np.array([-0.45, -0.12])
REGION_XY_HI = np.array([0.10, 0.35])
TABLE_XY_LO = np.array([-0.5, -0.6])
TABLE_XY_HI = np.array([0.5, 0.6])

BP_BINS = {"near": (0.16, 0.26), "medium": (0.26, 0.38), "far": (0.38, 0.55)}
EDGE_BINS = {"near": (0.05, 0.17), "medium": (0.17, 0.29), "far": (0.29, 0.42)}
CABINET_EDGE_BINS = {"near": (0.14, 0.20), "medium": (0.20, 0.26), "far": (0.26, 0.32)}
LEVELS = ["near", "medium", "far"]
CABINET_LEVELS = ["near", "medium", "far", "absent"]
COMBOS27 = [(bp, be, pe) for bp in LEVELS for be in LEVELS for pe in LEVELS]

TARGET_TOTAL = 100
MAX_ATTEMPTS = 400
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100


def dist_to_edge(xy):
    return min(xy[0] - TABLE_XY_LO[0], TABLE_XY_HI[0] - xy[0], xy[1] - TABLE_XY_LO[1], TABLE_XY_HI[1] - xy[1])


def label_for(combo27, cab_state):
    bp, be, pe = combo27
    return f"bp-{bp}_bowl-{be}_plate-{pe}_cab-{cab_state}"


class LiberoEnvScene1_4Factor:
    def __init__(self, libero_env):
        self.base_env = libero_env
        self.target_combo27 = None  # cabinet state is fixed by which bddl/env this is, not per-attempt
        self.rng = np.random.default_rng()

    def _obj_qpos_addr(self, name):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        return qs, vs, ve

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        qs, vs, ve = self._obj_qpos_addr(name)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _xy(self, name):
        sim = self.base_env.env.sim
        return sim.data.get_body_xpos(self.base_env.env.objects_dict[name].root_body)[:2].copy()

    def _sample_bowl_xy(self, edge_lo, edge_hi):
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = self.rng.uniform(REGION_XY_LO, REGION_XY_HI)
            if edge_lo <= dist_to_edge(cand) <= edge_hi:
                return cand
        return None

    def _sample_plate_xy(self, bowl_xy, bp_lo, bp_hi, edge_lo, edge_hi):
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(bp_lo, bp_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = bowl_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if not (REGION_XY_LO[0] <= cand[0] <= REGION_XY_HI[0] and REGION_XY_LO[1] <= cand[1] <= REGION_XY_HI[1]):
                continue
            if edge_lo <= dist_to_edge(cand) <= edge_hi:
                return cand
        return None

    def reset(self):
        ret = self.base_env.reset()
        bp_bin, bowl_bin, plate_bin = self.target_combo27
        bp_lo, bp_hi = BP_BINS[bp_bin]
        bowl_lo, bowl_hi = EDGE_BINS[bowl_bin]
        plate_lo, plate_hi = EDGE_BINS[plate_bin]

        placed = False
        for _ in range(MAX_OUTER_RETRIES):
            bowl_xy = self._sample_bowl_xy(bowl_lo, bowl_hi)
            if bowl_xy is None:
                continue
            plate_xy = self._sample_plate_xy(bowl_xy, bp_lo, bp_hi, plate_lo, plate_hi)
            if plate_xy is None:
                continue
            self._set_xy(BOWL_NAME, bowl_xy)
            self._set_xy(PLATE_NAME, plate_xy)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"could not jointly satisfy combo {self.target_combo27} in {MAX_OUTER_RETRIES} outer tries")

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

    def render(self, *a, **k):
        raise NotImplementedError

    def serialize(self):
        return {"env_name": "LIBERO_KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_4factor", "type": "libero"}


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def write_chunked_copy(src_group, dst_group):
    for key in src_group:
        if key == "obs":
            dst_obs = dst_group.create_group("obs")
            for ok in src_group["obs"]:
                data = src_group["obs"][ok][()]
                if ok in ("agentview_rgb", "eye_in_hand_rgb") and data.ndim == 4:
                    dst_obs.create_dataset(ok, data=data, chunks=(1,) + data.shape[1:], compression="gzip", compression_opts=1)
                else:
                    dst_obs.create_dataset(ok, data=data)
        elif isinstance(src_group[key], h5py.Group):
            src_group.copy(key, dst_group, name=key)
        else:
            dst_group.create_dataset(key, data=src_group[key][()])
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v


def write_distractor_episode(grp, name, rec, attrs):
    g = grp.create_group(name)
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
    g.create_dataset("actions", data=np.stack(rec["actions"]))
    g.create_dataset("states", data=np.stack(rec["states"]))
    n_ = len(rec["actions"])
    dones = np.zeros(n_, np.uint8); dones[-1] = 1
    g.create_dataset("dones", data=dones)
    for k, v in attrs.items():
        g.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)


def try_distractor_variant(env, raw_env, traj, seed):
    """Unchanged mechanism from run_scene1_3factor.py -- the cabinet is a
    FIXTURE (env.env.fixtures_dict), never touched by distractor_lib.Scene
    (which only iterates objects_dict), so this pass is orthogonal to the
    cabinet-edge-distance axis: it randomizes whatever ELSE might be a
    non-task object (none currently exist in this bddl, so this stays a
    no-op for SCENE1, same as the 3-factor run -- kept for forward
    compatibility / consistency with the merged pipeline)."""
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()
    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, {BOWL_NAME, PLATE_NAME})
    scene.measure_radii()
    if not scene.distractors:
        return None, "no_distractors"

    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
                               p_remove=0.3, table_z=TABLE_Z, vis=vis, xy_lo=(-0.5, -0.65), xy_hi=(0.5, 0.65))
    if result is None:
        return None, "no_valid_layout"
    st, removed, moved = result

    pre = {n: scene.sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy() for n in (BOWL_NAME, PLATE_NAME)}
    env.reset_to({"states": st})
    for _ in range(10):
        env.base_env.step(_DUMMY_ACTION)
    sim = raw_env.env.sim
    for n in (BOWL_NAME, PLATE_NAME):
        post = sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy()
        if np.linalg.norm(post - pre[n]) > 0.01:
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


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/mgbridge/out"
    label = sys.argv[2] if len(sys.argv) > 2 else "scene1_4factor"

    from libero.libero.envs import OffScreenRenderEnv
    import make_cabinet_bddl

    cabinet_bddl_paths = make_cabinet_bddl.write_all()

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    rng_combo = np.random.default_rng(42)
    per_combo27 = {c: [0, 0] for c in COMBOS27}  # [attempts, success] -- PRIMARY balance key, GLOBAL across all 4 passes
    succ_records = []  # (combo27, cabinet_state)
    distractor_records = []
    total_attempts = 0

    def pick_least_used(options, key_fn, rng):
        # ties broken RANDOMLY, not by list order -- min() alone always picks
        # the first element on a tie, which would bias early attempts toward
        # whichever combo happens to sort first.
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    pass_target = TARGET_TOTAL // len(CABINET_LEVELS)
    pass_max_attempts = MAX_ATTEMPTS // len(CABINET_LEVELS)

    for cab_state in CABINET_LEVELS:
        print(f"\n--- cabinet state: {cab_state} (bddl {cabinet_bddl_paths[cab_state]}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=cabinet_bddl_paths[cab_state], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnvScene1_4Factor(raw_env)
        env_interface = MG_LiberoBowlOnPlate(env=env.base_env)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)

        pass_successes = 0
        pass_attempts = 0
        while pass_successes < pass_target and pass_attempts < pass_max_attempts:
            combo27 = pick_least_used(COMBOS27, lambda c: (per_combo27[c][1], per_combo27[c][0]), rng_combo)
            env.target_combo27 = combo27
            per_combo27[combo27][0] += 1
            try:
                traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                                render=False, video_writer=None)
            except Exception as e:  # noqa: BLE001
                pass_attempts += 1
                total_attempts += 1
                if pass_attempts % 20 == 0:
                    print(f"[cab={cab_state} {pass_attempts}] exception for combo {combo27}: {type(e).__name__}: {e}", flush=True)
                continue
            pass_attempts += 1
            total_attempts += 1
            success = bool(traj["success"])
            if pass_attempts % 10 == 0 or success:
                print(f"[cab={cab_state} attempt {pass_attempts}] combo={label_for(combo27, cab_state)} success={success} "
                      f"(pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
            if not success:
                continue

            per_combo27[combo27][1] += 1
            pass_successes += 1
            succ_records.append((combo27, cab_state))
            MG_FileUtils.write_demo_to_hdf5(folder=tmp_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
                                             observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
                                             src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"])

            rec, info = try_distractor_variant(env, raw_env, traj, seed=4_000_000 + total_attempts)
            if rec is not None:
                distractor_records.append((len(succ_records) - 1, rec, info))

        raw_env.close()
        print(f"--- cabinet state {cab_state} done: {pass_successes}/{pass_attempts} ---", flush=True)

    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes ===")
    for c in COMBOS27:
        a, s = per_combo27[c]
        cab_breakdown = ", ".join(f"{cs}:{sum(1 for r in succ_records if r == (c, cs))}" for cs in CABINET_LEVELS)
        print(f"  bp-{c[0]}_bowl-{c[1]}_plate-{c[2]}: {s}/{a}  (cabinet: {cab_breakdown})")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_dir, "*.hdf5")))
    assert len(succ_files) == len(succ_records)
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": "put the black bowl on the plate"})
        data_grp.attrs["mimicgen_bddl_path"] = BDDL
        data_grp.attrs["mimicgen_num_attempts"] = total_attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_files)
        data_grp.attrs["stratification"] = json.dumps({
            "bowl_plate_distance_bins_m": BP_BINS, "desk_edge_distance_bins_m": EDGE_BINS,
            "cabinet_edge_distance_bins_m": CABINET_EDGE_BINS,
        })
        for i, (sf, (combo27, cab_state)) in enumerate(zip(succ_files, succ_records)):
            group_name = f"demo_{i:03d}_{label_for(combo27, cab_state)}"
            with h5py.File(sf, "r") as in_f:
                g = data_grp.create_group(group_name)
                write_chunked_copy(in_f["data/demo_0"], g)
                g.attrs["bowl_plate_bin"], g.attrs["bowl_edge_bin"], g.attrs["plate_edge_bin"] = combo27
                g.attrs["cabinet_state"] = cab_state

        dgrp = out_f.create_group("data_distractor")
        for i, rec, info in distractor_records:
            combo27, cab_state = succ_records[i]
            dname = f"distractor_{i:03d}_{label_for(combo27, cab_state)}"
            write_distractor_episode(dgrp, dname, rec, info)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"wrote {out_path}: {len(succ_files)} standard episodes, {len(distractor_records)} distractor episodes")


if __name__ == "__main__":
    main()
