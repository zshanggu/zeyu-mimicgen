"""KITCHEN_SCENE1_put_the_black_bowl_on_the_plate, stratified along THREE
independent axes at once (near/medium/far each): bowl-plate distance,
bowl-to-desk-edge distance, plate-to-desk-edge distance -- 27 combinations,
~100 episodes spread evenly across them. Each successful episode's demo
group is named to encode its combo directly (e.g.
"demo_003_bp-far_bowl-near_plate-med"), so it's visible straight in the data
viewer's episode list with NO viewer code changes needed (confirmed by
reading hdf5-lerobot-visualizer/src/libero_viewer/data_model.py:
read_task_metadata() lists demo groups by their literal HDF5 key).

Desk-edge distance: measured directly from the table's real collision geom
(table_collision, center (0,0,0.875), half-size [0.5,0.6,0.025) ->
x in [-0.5,0.5], y in [-0.6,0.6]), NOT the inflated ~0.938 figure from an
earlier broad keyword search that also picked up the table legs. Within the
already-established safe bowl/plate region (x in [-0.45,0.10], y in
[-0.12,0.35], clipped 5cm inside the table edge by construction -- see
env_interface.py's REGION_XY_LO/HI), the achievable distance-to-nearest-edge
range is ~0.05m (region boundary, = the table-edge safety margin) to ~0.40m
(the region's innermost reachable corner). Bins split that range into
thirds.

Placement per attempt (joint constraint satisfaction, not independent
sampling): sample bowl xy until its edge-distance lands in the target bowl
bin; then sample plate xy at the target bowl-plate distance/direction from
the bowl (reusing the existing direct-placement helper) until its edge-
distance ALSO lands in the target plate bin; if that search is exhausted,
resample the bowl position too (bounded outer retries).
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
PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"  # already built earlier this session for SCENE1

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.90

REGION_XY_LO = np.array([-0.45, -0.12])
REGION_XY_HI = np.array([0.10, 0.35])
TABLE_XY_LO = np.array([-0.5, -0.6])
TABLE_XY_HI = np.array([0.5, 0.6])

BP_BINS = {"near": (0.16, 0.26), "medium": (0.26, 0.38), "far": (0.38, 0.55)}
EDGE_BINS = {"near": (0.05, 0.17), "medium": (0.17, 0.29), "far": (0.29, 0.42)}
LEVELS = ["near", "medium", "far"]
COMBOS = [(bp, be, pe) for bp in LEVELS for be in LEVELS for pe in LEVELS]  # 27

TARGET_TOTAL = 100
MAX_ATTEMPTS = 400
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100


def dist_to_edge(xy):
    return min(xy[0] - TABLE_XY_LO[0], TABLE_XY_HI[0] - xy[0], xy[1] - TABLE_XY_LO[1], TABLE_XY_HI[1] - xy[1])


def label_for(combo):
    bp, be, pe = combo
    return f"bp-{bp}_bowl-{be}_plate-{pe}"


class LiberoEnvScene1_3Factor:
    def __init__(self, libero_env):
        self.base_env = libero_env
        self.target_combo = None  # (bp_bin, bowl_edge_bin, plate_edge_bin)
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
        bp_bin, bowl_bin, plate_bin = self.target_combo
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
            raise RuntimeError(f"could not jointly satisfy combo {self.target_combo} in {MAX_OUTER_RETRIES} outer tries")

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
        return {"env_name": "LIBERO_KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_3factor", "type": "libero"}


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
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()  # full reset REQUIRED before overlaying state -- see generic_pipeline.py docstring
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
    label = sys.argv[2] if len(sys.argv) > 2 else "scene1_3factor"

    from libero.libero.envs import OffScreenRenderEnv

    raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    env = LiberoEnvScene1_3Factor(raw_env)
    env_interface = MG_LiberoBowlOnPlate(env=env.base_env)

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)
    print(data_generator)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    per_combo = {c: [0, 0] for c in COMBOS}  # [attempts, success]
    succ_labels = []
    distractor_records = []
    attempts = 0
    combo_cycle_idx = 0

    while len(succ_labels) < TARGET_TOTAL and attempts < MAX_ATTEMPTS:
        # weighted round-robin: always pick the combo with the fewest successes so far (ties broken by fewest attempts)
        combo = min(COMBOS, key=lambda c: (per_combo[c][1], per_combo[c][0]))
        env.target_combo = combo
        per_combo[combo][0] += 1
        try:
            traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                            transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                            render=False, video_writer=None)
        except Exception as e:  # noqa: BLE001
            attempts += 1
            if attempts % 20 == 0:
                print(f"[{attempts}] exception for combo {combo}: {type(e).__name__}: {e}", flush=True)
            continue
        attempts += 1
        success = bool(traj["success"])
        if attempts % 10 == 0 or success:
            print(f"[attempt {attempts}] combo={label_for(combo)} success={success} "
                  f"(so far {len(succ_labels)}/{TARGET_TOTAL})", flush=True)
        if not success:
            continue

        per_combo[combo][1] += 1
        succ_labels.append(combo)
        MG_FileUtils.write_demo_to_hdf5(folder=tmp_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
                                         observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
                                         src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"])

        rec, info = try_distractor_variant(env, raw_env, traj, seed=3_000_000 + attempts)
        if rec is not None:
            distractor_records.append((len(succ_labels) - 1, rec, info))

    print(f"\n=== {label}: {len(succ_labels)}/{attempts} successes ===")
    for c in COMBOS:
        a, s = per_combo[c]
        print(f"  {label_for(c)}: {s}/{a}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_dir, "*.hdf5")))
    assert len(succ_files) == len(succ_labels)
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": "put the black bowl on the plate"})
        data_grp.attrs["mimicgen_bddl_path"] = BDDL
        data_grp.attrs["mimicgen_num_attempts"] = attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_files)
        data_grp.attrs["stratification"] = json.dumps({
            "bowl_plate_distance_bins_m": BP_BINS, "desk_edge_distance_bins_m": EDGE_BINS,
        })
        name_for_idx = {}
        for i, (sf, combo) in enumerate(zip(succ_files, succ_labels)):
            group_name = f"demo_{i:03d}_{label_for(combo)}"
            name_for_idx[i] = group_name
            with h5py.File(sf, "r") as in_f:
                g = data_grp.create_group(group_name)
                write_chunked_copy(in_f["data/demo_0"], g)
                g.attrs["bowl_plate_bin"], g.attrs["bowl_edge_bin"], g.attrs["plate_edge_bin"] = combo

        dgrp = out_f.create_group("data_distractor")
        for i, rec, info in distractor_records:
            dname = f"distractor_{i:03d}_{label_for(succ_labels[i])}"
            write_distractor_episode(dgrp, dname, rec, info)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    raw_env.close()
    print(f"wrote {out_path}: {len(succ_files)} standard episodes, {len(distractor_records)} distractor episodes")


if __name__ == "__main__":
    main()
