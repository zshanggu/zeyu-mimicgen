"""Merge of two previously-separate pieces of work:
  1. mimicgen trajectory augmentation (run_generation.py): retarget a grasp/
     place trajectory onto a newly-sampled bowl/plate pose.
  2. libero_rerender_script/rerender_libero90.py's distractor randomization:
     randomize/remove task-UNrelated objects (here: ketchup_1), rejecting
     placements that overlap task objects/fixtures/each other or come near
     the end-effector path.

These two don't compose for free: (2) was built to replay a FIXED, already-
known trajectory, so it can check distance-to-path before ever touching the
sim. Here the trajectory only exists after mimicgen generates it. So per
attempt this script:
  (a) generates the trajectory as usual (mimicgen, fresh bowl/plate pose),
  (b) if it succeeds, reads back the GENERATED eef path from its own
      recorded observations,
  (c) resets to the trajectory's initial state, samples a distractor layout
      (ketchup_1) that avoids the task objects/fixtures AND that path,
  (d) re-settles and replays the IDENTICAL actions open-loop with the
      distractor layout in place, and keeps the episode only if the task
      still succeeds (same "guarantee" the rerender script uses: moving only
      task-unrelated objects must not break the original trajectory).

Writes two separate hdf5 groups, same convention as rerender_libero90.py:
    data/demo_i                        standard episodes (as run_generation.py)
    data_distractor/distractor_i_0     same episode, ketchup randomized/removed

Usage:
    python3 run_generation_with_distractors.py <bddl_path> <num_attempts> <out_hdf5_dir> <label>
"""
import sys
import os
import glob
import json
import shutil
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, LiberoEnvForDataGen, BOWL_NAME, PLATE_NAME
import distractor_lib as DL

PREPARED_HDF5 = "/mgbridge/prepared_source_scene5.hdf5"
TASK_OBJECT_NAMES = {BOWL_NAME, PLATE_NAME}
TABLE_Z = 0.90
P_REMOVE = 0.3
MAX_DRIFT = 0.01
SETTLE_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(
        object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
        selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
        action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False,
    )
    task_spec.add_subtask(
        object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
        action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False,
    )
    return task_spec


def write_distractor_episode(grp, name, rec, attrs, compress=True):
    kw = dict(compression="gzip", compression_opts=1) if compress else {}
    n = len(rec["actions"])
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
        o.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:], **kw)
    g.create_dataset("actions", data=np.stack(rec["actions"]))
    g.create_dataset("states", data=np.stack(rec["states"]))
    n_ = len(rec["actions"])
    dones = np.zeros(n_, np.uint8); dones[-1] = 1
    g.create_dataset("dones", data=dones)
    for k, v in attrs.items():
        g.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)


def try_distractor_variant(env, raw_env, traj, seed):
    """Returns a rec dict (for write_distractor_episode) + info, or (None, reason).

    IMPORTANT: a plain `regenerate_obs_from_state` (raw sim state restore) is
    NOT enough to reproduce the original rollout -- verified empirically
    (diag_replay_determinism.py): even replaying the SAME actions from the
    SAME initial_state with zero perturbation fails unless a full env.reset()
    runs first, clearing whatever controller/interpolator state the
    just-finished generate() rollout left behind. Same gotcha the rerender
    script's own reset_to() docstring calls out for its hard-reset-then-
    overlay-state pattern.
    """
    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()
    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, TASK_OBJECT_NAMES)
    scene.measure_radii()
    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(
        scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
        p_remove=P_REMOVE, table_z=TABLE_Z, vis=vis,
        xy_lo=(-0.5, -0.6), xy_hi=(0.5, 0.6),
    )
    if result is None:
        return None, "no_distractors_or_no_valid_layout"

    st, removed, moved = result
    pre_bowl = raw_env.env.sim.data.get_body_xpos(raw_env.env.objects_dict[BOWL_NAME].root_body)[:2].copy()
    pre_plate = raw_env.env.sim.data.get_body_xpos(raw_env.env.objects_dict[PLATE_NAME].root_body)[:2].copy()

    env.reset_to({"states": st})
    obs = None
    for _ in range(SETTLE_STEPS):
        obs = env.base_env.step(DUMMY_ACTION)[0]
    sim = raw_env.env.sim
    post_bowl = sim.data.get_body_xpos(raw_env.env.objects_dict[BOWL_NAME].root_body)[:2].copy()
    post_plate = sim.data.get_body_xpos(raw_env.env.objects_dict[PLATE_NAME].root_body)[:2].copy()
    if np.linalg.norm(post_bowl - pre_bowl) > MAX_DRIFT or np.linalg.norm(post_plate - pre_plate) > MAX_DRIFT:
        return None, f"task_object_drift bowl={np.linalg.norm(post_bowl - pre_bowl):.4f} plate={np.linalg.norm(post_plate - pre_plate):.4f}"

    actions, states, ee_states, gripper_states, joint_states, agent, wrist = ([] for _ in range(7))
    import robosuite.utils.transform_utils as T
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

    success = env.is_success()["task"]
    if not success:
        return None, "replay_with_distractors_failed"

    rec = dict(actions=actions, states=states, ee_states=ee_states, gripper_states=gripper_states,
               joint_states=joint_states, agent=agent, wrist=wrist)
    info = dict(removed=removed, moved=moved, seed=int(seed))
    return rec, info


def main():
    bddl_path = sys.argv[1]
    num_attempts_target = int(sys.argv[2])
    out_dir = sys.argv[3]
    label = sys.argv[4]

    from libero.libero.envs import OffScreenRenderEnv

    raw_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)

    env = LiberoEnvForDataGen(raw_env)
    env_interface = MG_LiberoBowlOnPlate(env=env.base_env)

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)
    print(data_generator)

    tmp_succ_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_succ_dir, ignore_errors=True)
    os.makedirs(tmp_succ_dir, exist_ok=True)

    num_success = num_attempts = num_distractor_success = num_distractor_tried = 0
    distractor_records = []  # (name, rec, info)

    while num_attempts < num_attempts_target:
        try:
            traj = data_generator.generate(
                env=env, env_interface=env_interface, select_src_per_subtask=False,
                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                render=False, video_writer=None,
            )
        except Exception as e:  # noqa: BLE001
            num_attempts += 1
            print(f"[attempt {num_attempts}] EXCEPTION during generation: {type(e).__name__}: {e}")
            continue

        num_attempts += 1
        success = bool(traj["success"])
        print(f"[attempt {num_attempts}] success={success} steps={0 if traj['actions'] is None else len(traj['actions'])}")

        if not success:
            continue
        num_success += 1
        MG_FileUtils.write_demo_to_hdf5(
            folder=tmp_succ_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
            observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
            src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"],
        )

        num_distractor_tried += 1
        rec, info = try_distractor_variant(env, raw_env, traj, seed=1_000_000 + num_attempts)
        if rec is None:
            print(f"    distractor variant skipped: {info}")
        else:
            num_distractor_success += 1
            distractor_records.append((f"distractor_{num_success - 1}_0", rec, info))
            print(f"    distractor variant OK: removed={info['removed']} moved={list(info['moved'].keys())}")

    print(f"\n=== {label}: standard {num_success}/{num_attempts}  distractor {num_distractor_success}/{num_distractor_tried} ===")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_succ_dir, "*.hdf5")))
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = '{"language_instruction": "put the black bowl on the plate"}'
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = num_attempts
        data_grp.attrs["mimicgen_num_success"] = num_success
        for i, sf in enumerate(succ_files):
            with h5py.File(sf, "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=f"demo_{i}")

        dgrp = out_f.create_group("data_distractor")
        dgrp.attrs["num_tried"] = num_distractor_tried
        dgrp.attrs["num_success"] = num_distractor_success
        for name, rec, info in distractor_records:
            write_distractor_episode(dgrp, name, rec, info)

    print(f"wrote {out_path}: data/ {len(succ_files)} episodes, data_distractor/ {len(distractor_records)} episodes")


if __name__ == "__main__":
    main()
