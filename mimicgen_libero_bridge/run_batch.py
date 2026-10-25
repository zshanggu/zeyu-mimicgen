"""Batch driver: runs the generic mimicgen+distractor pipeline across a list
of LIBERO-90 "(On moving target)" tasks, ~TARGET_SUCCESSES episodes each.

Resumable (skips tasks whose output file already has >= TARGET_SUCCESSES
standard episodes), logs progress per task, monitors free disk space before
EVERY task and stops cleanly (not mid-file) if it drops below MIN_FREE_GB.

Usage: python3 run_batch.py <out_dir> [task_name_substring_to_start_at]
"""
import sys
import os
import gc
import glob
import json
import shutil
import time
import traceback
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

import generic_pipeline as GP

BDDL_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90"
SRC_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256"
SCRATCH_SRC = "/mgbridge/_scratch_prepared_source.hdf5"
TARGET_SUCCESSES = int(os.environ.get("MG_TARGET_SUCCESSES", 100))
MAX_ATTEMPTS = int(os.environ.get("MG_MAX_ATTEMPTS", 320))       # hard cap per task so one pathological task can't run forever
MIN_FREE_GB = 25
IMG_KEYS = ("agentview_rgb", "eye_in_hand_rgb")

TASKS = [
    # (task_name,)  -- goal structure (free-free vs region-target) is detected live, not hardcoded here
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
]


def free_gb(path="/data"):
    s = shutil.disk_usage(path)
    return s.free / 1e9


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="moving", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="target", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def prepare_source(bddl_path, src_hdf5, moving, target, target_kind):
    from libero.libero.envs import OffScreenRenderEnv
    import robomimic.utils.tensor_utils as TensorUtils

    shutil.copy(src_hdf5, SCRATCH_SRC)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    env.seed(0)
    env.reset()
    env_interface = GP.MG_LiberoGeneric(env=env, moving=moving, target=target, target_kind=target_kind)

    f = h5py.File(SCRATCH_SRC, "a")
    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    for ep in demo_keys:
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
        ep_grp["datagen_info"].attrs["env_interface_name"] = "MG_LiberoGeneric"
        ep_grp["datagen_info"].attrs["env_interface_type"] = "libero"
    f.close()
    env.close()


def write_chunked_copy(src_group, dst_group):
    """Copy an mimicgen-written demo group into dst_group, re-chunking image
    datasets as (1,H,W,C) gzip for real compression (the mimicgen writer's
    default h5py auto-chunking splits channels into separate chunks, which
    compresses far worse -- same fix already used in the distractor writer)."""
    for key in src_group:
        if key == "obs":
            dst_obs = dst_group.create_group("obs")
            for ok in src_group["obs"]:
                data = src_group["obs"][ok][()]
                if ok in IMG_KEYS and data.ndim == 4:
                    dst_obs.create_dataset(ok, data=data, chunks=(1,) + data.shape[1:],
                                            compression="gzip", compression_opts=1)
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


def run_one_task(task_name, out_dir, log):
    out_path = os.path.join(out_dir, f"{task_name}_demo.hdf5")
    if os.path.exists(out_path):
        with h5py.File(out_path, "r") as f:
            n = len(f["data"].keys()) if "data" in f else 0
        if n >= TARGET_SUCCESSES * 0.8:
            log(f"SKIP {task_name}: already has {n} episodes")
            return

    bddl_path = os.path.join(BDDL_DIR, f"{task_name}.bddl")
    src_hdf5 = os.path.join(SRC_DIR, f"{task_name}_demo.hdf5")
    if not os.path.exists(src_hdf5):
        log(f"SKIP {task_name}: no source demos at {src_hdf5}")
        return

    from libero.libero.envs import OffScreenRenderEnv

    t0 = time.time()
    probe_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
    cfg = GP.build_task_config(probe_env)
    probe_env.close()
    moving, target, target_kind, task_object_names = cfg["moving"], cfg["target"], cfg["target_kind"], cfg["task_object_names"]
    log(f"START {task_name}: moving={moving} target={target} ({target_kind})")

    prepare_source(bddl_path, src_hdf5, moving, target, target_kind)

    raw_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    moving_lo, moving_hi = GP.widen_object_region(raw_env, moving)
    target_lo = target_hi = None
    if target_kind == "object":
        target_lo, target_hi = GP.widen_object_region(raw_env, target, other_native_boxes=[(moving_lo[0], moving_hi[0], moving_lo[1], moving_hi[1])])
    log(f"  region moving=[{moving_lo.round(3)},{moving_hi.round(3)}] target=[{target_lo},{target_hi}]")

    env = GP.LiberoEnvForDataGenGeneric(raw_env, moving, target, target_kind, moving_lo, moving_hi, target_lo, target_hi)
    env_interface = GP.MG_LiberoGeneric(env=env.base_env, moving=moving, target=target, target_kind=target_kind)
    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=SCRATCH_SRC, filter_key=None, start=None, n=None)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=SCRATCH_SRC, demo_keys=all_demos)

    tmp_dir = "/tmp/mg_batch_succ"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    per_bin = {b: [0, 0] for b in GP.BIN_CYCLE}  # [attempts, success]
    distractor_ok = distractor_tried = 0
    succ_bins = []
    attempts = 0
    use_bins = (target_kind == "object") or True  # also apply to site targets (moving obj near/med/far from fixed target)

    while len(succ_bins) < TARGET_SUCCESSES and attempts < MAX_ATTEMPTS:
        target_bin = GP.BIN_CYCLE[attempts % 3]
        env.target_bin = target_bin
        per_bin[target_bin][0] += 1
        try:
            traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                            transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                            render=False, video_writer=None)
        except Exception as e:  # noqa: BLE001
            attempts += 1
            if attempts % 20 == 0:
                log(f"  [{attempts}] exception {type(e).__name__}: {e}")
            continue
        attempts += 1
        if not traj["success"]:
            continue
        per_bin[target_bin][1] += 1
        succ_bins.append(target_bin)
        MG_FileUtils.write_demo_to_hdf5(folder=tmp_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
                                         observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
                                         src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"])
        distractor_tried += 1
        rec, info = GP.try_distractor_variant(env, raw_env, traj, task_object_names, seed=2_000_000 + attempts)
        if rec is not None:
            distractor_ok += 1

    log(f"  generation done: {len(succ_bins)}/{attempts} successes, "
        f"bins={{{', '.join(f'{b}:{s}/{a}' for b,(a,s) in per_bin.items())}}}, distractor {distractor_ok}/{distractor_tried}")

    os.makedirs(out_dir, exist_ok=True)
    succ_files = sorted(glob.glob(os.path.join(tmp_dir, "*.hdf5")))
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": task_name})
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_files)
        for i, sf in enumerate(succ_files):
            with h5py.File(sf, "r") as in_f:
                g = data_grp.create_group(f"demo_{i}")
                write_chunked_copy(in_f["data/demo_0"], g)
                g.attrs["distance_bin"] = succ_bins[i]

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.remove(SCRATCH_SRC)
    raw_env.close()
    del data_generator, env, raw_env
    gc.collect()
    log(f"DONE {task_name}: {len(succ_files)} episodes -> {out_path} ({time.time()-t0:.0f}s, free={free_gb():.1f}GB)")


def main():
    out_dir = sys.argv[1]
    start_at = sys.argv[2] if len(sys.argv) > 2 else None
    log_path = os.path.join(out_dir, "_batch_log.txt")
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    tasks = TASKS
    if start_at:
        idx = next((i for i, t in enumerate(tasks) if start_at in t), 0)
        tasks = tasks[idx:]

    for task_name in tasks:
        if free_gb() < MIN_FREE_GB:
            log(f"ABORT BATCH: free space {free_gb():.1f}GB < {MIN_FREE_GB}GB floor, stopping before {task_name}")
            break
        try:
            run_one_task(task_name, out_dir, log)
        except Exception as e:  # noqa: BLE001
            log(f"TASK FAILED {task_name}: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            if os.path.exists(SCRATCH_SRC):
                os.remove(SCRATCH_SRC)
            continue

    log("BATCH COMPLETE")


if __name__ == "__main__":
    main()
