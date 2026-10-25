"""Same as run_generation.py, but cycles env.target_bin through
near/medium/far each attempt (env_interface.BOWL_PLATE_DIST_BINS) so bowl-
plate separation is deliberately covered rather than left to the implicit,
corner-rare distribution of independent uniform sampling (measured: median
0.32m, nothing past 0.53m, out of a shared region whose diagonal allows 0.72m).

Usage:
    python3 run_generation_stratified.py <bddl_path> <num_attempts> <out_hdf5_dir> <label>
"""
import sys
import os
import glob
import shutil
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, LiberoEnvForDataGen, BOWL_PLATE_DIST_BINS

PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"
BIN_CYCLE = ["near", "medium", "far"]


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

    num_attempts = 0
    per_bin_attempts = {b: 0 for b in BIN_CYCLE}
    per_bin_success = {b: 0 for b in BIN_CYCLE}
    bowl_plate_dist_success = []
    succ_bins = []  # parallel to the eventual demo_i ordering

    while num_attempts < num_attempts_target:
        target_bin = BIN_CYCLE[num_attempts % len(BIN_CYCLE)]
        env.target_bin = target_bin
        per_bin_attempts[target_bin] += 1
        try:
            traj = data_generator.generate(
                env=env, env_interface=env_interface, select_src_per_subtask=False,
                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                render=False, video_writer=None,
            )
        except Exception as e:  # noqa: BLE001
            num_attempts += 1
            print(f"[attempt {num_attempts} bin={target_bin}] EXCEPTION: {type(e).__name__}: {e}")
            continue

        num_attempts += 1
        success = bool(traj["success"])
        bowl_xy = traj["datagen_infos"][0].object_poses["bowl"][:2, 3].copy()
        plate_xy = traj["datagen_infos"][0].object_poses["plate"][:2, 3].copy()
        dist = float(np.linalg.norm(bowl_xy - plate_xy))
        print(f"[attempt {num_attempts} bin={target_bin}] success={success} dist={dist:.3f} "
              f"bowl=({bowl_xy[0]:.3f},{bowl_xy[1]:.3f}) plate=({plate_xy[0]:.3f},{plate_xy[1]:.3f})")

        if success:
            per_bin_success[target_bin] += 1
            bowl_plate_dist_success.append(dist)
            succ_bins.append(target_bin)
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_succ_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
                observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
                src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"],
            )

    total_success = sum(per_bin_success.values())
    print(f"\n=== {label}: {total_success}/{num_attempts} successes ===")
    for b in BIN_CYCLE:
        lo, hi = BOWL_PLATE_DIST_BINS[b]
        print(f"  bin {b} [{lo:.2f},{hi:.2f}]m: {per_bin_success[b]}/{per_bin_attempts[b]} successes")
    if bowl_plate_dist_success:
        arr = np.array(bowl_plate_dist_success)
        print(f"achieved bowl-plate dist (successes only): min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_succ_dir, "*.hdf5")))
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = '{"language_instruction": "put the black bowl on the plate"}'
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = num_attempts
        data_grp.attrs["mimicgen_num_success"] = total_success
        for i, sf in enumerate(succ_files):
            with h5py.File(sf, "r") as in_f:
                grp_name = f"demo_{i}"
                in_f.copy("data/demo_0", data_grp, name=grp_name)
                data_grp[grp_name].attrs["distance_bin"] = succ_bins[i]
    print(f"wrote {out_path} with {len(succ_files)} episodes")


if __name__ == "__main__":
    main()
