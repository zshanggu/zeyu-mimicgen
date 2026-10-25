"""Equivalent of mimicgen/scripts/generate_dataset.py's core loop, driving
LIBERO directly (see env_interface.LiberoEnvForDataGen for why). Usage:

    python3 run_generation.py <bddl_path> <num_attempts> <out_hdf5_dir> <label>

Writes every SUCCESSFUL generated episode into a single new hdf5 at
<out_hdf5_dir>/<label>.hdf5, and prints attempt/success counts.
"""
import sys
import os
import glob
import shutil
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")  # gdown/huggingface_hub stand-ins -- not installed, not needed here
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, LiberoEnvForDataGen

PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(
        object_ref="bowl",
        subtask_term_signal="grasp",
        subtask_term_offset_range=(5, 10),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )
    task_spec.add_subtask(
        object_ref="plate",
        subtask_term_signal=None,
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs=dict(nn_k=3),
        action_noise=0.05,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
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

    num_success = 0
    num_attempts = 0
    bowl_positions_success = []

    diagnose = os.environ.get("MG_DIAGNOSE") == "1"

    while num_attempts < num_attempts_target:
        try:
            traj = data_generator.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=False,
                transform_first_robot_pose=False,
                interpolate_from_last_target_pose=True,
                render=False,
                video_writer=None,
            )
        except Exception as e:  # noqa: BLE001 -- quick test: log and keep going like generate_dataset.py's except-and-continue
            num_attempts += 1
            print(f"[attempt {num_attempts}] EXCEPTION during generation: {type(e).__name__}: {e}")
            continue

        num_attempts += 1
        success = bool(traj["success"])
        bowl_pos_this_attempt = env_interface.get_object_poses()["bowl"][:3, 3].copy()
        bowl_start = traj["datagen_infos"][0].object_poses["bowl"][:3, 3].copy()
        plate_pos0 = traj["datagen_infos"][0].object_poses["plate"][:3, 3].copy()
        n_steps = 0 if traj["actions"] is None else len(traj["actions"])
        print(f"[attempt {num_attempts}] success={success} bowl_start=({bowl_start[0]:.3f},{bowl_start[1]:.3f},{bowl_start[2]:.3f}) "
              f"bowl_final=({bowl_pos_this_attempt[0]:.3f},{bowl_pos_this_attempt[1]:.3f},{bowl_pos_this_attempt[2]:.3f}) "
              f"plate=({plate_pos0[0]:.3f},{plate_pos0[1]:.3f},{plate_pos0[2]:.3f}) steps={n_steps} "
              f"lifted={bowl_pos_this_attempt[2] - bowl_start[2]:.3f} moved_xy={np.linalg.norm(bowl_pos_this_attempt[:2]-bowl_start[:2]):.3f}")
        if diagnose and num_attempts <= 3:
            import env_interface as _ei
            print(f"    final eef={env_interface.get_robot_eef_pose()[:3,3].round(3)} "
                  f"final grasped_obj={_ei._grasped_object_name(env.base_env)}")

        if success:
            num_success += 1
            bowl_positions_success.append(bowl_pos_this_attempt)
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_succ_dir,
                env=env,
                initial_state=traj["initial_state"],
                states=traj["states"],
                observations=traj["observations"],
                datagen_info=traj["datagen_infos"],
                actions=traj["actions"],
                src_demo_inds=traj["src_demo_inds"],
                src_demo_labels=traj["src_demo_labels"],
            )

    print(f"\n=== {label}: {num_success}/{num_attempts} successes ===")
    if bowl_positions_success:
        arr = np.array(bowl_positions_success)
        print(f"successful bowl xy range: x[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] y[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]")

    # merge all individual per-episode hdf5s (written by write_demo_to_hdf5, one per success)
    # into a single final hdf5 with demo_0..demo_{N-1}, matching the source dataset layout.
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_succ_dir, "*.hdf5")))
    print(f"merging {len(succ_files)} successful episode files into {out_path}")
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = '{"language_instruction": "put the black bowl on the plate"}'
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["mimicgen_num_attempts"] = num_attempts
        data_grp.attrs["mimicgen_num_success"] = num_success
        for i, sf in enumerate(succ_files):
            with h5py.File(sf, "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=f"demo_{i}")
    print(f"wrote {out_path} with {len(succ_files)} episodes")


if __name__ == "__main__":
    main()
