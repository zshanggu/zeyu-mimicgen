"""Isolate: does reset_to(initial_state) + open-loop replay of the SAME
recorded actions reproduce success, with ZERO distractor perturbation at all?
If this alone fails, the bug is in the replay mechanic, not in distractor
placement."""
import sys
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, LiberoEnvForDataGen

PREPARED_HDF5 = "/mgbridge/prepared_source_scene5.hdf5"
BDDL = "/mgbridge/KITCHEN_SCENE5_put_the_black_bowl_on_the_plate_widened.bddl"
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def main():
    from libero.libero.envs import OffScreenRenderEnv
    raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
    raw_env.seed(1234)
    env = LiberoEnvForDataGen(raw_env)
    env_interface = MG_LiberoBowlOnPlate(env=env.base_env)
    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)

    tries = 0
    while tries < 20:
        traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                        transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                        render=False, video_writer=None)
        tries += 1
        if not traj["success"]:
            print(f"try {tries}: original generation failed, skip")
            continue
        print(f"try {tries}: original generation SUCCEEDED, testing pure replay determinism...")

        initial_state = traj["initial_state"]["states"]
        raw_env.seed(999)
        raw_env.reset()  # full hard reset first, to clear controller/interpolator state left over
        # from generate()'s own rollout -- THEN overlay the exact recorded initial state.
        env.reset_to({"states": initial_state})
        for _ in range(10):
            env.base_env.step(DUMMY_ACTION)

        for action in traj["actions"]:
            a = action.tolist() if hasattr(action, "tolist") else list(action)
            env.base_env.step(a)

        success = env.is_success()["task"]
        print(f"  pure replay (no perturbation at all) success = {success}")
        break

    raw_env.close()


if __name__ == "__main__":
    main()
