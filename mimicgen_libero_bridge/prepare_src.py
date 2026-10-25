"""Equivalent of mimicgen/scripts/prepare_src_dataset.py, but driving LIBERO's
OffScreenRenderEnv directly instead of going through robomimic's EnvUtils
(which needs robosuite.make(), incompatible with LIBERO's bddl-driven envs).

Copies the source hdf5, replays every demo frame-by-frame via
env.regenerate_obs_from_state(states[t]), and writes a "datagen_info" group
per demo (eef_pose, object_poses, subtask_term_signals, target_pose,
gripper_action) -- exactly what mimicgen.utils.file_utils.parse_source_dataset
expects to find.
"""
import sys
import shutil
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import robomimic.utils.tensor_utils as TensorUtils

from env_interface import MG_LiberoBowlOnPlate

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
SRC_HDF5 = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5"
PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"

ENV_INTERFACE_NAME = "MG_LiberoBowlOnPlate"
ENV_INTERFACE_TYPE = "libero"


def main():
    from libero.libero.envs import OffScreenRenderEnv

    shutil.copy(SRC_HDF5, PREPARED_HDF5)

    env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
    env.seed(0)
    env.reset()

    env_interface = MG_LiberoBowlOnPlate(env=env)

    f = h5py.File(PREPARED_HDF5, "a")
    demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    print(f"preparing {len(demo_keys)} demos")

    total_grasp_frames = 0
    for ep in tqdm(demo_keys):
        ep_grp = f[f"data/{ep}"]
        states = ep_grp["states"][()]
        actions = ep_grp["actions"][()]

        all_infos = []
        for t in range(len(states)):
            env.regenerate_obs_from_state(states[t])
            info = env_interface.get_datagen_info(action=actions[t]).to_dict()
            all_infos.append(info)

        total_grasp_frames += sum(int(i["subtask_term_signals"]["grasp"]) for i in all_infos)

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
        ep_grp["datagen_info"].attrs["env_interface_name"] = ENV_INTERFACE_NAME
        ep_grp["datagen_info"].attrs["env_interface_type"] = ENV_INTERFACE_TYPE

    f.close()
    print(f"done. total grasp=1 frames across all demos: {total_grasp_frames} (should be > 0 per demo, else grasp signal never fires)")
    print(f"wrote {PREPARED_HDF5}")


if __name__ == "__main__":
    main()
