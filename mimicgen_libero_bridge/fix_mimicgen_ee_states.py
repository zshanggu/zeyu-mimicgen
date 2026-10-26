"""One-time in-place fix: the mimicgen pipeline's obs/ee_states was written
as ee_pos(3)+ee_ori_quat(4)=7 dims, but the established dataset convention
(matching libero90_openvla_256_ori and convert_libero_subskills_to_lerobot.py's
expected 8-dim state = 6-dim ee_states + 2-dim gripper_states) is
ee_pos(3)+axis_angle(3)=6 dims. ee_ori (the raw quaternion) is still stored
in every file, so this recomputes ee_states from it in place -- no
regeneration of any trajectory needed, nothing else in the file changes."""
import math
import h5py
import numpy as np


def quat2axisangle(quat):
    """xyzw quaternion -> axis-angle (3,). Verbatim copy of
    robosuite.utils.transform_utils.quat2axisangle (confirmed via
    inspect.getsource against the live package -- robosuite quaternions are
    (x,y,z,w), scalar LAST, not the MuJoCo-native wxyz order), inlined to
    avoid a robosuite dependency for this one-off fix. obs/ee_ori in these
    files is stored straight from robosuite's own robot0_eef_quat, so this
    is the correct convention for it."""
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


FILES = [
    "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_mimicgen/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5",
    "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_mimicgen/KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate_demo.hdf5",
    "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_mimicgen/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket_demo.hdf5",
]

for path in FILES:
    print(f"=== {path} ===", flush=True)
    with h5py.File(path, "a") as f:
        data_grp = f["data"]
        for i, ep in enumerate(data_grp.keys()):
            obs = data_grp[f"{ep}/obs"]
            ee_pos = obs["ee_pos"][()]
            ee_ori_quat = obs["ee_ori"][()]
            cur_shape = obs["ee_states"].shape
            if cur_shape[-1] == 6:
                continue  # already fixed (idempotent re-run safety)
            assert cur_shape[-1] == 7, f"{ep}: unexpected ee_states shape {cur_shape}"
            axis_angle = np.stack([quat2axisangle(q) for q in ee_ori_quat], axis=0)
            new_ee_states = np.hstack([ee_pos, axis_angle]).astype(np.float32)
            del obs["ee_states"]
            obs.create_dataset("ee_states", data=new_ee_states)
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(data_grp)} episodes fixed", flush=True)
        print(f"  done: {len(data_grp)} episodes", flush=True)
