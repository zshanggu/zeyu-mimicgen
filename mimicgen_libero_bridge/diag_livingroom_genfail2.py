"""Sanity test: force the moving/target placement to stay essentially at
butter's/basket's NATIVE spawn position (near-zero retargeting transform),
to check whether the generated trajectory's grasp-approach failure is a
transform-magnitude issue (big displacement breaks it) or something
structurally broken in this task's retargeting regardless of distance."""
import sys
sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.datagen.data_generator import DataGenerator

import run_livingroom_task_5factor as M

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
SCRATCH = "/mgbridge/_scratch_prepared_source_LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.hdf5"

task_spec = M.build_task_spec()
all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=SCRATCH, filter_key=None, start=None, n=None)
data_generator = DataGenerator(task_spec=task_spec, dataset_path=SCRATCH, demo_keys=all_demos)

raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
raw_env.seed(1234)
env = M.LiberoEnv5FactorGeneric(raw_env, "butter_1", "basket_1")
env_interface = M.MG_LiberoBowlPlateGeneric(env=env.base_env, moving="butter_1", target="basket_1")

# monkeypatch: skip the bin-based placement entirely, nudge only ~1cm from native
orig_reset = env.base_env.reset


def patched_reset():
    ret = orig_reset()
    env._set_xy("butter_1", np.array([0.045, 0.038]) + np.array([0.01, 0.01]))
    env._set_xy("basket_1", np.array([-0.0009, 0.261]) + np.array([0.01, -0.01]))
    for _ in range(M.NUM_SETTLE_STEPS):
        ret = env.base_env.step(M._DUMMY_ACTION)
    return ret


env.reset = patched_reset

traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                render=False, video_writer=None)
print("traj success:", traj["success"])

sim = raw_env.env.sim
butter_pos = sim.data.get_body_xpos(raw_env.env.objects_dict["butter_1"].root_body)
basket_pos = sim.data.get_body_xpos(raw_env.env.objects_dict["basket_1"].root_body)
print("final butter pos:", butter_pos, "final basket pos:", basket_pos)

env.reset_to(traj["initial_state"])
sim = raw_env.env.sim
butter_body = raw_env.env.objects_dict["butter_1"].root_body
for t, a in enumerate(traj["actions"]):
    env.step(a.tolist() if hasattr(a, "tolist") else list(a))
    sim = raw_env.env.sim
    if t % 10 == 0:
        bp = sim.data.get_body_xpos(butter_body)
        gp = sim.data.get_site_xpos("gripper0_grip_site")
        print(f"  t={t:3d} butter_z={bp[2]:.4f} grip_cmd={a[-1]:+.2f} dist={np.linalg.norm(bp-gp):.4f} gp={np.round(gp,3)} bp={np.round(bp,3)}")

raw_env.close()
