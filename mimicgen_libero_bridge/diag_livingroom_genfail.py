"""Generate ONE trajectory via the same DataGenerator/env_interface path as
run_livingroom_task_5factor.py, then inspect exactly why it fails (butter
final position vs basket, whether the generated replay's gripper ever
closes, closest approach distance) -- the small-scale test is failing 0/40,
which is systematic, not bad luck, so debug directly instead of burning the
rest of the attempt budget blind."""
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

env.target_primary = ("far", "bottom", "top")
env.target_secondary = ("left", "right")

traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                render=False, video_writer=None)
print("traj success:", traj["success"])
print("n actions:", len(traj["actions"]))

sim = raw_env.env.sim
butter_pos = sim.data.get_body_xpos(raw_env.env.objects_dict["butter_1"].root_body)
basket_pos = sim.data.get_body_xpos(raw_env.env.objects_dict["basket_1"].root_body)
print("final butter pos:", butter_pos)
print("final basket pos:", basket_pos)
print("final butter-basket xy dist:", np.linalg.norm(butter_pos[:2] - basket_pos[:2]))
print("is_success:", raw_env.env._check_success())

# did the gripper ever close during the generated traj, and did butter ever leave the table?
zs = []
grip_cmds = []
for a in traj["actions"]:
    grip_cmds.append(a[-1])
print("gripper cmd range in generated traj:", min(grip_cmds), max(grip_cmds))

# replay again from initial_state and track butter z + grip distance over the generated actions
env.reset_to(traj["initial_state"])
sim = raw_env.env.sim
butter_body = raw_env.env.objects_dict["butter_1"].root_body
for t, a in enumerate(traj["actions"]):
    env.step(a.tolist() if hasattr(a, "tolist") else list(a))
    sim = raw_env.env.sim
    if 40 <= t <= 90:
        bp = sim.data.get_body_xpos(butter_body)
        gp = sim.data.get_site_xpos("gripper0_grip_site")
        print(f"  t={t:3d} butter_z={bp[2]:.4f} grip_cmd={a[-1]:+.2f} dist={np.linalg.norm(bp-gp):.4f} gp={np.round(gp,3)} bp={np.round(bp,3)}")

raw_env.close()
