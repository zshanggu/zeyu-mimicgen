"""Verify the user's proposed workspace extremes (X up to ~0.13, Y up to
~+-0.55) two ways: (1) does mimicgen's actual retargeted grasp+place
trajectory succeed with bowl/plate forced to these extreme corners (real
reach test, not a simplified P-controller probe), (2) is the position still
inside the agentview camera's visible frame (reuse the precise raycast
already done interactively)."""
import sys
import json
import numpy as np

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, BOWL_NAME, PLATE_NAME

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"
NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# corners to test: (bowl_xy, plate_xy) -- plate placed 0.2m away from bowl,
# toward table center, so the GRASP point (bowl) is the one actually at the
# extreme being tested each time.
CORNERS = {
    "y040 (-0.15, 0.40)": (np.array([-0.15, 0.40]), np.array([-0.15, 0.15])),
    "y045 (-0.15, 0.45)": (np.array([-0.15, 0.45]), np.array([-0.15, 0.20])),
    "y050 (-0.15, 0.50)": (np.array([-0.15, 0.50]), np.array([-0.15, 0.25])),
    "yneg040 (-0.15, -0.40)": (np.array([-0.15, -0.40]), np.array([-0.15, -0.15])),
}


def camera_y_bound(sim, x):
    cid = sim.model.camera_name2id("agentview")
    cam_pos = sim.data.cam_xpos[cid].copy()
    cam_mat = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    fovy = sim.model.cam_fovy[cid]
    table_z = 0.90
    MARGIN_PX, size = 20, 256
    m = 1 - 2 * MARGIN_PX / size

    def hit(u, v):
        t = np.tan(np.radians(fovy) / 2)
        d = np.array([u * t, v * t, -1.0]); d /= np.linalg.norm(d)
        d = cam_mat @ d
        tt = (table_z - cam_pos[2]) / d[2]
        return cam_pos + tt * d

    lo, hi = -m, m
    for _ in range(40):
        mid = (lo + hi) / 2
        p = hit(0, mid)
        if p[0] > x:
            lo = mid
        else:
            hi = mid
    v = (lo + hi) / 2
    return hit(m, v)[1]


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


class FixedPosEnv:
    def __init__(self, libero_env):
        self.base_env = libero_env
        self.target_bowl_xy = None
        self.target_plate_xy = None

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def reset(self):
        ret = self.base_env.reset()
        self._set_xy(BOWL_NAME, self.target_bowl_xy)
        self._set_xy(PLATE_NAME, self.target_plate_xy)
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
        return {"env_name": "check_farthest", "type": "libero"}


def main():
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    import os

    outdir = "/mgbridge/out/farthest_check"
    os.makedirs(outdir, exist_ok=True)

    raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=400)
    raw_env.seed(1234)
    env = FixedPosEnv(raw_env)
    env_interface = MG_LiberoBowlOnPlate(env=env.base_env)
    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)
    data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)

    for name, (bowl_xy, plate_xy) in CORNERS.items():
        y_bound = camera_y_bound(raw_env.env.sim, bowl_xy[0])
        in_camera = abs(bowl_xy[1]) <= y_bound
        print(f"\n=== {name}: bowl target={bowl_xy} camera_y_bound(at this x)={y_bound:.3f} in_camera={in_camera} ===")

        successes, tries = 0, 0
        last_img = None
        for attempt in range(5):
            env.target_bowl_xy = bowl_xy
            env.target_plate_xy = plate_xy
            try:
                traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                                render=False, video_writer=None)
            except Exception as e:
                tries += 1
                print(f"  attempt {attempt}: EXCEPTION {type(e).__name__}: {e}")
                continue
            tries += 1
            success = bool(traj["success"])
            if success:
                successes += 1
            print(f"  attempt {attempt}: success={success}")
            if traj["observations"]:
                last_img = traj["observations"][0]["agentview_rgb"]
        print(f"  --> {successes}/{tries} successes")
        if last_img is not None:
            safe_name = name.split(" ")[0].replace("(", "").replace(")", "").replace(",", "")
            Image.fromarray(last_img).save(f"{outdir}/{safe_name}.png")

    raw_env.close()


if __name__ == "__main__":
    main()
