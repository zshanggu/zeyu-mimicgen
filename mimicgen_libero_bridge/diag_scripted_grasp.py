"""Trace one scripted_pick_place attempt step by step to see exactly where
the real-physics grasp fails: does the gripper actually reach butter's true
position, does contact register, does it slip after closing."""
import sys
sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import numpy as np
from libero.libero.envs import OffScreenRenderEnv
import run_livingroom_task_5factor as BASE

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"

raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=1000)
raw_env.seed(1234)
env = BASE.LiberoEnv5FactorGeneric(raw_env, "butter_1", "basket_1")
env.target_primary = ("far", "bottom", "top")
env.target_secondary = ("left", "right")
env.reset()

sim = raw_env.env.sim
moving_body = raw_env.env.objects_dict["butter_1"].root_body
moving_xy = sim.data.get_body_xpos(moving_body)[:2].copy()
moving_z0 = float(sim.data.get_body_xpos(moving_body)[2])
print("butter xy:", moving_xy, "z0:", moving_z0)

HOVER_Z = 0.70
POS_GAIN = 4.0
MAX_ACTION_DELTA = 0.15


def move_to(target_xyz, gripper_cmd, max_steps=120, tol=0.012, verbose_every=20):
    sim = raw_env.env.sim
    for t in range(max_steps):
        eef = sim.data.get_site_xpos("gripper0_grip_site")
        err = np.asarray(target_xyz) - eef
        d = float(np.linalg.norm(err))
        if d < tol:
            print(f"  converged at step {t}, d={d:.4f}")
            return d
        delta = np.clip(err * POS_GAIN, -MAX_ACTION_DELTA, MAX_ACTION_DELTA)
        action = np.zeros(7)
        action[:3] = delta
        action[6] = gripper_cmd
        raw_env.step(list(action))
        sim = raw_env.env.sim
        if t % verbose_every == 0:
            print(f"  t={t} eef={np.round(eef,3)} target={np.round(target_xyz,3)} d={d:.4f}")
    print(f"  DID NOT CONVERGE after {max_steps} steps, final d={d:.4f}")
    return d


print("--- hover above butter ---")
move_to((moving_xy[0], moving_xy[1], HOVER_Z), -1.0)

print("--- descend to grasp height (tight tolerance) ---")
move_to((moving_xy[0], moving_xy[1], moving_z0), -1.0, max_steps=200, tol=0.003)

sim = raw_env.env.sim
eef = sim.data.get_site_xpos("gripper0_grip_site")
bp = sim.data.get_body_xpos(moving_body)
print("pre-close: eef=", eef, "butter=", bp, "dist=", np.linalg.norm(eef - bp))

print("--- closing gripper ---")
for i in range(20):
    action = np.zeros(7); action[6] = 1.0
    raw_env.step(list(action))
    sim = raw_env.env.sim
    bp = sim.data.get_body_xpos(moving_body)
    eef = sim.data.get_site_xpos("gripper0_grip_site")
    ncon_butter = 0
    for c in range(sim.data.ncon):
        cc = sim.data.contact[c]
        b1 = sim.model.body_id2name(sim.model.geom_bodyid[cc.geom1])
        b2 = sim.model.body_id2name(sim.model.geom_bodyid[cc.geom2])
        if b1 == moving_body or b2 == moving_body:
            ncon_butter += 1
    grip_qpos = sim.data.qpos[sim.model.get_joint_qpos_addr("gripper0_finger_joint1")[0]] if False else None
    print(f"  close step {i}: butter_z={bp[2]:.4f} eef_z={eef[2]:.4f} dist={np.linalg.norm(eef-bp):.4f} ncon_with_butter={ncon_butter}")

print("--- lifting ---")
move_to((moving_xy[0], moving_xy[1], HOVER_Z), 1.0, verbose_every=20)
sim = raw_env.env.sim
bp = sim.data.get_body_xpos(moving_body)
print("final butter z:", bp[2], "(started at", moving_z0, ")")

raw_env.close()
