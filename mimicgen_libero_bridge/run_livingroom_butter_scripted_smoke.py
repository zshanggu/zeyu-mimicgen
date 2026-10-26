"""Scripted (non-MimicGen) pick-and-place generation for
LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.

Why this exists instead of reusing run_livingroom_task_5factor.py's
MimicGen-based pipeline: butter's native source demos hold the object via
something other than real contact physics (confirmed exhaustively in
diag_livingroom_grasp*.py / diag_livingroom_genfail*.py -- zero finger-pad
contact registers at ANY timestep in ANY of the 49 source demos, even with
collision margin widened to 1cm; sim.model.neq == 0, so no MuJoCo weld
constraint either; the gripper's own eef/grip-site frame sits a roughly
CONSTANT ~0.11-0.23m from the object throughout the whole "carry" -- compare
to the working KITCHEN_SCENE1 bowl grasp, where the same site sits only
~3.8-5cm from the bowl with real finger contact). MimicGen's retargeting
reproduces the recorded relative offset by construction, so no amount of
task_spec tuning (action noise, interpolation steps, settle steps) can ever
close that gap -- it's not a precision problem, the reference motion itself
never touches the object. A positional "weld" override was tried and
rejected (replicates the native data's own apparent non-physical trick,
doesn't belong in generated training data).

This script instead throws out MimicGen's subtask-replay machinery entirely
for butter and drives a simple closed-loop proportional controller (same
P-control pattern as check_livingroom_reach.py's reach probe, already
validated against this exact arena): hover above the object, descend so the
controller's own eef/grip site lands AT the object's true position (zero
target offset -- matches the bowl task's real-grasp convention), close the
gripper, verify a REAL lift (object height rises with the gripper, not just
assumed), carry to above the basket, descend to just above the rim, release,
retreat. Every step is produced by actually stepping the simulator under
plain env.step() physics -- there is no object-position override anywhere
in this file.

Region/bin stratification (5-factor: dist/movingX/movingY/targetX/targetY),
camera-safe placement, and the mandatory per-episode distractor
randomization are reused UNCHANGED from run_livingroom_task_5factor.py
(imported as BASE) -- none of that logic is butter-specific or physics-
mechanism-specific, it's shared with every other living-room task.
"""
import sys
import os
import json
import shutil
import numpy as np
import h5py

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import distractor_lib as DL
import run_livingroom_task_5factor as BASE

BDDL_DIR = BASE.BDDL_DIR
TABLE_Z = BASE.TABLE_Z
NUM_SETTLE_STEPS = BASE.NUM_SETTLE_STEPS
_DUMMY_ACTION = BASE._DUMMY_ACTION
PRIMARY_COMBOS = BASE.PRIMARY_COMBOS
SECONDARY_COMBOS = BASE.SECONDARY_COMBOS
DIST_BINS = BASE.DIST_BINS
X_BINS = BASE.X_BINS
Y_BINS = BASE.Y_BINS
label_for = BASE.label_for
randomize_distractors = BASE.randomize_distractors
write_episode = BASE.write_episode

# Measured directly against the live sim (see conversation diagnostics):
# basket rim top z=0.6415, tallest distractor (milk/orange_juice) top
# z=0.565 -- HOVER_Z/PLACE_HOVER_Z clear both with margin.
HOVER_Z = 0.70
RELEASE_Z = 0.68  # just above the basket rim (0.6415), drop in rather than crash into the wall

POS_GAIN = 10.0
MAX_ACTION_DELTA = 0.25
WAYPOINT_MAX_STEPS = 50
POS_TOL = 0.015

# Butter's real footprint is 7.6cm(X) x 3.95cm(Y) x 1.74cm(Z) -- it lies FLAT
# on the table, not standing up (measured directly from world-frame AABB
# corners, not assumed). The Panda gripper's fingers open ~4.17cm apart along
# world Y (measured directly too) -- so closing on butter's 3.95cm Y-width
# leaves only ~1mm clearance per side. diag_scripted_grasp.py confirmed:
# a 1.2cm tolerance at the grasp descent is FAR too loose for that
# tolerance -- the gripper closes near but not precisely enough, registers a
# few contacts, and never actually lifts the object (sits static at its
# resting z through 20 hold steps with the gripper fully closed). Tightening
# just the final grasp-descent waypoint to GRASP_POS_TOL=0.003 (3mm) made the
# real lift work immediately (0.445 -> 0.671, a genuine 22.5cm rigid-body
# lift under real contact physics, confirmed via sim.data.contact showing 9
# stable finger<->butter contacts through the whole hold). Transit waypoints
# (hover, carry, release) don't need this precision and keep the looser
# POS_TOL/WAYPOINT_MAX_STEPS.
#
# Episode length fix: the P-controller's near-target convergence rate is
# roughly exponential in gain, and POS_GAIN=4.0 made the grasp descent alone
# take 164-250 steps (vs ~120-160 for an ENTIRE native demo) -- confirmed via
# check_lr_offset_and_reach.py that this was a slow-convergence problem, not
# a reach limit: the SAME target that "didn't converge" in 150 steps at
# gain=4 converged to <3mm in well under 400 steps at gain=8.0. Bumping
# POS_GAIN to 8.0 (with MAX_ACTION_DELTA raised to 0.2 to match) cuts the
# grasp-descent budget accordingly.
GRASP_POS_TOL = 0.003
GRASP_DESCENT_MAX_STEPS = 120

GRASP_CLOSE_STEPS = 8
RELEASE_STEPS = 5
POST_RELEASE_SETTLE_STEPS = 10
LIFT_CHECK_DZ = 0.04  # real height gain required to count as "actually lifted"

TARGET_TOTAL = 4
MAX_ATTEMPTS = 30  # smoke test
N_DISTRACTOR_RETRIES = 3


def _get_obs(raw_env):
    raw = raw_env.env._get_observations()
    obs = dict(raw)
    obs["agentview_rgb"] = raw["agentview_image"]
    obs["eye_in_hand_rgb"] = raw["robot0_eye_in_hand_image"]
    obs["ee_pos"] = raw["robot0_eef_pos"]
    obs["ee_ori"] = raw["robot0_eef_quat"]
    obs["gripper_states"] = raw["robot0_gripper_qpos"]
    obs["joint_states"] = raw["robot0_joint_pos"]
    return obs


def _record_step(raw_env, action, rec, all_object_names):
    import robosuite.utils.transform_utils as T
    sim = raw_env.env.sim
    rec["states"].append(sim.get_state().flatten())
    rec["actions"].append(np.array(action, dtype=np.float64))
    o = _get_obs(raw_env)
    rec["gripper_states"].append(o["gripper_states"])
    rec["joint_states"].append(o["joint_states"])
    rec["ee_states"].append(np.hstack((o["ee_pos"], T.quat2axisangle(o["ee_ori"]))))
    rec["agent"].append(o["agentview_rgb"])
    rec["wrist"].append(o["eye_in_hand_rgb"])
    for n in all_object_names:
        body = raw_env.env.objects_dict[n].root_body
        rec["object_poses"][n].append((sim.data.get_body_xpos(body).copy(), sim.data.get_body_xquat(body).copy()))
    raw_env.step(list(action))


def _move_to(raw_env, target_xyz, gripper_cmd, rec, all_object_names, max_steps=WAYPOINT_MAX_STEPS, tol=POS_TOL):
    sim = raw_env.env.sim
    d = None
    for _ in range(max_steps):
        eef = sim.data.get_site_xpos("gripper0_grip_site")
        err = np.asarray(target_xyz) - eef
        d = float(np.linalg.norm(err))
        if d < tol:
            break
        delta = np.clip(err * POS_GAIN, -MAX_ACTION_DELTA, MAX_ACTION_DELTA)
        action = np.zeros(7)
        action[:3] = delta
        action[6] = gripper_cmd
        _record_step(raw_env, action, rec, all_object_names)
        sim = raw_env.env.sim
    return d


def _hold(raw_env, gripper_cmd, n_steps, rec, all_object_names):
    for _ in range(n_steps):
        action = np.zeros(7)
        action[6] = gripper_cmd
        _record_step(raw_env, action, rec, all_object_names)


def scripted_pick_place(raw_env, moving_name, target_name, all_object_names):
    sim = raw_env.env.sim
    moving_body = raw_env.env.objects_dict[moving_name].root_body
    moving_xy = sim.data.get_body_xpos(moving_body)[:2].copy()
    moving_z0 = float(sim.data.get_body_xpos(moving_body)[2])
    target_body = raw_env.env.objects_dict[target_name].root_body
    target_xy = sim.data.get_body_xpos(target_body)[:2].copy()

    rec = dict(states=[], actions=[], gripper_states=[], joint_states=[], ee_states=[], agent=[], wrist=[],
               object_poses={n: [] for n in all_object_names})

    _move_to(raw_env, (moving_xy[0], moving_xy[1], HOVER_Z), -1.0, rec, all_object_names)
    d = _move_to(raw_env, (moving_xy[0], moving_xy[1], moving_z0), -1.0, rec, all_object_names,
                 max_steps=GRASP_DESCENT_MAX_STEPS, tol=GRASP_POS_TOL)
    if d is None or d > GRASP_POS_TOL * 2:
        return None, "grasp_descent_did_not_converge"
    _hold(raw_env, 1.0, GRASP_CLOSE_STEPS, rec, all_object_names)
    _move_to(raw_env, (moving_xy[0], moving_xy[1], HOVER_Z), 1.0, rec, all_object_names)

    sim = raw_env.env.sim
    cur_z = float(sim.data.get_body_xpos(moving_body)[2])
    if cur_z < moving_z0 + LIFT_CHECK_DZ:
        return None, "grasp_failed_no_real_lift"

    _move_to(raw_env, (target_xy[0], target_xy[1], HOVER_Z), 1.0, rec, all_object_names)
    _move_to(raw_env, (target_xy[0], target_xy[1], RELEASE_Z), 1.0, rec, all_object_names)
    _hold(raw_env, -1.0, RELEASE_STEPS, rec, all_object_names)
    _move_to(raw_env, (target_xy[0], target_xy[1], HOVER_Z), -1.0, rec, all_object_names)
    _hold(raw_env, -1.0, POST_RELEASE_SETTLE_STEPS, rec, all_object_names)

    if not bool(raw_env.env._check_success()):
        return None, "final_task_check_failed"

    return rec, None


def _traj_from_rec(rec):
    """Build the minimal traj dict shape randomize_distractors() expects
    (it only reads initial_state/observations[*]["ee_pos"]/actions)."""
    observations = [{"ee_pos": ee[:3]} for ee in rec["ee_states"]]
    return dict(initial_state=dict(states=rec["states"][0]), actions=rec["actions"], observations=observations)


def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/mgbridge/out"
    label = sys.argv[3] if len(sys.argv) > 3 else task_name

    bddl_path = os.path.join(BDDL_DIR, f"{task_name}.bddl")
    assert os.path.exists(bddl_path), bddl_path
    moving, target = BASE.parse_task_goal(bddl_path)
    print(f"task={task_name} moving={moving} target={target}", flush=True)

    from libero.libero.envs import OffScreenRenderEnv

    probe_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=64, camera_widths=64, horizon=10)
    probe_env.seed(0); probe_env.reset()
    all_objects = set(probe_env.env.objects_dict.keys())
    probe_env.close()
    task_object_names = {moving, target}
    print(f"objects_dict={all_objects}, task_object_names={task_object_names}, "
          f"distractors={all_objects - task_object_names}", flush=True)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    raw_env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=256, camera_widths=256, horizon=1000)
    raw_env.seed(1234)
    env = BASE.LiberoEnv5FactorGeneric(raw_env, moving, target)

    rng_combo = np.random.default_rng(42)
    per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}
    per_primary_secondary = {}
    succ_records = []
    total_attempts = 0
    distractor_discards = 0
    grasp_fail_count = 0
    place_fail_count = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    while len(succ_records) < TARGET_TOTAL and total_attempts < MAX_ATTEMPTS:
        primary = pick_least_used(PRIMARY_COMBOS, lambda p: (per_primary[p][1], per_primary[p][0]), rng_combo)
        for sc in SECONDARY_COMBOS:
            per_primary_secondary.setdefault((primary, sc), [0, 0])
        secondary = pick_least_used(SECONDARY_COMBOS, lambda sc: tuple(per_primary_secondary[(primary, sc)]), rng_combo)
        env.target_primary = primary
        env.target_secondary = secondary
        per_primary[primary][0] += 1
        per_primary_secondary[(primary, secondary)][0] += 1

        try:
            env.reset()
        except RuntimeError:
            total_attempts += 1
            continue

        all_object_names = list(raw_env.env.objects_dict.keys())
        rec, fail_reason = scripted_pick_place(raw_env, moving, target, all_object_names)
        total_attempts += 1
        if rec is None:
            if fail_reason == "grasp_failed_no_real_lift":
                grasp_fail_count += 1
            else:
                place_fail_count += 1
            if total_attempts % 10 == 0:
                print(f"[attempt {total_attempts}] {label_for(primary, secondary)} failed ({fail_reason}) "
                      f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
            continue

        traj = _traj_from_rec(rec)
        dist_rec = info = None
        for retry in range(N_DISTRACTOR_RETRIES):
            dist_rec, info = randomize_distractors(env, raw_env, traj, task_object_names,
                                                     seed=8_000_000 + total_attempts * 10 + retry)
            if dist_rec is not None:
                break
        if dist_rec is None:
            distractor_discards += 1
            print(f"[attempt {total_attempts}] {label_for(primary, secondary)} "
                  f"grasp/place succeeded but distractor randomization failed ({info}) after "
                  f"{N_DISTRACTOR_RETRIES} tries -- discarded", flush=True)
            continue

        per_primary[primary][1] += 1
        per_primary_secondary[(primary, secondary)][1] += 1
        idx = len(succ_records)
        succ_records.append((primary, secondary, info))
        write_episode(os.path.join(tmp_dir, f"{idx:04d}.hdf5"), dist_rec,
                      attrs=dict(dist_bin=primary[0], moving_x_bin=primary[1], target_x_bin=primary[2],
                                 moving_y_bin=secondary[0], target_y_bin=secondary[1],
                                 removed=info["removed"], moved=info["moved"], distractor_seed=info["seed"]))
        print(f"[attempt {total_attempts}] {label_for(primary, secondary)} success=True "
              f"distractors: removed={info['removed']} moved={list(info['moved'].keys())} "
              f"(total {len(succ_records)}/{TARGET_TOTAL})", flush=True)

    raw_env.close()
    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes "
          f"(grasp_fail={grasp_fail_count}, place_fail={place_fail_count}, "
          f"distractor_discards={distractor_discards}) ===")
    for p in PRIMARY_COMBOS:
        a, s = per_primary[p]
        print(f"  dist-{p[0]}_movX-{p[1]}_tgtX-{p[2]}: {s}/{a}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": task_name})
        data_grp.attrs["mimicgen_bddl_path"] = bddl_path
        data_grp.attrs["num_attempts"] = total_attempts
        data_grp.attrs["num_success"] = len(succ_records)
        data_grp.attrs["moving_object"] = moving
        data_grp.attrs["target_object"] = target
        data_grp.attrs["distractors_always_randomized"] = True
        data_grp.attrs["generation_method"] = "scripted_physics_pick_place"
        data_grp.attrs["stratification"] = json.dumps({
            "dist_bins_m": DIST_BINS, "x_bins_m": X_BINS, "y_bins_m": Y_BINS,
        })
        for i, (primary, secondary, info) in enumerate(succ_records):
            group_name = f"demo_{i:03d}_{label_for(primary, secondary)}"
            with h5py.File(os.path.join(tmp_dir, f"{i:04d}.hdf5"), "r") as in_f:
                in_f.copy("data/demo_0", data_grp, name=group_name)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"wrote {out_path}: {len(succ_records)} episodes, every one with randomized distractors "
          f"(scripted real-physics grasp, no retargeting/weld)")


if __name__ == "__main__":
    main()
