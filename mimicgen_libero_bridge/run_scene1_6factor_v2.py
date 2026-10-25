"""v2 of run_scene1_6factor.py: moves the agentview camera back and up
(CAMERA_X_OFFSET, CAMERA_Z_OFFSET) to fit a much larger table area in frame,
and widens the bowl/plate region to match the ACTUAL robot workspace
instead of the old, overly conservative one -- both re-measured directly,
not assumed:

1. Camera offset: sim.model.cam_pos is a plain mutable array (unlike
   placement_initializer, cameras aren't rebuilt by bddl region sampling --
   but they DO reset to the XML default on every env.reset(), confirmed
   empirically across 3 resets), so it must be reapplied every reset, same
   pattern as everything else that doesn't survive a hard reset. With the
   offset applied, camera_y_bound(x) (same raycast technique as v1) grows
   from ~0.31-0.46 to ~0.52-0.68 depending on x -- confirmed both
   analytically and by rendering the extreme corners (check_camera_offset_
   bound.py): y=0.55 at x=-0.15 is cleanly visible; the tight corner
   x=0.13+y=0.55 together still clips (0.524 bound < 0.55 target there) --
   the camera-safety filter (unchanged mechanism from v1, just fed the new
   cam_pos/cam_mat) handles this automatically, same as before.

2. Y region now DEPENDS ON CABINET STATE. The old y_min=-0.12 floor was
   NEVER a reach or camera limit -- it was purely cabinet-collision
   clearance. Verified directly (check run above): with the cabinet
   genuinely ABSENT (make_cabinet_bddl.py's own "absent" bddl, not just a
   smaller bddl region), bowl placed at y=-0.40/-0.45/-0.55 all succeed
   5/5 or 4/5 -- the SAME reach the +y side already had. The earlier
   y=-0.40 failure (0/5, from check_farthest_position.py) used the STOCK
   bddl, which still has the cabinet sitting right where that position
   would land -- that was a collision, not a reach limit. So:
     - cabinet PRESENT (near/medium/far): Y range stays conservative on
       the negative side (y_min=-0.12, same clearance as before) but
       extends further positive, out to the real reach ceiling (~0.48,
       see below) instead of the old 0.35.
     - cabinet ABSENT: nothing to collide with, so Y range is symmetric,
       using the same ~0.48 ceiling on both sides.

3. Reach ceiling pinned down directly (check_farthest_position.py):
   y=0.45 at x=-0.15 succeeds 5/5, y=0.50 drops to 4/5 (right at the edge)
   -- used y_max=0.48 as a small safety margin under that.

4. X ceiling: x=0.13 confirmed safe both for reach (3/5, consistent with
   it being near the edge of what's reachable) and camera (clean image,
   no clipping) -- widened from the old 0.10 to 0.13.
"""
import sys
import os
import glob
import json
import shutil
import numpy as np
import h5py

sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator

from env_interface import MG_LiberoBowlOnPlate, BOWL_NAME, PLATE_NAME
import distractor_lib as DL

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate.bddl"
PREPARED_HDF5 = "/mgbridge/prepared_source.hdf5"

NUM_SETTLE_STEPS = 30
_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TABLE_Z = 0.90

CAMERA_X_OFFSET = 0.5
CAMERA_Z_OFFSET = 0.4
CAMERA_MARGIN_PX = 20
CAMERA_IMG_SIZE = 256

REGION_X_LO, REGION_X_HI = -0.45, 0.13
Y_MAX = 0.48          # reach ceiling, both cabinet-present and absent
Y_MIN_CABINET_PRESENT = -0.12   # cabinet-collision clearance, unchanged from v1
Y_MIN_CABINET_ABSENT = -0.48    # symmetric, nothing to collide with

LEVELS = ["near", "medium", "far"]
X_LEVELS = ["bottom", "center", "top"]
Y_LEVELS = ["left", "center", "right"]
CABINET_LEVELS = ["near", "medium", "far", "absent"]

_x_edges = np.linspace(REGION_X_LO, REGION_X_HI, 4)
X_BINS = {lvl: (float(_x_edges[i]), float(_x_edges[i + 1])) for i, lvl in enumerate(X_LEVELS)}


def y_bins_for(cabinet_state):
    y_lo = Y_MIN_CABINET_ABSENT if cabinet_state == "absent" else Y_MIN_CABINET_PRESENT
    edges = np.linspace(y_lo, Y_MAX, 4)
    return {lvl: (float(edges[i]), float(edges[i + 1])) for i, lvl in enumerate(Y_LEVELS)}


BP_BINS = {"near": (0.16, 0.26), "medium": (0.26, 0.38), "far": (0.38, 0.55)}
CABINET_EDGE_BINS = {"near": (0.14, 0.20), "medium": (0.20, 0.26), "far": (0.26, 0.32)}

PRIMARY_COMBOS = [(bp, bx, px) for bp in LEVELS for bx in X_LEVELS for px in X_LEVELS]  # 27
SECONDARY_COMBOS = [(by, py) for by in Y_LEVELS for py in Y_LEVELS]  # 9

TARGET_TOTAL = 100
MAX_ATTEMPTS = 400
MAX_OUTER_RETRIES = 25
MAX_PLACEMENT_TRIES = 100


def camera_y_bound(cam_pos, cam_mat, fovy, x, table_z=TABLE_Z):
    m = 1 - 2 * CAMERA_MARGIN_PX / CAMERA_IMG_SIZE

    def hit(u, v):
        t = np.tan(np.radians(fovy) / 2)
        d = np.array([u * t, v * t, -1.0]); d /= np.linalg.norm(d)
        d = cam_mat @ d
        tt = (table_z - cam_pos[2]) / d[2]
        return cam_pos + tt * d

    lo, hi = -m, m
    for _ in range(40):
        mid = (lo + hi) / 2
        if hit(0, mid)[0] > x:
            lo = mid
        else:
            hi = mid
    v = (lo + hi) / 2
    return hit(m, v)[1]


def label_for(primary, secondary, cab_state):
    bp, bx, px = primary
    by, py = secondary
    return f"bp-{bp}_bowlX-{bx}_bowlY-{by}_plateX-{px}_plateY-{py}_cab-{cab_state}"


class LiberoEnvScene1_6FactorV2:
    def __init__(self, libero_env, cabinet_state):
        self.base_env = libero_env
        self.cabinet_state = cabinet_state
        self.y_bins = y_bins_for(cabinet_state)
        self.target_primary = None
        self.target_secondary = None
        self.rng = np.random.default_rng()
        self._cam_params = None

    def _apply_camera_offset(self):
        sim = self.base_env.env.sim
        cid = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[cid] = sim.model.cam_pos[cid] + np.array([CAMERA_X_OFFSET, 0.0, CAMERA_Z_OFFSET])
        sim.forward()
        self._cam_params = (sim.data.cam_xpos[cid].copy(), sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                             float(sim.model.cam_fovy[cid]))

    def _camera_safe(self, xy):
        cam_pos, cam_mat, fovy = self._cam_params
        return abs(xy[1]) <= camera_y_bound(cam_pos, cam_mat, fovy, xy[0])

    def _obj_qpos_addr(self, name):
        sim = self.base_env.env.sim
        joint = self.base_env.env.objects_dict[name].joints[0]
        qs, _ = sim.model.get_joint_qpos_addr(joint)
        vs, ve = sim.model.get_joint_qvel_addr(joint)
        return qs, vs, ve

    def _set_xy(self, name, xy):
        sim = self.base_env.env.sim
        qs, vs, ve = self._obj_qpos_addr(name)
        sim.data.qpos[qs:qs + 2] = xy
        sim.data.qvel[vs:ve] = 0.0
        sim.forward()

    def _sample_box(self, x_bin, y_bin):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = self.y_bins[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            cand = self.rng.uniform([x_lo, y_lo], [x_hi, y_hi])
            if self._camera_safe(cand):
                return cand
        return None

    def _sample_plate_xy(self, bowl_xy, bp_lo, bp_hi, x_bin, y_bin):
        x_lo, x_hi = X_BINS[x_bin]
        y_lo, y_hi = self.y_bins[y_bin]
        for _ in range(MAX_PLACEMENT_TRIES):
            d = self.rng.uniform(bp_lo, bp_hi)
            theta = self.rng.uniform(0, 2 * np.pi)
            cand = bowl_xy + d * np.array([np.cos(theta), np.sin(theta)])
            if x_lo <= cand[0] <= x_hi and y_lo <= cand[1] <= y_hi and self._camera_safe(cand):
                return cand
        return None

    def reset(self):
        ret = self.base_env.reset()
        self._apply_camera_offset()
        bp_bin, bowl_x_bin, plate_x_bin = self.target_primary
        bowl_y_bin, plate_y_bin = self.target_secondary
        bp_lo, bp_hi = BP_BINS[bp_bin]

        placed = False
        for _ in range(MAX_OUTER_RETRIES):
            bowl_xy = self._sample_box(bowl_x_bin, bowl_y_bin)
            if bowl_xy is None:
                continue
            plate_xy = self._sample_plate_xy(bowl_xy, bp_lo, bp_hi, plate_x_bin, plate_y_bin)
            if plate_xy is None:
                continue
            self._set_xy(BOWL_NAME, bowl_xy)
            self._set_xy(PLATE_NAME, plate_xy)
            placed = True
            break
        if not placed:
            raise RuntimeError(f"could not jointly satisfy {self.target_primary}/{self.target_secondary} in {MAX_OUTER_RETRIES} outer tries")

        for _ in range(NUM_SETTLE_STEPS):
            ret = self.base_env.step(_DUMMY_ACTION)
        return ret

    def get_state(self):
        return {"states": self.base_env.get_sim_state()}

    def reset_to(self, state):
        ret = self.base_env.regenerate_obs_from_state(state["states"])
        self._apply_camera_offset()
        return ret

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
        return {"env_name": "LIBERO_KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_6factor_v2", "type": "libero"}


def build_task_spec():
    task_spec = MG_TaskSpec()
    task_spec.add_subtask(object_ref="bowl", subtask_term_signal="grasp", subtask_term_offset_range=(5, 10),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    task_spec.add_subtask(object_ref="plate", subtask_term_signal=None, subtask_term_offset_range=(0, 0),
                           selection_strategy="nearest_neighbor_object", selection_strategy_kwargs=dict(nn_k=3),
                           action_noise=0.05, num_interpolation_steps=5, num_fixed_steps=0, apply_noise_during_interpolation=False)
    return task_spec


def write_chunked_copy(src_group, dst_group):
    for key in src_group:
        if key == "obs":
            dst_obs = dst_group.create_group("obs")
            for ok in src_group["obs"]:
                data = src_group["obs"][ok][()]
                if ok in ("agentview_rgb", "eye_in_hand_rgb") and data.ndim == 4:
                    dst_obs.create_dataset(ok, data=data, chunks=(1,) + data.shape[1:], compression="gzip", compression_opts=1)
                else:
                    dst_obs.create_dataset(ok, data=data)
        elif isinstance(src_group[key], h5py.Group):
            src_group.copy(key, dst_group, name=key)
        else:
            dst_group.create_dataset(key, data=src_group[key][()])
    for k, v in src_group.attrs.items():
        dst_group.attrs[k] = v


def write_distractor_episode(grp, name, rec, attrs):
    g = grp.create_group(name)
    o = g.create_group("obs")
    ee = np.stack(rec["ee_states"], 0)
    o.create_dataset("gripper_states", data=np.stack(rec["gripper_states"], 0))
    o.create_dataset("joint_states", data=np.stack(rec["joint_states"], 0))
    o.create_dataset("ee_states", data=ee)
    o.create_dataset("ee_pos", data=ee[:, :3])
    o.create_dataset("ee_ori", data=ee[:, 3:])
    for key, frames in (("agentview_rgb", rec["agent"]), ("eye_in_hand_rgb", rec["wrist"])):
        arr = np.stack(frames, 0)
        o.create_dataset(key, data=arr, chunks=(1,) + arr.shape[1:], compression="gzip", compression_opts=1)
    g.create_dataset("actions", data=np.stack(rec["actions"]))
    g.create_dataset("states", data=np.stack(rec["states"]))
    n_ = len(rec["actions"])
    dones = np.zeros(n_, np.uint8); dones[-1] = 1
    g.create_dataset("dones", data=dones)
    for k, v in attrs.items():
        g.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)


def try_distractor_variant(env, raw_env, traj, seed):
    import robosuite.utils.transform_utils as T

    initial_state = traj["initial_state"]["states"]
    path_xyz = np.array([o["ee_pos"] for o in traj["observations"]]) if traj["observations"] else None

    raw_env.seed(seed)
    raw_env.reset()
    env.reset_to({"states": initial_state})
    scene = DL.Scene(raw_env, {BOWL_NAME, PLATE_NAME})
    scene.measure_radii()
    if not scene.distractors:
        return None, "no_distractors"

    rng = np.random.default_rng(seed)
    vis = scene.visibility_fn()
    result = DL.sample_layout(scene, state0=initial_state.copy(), path_xyz=path_xyz, rng=rng,
                               p_remove=0.3, table_z=TABLE_Z, vis=vis, xy_lo=(-0.5, -0.65), xy_hi=(0.5, 0.65))
    if result is None:
        return None, "no_valid_layout"
    st, removed, moved = result

    pre = {n: scene.sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy() for n in (BOWL_NAME, PLATE_NAME)}
    env.reset_to({"states": st})
    for _ in range(10):
        env.base_env.step(_DUMMY_ACTION)
    sim = raw_env.env.sim
    for n in (BOWL_NAME, PLATE_NAME):
        post = sim.data.get_body_xpos(raw_env.env.objects_dict[n].root_body)[:2].copy()
        if np.linalg.norm(post - pre[n]) > 0.01:
            return None, f"task_object_drift({n})"

    actions, states, ee_states, gripper_states, joint_states, agent, wrist = ([] for _ in range(7))
    for action in traj["actions"]:
        states.append(sim.get_state().flatten())
        actions.append(action)
        o = env.get_observation()
        gripper_states.append(o["gripper_states"])
        joint_states.append(o["joint_states"])
        ee_states.append(np.hstack((o["ee_pos"], T.quat2axisangle(o["ee_ori"]))))
        agent.append(o["agentview_rgb"])
        wrist.append(o["eye_in_hand_rgb"])
        env.base_env.step(action.tolist() if hasattr(action, "tolist") else list(action))

    if not env.is_success()["task"]:
        return None, "replay_failed"
    rec = dict(actions=actions, states=states, ee_states=ee_states, gripper_states=gripper_states,
               joint_states=joint_states, agent=agent, wrist=wrist)
    return rec, dict(removed=removed, moved=moved, seed=int(seed))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/mgbridge/out"
    label = sys.argv[2] if len(sys.argv) > 2 else "scene1_6factor_v2"

    from libero.libero.envs import OffScreenRenderEnv
    import make_cabinet_bddl

    cabinet_bddl_paths = make_cabinet_bddl.write_all()

    task_spec = build_task_spec()
    all_demos = MG_FileUtils.get_all_demos_from_dataset(dataset_path=PREPARED_HDF5, filter_key=None, start=None, n=None)

    tmp_dir = f"/tmp/mg_succ_{label}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    rng_combo = np.random.default_rng(42)
    per_primary = {p: [0, 0] for p in PRIMARY_COMBOS}
    per_primary_secondary = {}
    succ_records = []
    distractor_records = []
    total_attempts = 0

    def pick_least_used(options, key_fn, rng):
        best = min(key_fn(o) for o in options)
        tied = [o for o in options if key_fn(o) == best]
        return tied[rng.integers(len(tied))]

    pass_target = TARGET_TOTAL // len(CABINET_LEVELS)
    pass_max_attempts = MAX_ATTEMPTS // len(CABINET_LEVELS)

    for cab_state in CABINET_LEVELS:
        print(f"\n--- cabinet state: {cab_state} (bddl {cabinet_bddl_paths[cab_state]}) ---", flush=True)
        raw_env = OffScreenRenderEnv(bddl_file_name=cabinet_bddl_paths[cab_state], camera_heights=256, camera_widths=256, horizon=400)
        raw_env.seed(1234)
        env = LiberoEnvScene1_6FactorV2(raw_env, cab_state)
        env_interface = MG_LiberoBowlOnPlate(env=env.base_env)
        data_generator = DataGenerator(task_spec=task_spec, dataset_path=PREPARED_HDF5, demo_keys=all_demos)

        pass_successes = 0
        pass_attempts = 0
        while pass_successes < pass_target and pass_attempts < pass_max_attempts:
            primary = pick_least_used(PRIMARY_COMBOS, lambda p: (per_primary[p][1], per_primary[p][0]), rng_combo)
            for sc in SECONDARY_COMBOS:
                per_primary_secondary.setdefault((primary, cab_state, sc), [0, 0])
            secondary = pick_least_used(SECONDARY_COMBOS,
                                         lambda sc: tuple(per_primary_secondary[(primary, cab_state, sc)]), rng_combo)
            env.target_primary = primary
            env.target_secondary = secondary
            per_primary[primary][0] += 1
            per_primary_secondary[(primary, cab_state, secondary)][0] += 1
            try:
                traj = data_generator.generate(env=env, env_interface=env_interface, select_src_per_subtask=False,
                                                transform_first_robot_pose=False, interpolate_from_last_target_pose=True,
                                                render=False, video_writer=None)
            except Exception as e:  # noqa: BLE001
                pass_attempts += 1
                total_attempts += 1
                if pass_attempts % 20 == 0:
                    print(f"[cab={cab_state} {pass_attempts}] exception for {primary}/{secondary}: {type(e).__name__}: {e}", flush=True)
                continue
            pass_attempts += 1
            total_attempts += 1
            success = bool(traj["success"])
            if pass_attempts % 10 == 0 or success:
                print(f"[cab={cab_state} attempt {pass_attempts}] {label_for(primary, secondary, cab_state)} success={success} "
                      f"(pass {pass_successes}/{pass_target}, total {len(succ_records)}/{TARGET_TOTAL})", flush=True)
            if not success:
                continue

            per_primary[primary][1] += 1
            per_primary_secondary[(primary, cab_state, secondary)][1] += 1
            pass_successes += 1
            succ_records.append((primary, secondary, cab_state))
            MG_FileUtils.write_demo_to_hdf5(folder=tmp_dir, env=env, initial_state=traj["initial_state"], states=traj["states"],
                                             observations=traj["observations"], datagen_info=traj["datagen_infos"], actions=traj["actions"],
                                             src_demo_inds=traj["src_demo_inds"], src_demo_labels=traj["src_demo_labels"])

            rec, info = try_distractor_variant(env, raw_env, traj, seed=6_000_000 + total_attempts)
            if rec is not None:
                distractor_records.append((len(succ_records) - 1, rec, info))

        raw_env.close()
        print(f"--- cabinet state {cab_state} done: {pass_successes}/{pass_attempts} ---", flush=True)

    print(f"\n=== {label}: {len(succ_records)}/{total_attempts} successes ===")
    for p in PRIMARY_COMBOS:
        a, s = per_primary[p]
        print(f"  bp-{p[0]}_bowlX-{p[1]}_plateX-{p[2]}: {s}/{a}")
    from collections import Counter
    print("cabinet state distribution:", dict(Counter(r[2] for r in succ_records)))
    print("bowl Y distribution:", dict(Counter(r[1][0] for r in succ_records)))
    print("plate Y distribution:", dict(Counter(r[1][1] for r in succ_records)))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{label}.hdf5")
    succ_files = sorted(glob.glob(os.path.join(tmp_dir, "*.hdf5")))
    assert len(succ_files) == len(succ_records)
    with h5py.File(out_path, "w") as out_f:
        data_grp = out_f.create_group("data")
        data_grp.attrs["problem_info"] = json.dumps({"language_instruction": "put the black bowl on the plate"})
        data_grp.attrs["mimicgen_bddl_path"] = BDDL
        data_grp.attrs["mimicgen_num_attempts"] = total_attempts
        data_grp.attrs["mimicgen_num_success"] = len(succ_files)
        data_grp.attrs["stratification"] = json.dumps({
            "bowl_plate_distance_bins_m": BP_BINS,
            "x_bins_m": X_BINS,
            "y_bins_cabinet_present_m": y_bins_for("near"),
            "y_bins_cabinet_absent_m": y_bins_for("absent"),
            "cabinet_edge_distance_bins_m": CABINET_EDGE_BINS,
            "camera_offset": {"x": CAMERA_X_OFFSET, "z": CAMERA_Z_OFFSET},
            "note": "X: bottom=near robot, top=away. Y: left/right, +y=right (confirmed by direct "
                    "image check). Y range is wider when cabinet is absent (no collision risk).",
        })
        for i, (sf, (primary, secondary, cab_state)) in enumerate(zip(succ_files, succ_records)):
            group_name = f"demo_{i:03d}_{label_for(primary, secondary, cab_state)}"
            with h5py.File(sf, "r") as in_f:
                g = data_grp.create_group(group_name)
                write_chunked_copy(in_f["data/demo_0"], g)
                g.attrs["bowl_plate_bin"], g.attrs["bowl_x_bin"], g.attrs["plate_x_bin"] = primary
                g.attrs["bowl_y_bin"], g.attrs["plate_y_bin"] = secondary
                g.attrs["cabinet_state"] = cab_state

        dgrp = out_f.create_group("data_distractor")
        for i, rec, info in distractor_records:
            primary, secondary, cab_state = succ_records[i]
            dname = f"distractor_{i:03d}_{label_for(primary, secondary, cab_state)}"
            write_distractor_episode(dgrp, dname, rec, info)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"wrote {out_path}: {len(succ_files)} standard episodes, {len(distractor_records)} distractor episodes")


if __name__ == "__main__":
    main()
