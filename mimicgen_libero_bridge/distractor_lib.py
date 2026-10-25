"""Distractor (task-unrelated object) randomization, ported from
libero_rerender_script/rerender_libero90.py's Scene/sample_layout so the same
proven mechanism (reject placements overlapping task objects/fixtures/other
distractors or the recorded end-effector path) works against a MimicGen-
GENERATED trajectory instead of a pre-recorded one.

Difference from the original: that script replays a FIXED original demo, so
the eef path is known before any distractor is placed. Here the path only
exists after mimicgen generates it, so the driver must generate first, then
place distractors avoiding that path, then re-settle + replay the same
(successful) actions open-loop to confirm the task still succeeds with the
distractors present (see run_generation_with_distractors.py).
"""
import numpy as np

FAR_AWAY = np.array([0.0, 3.5, 0.5])  # parked outside every camera's view


def quat_mul(a, b):  # MuJoCo wxyz
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def yaw_quat(theta):
    return np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Scene:
    """Per-task facts read once from the env: who is task-related, object radii,
    flat-state indices. `env` is a LIBERO OffScreenRenderEnv (or anything with a
    matching `.env`/`.sim` shape) -- same contract as the original."""

    def __init__(self, env, task_object_names):
        e = env.env
        self.env = env  # NOTE: a hard env.reset() rebuilds the sim -- never cache `sim` across resets
        m = e.sim.model

        self.task_objs = [n for n in e.objects_dict if n in task_object_names]
        self.distractors = [n for n in e.objects_dict if n not in task_object_names]

        table_names = [n for n in list(e.fixtures_dict) + ["kitchen_table", "living_room_table", "study_table", "main_table"] if n.endswith("table")]
        on_table = set()
        for pred in e.parsed_problem["initial_state"]:
            if pred[0] == "on" and len(pred) == 3 and any(pred[2].startswith(t + "_") for t in table_names):
                on_table.add(pred[1])
        self.on_table = on_table

        nq = m.nq
        self.q_idx, self.v_idx = {}, {}
        for n, obj in e.objects_dict.items():
            j = obj.joints[0]
            qs, _ = m.get_joint_qpos_addr(j)
            vs, ve = m.get_joint_qvel_addr(j)
            self.q_idx[n] = 1 + qs  # sim.get_state().flatten() = [time, qpos, qvel]
            self.v_idx[n] = (1 + nq + vs, 1 + nq + ve)

        bodies = [(i, m.body_id2name(i) or "") for i in range(m.nbody)]
        self.geoms, self.fixture_geoms = {}, {}
        for n in list(e.objects_dict) + list(e.fixtures_dict):
            ids = {i for i, bn in bodies if bn.startswith(n + "_") or bn == n}
            gl = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids]
            (self.geoms if n in e.objects_dict else self.fixture_geoms)[n] = gl
        self.radius = {}

    @property
    def e(self):
        return self.env.env

    @property
    def sim(self):
        return self.env.env.sim

    def measure_radii(self):
        """Footprints from the real collision geometry at the current pose."""
        d, m = self.sim.data, self.sim.model
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        self.boxes = {}
        for table in (self.geoms, self.fixture_geoms):
            for n, gl in table.items():
                coll = [g for g in gl if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0] or list(gl)
                if not coll:
                    self.boxes[n] = (np.zeros(2) - 0.05, np.zeros(2) + 0.05)
                    self.radius[n] = 0.05
                    continue
                pts = []
                for g in coll:
                    c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                    corners = c + signs * h
                    pts.append(d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T)
                pts = np.concatenate(pts)[:, :2]
                lo, hi = pts.min(0), pts.max(0)
                self.boxes[n] = (lo, hi)
                origin = self.origin_xy(n)
                self.radius[n] = float(np.max(np.linalg.norm(pts - origin, axis=1)))

    def visibility_fn(self, cam="agentview", size=256, margin=14):
        m, d = self.sim.model, self.sim.data
        cid = m.camera_name2id(cam)
        pos, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
        f = 0.5 * size / np.tan(np.radians(m.cam_fovy[cid]) / 2)

        def vis(x, y, z):
            pc = R.T @ (np.array([x, y, z]) - pos)
            if pc[2] >= -1e-6:
                return False
            u, v = size / 2 + f * pc[0] / -pc[2], size / 2 - f * pc[1] / -pc[2]
            return margin <= u <= size - margin and margin <= v <= size - margin

        return vis

    def origin_xy(self, name):
        if name in self.q_idx:
            return self.sim.data.qpos[self.q_idx[name] - 1: self.q_idx[name] + 1].copy()
        return self.xy(name)

    def xy(self, name):
        table = self.geoms if name in self.geoms else self.fixture_geoms
        gl = table[name]
        return np.mean([self.sim.data.geom_xpos[g][:2] for g in gl], axis=0) if gl else np.zeros(2)

    def positions(self, names):
        return {n: self.sim.data.qpos[self.q_idx[n] - 1: self.q_idx[n] + 2].copy() for n in names}

    def disable_collisions(self, name):
        m = self.sim.model
        for g in self.geoms[name]:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0


def _dist_to_box(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi)))


def sample_layout(scene, state0, path_xyz, rng, p_remove, table_z, vis=None,
                   xy_lo=(-0.40, -0.50), xy_hi=(0.40, 0.50)):
    """Return (perturbed flat state, removed list, moved dict) or None if
    nothing changed. state0 = sim.get_state().flatten() at the pose scene's
    boxes/radii were measured at. xy_lo/xy_hi: hard clip on the candidate
    sampling box (table-relative; override per-scene if the default -0.4..0.4 /
    -0.5..0.5 box doesn't fit -- it's a LIBERO-kitchen-arena default, not a
    physical constant)."""
    st = state0.copy()
    removed, moved = [], {}
    objs = scene.distractors
    if not objs:
        return None

    obstacles = [(*scene.boxes[n], 0.03) for n in scene.task_objs if n in scene.boxes]
    obstacles += [(*scene.boxes[n], 0.03) for n in scene.fixture_geoms]
    still = [n for n in objs]

    everything = [scene.boxes[n][0] for n in scene.boxes] + [scene.boxes[n][1] for n in scene.boxes]
    lo = np.min(everything, axis=0) - 0.15
    hi = np.max(everything, axis=0) + 0.15
    lo = np.maximum(lo, xy_lo)
    hi = np.minimum(hi, xy_hi)

    near_table = path_xyz[path_xyz[:, 2] < table_z + 0.25][:, :2] if path_xyz is not None and len(path_xyz) else np.zeros((0, 2))

    order = list(rng.permutation(objs))
    for n in order:
        qi = scene.q_idx[n]
        orig_xy = state0[qi:qi + 2].copy()
        orig_yaw = yaw_of(state0[qi + 3:qi + 7])
        still.remove(n)
        if rng.random() < p_remove:
            k = len(removed)
            st[qi:qi + 3] = FAR_AWAY + np.array([0.3 * k, 0.0, 0.0])
            st[scene.v_idx[n][0]:scene.v_idx[n][1]] = 0.0
            removed.append(n)
            continue
        if n not in scene.on_table:
            obstacles.append((*scene.boxes[n], 0.02))
            continue
        placed = False
        r = scene.radius[n]
        others = obstacles + [(*scene.boxes[o], 0.02) for o in still]
        for _ in range(80):
            cand = rng.uniform(lo, hi)
            if vis is not None and not vis(cand[0], cand[1], float(state0[qi + 2])):
                continue
            if any(_dist_to_box(cand, blo, bhi) < r + mg for blo, bhi, mg in others):
                continue
            if len(near_table) and np.min(np.linalg.norm(near_table - cand, axis=1)) < r + 0.05:
                continue
            yaw = rng.uniform(0, 2 * np.pi)
            st[qi:qi + 2] = cand
            st[qi + 3:qi + 7] = quat_mul(yaw_quat(yaw), state0[qi + 3:qi + 7])
            st[scene.v_idx[n][0]:scene.v_idx[n][1]] = 0.0
            wrap = lambda a: float((a + np.pi) % (2 * np.pi) - np.pi)
            moved[n] = {"from": [float(orig_xy[0]), float(orig_xy[1]), wrap(orig_yaw)],
                        "to": [float(cand[0]), float(cand[1]), wrap(orig_yaw + yaw)]}
            obstacles.append((cand - r, cand + r, 0.02))
            placed = True
            break
        if not placed:
            obstacles.append((*scene.boxes[n], 0.02))
    if not removed and not moved:
        return None
    return st, removed, moved
