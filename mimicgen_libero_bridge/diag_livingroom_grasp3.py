"""Compare butter_1's root_body xpos against its own geom_xpos (collision/
visual geometry), and against the gripper's closest finger geom xpos -- to
find whether there's a body-origin vs mesh-centroid offset (vs. the gripper
actually being close to the GEOMS even though get_body_xpos reports far)."""
import sys
import h5py
import numpy as np

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"
SRC = "/mgbridge/_scratch_prepared_source_LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.hdf5"

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(0)
env.reset()
sim = env.env.sim
m = sim.model

butter = env.env.objects_dict["butter_1"]
print("butter root_body:", butter.root_body, "joints:", butter.joints)

# all bodies whose name starts with butter_1
butter_body_ids = [i for i in range(m.nbody) if (m.body_id2name(i) or "").startswith("butter_1")]
print("butter-prefixed bodies:", [(i, m.body_id2name(i)) for i in butter_body_ids])

# all geoms belonging to those bodies
butter_geom_ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] in butter_body_ids]
print("butter geoms:", [(g, m.geom_id2name(g)) for g in butter_geom_ids])

f = h5py.File(SRC, "r")
states = f["data/demo_0/states"][()]
f.close()

t = 122  # the closest-approach-by-body-origin timestep found previously
env.regenerate_obs_from_state(states[t])
sim = env.env.sim

body_pos = sim.data.get_body_xpos("butter_1_main")
print(f"\nat t={t}: butter_1_main body xpos = {body_pos}")
for g in butter_geom_ids:
    print(f"  geom {m.geom_id2name(g)} (id {g}) xpos = {sim.data.geom_xpos[g]}  size={m.geom_size[g]}  aabb={m.geom_aabb[g]}")

grip_site = sim.data.get_site_xpos("gripper0_grip_site")
print(f"\ngripper0_grip_site xpos = {grip_site}")
for g in butter_geom_ids:
    d = np.linalg.norm(sim.data.geom_xpos[g] - grip_site)
    print(f"  dist grip_site <-> geom {m.geom_id2name(g)}: {d:.4f}")

env.close()
