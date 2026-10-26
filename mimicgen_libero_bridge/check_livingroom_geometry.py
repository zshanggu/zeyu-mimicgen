"""Probe LIVING_ROOM_SCENE2's arena geometry: fixtures, robot base, table
extent, camera params, and native per-object init regions -- analogous to the
kitchen-scene geometry investigation, since this task uses a different table
fixture (living_room_table) and this is a first pass, not a known-identical
arena."""
import sys
import numpy as np

sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket.bddl"

env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=10)
env.seed(0)
env.reset()
e = env.env
sim = e.sim

print("fixtures_dict:", list(e.fixtures_dict.keys()))
print("objects_dict:", list(e.objects_dict.keys()))

for fname in e.fixtures_dict:
    fx = e.fixtures_dict[fname]
    print(f"fixture {fname}: joints={fx.joints}")

base_id = sim.model.body_name2id("robot0_base")
print("robot0_base pos:", sim.data.body_xpos[base_id])

# table geom extent -- find any geom whose body name mentions table
for i in range(sim.model.ngeom):
    bid = sim.model.geom_bodyid[i]
    bname = sim.model.body_id2name(bid)
    if bname and "table" in bname.lower():
        gname = sim.model.geom_id2name(i)
        print(f"geom {gname} body={bname} pos={sim.data.geom_xpos[i]} size={sim.model.geom_size[i]} type={sim.model.geom_type[i]}")

cid = sim.model.camera_name2id("agentview")
print("agentview cam_pos (native):", sim.model.cam_pos[cid])
print("agentview cam_xpos (world):", sim.data.cam_xpos[cid])
print("agentview cam_xmat:", sim.data.cam_xmat[cid].reshape(3, 3))
print("agentview fovy:", sim.model.cam_fovy[cid])

for name in ["butter_1", "basket_1", "milk_1", "cream_cheese_1", "tomato_sauce_1",
             "orange_juice_1", "alphabet_soup_1", "ketchup_1"]:
    body = e.objects_dict[name].root_body
    print(f"{name}: pos={sim.data.get_body_xpos(body)} joints={e.objects_dict[name].joints}")

env.close()
