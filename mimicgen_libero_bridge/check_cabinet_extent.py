import sys
sys.path.insert(0, "/mgbridge")
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(999)
env.reset()
sim = env.env.sim

# Find all geoms belonging to the cabinet body subtree and their bounding extents
all_body_names = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
cab_related = [n for n in all_body_names if n and "cabinet" in n.lower()]
print("cabinet-related body names:", cab_related)

ys = []
for i in range(sim.model.ngeom):
    body_id = sim.model.geom_bodyid[i]
    body_name = sim.model.body_id2name(body_id) or ""
    if "cabinet" in body_name.lower():
        gpos = sim.data.geom_xpos[i]
        gsize = sim.model.geom_size[i]
        ys.append((gpos[1], gsize[1] if len(gsize) > 1 else 0, body_name))
for y, sy, name in sorted(ys):
    print(f"  geom body={name} y={y:.3f} size_y={sy:.3f} -> extent approx [{y-sy:.3f},{y+sy:.3f}]")

# also print table surface height reference and bowl resting height for comparison
bowl = env.env.objects_dict["akita_black_bowl_1"]
print("bowl root body z (at reset, before step):", sim.data.body_xpos[sim.model.body_name2id(bowl.root_body)])
env.close()
