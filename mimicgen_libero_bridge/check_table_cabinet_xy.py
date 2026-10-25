import sys
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(999)
env.reset()
sim = env.env.sim

def dump(keyword):
    xs, ys = [], []
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        body_name = sim.model.body_id2name(body_id) or ""
        if keyword in body_name.lower():
            gpos = sim.data.geom_xpos[i]
            gsize = sim.model.geom_size[i]
            sx = gsize[0] if len(gsize) > 0 else 0
            sy = gsize[1] if len(gsize) > 1 else 0
            xs.append((gpos[0]-sx, gpos[0]+sx))
            ys.append((gpos[1]-sy, gpos[1]+sy))
    if xs:
        print(f"{keyword}: x range [{min(x[0] for x in xs):.3f},{max(x[1] for x in xs):.3f}]  y range [{min(y[0] for y in ys):.3f},{max(y[1] for y in ys):.3f}]")
    else:
        print(f"{keyword}: no geoms found")

dump("table")
dump("cabinet")
env.close()
