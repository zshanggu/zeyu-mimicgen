"""Validate the shared bowl/plate region: (1) no spawn overlap (2) real
relative-arrangement diversity (bowl can land on either side of plate, not
always the same side) (3) no cabinet collision."""
import sys
import numpy as np
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(123)

bowl_pos, plate_pos = [], []
min_dists = []
for trial in range(60):
    env.reset()
    sim = env.env.sim
    bowl_obj = env.env.objects_dict["akita_black_bowl_1"]
    plate_obj = env.env.objects_dict["plate_1"]
    bp = sim.data.get_body_xpos(bowl_obj.root_body).copy()
    pp = sim.data.get_body_xpos(plate_obj.root_body).copy()
    bowl_pos.append(bp)
    plate_pos.append(pp)
    min_dists.append(np.linalg.norm(bp[:2] - pp[:2]))

env.close()
bowl_pos = np.array(bowl_pos)
plate_pos = np.array(plate_pos)
min_dists = np.array(min_dists)

print(f"bowl  x[{bowl_pos[:,0].min():.3f},{bowl_pos[:,0].max():.3f}] y[{bowl_pos[:,1].min():.3f},{bowl_pos[:,1].max():.3f}]")
print(f"plate x[{plate_pos[:,0].min():.3f},{plate_pos[:,0].max():.3f}] y[{plate_pos[:,1].min():.3f},{plate_pos[:,1].max():.3f}]")
print(f"min bowl-plate center distance over {len(min_dists)} trials: {min_dists.min():.3f} (bowl r~0.05, plate r~0.09 -> need > ~0.14 to not overlap)")
print(f"mean dist {min_dists.mean():.3f}")

rel = plate_pos[:, :2] - bowl_pos[:, :2]  # plate relative to bowl
n_plate_more_negx = (rel[:, 0] < 0).sum()  # bowl is at larger x than plate -> "bowl toward +x side"
n_plate_more_posx = (rel[:, 0] > 0).sum()
n_plate_more_negy = (rel[:, 1] < 0).sum()
n_plate_more_posy = (rel[:, 1] > 0).sum()
print(f"bowl at larger-x than plate: {n_plate_more_negx}/60   bowl at smaller-x than plate: {n_plate_more_posx}/60")
print(f"bowl at larger-y than plate: {n_plate_more_negy}/60   bowl at smaller-y than plate: {n_plate_more_posy}/60")

cabinet_x = (-0.12, 0.121)
cabinet_y = (-0.525, -0.191)
for name, arr in [("bowl", bowl_pos), ("plate", plate_pos)]:
    overlap = ((arr[:,0] > cabinet_x[0]) & (arr[:,0] < cabinet_x[1]) & (arr[:,1] > cabinet_y[0]) & (arr[:,1] < cabinet_y[1])).sum()
    print(f"{name} spawns inside cabinet footprint: {overlap}/60")
