"""Sanity check: what xy does LIBERO's own region sampler actually produce for
bowl/plate on repeated resets of the widened bddl, with no mimicgen involved at all?
Compares that against the declared bddl region bounds."""
import sys
import numpy as np
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/mgbridge/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_widened.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(4242)

bowl_pos = []
plate_pos = []
for trial in range(30):
    env.reset()
    sim = env.env.sim
    bowl_body = "akita_black_bowl_1_main"
    plate_body = "plate_1_main"
    # use objects_dict like env_interface.py does
    bowl_obj = env.env.objects_dict["akita_black_bowl_1"]
    plate_obj = env.env.objects_dict["plate_1"]
    bp = sim.data.get_body_xpos(bowl_obj.root_body).copy()
    pp = sim.data.get_body_xpos(plate_obj.root_body).copy()
    bowl_pos.append(bp)
    plate_pos.append(pp)

env.close()
bowl_pos = np.array(bowl_pos)
plate_pos = np.array(plate_pos)
print("bowl x", bowl_pos[:,0].min(), bowl_pos[:,0].max(), "y", bowl_pos[:,1].min(), bowl_pos[:,1].max())
print("plate x", plate_pos[:,0].min(), plate_pos[:,0].max(), "y", plate_pos[:,1].min(), plate_pos[:,1].max())
print("bowl samples:\n", bowl_pos[:10].round(3))
print("plate samples:\n", plate_pos[:10].round(3))
