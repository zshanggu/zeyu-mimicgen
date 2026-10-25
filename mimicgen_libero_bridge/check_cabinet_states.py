import sys
from PIL import Image
import os
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
from run_scene1_4factor import LiberoEnvScene1_4Factor, BDDL, CABINET_LEVELS, dist_to_edge

raw_env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, horizon=50)
raw_env.seed(7)
env = LiberoEnvScene1_4Factor(raw_env)
env.target_combo27 = ("near", "near", "near")

outdir = "/mgbridge/out/cabinet_state_check"
os.makedirs(outdir, exist_ok=True)

for state in CABINET_LEVELS:
    env.target_cabinet_state = state
    env.reset()
    sim = raw_env.env.sim
    cab_xy = sim.data.get_body_xpos(raw_env.env.fixtures_dict["wooden_cabinet_1"].root_body)[:2]
    d = dist_to_edge(cab_xy)
    print(f"cabinet_state={state} xy={cab_xy.round(3)} dist_to_edge={d:.3f}")
    obs = raw_env.env._get_observations()
    Image.fromarray(obs["agentview_image"]).save(f"{outdir}/cab_{state}.png")

raw_env.close()
