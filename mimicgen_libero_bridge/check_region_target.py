import sys
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv

BDDL = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90/KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet.bddl"
env = OffScreenRenderEnv(bddl_file_name=BDDL, camera_heights=64, camera_widths=64, horizon=400)
env.seed(0)
env.reset()
e = env.env
print("objects_dict:", list(e.objects_dict.keys()))
print("fixtures_dict:", list(e.fixtures_dict.keys()))
print("obj_of_interest:", e.obj_of_interest)
print("object_sites_dict keys:", list(e.object_sites_dict.keys()) if hasattr(e, "object_sites_dict") else "NO object_sites_dict attr")
goal = e.parsed_problem["goal_state"]
print("goal_state:", goal)

target_name = "wooden_cabinet_1_top_side"
if hasattr(e, "object_sites_dict") and target_name in e.object_sites_dict:
    site = e.object_sites_dict[target_name]
    print("site object:", site, "name attr:", getattr(site, "name", None))
    sim = e.sim
    sid = sim.model.site_name2id(site.name if hasattr(site, "name") else target_name)
    print("site world pos:", sim.data.site_xpos[sid])
env.close()
