import sys
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")
from libero.libero.envs import OffScreenRenderEnv
import generic_pipeline as GP

TASKS = [
    ("LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate", "red_coffee_mug_1", "plate_1"),
    ("KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet", "akita_black_bowl_1", None),
]
BDDL_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/zeyu_openpi/third_party/libero/libero/libero/bddl_files/libero_90"

for task_name, obj_a, obj_b in TASKS:
    bddl = f"{BDDL_DIR}/{task_name}.bddl"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64, horizon=10)
    e = env.env
    print(f"=== {task_name} ===")
    print("all sampler names:", list(e.placement_initializer.samplers.keys()))
    if e.conditional_placement_initializer is not None:
        print("conditional_placement_initializer samplers:", list(getattr(e.conditional_placement_initializer, 'samplers', {}).keys()))
    for name in [obj_a, obj_b]:
        if name is None:
            continue
        s = GP._find_table_sampler(e, name)
        if s is None:
            print(f"  {name}: NO table-middle sampler found")
        else:
            print(f"  {name}: x_ranges={s.x_ranges} y_ranges={s.y_ranges}")
    env.close()
