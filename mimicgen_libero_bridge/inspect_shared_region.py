import h5py
import numpy as np
from PIL import Image
import os

path = "/mgbridge/out/shared_region_256.hdf5"
outdir = "/mgbridge/wrist_check/real_episodes"
os.makedirs(outdir, exist_ok=True)

with h5py.File(path, "r") as f:
    data = f["data"]
    print("attrs:", dict(data.attrs))
    demo_keys = list(data.keys())
    bowl_pos = np.array([data[k]["obs"]["akita_black_bowl_1_pos"][0] for k in demo_keys])
    plate_pos = np.array([data[k]["obs"]["plate_1_pos"][0] for k in demo_keys])
    print("num demos", len(demo_keys))
    print("bowl x", bowl_pos[:,0].min(), bowl_pos[:,0].max(), "y", bowl_pos[:,1].min(), bowl_pos[:,1].max())
    print("plate x", plate_pos[:,0].min(), plate_pos[:,0].max(), "y", plate_pos[:,1].min(), plate_pos[:,1].max())
    rel = plate_pos[:,:2] - bowl_pos[:,:2]
    print("bowl x > plate x:", (rel[:,0]<0).sum(), "/", len(demo_keys))
    print("bowl x < plate x:", (rel[:,0]>0).sum(), "/", len(demo_keys))
    print("bowl y > plate y:", (rel[:,1]<0).sum(), "/", len(demo_keys))
    print("bowl y < plate y:", (rel[:,1]>0).sum(), "/", len(demo_keys))
    dist = np.linalg.norm(rel, axis=1)
    print("min bowl-plate dist", dist.min())

    # wrist-view check: pick extreme-bowl-position demos and dump their frame-0 wrist image
    dist_from_center = np.linalg.norm(bowl_pos[:, :2] - np.array([-0.1, 0.0]), axis=1)
    order = np.argsort(-dist_from_center)
    picks = [demo_keys[i] for i in order[:6]] + [demo_keys[i] for i in order[-3:]]
    for k in picks:
        wrist0 = data[k]["obs"]["eye_in_hand_rgb"][0]
        agent0 = data[k]["obs"]["agentview_rgb"][0]
        bp = data[k]["obs"]["akita_black_bowl_1_pos"][0]
        pp = data[k]["obs"]["plate_1_pos"][0]
        Image.fromarray(wrist0).save(f"{outdir}/{k}_wrist_b{bp[0]:.2f}_{bp[1]:.2f}_p{pp[0]:.2f}_{pp[1]:.2f}.png")
        Image.fromarray(agent0).save(f"{outdir}/{k}_agent.png")
    print("picked demos for wrist check:", picks)
