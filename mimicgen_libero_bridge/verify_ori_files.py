import glob
import os
import h5py

DST_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_ori"

bad = []
files = sorted(glob.glob(os.path.join(DST_DIR, "*.hdf5")))
for i, path in enumerate(files):
    name = os.path.basename(path)
    try:
        with h5py.File(path, "r") as f:
            demos = list(f["data"].keys())
            g = f["data"][demos[0]]
            _ = g["obs"]["agentview_rgb"][0]
            _ = g["obs"]["agentview_rgb"][-1]
    except Exception as e:
        bad.append(name)
        print(f"[{i+1}/{len(files)}] BAD {name}: {e}")
        continue
print(f"checked {len(files)} files, {len(bad)} bad")
for b in bad:
    print("BAD:", b)
