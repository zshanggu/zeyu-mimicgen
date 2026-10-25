"""Build libero90_openvla_256_ori/ from libero90_openvla_256/, keeping only
the standard `data/` group from each task file and dropping `data_distractor/`
(h5py's Group.copy preserves existing chunking/compression as-is, so this is
a pure I/O operation, no re-encoding)."""
import sys
import glob
import os
import time
import h5py

SRC_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256"
DST_DIR = "/data/zeyu/PHD_LAB/Amazon_Project/libero_modified_hdf5/libero90_openvla_256_ori"


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.hdf5")))
    print(f"{len(files)} source files")
    total_std = 0
    for i, src_path in enumerate(files):
        name = os.path.basename(src_path)
        dst_path = os.path.join(DST_DIR, name)
        if os.path.exists(dst_path):
            print(f"[{i+1}/{len(files)}] SKIP (exists) {name}")
            continue
        t0 = time.time()
        tmp_path = dst_path + ".tmp"
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            src.copy("data", dst, name="data")
            n = len(dst["data"].keys())
            total_std += n
        os.rename(tmp_path, dst_path)
        print(f"[{i+1}/{len(files)}] {name}: {n} episodes ({time.time()-t0:.1f}s)", flush=True)
    print(f"DONE. total standard episodes copied this run: {total_std}")


if __name__ == "__main__":
    main()
