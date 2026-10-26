"""Binary-search the max +x (toward-camera direction) at which a point at
butter's resting height (z=0.44) stays within the agentview frame margin,
confirmed empirically to be the binding constraint (not y) for this scene's
camera -- see check_livingroom_camera_corners2.py results: x=0.22 clips the
top edge even at y=0, while y=+-0.24 at x=0 is comfortably in frame."""
import numpy as np

CAM_POS = np.array([0.60657737, 0.0, 0.96])
CAM_MAT = np.array([
    [-1.72339050e-06, -5.28769744e-01, 8.48765314e-01],
    [1.00000000e+00, -7.82314965e-07, 1.54309560e-06],
    [-1.51940455e-07, 8.48765314e-01, 5.28769744e-01],
])
FOVY = 45.0
IMG = 256
MARGIN_PX = 20
OBJ_Z = 0.44


def project(world_pt):
    rel = world_pt - CAM_POS
    cam_frame = CAM_MAT.T @ rel  # local camera coords
    x_c, y_c, z_c = cam_frame
    t = np.tan(np.radians(FOVY) / 2)
    u = x_c / (-z_c) / t
    v = y_c / (-z_c) / t
    return u, v  # NDC in [-1,1] ~ [left,right]/[down,up] of frame (sign TBD)


m = 1 - 2 * MARGIN_PX / IMG
for x in [0.0, 0.05, 0.10, 0.13, 0.15, 0.18, 0.20, 0.22]:
    for y in [0.0, 0.24, -0.24]:
        u, v = project(np.array([x, y, OBJ_Z]))
        flag = "OUT" if max(abs(u), abs(v)) > m else "ok"
        print(f"x={x:+.2f} y={y:+.2f} -> u={u:+.3f} v={v:+.3f} {flag}")
