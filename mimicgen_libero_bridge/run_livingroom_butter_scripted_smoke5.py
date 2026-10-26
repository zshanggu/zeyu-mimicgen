"""Smoke test forcing the basket (target) to the exact user-specified corner
[0.1759, -0.3800, 0.4322], with the camera reverted to native (no offset,
cam_pos [0.6065774, 0, 0.96]) and the sped-up controller (gain=14). Monkey-
patches LiberoEnv5FactorGeneric._sample_target_xy so EVERY attempt places
the basket at that exact point (moving/butter still randomized normally via
the 5-factor stratification) -- this directly exercises the corner the user
asked to validate, rather than relying on it coming up by chance."""
import sys
sys.path.insert(0, "/mgbridge/_stubs")
sys.path.insert(0, "/data/zeyu/PHD_LAB/Amazon_Project/zeyu-mimicgen")
sys.path.insert(0, "/mgbridge")

import numpy as np
import run_livingroom_task_5factor as BASE
import run_livingroom_butter_scripted as M

FORCED_TARGET_XY = np.array([0.1759, -0.3800])

_orig_sample_target_xy = BASE.LiberoEnv5FactorGeneric._sample_target_xy


def _forced_sample_target_xy(self, moving_xy, d_lo, d_hi, x_bin, y_bin):
    return FORCED_TARGET_XY.copy()


BASE.LiberoEnv5FactorGeneric._sample_target_xy = _forced_sample_target_xy

M.TARGET_TOTAL = 4
M.MAX_ATTEMPTS = 30

if __name__ == "__main__":
    M.main()
