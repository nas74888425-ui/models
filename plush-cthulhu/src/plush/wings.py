"""Membrane wings: scalloped bat-wing sheet + finger bones + shoulder root."""
import numpy as np
import math
from sdf_lib import smin, ell, sphere, round_cone, chain, poly2d

BASE = np.array([0.35, 2.35])          # (x, z) where the wing meets the shoulder

# hand-authored bat wing outline (x, z), ordered counter-clockwise around the shape
WING_POLY = [
    (0.35, 2.35),   # root, top
    (0.95, 2.64),   # top arch
    (1.78, 2.56),
    (2.30, 2.28),   # finger tip 1 (outermost)
    (1.96, 2.06),
    (2.16, 1.62),   # finger tip 2
    (1.62, 1.72),   # scallop notch
    (1.78, 1.16),   # finger tip 3
    (1.22, 1.40),   # scallop notch
    (1.28, 0.96),   # finger tip 4 (lowest)
    (0.80, 1.28),   # scallop notch
    (0.54, 1.62),   # inner bottom
    (0.40, 2.00),   # inner edge back to the root
]

TIPS = [(2.30, 2.28), (2.16, 1.62), (1.78, 1.16), (1.28, 0.96)]
TIP_R = (0.075, 0.068, 0.060, 0.052)

Y_BASE = 0.52
SWEEP = 0.16


def _yc(x):
    return Y_BASE + SWEEP * (x - 0.4)


def wing_sdf(p, thick=0.05):
    pm = np.stack([np.abs(p[:, 0]), p[:, 1], p[:, 2]], -1)
    x, y, z = pm[:, 0], pm[:, 1], pm[:, 2]
    m = np.maximum(poly2d(x, z, WING_POLY), np.abs(y - _yc(x)) - thick)
    bones = None
    for tip, r in zip(TIPS, TIP_R):
        A = np.array([BASE[0], _yc(BASE[0]), BASE[1]], np.float32)
        B = np.array([tip[0], _yc(tip[0]), tip[1]], np.float32)
        s = round_cone(pm, A, B, r, r * 0.65)
        bones = s if bones is None else smin(bones, s, 0.02)
    root = ell(pm, (0.40, _yc(0.40), 2.30), (0.20, 0.17, 0.24))
    d = smin(m, bones, 0.03)
    return smin(d, root, 0.08)
