"""Bat-like felt wings, traced from the reference (behind the shoulders)."""
import numpy as np

from sdf_lib import smin, smax, poly2d, ell

# right-hand wing outline in model x/z, traced from the reference
WING_POLY = [
    (0.58, 2.32), (0.86, 2.34), (1.16, 2.26), (1.42, 2.10), (1.60, 1.90),
    (1.70, 1.66), (1.56, 1.66), (1.54, 1.44), (1.40, 1.48), (1.32, 1.28),
    (1.18, 1.38), (1.06, 1.22), (0.94, 1.40), (0.80, 1.48), (0.68, 1.62),
    (0.58, 1.90),
]
Y_BASE = 0.34          # how far behind the body the wing sits
THICK = 0.055

# finger struts: from the shoulder joint to the wing tips (drawn as felt ribs)
TIPS = [(1.66, 1.70), (1.52, 1.42), (1.30, 1.26), (1.00, 1.36)]
ROOT = (0.62, 2.20)


def wing_sdf(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    plate = poly2d(q[:, 0], q[:, 2], WING_POLY)
    # gentle curve across y: the wing wraps slightly around the back
    yc = Y_BASE + 0.12 * np.clip((q[:, 0] - 0.6) / 1.1, 0, 1) ** 2
    thick = np.abs(q[:, 1] - yc) - THICK
    d = smax(plate, thick, 0.03)
    # felt ribs along the fingers, slightly proud of the membrane
    for tx, tz in TIPS:
        a = np.array([ROOT[0], ROOT[1]])
        b = np.array([tx, tz])
        ab = b - a
        t = np.clip(((q[:, 0] - a[0]) * ab[0] + (q[:, 2] - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        px = a[0] + t * ab[0] - q[:, 0]
        pz = a[1] + t * ab[1] - q[:, 2]
        dist = np.sqrt(px ** 2 + pz ** 2) - 0.028
        rib = smax(dist, np.abs(q[:, 1] - yc) - THICK * 1.8, 0.02)
        d = np.minimum(d, rib)
    return d
