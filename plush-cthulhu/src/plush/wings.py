"""Bat-like felt wings, traced from the reference (behind the shoulders)."""
import numpy as np

from sdf_lib import smin, smax, poly2d, ell

# right-hand wing outline in model x/z, traced from the reference
WING_POLY = [
    (0.70, 2.46), (1.06, 2.50), (1.44, 2.42), (1.74, 2.24), (1.94, 1.98),
    (2.00, 1.66), (1.92, 1.36), (1.80, 1.48), (1.72, 1.14), (1.56, 1.32),
    (1.46, 0.98), (1.28, 1.18), (1.16, 0.92), (1.00, 1.14), (0.88, 0.98),
    (0.74, 1.30), (0.66, 1.72),
]
Y_BASE = 0.40          # how far behind the body the wing sits
THICK = 0.055

# finger struts: from the shoulder joint to the wing tips (drawn as felt ribs)
TIPS = [(2.00, 1.72), (1.74, 1.24), (1.46, 1.02), (1.14, 1.18)]
ROOT = (0.76, 2.34)


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
