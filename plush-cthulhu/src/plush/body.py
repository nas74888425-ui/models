"""Anatomy of the felt Cthulhu baby, measured off the reference image.

Coordinate system (identical for every module):
    1 unit  = 100 px of the reference
    x = (Xpx - 341.5) / 100      (+x = the doll's left, i.e. right on screen)
    z = (444   - Ypx) / 100      (0 = ground, +z = up)
    y                            depth, -y = towards the camera

Reference landmarks driving the constants below (image pixels):
    head      egg shaped, top 40 (pink tuft), bottom ~255, x 232..452 at eye height
    eyes      centres (294,172)/(389,172), sclera r 33, iris r 26, pupils huge
    muzzle    nose at (341,186), 8 tentacles, tips down to z = 1.72
    ears      thin pointed plates, tips at x 180 / 505, z 2.55..3.55
    wings     x 164..520, z 1.55..2.80, behind the shoulders
    body      shoulders y 280, belly y 330, navel (341,331)
    nappy     white terry cloth, y 336..415
    legs      thighs x = +-1.15, feet in front of them, pink pads face the camera
"""
import numpy as np

from sdf_lib import smin, smax, ell, ell_rot, sphere, round_cone, chain, poly2d

# ------------------------------------------------------------------ landmarks
HEAD_C = np.array([0.0, 0.10, 2.86], np.float32)
HEAD_R = np.array([1.10, 1.00, 1.22], np.float32)

EYE_C = np.array([0.425, -1.02, 3.00], np.float32)
EYE_R = 0.350

MUZZLE_C = np.array([0.0, -0.80, 2.48], np.float32)
MUZZLE_R = np.array([0.58, 0.44, 0.48], np.float32)

CHEST_C = np.array([0.0, 0.00, 1.66], np.float32)
CHEST_R = np.array([0.72, 0.62, 0.58], np.float32)
BELLY_C = np.array([0.0, -0.06, 1.12], np.float32)
BELLY_R = np.array([0.82, 0.66, 0.56], np.float32)

DIA_TOP, DIA_BOT = 1.12, 0.16
NAVEL_C = np.array([0.0, -0.62, 1.30], np.float32)

# thigh centre / foot centre (screen right side, mirrored for the left)
THIGH_C = np.array([0.98, -0.02, 0.60], np.float32)
THIGH_R = np.array([0.45, 0.40, 0.46], np.float32)
FOOT_C = np.array([1.12, -0.56, 0.44], np.float32)
FOOT_R = np.array([0.58, 0.34, 0.52], np.float32)

HAND_C = np.array([1.04, -1.06, 1.00], np.float32)

# ------------------------------------------------------------------ tentacles
# Eight tentacles hanging off the lower rim of the muzzle, traced from the
# reference: thin, mostly vertical, the outer pair curling outwards.
# (root, mid, tip) in model space; radii in TENT_R
_TENT_RAW = [
    # (points, radius): 8 slim felt tentacles with clear gaps between them
    ([(0.05, -1.18, 2.46), (-0.03, -1.34, 2.20), (0.01, -1.42, 1.88), (0.12, -1.38, 1.62)], 0.098),
    ([(0.30, -1.17, 2.48), (0.27, -1.35, 2.18), (0.32, -1.44, 1.84), (0.45, -1.40, 1.58)], 0.104),
    ([(0.56, -1.10, 2.50), (0.57, -1.30, 2.16), (0.66, -1.39, 1.82), (0.80, -1.34, 1.60)], 0.098),
    ([(0.78, -1.00, 2.52), (0.85, -1.18, 2.20), (0.97, -1.25, 1.92), (1.12, -1.16, 1.76)], 0.088),
]
TENT = {i + 1: (np.array(pts, np.float32), rr) for i, (pts, rr) in enumerate(_TENT_RAW)}
TENT_R = {k: v[1] for k, v in TENT.items()}


def _sym(p):
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0])
    return q


def torso(p):
    d = ell(p, CHEST_C, CHEST_R)
    d = smin(d, ell(p, BELLY_C, BELLY_R), 0.22)
    d = smax(d, -p[:, 2] + 0.08, 0.16)
    return d


def legs(p):
    """seated: fat felt thigh, and a rounded foot in front of it, sole towards us"""
    d = np.full(len(p), 1e9, np.float32)
    q = _sym(p)
    thigh = ell(q, THIGH_C, THIGH_R)
    foot = ell(q, FOOT_C, FOOT_R)
    d = smin(thigh, foot, 0.12)
    return smin(d, _sym_neg(p), 0.20)


def _sym_neg(p):
    """helper: mirror the leg field to both sides"""
    q = p.copy(); q[:, 0] = -np.abs(q[:, 0]) - 0.0
    return np.full(len(p), 1e9)


def legs_both(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    d = smin(ell(q, THIGH_C, THIGH_R), ell(q, FOOT_C, FOOT_R), 0.12)
    d = smin(d, ell(q, (1.02, -0.24, 0.86), (0.46, 0.40, 0.34)), 0.18)   # shin bridge
    return d


def toes(p):
    """four toe bumps on the upper edge of each sole"""
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    d = np.full(len(p), 1e9, np.float32)
    for tx, tz, tr in [(-0.20, 0.06, 0.062), (-0.06, 0.09, 0.070), (0.07, 0.08, 0.066), (0.18, 0.03, 0.056)]:
        d = smin(d, sphere(q, (1.10 + tx * 0.95, -0.92, 0.52 + tz), tr * 1.2), 0.04)
    return d


def arms(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    sh = np.array([0.50, -0.02, 1.92], np.float32)
    el = np.array([0.98, -0.60, 1.52], np.float32)
    wr = np.array([1.08, -1.02, 1.20], np.float32)
    c = chain(q, [sh, el, wr], [0.235, 0.205, 0.19], 0.07)
    # little fist on the knee
    c = smin(c, ell(q, HAND_C, np.array([0.21, 0.20, 0.22], np.float32)), 0.10)
    return c


def head(p):
    d = ell(p, HEAD_C, HEAD_R)
    d = smin(d, ell(p, (0.0, -0.34, 3.00), (1.00, 0.88, 1.10)), 0.20)
    # face plate: brings the brow/eye area forward so the eyes sit in the face
    d = smin(d, ell(p, (0.0, -0.38, 2.98), (1.06, 0.76, 0.92)), 0.24)
    # (brows are stitched in the paint pass - no geometry bump)
    for s in (1.0, -1.0):
        d = smin(d, ell(p, (0.64 * s, -0.80, 2.56), (0.40, 0.34, 0.36)), 0.20)
    d = smin(d, ell(p, MUZZLE_C, MUZZLE_R), 0.30)
    d = smin(d, ell(p, (0.0, -0.68, 2.26), (0.40, 0.34, 0.24)), 0.20)   # chin pad
    return d


EAR_POLY = [(0.90, 2.62), (1.16, 2.62), (1.40, 2.70), (1.60, 2.86), (1.70, 3.06),
            (1.66, 3.28), (1.56, 3.42), (1.44, 3.24), (1.30, 3.04), (1.14, 2.86), (0.98, 2.72), (0.90, 2.68)]


def ears(p):
    """big pointed felt plates sticking out and up (pink lining inside)"""
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    plate = poly2d(q[:, 0], q[:, 2], EAR_POLY)
    thick = np.abs(q[:, 1] - 0.14) - 0.058
    d = smax(plate, thick, 0.05)
    # thicken the root so the ear grows out of the head
    d = smin(d, ell(q, (0.96, 0.10, 2.82), (0.30, 0.17, 0.32)), 0.12)
    return d


def tentacles(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    d = np.full(len(p), 1e9, np.float32)
    for i in range(1, 5):
        pts, r = TENT[i]
        c = chain(q, list(pts), [r * 0.95, r * 1.0, r * 0.92, r * 0.66], 0.03)
        d = smin(d, c, 0.028)
    return d


def nose(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    return sphere(q, (0.16, -1.19, 2.68), 0.075)


def neck(p):
    """soft felt column that blends the head into the shoulders (no lollipop gap)"""
    return ell(p, (0.0, 0.00, 2.16), (0.60, 0.56, 0.42))


def body_sdf_raw(p):
    d = torso(p)
    d = smin(d, legs_both(p), 0.20)
    d = smin(d, toes(p), 0.08)
    d = smin(d, arms(p), 0.16)
    d = smin(d, neck(p), 0.28)
    d = smin(d, head(p), 0.14)
    d = smin(d, ears(p), 0.10)
    d = smin(d, tentacles(p), 0.08)
    d = smin(d, nose(p), 0.06)
    return d


def eyes_sockets(p):
    q = p.copy(); q[:, 0] = np.abs(p[:, 0])
    return sphere(q, EYE_C + np.array([0, 0.09, 0]), EYE_R + 0.012)


def body_sdf(p):
    return smax(body_sdf_raw(p), -eyes_sockets(p), 0.025)


def head_only(p):
    return smin(head(p), ears(p), 0.10)
