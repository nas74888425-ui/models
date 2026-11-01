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
HEAD_C = np.array([0.0, 0.12, 2.90], np.float32)
HEAD_R = np.array([1.10, 1.00, 1.22], np.float32)

EYE_C = np.array([0.475, -1.00, 3.12], np.float32)
EYE_R = 0.325

MUZZLE_C = np.array([0.0, -0.80, 2.48], np.float32)
MUZZLE_R = np.array([0.58, 0.44, 0.48], np.float32)

CHEST_C = np.array([0.0, 0.00, 1.72], np.float32)
CHEST_R = np.array([0.62, 0.56, 0.56], np.float32)
BELLY_C = np.array([0.0, -0.04, 1.16], np.float32)
BELLY_R = np.array([0.74, 0.62, 0.54], np.float32)

DIA_TOP, DIA_BOT = 1.12, 0.16
NAVEL_C = np.array([0.0, -0.62, 1.30], np.float32)

# thigh centre / foot centre (screen right side, mirrored for the left)
THIGH_C = np.array([1.06, 0.00, 0.54], np.float32)
THIGH_R = np.array([0.45, 0.40, 0.46], np.float32)
FOOT_C = np.array([1.10, -0.50, 0.42], np.float32)
FOOT_R = np.array([0.46, 0.36, 0.42], np.float32)

HAND_C = np.array([1.12, -1.20, 1.16], np.float32)

# ------------------------------------------------------------------ tentacles
# Eight tentacles hanging off the lower rim of the muzzle, traced from the
# reference: thin, mostly vertical, the outer pair curling outwards.
# (root, mid, tip) in model space; radii in TENT_R
_TENT_RAW = [
    # inner pair: hangs straight down over the mouth
    ([(0.08, -1.20, 2.64), (0.10, -1.27, 2.38), (0.13, -1.27, 2.10), (0.17, -1.22, 1.88)], 0.082),
    ([(0.25, -1.19, 2.62), (0.27, -1.27, 2.34), (0.30, -1.29, 2.04), (0.34, -1.24, 1.80)], 0.088),
    ([(0.42, -1.14, 2.60), (0.46, -1.23, 2.32), (0.52, -1.25, 2.02), (0.60, -1.20, 1.76)], 0.080),
    # outer pair: hooks outwards at the tip, like the reference
    ([(0.58, -1.06, 2.58), (0.66, -1.15, 2.32), (0.76, -1.17, 2.06), (0.92, -1.09, 1.86)], 0.072),
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
    d = smin(d, ell(q, (1.04, -0.26, 0.80), (0.40, 0.36, 0.30)), 0.16)   # shin bridge
    return d


def toes(p):
    """four toe bumps on the upper edge of each sole"""
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    d = np.full(len(p), 1e9, np.float32)
    for tx, tz, tr in [(-0.20, 0.06, 0.062), (-0.06, 0.09, 0.070), (0.07, 0.08, 0.066), (0.18, 0.03, 0.056)]:
        d = smin(d, sphere(q, (1.10 + tx * 0.85, -0.84, 0.44 + tz), tr), 0.04)
    return d


def arms(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    sh = np.array([0.50, -0.02, 1.92], np.float32)
    el = np.array([0.98, -0.60, 1.52], np.float32)
    wr = np.array([1.08, -1.02, 1.20], np.float32)
    c = chain(q, [sh, el, wr], [0.24, 0.215, 0.19], 0.06)
    # little fist on the knee
    c = smin(c, ell(q, HAND_C, np.array([0.24, 0.21, 0.24], np.float32)), 0.10)
    return c


def head(p):
    d = ell(p, HEAD_C, HEAD_R)
    d = smin(d, ell(p, (0.0, -0.34, 3.00), (1.00, 0.88, 1.10)), 0.20)
    # face plate: brings the brow/eye area forward so the eyes sit in the face
    d = smin(d, ell(p, (0.0, -0.34, 3.04), (1.03, 0.74, 0.94)), 0.24)
    # brow ridges: two soft felt pads above the eyes (reference has clear brows)
    for s_ in (1.0, -1.0):
        d = smin(d, ell_rot(p, (0.50 * s_, -1.02, 3.38), (0.30, 0.16, 0.10),
                            np.array([0.0, 0.0, -0.16 * s_], np.float32)), 0.16)
    for s in (1.0, -1.0):
        d = smin(d, ell(p, (0.60 * s, -0.66, 2.64), (0.30, 0.28, 0.30)), 0.18)
    d = smin(d, ell(p, MUZZLE_C, MUZZLE_R), 0.30)
    d = smin(d, ell(p, (0.0, -0.68, 2.26), (0.40, 0.34, 0.24)), 0.20)   # chin pad
    return d


def ears(p):
    """thin pointed felt plates sticking out and up, pink lining inside"""
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    base = np.array([0.92, 0.12, 2.62], np.float32)
    mid = np.array([1.34, 0.18, 2.98], np.float32)
    tip = np.array([1.62, 0.24, 3.42], np.float32)
    c = chain(q, [base, mid, tip], [0.34, 0.27, 0.055], 0.10)
    # flatten along y: ears are plates, not tubes
    c = np.maximum(c, np.abs(q[:, 1] - 0.16) - 0.105)
    # slight forward cup
    c = smin(c, ell(q, (1.22, 0.10, 2.96), (0.34, 0.16, 0.34)), 0.12)
    return c


def tentacles(p):
    q = p.copy(); q[:, 0] = np.abs(q[:, 0])
    d = np.full(len(p), 1e9, np.float32)
    for i in range(1, 5):
        pts, r = TENT[i]
        c = chain(q, list(pts), [r * 1.15, r * 1.0, r * 0.80, r * 0.34], 0.02)
        d = smin(d, c, 0.045)
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
    d = np.minimum(d, nose(p))
    return d


def eyes_sockets(p):
    q = p.copy(); q[:, 0] = np.abs(p[:, 0])
    return sphere(q, EYE_C + np.array([0, 0.09, 0]), EYE_R + 0.012)


def body_sdf(p):
    return smax(body_sdf_raw(p), -eyes_sockets(p), 0.025)


def head_only(p):
    return smin(head(p), ears(p), 0.10)
