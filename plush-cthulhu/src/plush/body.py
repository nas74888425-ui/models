"""SDF anatomy of the felt plush baby-cthulhu (seated pose, from the reference).

Axes:  +X = character's right (image x),  +Y = backwards,  +Z = up.
Scale: 1 unit = 100 px of the 706x474 reference.  Origin under the bottom.
The character sits: legs forward (+ feet toward -Y), arms resting on the thighs.
"""
import numpy as np
from sdf_lib import smin, smax, ell, ell_rot, sphere, round_cone, chain

# ------------------------------------------------------------------ key masses
HEAD_C = np.array([0.0, 0.02, 2.94])
HEAD_R = np.array([1.26, 1.10, 0.90])
EYE_C = np.array([0.56, -0.92, 2.58])
EYE_R = 0.37

# feet centres (front paws), used by the animation rig
FOOT_C = np.array([0.98, -1.95, 0.36])


def _sym(p):
    return np.stack([np.abs(p[:, 0]), p[:, 1], p[:, 2]], -1)


# ---------------------------------------------------------------------- torso
def torso(p):
    pm = _sym(p)
    # single soft pear-shaped body (no separate pelvis: the plush is one piece)
    d = ell(pm, (0.0, 0.06, 1.18), (0.94, 0.86, 1.00))                  # belly barrel
    d = smin(d, ell(pm, (0.0, 0.02, 0.55), (1.02, 0.94, 0.55)), 0.32)   # seated base
    d = smin(d, ell(pm, (0.0, -0.34, 1.32), (0.80, 0.60, 0.60)), 0.32)  # tummy
    d = smin(d, ell(pm, (0.0, 0.02, 1.92), (0.74, 0.70, 0.34)), 0.30)   # shoulder shelf / neck
    return d


# ----------------------------------------------------------------------- legs
def legs(p):
    pm = _sym(p)
    # thigh going forward-out, knee at the front, foot pointing up-forward
    thigh = chain(pm, [(0.60, -0.20, 0.78), (0.92, -0.90, 0.66), (1.10, -1.45, 0.56)],
                  [0.54, 0.50, 0.46], 0.10)
    shin = chain(pm, [(1.10, -1.45, 0.56), (1.12, -1.72, 0.56)], [0.46, 0.44], 0.10)
    d = smin(thigh, shin, 0.20)
    # paw/foot: rounded pad facing the camera, slightly tilted up
    foot = ell_rot(pm, (1.12, -1.92, 0.52), (0.50, 0.32, 0.48), (0.55, 0, 0))
    d = smin(d, foot, 0.16)
    return d


def toes(p):
    """four toe bumps on each paw (geometry only, colour handled in bake)"""
    pm = _sym(p)
    d = None
    for i, (tx, tz, tr) in enumerate([(0.80, 0.66, 0.155), (1.02, 0.74, 0.165),
                                     (1.24, 0.72, 0.16), (1.44, 0.60, 0.14)]):
        ty = -2.24 + 0.02 * i
        s = sphere(pm, (tx, ty, tz), tr)
        d = s if d is None else smin(d, s, 0.06)
    return d


# ---------------------------------------------------------------------- arms
def arms(p):
    pm = _sym(p)
    # upper arm clamped to the side, forearm resting on the thigh, paw hanging
    d = chain(pm, [(0.88, -0.16, 1.68), (1.34, -0.60, 1.32), (1.36, -1.00, 1.00)],
              [0.38, 0.35, 0.32], 0.10)
    paw = ell_rot(pm, (1.38, -1.18, 0.82), (0.32, 0.27, 0.32), (0.5, 0, 0))
    d = smin(d, paw, 0.12)
    return d


def paw_fingers(p):
    pm = _sym(p)
    d = None
    for i, (dx, dy, dz, r) in enumerate([(-0.12, -0.16, -0.06, 0.10),
                                         (0.02, -0.20, -0.05, 0.105),
                                         (0.16, -0.17, -0.02, 0.10),
                                         (0.26, -0.10, 0.04, 0.09)]):
        s = sphere(pm, (1.32 + dx, -1.24 + dy, 0.72 + dz), r)
        d = s if d is None else smin(d, s, 0.05)
    return d


# ---------------------------------------------------------------------- head
def head(p):
    pm = _sym(p)
    d = ell(pm, HEAD_C, HEAD_R)                                   # cranium dome
    d = smin(d, ell(pm, (0.0, 0.34, 2.68), (1.14, 0.96, 0.84)), 0.30)   # back of the skull
    d = smin(d, ell(pm, (0.0, -0.10, 2.40), (1.24, 1.04, 0.68)), 0.30)  # cheeks / jaw
    d = smin(d, ell(pm, (0.0, -0.86, 2.26), (0.56, 0.44, 0.34)), 0.24)  # muzzle
    # brow ridge over the eyes
    d = smin(d, ell(pm, (0.48, -0.84, 2.66), (0.36, 0.30, 0.13)), 0.10)
    # flat-ish top / forehead
    d = smin(d, ell(pm, (0.0, -0.04, 3.34), (1.02, 0.90, 0.50)), 0.28)
    return d


def ears(p):
    pm = _sym(p)
    base = ell_rot(pm, (1.10, 0.14, 2.98), (0.36, 0.17, 0.44), (0.0, 0.05, 0.12))
    mid = ell_rot(pm, (1.52, 0.20, 3.16), (0.42, 0.15, 0.34), (0.0, -0.28, 0.30))
    tip = ell_rot(pm, (1.92, 0.26, 3.34), (0.26, 0.10, 0.20), (0.0, -0.50, 0.40))
    d = smin(base, mid, 0.14)
    return smin(d, tip, 0.12)


def tentacles(p):
    """8 tentacles hanging from the muzzle; return (sdf, ids) where ids are centre-line indices"""
    pm = _sym(p)
    d = None
    for cl, radii in TENTACLES:
        c = chain(pm, cl, radii, 0.09)
        d = c if d is None else smin(d, c, 0.06)
    return d


def TENT(i):
    return TENTACLES[i]


# centre-lines in the +x half; 4 pairs -> ends in loops / curls
TENTACLES = [
    # inner pair: hangs in front of the chin, tip curls outward
    ([(0.18, -1.16, 2.20), (0.20, -1.34, 2.00), (0.24, -1.46, 1.80),
      (0.38, -1.48, 1.68), (0.52, -1.42, 1.70)], [0.140, 0.136, 0.124, 0.100, 0.070]),
    # second pair (longest)
    ([(0.52, -1.14, 2.20), (0.56, -1.32, 1.96), (0.62, -1.44, 1.72),
      (0.74, -1.46, 1.56), (0.88, -1.40, 1.52)], [0.145, 0.140, 0.128, 0.104, 0.072]),
    # third pair, curls up at the tip
    ([(0.86, -1.04, 2.16), (0.96, -1.24, 1.94), (1.04, -1.32, 1.74),
      (1.16, -1.30, 1.62), (1.26, -1.20, 1.66)], [0.140, 0.136, 0.124, 0.100, 0.070]),
    # outer pair, sweeps out to the cheek
    ([(1.08, -0.84, 2.12), (1.22, -1.00, 1.94), (1.34, -1.06, 1.80),
      (1.46, -1.00, 1.78)], [0.132, 0.126, 0.112, 0.082]),
]


def body_sdf_raw(p):
    d = torso(p)
    d = smin(d, legs(p), 0.16)
    d = smin(d, toes(p), 0.06)
    d = smin(d, arms(p), 0.10)
    d = smin(d, paw_fingers(p), 0.06)
    h = head(p)
    d = smin(d, h, 0.30)
    d = smin(d, ears(p), 0.16)
    d = smin(d, tentacles(p), 0.07)
    return d


def body_sdf(p):
    d = body_sdf_raw(p)
    d = smax(d, -eyes_carve(p), 0.03)
    return d


def eyes_carve(p):
    """local socket + straight upper lid line, limited to a small ball around each eye"""
    pm = _sym(p)
    c = EYE_C
    sock = sphere(pm, c + np.array([0.0, 0.12, 0.0]), EYE_R + 0.02)
    # lid plane, but only inside a ball around the eye so the face keeps its shape
    lid = (c[2] + 0.26) - (pm[:, 2] - c[2]) * 0.42 - c[2]
    ball = sphere(pm, c, 0.62)
    lid_local = smax(lid, ball, 0.05)
    out = smax(sock, lid_local, 0.05)
    return out


def head_only(p):
    d = head(p)
    d = smin(d, ears(p), 0.16)
    return d


def eyes_sockets(p):
    """carve shallow sockets so eyeballs get a fabric lid rim"""
    pm = _sym(p)
    return sphere(pm, EYE_C + np.array([0.0, 0.14, 0.0]), EYE_R + 0.04)
