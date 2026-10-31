"""Separate parts: eyes, wings, diaper, navel."""
import numpy as np
from sdf_lib import smin, smax, ell, ell_rot, sphere
import plush.paint as paint
import plush.body as body

# ------------------------------------------------------------------------ eyes
EYE_C = body.EYE_C.copy()
EYE_R = body.EYE_R
FWD = np.array([0.0, -1.0, 0.0])


def eye_sdf(p, side=1.0):
    c = EYE_C * np.array([side, 1, 1])
    return sphere(p, c, EYE_R)


def eye_colors(V, side):
    """brown iris with dark rim, big black pupil, two catchlights, sclera with a soft shadow"""
    c = EYE_C * np.array([side, 1, 1])
    n = V - c
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    g = FWD + np.array([0.16 * side, 0, -0.02])
    g /= np.linalg.norm(g)
    ang = np.degrees(np.arccos(np.clip(n @ g, -1, 1)))
    N = len(V)
    col = np.tile(paint.hex2rgb(paint.WHITE), (N, 1))
    # sclera: slightly warm, darker toward the upper lid
    up = smoothstep_h(n[:, 2])
    col = paint.mix(col, '#D8D2C4', 0.5 * up)
    # iris
    t_iris = paint.smoothstep(61.0, 55.0, ang)
    tt = paint.smoothstep(22.0, 52.0, ang)
    iris = paint.mix(np.tile(paint.hex2rgb('#B07B33'), (N, 1)),
                     np.tile(paint.hex2rgb('#5A3612'), (N, 1)), tt)
    # radial fibres
    fib = 0.5 + 0.5 * np.sin(np.arctan2(n[:, 0], n[:, 2]) * 60)
    iris = paint.mix(iris, np.tile(paint.hex2rgb('#C08E45'), (N, 1)), 0.18 * fib)
    col = paint.mix(col, iris, t_iris)
    # dark rim of the iris
    rim = paint.smoothstep(4.0, 0.0, np.abs(ang - 58.0))
    col = paint.mix(col, '#3A2308', 0.8 * rim)
    # pupil
    col = paint.mix(col, paint.BLACK, paint.smoothstep(33.0, 27.0, ang))
    # catchlights: big upper-left, small lower-right
    for (dirv, size, thr) in [((np.array([-0.30 * side, -0.55, 0.52])), 0.0, 0.982),
                              ((np.array([0.24 * side, -0.55, -0.34])), 0.0, 0.994)]:
        d = dirv / np.linalg.norm(dirv)
        col = paint.mix(col, '#FFFFFF', paint.smoothstep(thr, thr + 0.012, n @ d))
    return col


def smoothstep_h(v):
    return paint.smoothstep(0.0, 1.0, v)


# ---------------------------------------------------------------------- wings
D = None  # placeholder to keep linters calm


def wing_colors(V):
    """dusty pink membrane with sparse darker speckles, teal felt finger bones"""
    from plush.wings import wing_sdf, WING_POLY
    from sdf_lib import poly2d
    pm = np.stack([np.abs(V[:, 0]), V[:, 1], V[:, 2]], -1)
    x, z = pm[:, 0], pm[:, 2]
    N = len(V)
    # distance to the finger bones: bones are inside the tube radius
    from plush.wings import TIPS, BASE, TIP_R, _yc
    dmin = np.full(N, 1e9, np.float32)
    for tip, r in zip(TIPS, TIP_R):
        A = np.array([BASE[0], _yc(BASE[0]), BASE[1]], np.float32)
        B = np.array([tip[0], _yc(tip[0]), tip[1]], np.float32)
        ab = B - A
        t = np.clip(((V - A) @ ab) / (ab @ ab), 0, 1)
        d = np.linalg.norm(V - (A + t[:, None] * ab), axis=1) - r
        dmin = np.minimum(dmin, d)
    bone = paint.smoothstep(0.030, 0.008, dmin)
    col = np.tile(paint.hex2rgb('#7FB2A2'), (N, 1))
    membrane = paint.smoothstep(0.0, 0.02, dmin)
    pink = np.tile(paint.hex2rgb('#C98FA8'), (N, 1))
    edge = paint.smoothstep(-0.03, 0.12, poly2d(x, z, WING_POLY))
    pink = paint.mix(pink, '#9C6880', 0.55 * edge)
    rnd = paint.hash3(np.round(V * 30.0) / 30.0, 4.3)
    pink = paint.mix(pink, '#7E5266', 0.75 * paint.smoothstep(0.90, 0.96, rnd))
    col = paint.mix(col, pink, membrane)
    return col


# --------------------------------------------------------------------- diaper
DIA_TOP = 1.10
DIA_BOT = 0.04


def diaper_sdf(p, thickness=0.24):
    """cloth nappy: a shell offset from the hips, rolled waistband, rising sides"""
    b = body.torso(p)
    z = p[:, 2]
    pm = np.stack([np.abs(p[:, 0]), p[:, 1], p[:, 2]], -1)
    band = np.clip(1.0 - np.abs(z - DIA_TOP) / 0.22, 0, 1)
    t = thickness + 0.022 * band
    d = np.maximum(-(b + t), b - t)
    # top rim
    d = np.maximum(d, z - DIA_TOP)
    # bottom edge rises toward the sides so the thighs stay free
    z_bot = 0.26 + 0.34 * np.clip((pm[:, 0] - 0.55) / 0.7, 0, 1) - 0.10 * np.clip(-(p[:, 1] + 0.6) / 1.0, 0, 1)
    d = np.maximum(d, z_bot - z)
    return d


def diaper_colors(V):
    N = len(V)
    col = np.tile(paint.hex2rgb('#E4DCC6'), (N, 1))
    n = paint.felt_noise(V, 8.0, 2.2)
    col = paint.mix(col, '#D3C9AF', 0.35 * n)
    # vertical cloth folds
    fold = 0.5 + 0.5 * np.sin(np.arctan2(V[:, 0], -(V[:, 1] + 0.2)) * 16)
    col = paint.mix(col, '#C9BFA4', 0.25 * fold)
    # stitch line along the waistband
    dash = 0.5 + 0.5 * np.sign(np.sin(np.arctan2(V[:, 0], -(V[:, 1] + 0.2)) * 40))
    seam = paint.smoothstep(0.035, 0.012, np.abs(V[:, 2] - (DIA_TOP - 0.055)))
    col = paint.mix(col, paint.STITCH, 0.75 * seam * dash)
    return col


# ---------------------------------------------------------------------- navel
def navel_sdf(p):
    return sphere(p, (0.0, -0.90, 1.18), 0.085)


def navel_colors(V):
    return np.tile(paint.hex2rgb('#4F8377'), (len(V), 1))
