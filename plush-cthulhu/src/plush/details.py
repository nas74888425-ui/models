"""Eyes, nappy and per-object colours."""
import numpy as np

import plush.body as body
import plush.paint as paint
from sdf_lib import smin, smax, ell, sphere
from plush.paint import smoothstep

EYE_R = body.EYE_R


def eye_center(side):
    return body.EYE_C * np.array([side, 1, 1], np.float32)


def eye_sdf(p, side):
    return sphere(p, eye_center(side), EYE_R)


def eye_colors(V, side):
    """big amber eye: huge pupil, thin amber iris ring, cream sclera, two highlights"""
    c = eye_center(side)
    rel = V - c
    rel /= (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9)
    gaze = np.array([-0.06 * side, -1.0, 0.05], np.float32)
    gaze /= np.linalg.norm(gaze)
    ang = np.degrees(np.arccos(np.clip(rel @ gaze, -1, 1)))

    sclera = paint.hex2rgb(paint.EYE_WHITE)
    iris = paint.hex2rgb(paint.IRIS)
    iris_deep = paint.hex2rgb(paint.IRIS_DEEP)
    pupil = paint.hex2rgb(paint.PUPIL)

    col = np.tile(sclera, (len(V), 1))
    # iris: amber, darker towards the edge
    t_iris = paint.smoothstep(53.0, 46.0, ang)          # 1 inside the iris
    amber = paint.mix(np.tile(iris_deep, (len(V), 1)), np.tile(iris, (len(V), 1)),
                      paint.smoothstep(50.0, 34.0, ang))
    col = paint.mix(col, amber, t_iris)
    # pupil
    col = paint.mix(col, pupil, paint.smoothstep(34.0, 30.0, ang))
    # tiny dark rim where the iris meets the sclera (as in the photos)
    col = paint.mix(col, iris_deep, 0.55 * paint.smoothstep(54.0, 51.0, ang) * paint.smoothstep(47.0, 51.0, ang))

    # highlights: one big soft, one small, both up and to the doll's left
    for (dx, dz, rad, amp) in [(-0.33, 0.34, 0.30, 0.95), (0.16, 0.44, 0.14, 0.85)]:
        h = np.array([dx * side, -0.42, dz], np.float32)
        h /= np.linalg.norm(h)
        a = np.arccos(np.clip(rel @ h, -1, 1))
        col = paint.mix(col, np.ones(3, np.float32), amp * paint.smoothstep(rad, rad * 0.55, a))
    return np.clip(col, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ nappy
def diaper_sdf(p, thickness=0.055):
    """small terry-cloth nappy: a shell around the hips only (kept off the legs)"""
    b = body.torso(p)
    z = p[:, 2]
    shell = np.maximum(-(b + thickness), b - 0.02)
    # widen the nappy across the hips (a soft padded mass rather than a glove)
    pad = ell(p, (0.0, -0.26, 0.64), (0.80, 0.58, 0.40))
    shell = smin(shell, pad, 0.14)
    shell = np.maximum(shell, z - (body.DIA_TOP - 0.06))
    # rounded leg holes: hollow out a fat cylinder around each thigh
    q = p.copy()
    q[:, 0] = np.abs(q[:, 0]) - 1.02
    leg = np.sqrt(q[:, 0] ** 2 + (q[:, 1] + 0.02) ** 2) - 0.50
    shell = np.maximum(shell, -leg)
    shell = np.maximum(shell, np.abs(p[:, 0]) - 0.80)
    z_bot = 0.16 + 0.26 * np.clip((np.abs(p[:, 0]) - 0.30) / 0.45, 0, 1)
    shell = np.maximum(shell, z_bot - z)
    # smooth folded waistband: gently thicker just below the top edge
    shell = shell - 0.030 * smoothstep(0.09, 0.0, np.abs(z - (body.DIA_TOP - 0.075)))
    return shell


def diaper_colors(V):
    base = paint.hex2rgb(paint.NAPPY)
    shade = paint.hex2rgb(paint.NAPPY_SHADE)
    n = paint.value_noise(V, 22.0, 5.0)
    n2 = paint.value_noise(V, 70.0, 2.0)
    col = base[None, :] * (0.90 + 0.16 * n[:, None] + 0.08 * n2[:, None])
    # darker towards the bottom and the creases
    col = paint.mix(col, shade[None, :], 0.5 * paint.smoothstep(0.75, 0.25, V[:, 2]))
    return np.clip(col, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ navel
def navel_colors(V):
    base = paint.hex2rgb(paint.TEAL)
    dark = paint.hex2rgb(paint.TEAL_DARK)
    d = np.linalg.norm(V - body.NAVEL_C, axis=1)
    col = base[None, :] * (0.9 + 0.2 * paint.value_noise(V, 20.0, 1.0))[:, None]
    col = paint.mix(col, dark[None, :], paint.smoothstep(0.09, 0.0, d))
    return np.clip(col, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ wings
def wing_colors(V):
    membrane = paint.hex2rgb(paint.WING_MEMBRANE)
    edge = paint.hex2rgb(paint.WING_EDGE)
    speck = paint.hex2rgb(paint.WING_SPECK)
    bone = paint.hex2rgb(paint.WING_BONE)
    n = paint.value_noise(V, 9.0, 3.3)
    col = membrane[None, :] * (0.86 + 0.28 * n[:, None])
    # dark speckles
    s = paint.value_noise(V, 46.0, 11.0)
    col = paint.mix(col, speck[None, :], 0.85 * paint.smoothstep(0.72, 0.86, s))
    # darker scalloped edge (near the outline)
    dist_edge = np.abs(np.linalg.norm(V[:, [0, 2]] - np.array([1.24, 1.86], np.float32), axis=1) - 0.62)
    col = paint.mix(col, edge[None, :], 0.5 * paint.smoothstep(0.16, 0.02, dist_edge))
    # felt bones along the fingers
    from plush.wings import ROOT, TIPS
    a = np.array(ROOT, np.float32)
    for tx, tz in TIPS:
        b = np.array([tx, tz], np.float32)
        ab = b - a
        t = np.clip(((V[:, [0, 2]] - a) @ ab) / (ab @ ab), 0, 1)
        dist = np.linalg.norm(V[:, [0, 2]] - (a + t[:, None] * ab), axis=1)
        col = paint.mix(col, bone[None, :], 0.9 * paint.smoothstep(0.035, 0.012, dist))
    return np.clip(col, 0, 1).astype(np.float32)
