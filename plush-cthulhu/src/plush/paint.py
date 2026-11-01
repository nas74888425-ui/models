"""Vertex-colour painting: felt, applique spots, stitched stars, pads, stitches."""
import numpy as np

# ------------------------------------------------------------------ palette
TEAL = '#679684'
TEAL_DARK = '#487563'
TEAL_LIGHT = '#7FAE97'
PINK = '#E2A8C0'
PINK_DEEP = '#C4869F'
CREAM_STAR = '#E8C878'
STITCH = '#C08A3E'
NAPPY = '#E9E3D4'
NAPPY_SHADE = '#CFC7B6'
EYE_WHITE = '#DFD3C0'
IRIS = '#9C6B25'
IRIS_DEEP = '#54300D'
PUPIL = '#0E0B08'
WING_MEMBRANE = '#BC8CA1'
WING_EDGE = '#A86F88'
WING_SPECK = '#7E5266'
WING_BONE = '#6E9C8C'


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def mix(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim == 1:
        t = t[:, None]
    return a * (1 - t) + b * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def hash3(x, y, z, seed=0.0):
    v = np.sin(x * 127.1 + y * 311.7 + z * 74.7 + seed * 13.3) * 43758.5453
    return v - np.floor(v)


def value_noise(P, freq, seed=0.0):
    """cheap smooth 3D value noise, P is (N,3)"""
    q = P * freq
    i = np.floor(q)
    f = q - i
    f = f * f * (3 - 2 * f)
    acc = np.zeros(len(P), np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) *
                     (f[:, 1] if dy else 1 - f[:, 1]) *
                     (f[:, 2] if dz else 1 - f[:, 2]))
                acc += w * hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return acc


def felt_noise(P, seed=0.0):
    n = (value_noise(P, 6.0, seed) * 0.55 + value_noise(P, 17.0, seed + 3.1) * 0.30 +
         value_noise(P, 43.0, seed + 7.7) * 0.15)
    return n


_STAR_PTS = None


def _star_poly(r_in_ratio=0.44):
    global _STAR_PTS
    if _STAR_PTS is None:
        pts = []
        for k in range(10):
            a = np.pi / 2 + k * np.pi / 5
            r = 1.0 if k % 2 == 0 else r_in_ratio
            pts.append((float(np.cos(a) * r), float(np.sin(a) * r)))
        _STAR_PTS = pts
    return _STAR_PTS


def star_field(u, v, cell=0.3, r_out=0.1, r_in=0.044, density=0.5, seed=0.0):
    """clean 5-point felt stars on a jittered grid in (u, v) space"""
    from sdf_lib import poly2d
    poly = _star_poly(r_in / r_out)
    out = np.zeros(len(u), np.float32)
    cu = np.floor(u / cell)
    cv = np.floor(v / cell)
    for du_ in (-1, 0, 1):
        for dv_ in (-1, 0, 1):
            cx_i = cu + du_
            cy_i = cv + dv_
            h1 = hash3(cx_i, cy_i, 0.0, seed)
            keep = (h1 < density).astype(np.float32)
            cx = (cx_i + 0.2 + 0.6 * hash3(cx_i, cy_i, 1.0, seed)) * cell
            cy = (cy_i + 0.2 + 0.6 * hash3(cx_i, cy_i, 2.0, seed)) * cell
            rot = hash3(cx_i, cy_i, 3.0, seed) * 2 * np.pi
            dx = (u - cx) / r_out
            dy = (v - cy) / r_out
            near = (np.abs(dx) < 1.5) & (np.abs(dy) < 1.5) & (keep > 0)
            if not near.any():
                continue
            cr, sr = np.cos(-rot[near]), np.sin(-rot[near])
            xr = dx[near] * cr - dy[near] * sr
            yr = dx[near] * sr + dy[near] * cr
            d = poly2d(xr, yr, poly)
            out[near] = np.maximum(out[near], smoothstep(0.02, -0.07, d))
    return out


# ------------------------------------------------------------------ props
def _limb_coords(V, A, B):
    """cylindrical unwrap of a limb: returns (t along axis, angle, radius)"""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    ab = B - A
    L = np.linalg.norm(ab)
    ab = ab / L
    ref = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(ab @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], np.float32)
    u1 = np.cross(ab, ref)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(ab, u1)
    rel = V - A
    t = rel @ ab / L
    perp = rel - (t[:, None] * L) * ab
    ang = np.arctan2(perp @ u2, perp @ u1)
    rad = np.linalg.norm(perp, axis=1)
    return t, ang, rad


def _mask(t, ang, rad, t0, t1, r_far, front=True):
    m = smoothstep(t0 - 0.04, t0 + 0.06, t) * smoothstep(t1 + 0.06, t1 - 0.04, t)
    m = m * smoothstep(r_far + 0.14, r_far - 0.02, rad)
    return m


def paint_body(V):
    P = V
    base = hex2rgb(TEAL)
    dark = hex2rgb(TEAL_DARK)
    light = hex2rgb(TEAL_LIGHT)
    pink = hex2rgb(PINK)
    cream = hex2rgb(CREAM_STAR)
    stitch = hex2rgb(STITCH)

    n = felt_noise(P)
    col = base[None, :] * (0.94 + 0.12 * n[:, None])
    # fine fibre speckle so the felt does not read as smooth plastic
    fib = value_noise(P, 140.0, 17.0)
    col = col * (0.98 + 0.04 * fib[:, None])
    # soft shading: slightly darker in the crevices (lower/behind)
    col = mix(col, dark[None, :], 0.10 * smoothstep(2.0, 0.1, P[:, 2]))

    # ---------------- felt applique spots on the head (regular, like the reference)
    head_c = np.array([0.0, 0.10, 3.02], np.float32)
    rel = P - head_c
    # spherical directions
    r = np.linalg.norm(rel, axis=1) + 1e-6
    d = rel / r[:, None]
    # spots laid out on a fibonacci sphere so they are evenly spread
    N = 40
    idx = np.arange(N)
    phi = np.arccos(1 - 2 * (idx + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    size = 0.070 + 0.045 * hash3(idx.astype(np.float64), 3.1, 7.7)
    spot = np.zeros(len(P), np.float32)
    for k in range(N):
        if hash3(idx[k], 1.0, 2.0) < 0.10:
            continue
        ang = np.arccos(np.clip(d @ dirs[k], -1, 1))
        rad = r * np.sin(ang)
        s = size[k]
        spot = np.maximum(spot, smoothstep(s, s * 0.86, rad) *
                          smoothstep(s * 1.02, s * 0.9, rad) * 0 + smoothstep(s, s * 0.88, rad))
    # only on the head (not on limbs / body)
    head_m = smoothstep(2.30, 2.55, P[:, 2]) * smoothstep(1.35, 1.10, np.abs(P[:, 0])) * smoothstep(-1.30, -1.15, P[:, 1])
    # keep the spots off the eyes and the muzzle
    for s_ in (1, -1):
        de = np.linalg.norm(P - (np.array([0.475 * s_, -1.00, 3.12], np.float32)), axis=1)
        spot = spot * smoothstep(0.40, 0.52, de)
    spot = spot * head_m
    col = mix(col, pink[None, :] * (0.97 + 0.06 * n[:, None]), 1.0 * spot)
    # pink felted tuft on the very top
    tuft = smoothstep(3.86, 3.96, P[:, 2]) * smoothstep(0.50, 0.28, np.sqrt(P[:, 0] ** 2 + (P[:, 1] - 0.05) ** 2))
    col = mix(col, pink[None, :], 0.95 * tuft)

    # ---------------- cheek blush (soft pink shading, as in the felt photo)
    for s_ in (1, -1):
        cc = np.array([0.74 * s_, -0.92, 2.74], np.float32)
        dd = np.linalg.norm((P - cc) * np.array([1.0, 0.75, 1.0], np.float32), axis=1)
        col = mix(col, pink[None, :], 0.5 * smoothstep(0.44, 0.06, dd))

    # ---------------- embroidered eyebrows above the eyes
    for s_ in (1, -1):
        ex, ez = 0.455 * s_, 3.40
        # a thin arc: distance to a circle of radius 0.235 centred slightly below the brow line
        rr = np.sqrt(((P[:, 0] - ex) / 1.0) ** 2 + ((P[:, 2] - (ez - 0.30)) / 1.0) ** 2)
        arc = smoothstep(0.034, 0.016, np.abs(rr - 0.30))
        arc = arc * smoothstep(0.30, 0.20, np.abs(P[:, 0] - ex))
        arc = arc * smoothstep(-0.75, -0.95, P[:, 1])
        col = mix(col, hex2rgb('#6B4B2E')[None, :], 0.6 * arc)

    # ---------------- pink sucker dots on the tentacles (reference detail)
    from plush.body import _TENT_RAW
    for pts, rr in _TENT_RAW:
        for side in (1, -1):
            pl = np.array([[x * side, y, z] for (x, y, z) in pts], np.float32)
            for k in range(len(pl) - 1):
                a, bb_ = pl[k], pl[k + 1]
                ab = bb_ - a
                t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
                proj = a + t[:, None] * ab
                d = np.linalg.norm(P - proj, axis=1)
                on = smoothstep(rr * 1.25, rr * 1.05, d) * smoothstep(-0.9, -1.0, P[:, 1] - proj[:, 1] * 0)
                # only the front-facing side of the tentacle
                front = smoothstep(0.0, -0.25, P[:, 1] - proj[:, 1])
                ph = (t * np.linalg.norm(ab)) / 0.085
                dot = smoothstep(0.42, 0.22, np.abs((ph % 1.0) - 0.5))
                col = mix(col, hex2rgb('#D8A2B8')[None, :], 0.75 * on * front * dot)

    # ---------------- round pink suckers along the front of every tentacle
    from plush.body import _TENT_RAW
    sucker = np.zeros(len(P), np.float32)
    for pts, rr in _TENT_RAW:
        for side in (1, -1):
            pl = np.array([[x * side, y, z] for (x, y, z) in pts], np.float32)
            # walk the curve and drop a cup every 0.115 units
            step = 0.115
            for k in range(len(pl) - 1):
                a, b = pl[k], pl[k + 1]
                L = np.linalg.norm(b - a)
                n_steps = max(1, int(L / step))
                for m in range(n_steps + 1):
                    t = m / n_steps
                    c = a * (1 - t) + b * t
                    # cups sit slightly to the front (-y) of the centre line
                    c = c + np.array([0.0, -rr * 0.80, 0.0], np.float32)
                    rad = rr * 0.31
                    d = np.linalg.norm(P - c, axis=1)
                    sucker = np.maximum(sucker, smoothstep(rad, rad * 0.55, d))
    col = mix(col, hex2rgb('#DFA6BC')[None, :], 0.75 * sucker)

    # ---------------- nostril slits between the eyes
    for s_ in (1, -1):
        c = np.array([0.145 * s_, -1.24, 2.62], np.float32)
        dd = np.sqrt(((P[:, 0] - c[0]) / 0.062) ** 2 + ((P[:, 2] - c[2]) / 0.150) ** 2)
        col = mix(col, hex2rgb('#2E4A3F')[None, :], 0.85 * smoothstep(1.05, 0.45, dd) * smoothstep(-0.90, -1.05, P[:, 1]))

    # ---------------- navel dimple
    dn = np.linalg.norm((P - np.array([0.0, -0.70, 1.30], np.float32)) * np.array([1.0, 0.35, 1.0], np.float32), axis=1)
    col = mix(col, hex2rgb('#3D6353')[None, :], 0.85 * smoothstep(0.085, 0.035, dn))

    # ---------------- pink pads on the soles (thumbs + toes)
    front = smoothstep(-0.60, -0.78, P[:, 1])
    for s in (1, -1):
        # big pad on the sole
        c = np.array([1.12 * s, -0.86, 0.30], np.float32)
        dd = np.sqrt(((P[:, 0] - c[0]) / 1.0) ** 2 + ((P[:, 2] - c[2]) / 0.85) ** 2)
        col = mix(col, pink[None, :], 0.96 * smoothstep(0.215, 0.175, dd) * front)
        # four toe pads along the top edge of the sole
        for tx, tz, tr in [(-0.255, 0.63, 0.080), (-0.085, 0.68, 0.090), (0.085, 0.665, 0.085), (0.245, 0.60, 0.072)]:
            c2 = np.array([1.12 * s + tx, -0.87, tz], np.float32)
            dd = np.sqrt((P[:, 0] - c2[0]) ** 2 + (P[:, 2] - c2[2]) ** 2)
            col = mix(col, pink[None, :], 0.95 * smoothstep(tr, tr * 0.75, dd) * front)

    # ---------------- pink lining in the ears (inner face of the plate)
    ax = np.abs(P[:, 0])
    from plush.body import EAR_POLY
    from sdf_lib import poly2d
    d_ear = poly2d(ax, P[:, 2], EAR_POLY)
    ear_m = (smoothstep(0.02, -0.02, d_ear) * smoothstep(0.125, 0.085, P[:, 1]))   # inner face only
    rim = smoothstep(0.075, 0.030, np.abs(d_ear)) * ear_m
    cup = smoothstep(-0.030, -0.075, d_ear) * ear_m
    col = mix(col, pink[None, :] * 0.94, 0.8 * cup)
    col = mix(col, hex2rgb(TEAL_DARK)[None, :], 0.45 * rim)
    # a stitched pink edge line along the ear rim
    edge_l = smoothstep(0.020, 0.006, np.abs(P[:, 2] - (2.66 + 0.62 * (ax - 1.05)))) * smoothstep(1.10, 1.25, ax) * smoothstep(3.25, 3.05, P[:, 2])
    col = mix(col, hex2rgb(STITCH)[None, :], 0.22 * edge_l * smoothstep(0.16, 0.10, P[:, 1]))

    # ---------------- embroidered stars on the legs / arms / belly
    for s in (1, -1):
        # thigh: unwrap around a vertical axis
        t, ang, rad = _limb_coords(P, (1.06 * s, -0.10, 1.30), (1.10 * s, -0.10, 0.20))
        m = _mask(t, ang, rad, 0.02, 0.80, 0.55)
        st = star_field(ang * 0.55 + 3.0 * s, t * 1.30, cell=0.23, r_out=0.092, r_in=0.040, density=0.45, seed=s)
        col = mix(col, cream[None, :], 0.95 * st * m)
        # arm
        t, ang, rad = _limb_coords(P, (0.50 * s, -0.02, 1.95), (1.00 * s, -0.56, 1.10))
        m = _mask(t, ang, rad, 0.05, 0.90, 0.25)
        st = star_field(ang * 0.35 + 5.0 * s, t * 1.10, cell=0.19, r_out=0.082, r_in=0.035, density=0.45, seed=2 + s)
        col = mix(col, cream[None, :], 0.95 * st * m)
    # belly stars
    belly_m = smoothstep(1.30, 1.45, P[:, 2]) * smoothstep(1.95, 1.80, P[:, 2]) * smoothstep(-0.30, -0.55, P[:, 1])
    st = star_field(P[:, 0] * 1.1, P[:, 2] * 1.1, cell=0.22, r_out=0.094, r_in=0.040, density=0.45, seed=9.0)
    col = mix(col, cream[None, :], 0.9 * st * belly_m)

    # ---------------- stitched seams: dashed line down the front centre + round the limbs
    def stitches(mask_line, along, spacing=0.085, width=0.012, phase=0.0):
        dash = smoothstep(0.45, 0.30, np.abs(((along / spacing + phase) % 1.0) - 0.5))
        return mask_line * dash * smoothstep(width, 0.0, np.abs(((along / spacing + phase + 0.25) % 1.0) - 0.5) * 0 + 0.0) + mask_line * dash

    # centre seam from the chest to the nappy
    line = smoothstep(0.020, 0.006, np.abs(P[:, 0])) * smoothstep(1.05, 1.15, P[:, 2]) * smoothstep(1.95, 1.85, P[:, 2])
    dd = smoothstep(0.55, 0.75, np.abs(P[:, 1] + 0.55))
    dash = smoothstep(0.42, 0.22, np.abs(((P[:, 2] / 0.075) % 1.0) - 0.5))
    col = mix(col, stitch[None, :], 0.85 * line * dash * dd)
    # side seams on the arms (outer silhouette)
    for s in (1, -1):
        t, ang, rad = _limb_coords(P, (0.50 * s, -0.02, 1.95), (1.00 * s, -0.56, 1.10))
        m = smoothstep(0.05, 0.0, np.abs(ang - np.pi / 2 * s)) * smoothstep(0.25, 0.18, rad) * \
            smoothstep(0.10, 0.20, t) * smoothstep(0.95, 0.80, t)
        dash = smoothstep(0.40, 0.20, np.abs(((t / 0.085) % 1.0) - 0.5))
        col = mix(col, stitch[None, :], 0.9 * m * dash)
        # leg side seam
        t, ang, rad = _limb_coords(P, (1.06 * s, -0.10, 1.30), (1.10 * s, -0.10, 0.30))
        m = smoothstep(0.10, 0.0, np.abs(np.abs(ang) - np.pi / 2)) * smoothstep(0.60, 0.50, rad) * \
            smoothstep(0.05, 0.15, t) * smoothstep(0.85, 0.70, t)
        dash = smoothstep(0.40, 0.20, np.abs(((t / 0.08) % 1.0) - 0.5))
        col = mix(col, stitch[None, :], 0.9 * m * dash)

    return np.clip(col, 0, 1).astype(np.float32)
