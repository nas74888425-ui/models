"""Vertex-colour painting of the felt doll: flat felt colours + embroidered patterns.

Everything is analytic (no textures, no UVs) so it survives export to glTF as COLOR_0
and can be animated/deformed freely.
"""
import numpy as np

# ------------------------------------------------------------------ palette (sRGB)
TEAL = '#6FA595'
TEAL_DARK = '#4E8175'
TEAL_LIGHT = '#86BCA9'
PINK = '#D9A6BC'
PINK_DEEP = '#C58AA6'
CREAM = '#E9E1CD'
CREAM_STAR = '#EBD394'
STITCH = '#C98B45'
BROWN_IRIS = '#8A5B22'
IRIS_RIM = '#4A2C0B'
BLACK = '#120C06'
WHITE = '#F2EFE7'


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def mix(a, b, t):
    """a,b: (N,3) or hex str; t: (N,)"""
    a = np.asarray(a if not isinstance(a, str) else np.tile(hex2rgb(a), (len(t), 1)), np.float32)
    b = np.asarray(b if not isinstance(b, str) else np.tile(hex2rgb(b), (len(t), 1)), np.float32)
    if a.ndim == 1:
        a = np.tile(a, (len(t), 1))
    if b.ndim == 1:
        b = np.tile(b, (len(t), 1))
    t = np.clip(t, 0, 1)[:, None]
    return a * (1 - t) + b * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def hash3(p, seed=0.0):
    """deterministic pseudo-random in [0,1) per point"""
    x = np.sin(p[:, 0] * 127.1 + p[:, 1] * 311.7 + p[:, 2] * 74.7 + seed) * 43758.5453
    return x - np.floor(x)


def felt_noise(p, scale=6.0, seed=0.0):
    """soft fibrous variation of the felt tone"""
    x, y, z = p[:, 0] * scale, p[:, 1] * scale, p[:, 2] * scale
    n = (np.sin(x * 1.7 + seed) * np.cos(y * 2.3 - seed * 0.3) * np.sin(z * 1.9 + 0.7)
         + 0.55 * np.sin(x * 3.9 - 0.4) * np.cos(z * 3.1 + seed * 0.5)
         + 0.3 * np.sin(y * 7.3 + 1.3) * np.cos(x * 6.7))
    return np.clip(0.5 + 0.4 * n, 0, 1)


# ------------------------------------------------------------------ star helper
def star_field(u, v, cell=0.62, r_out=0.19, r_in=0.082, density=0.36):
    """crisp 5-point stars scattered on a 2D parameter plane; returns coverage in [0,1]"""
    from sdf_lib import star2d
    out = np.zeros(u.shape, np.float32)
    iu = np.floor(u / cell)
    iv = np.floor(v / cell)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            ci = iu + du
            cv = iv + dv
            h1 = np.sin(ci * 12.9898 + cv * 78.233) * 43758.5453
            h1 = h1 - np.floor(h1)
            h2 = np.sin(ci * 39.3467 + cv * 11.1351) * 43758.5453
            h2 = h2 - np.floor(h2)
            h3 = np.sin(ci * 21.7893 + cv * 63.7264) * 43758.5453
            h3 = h3 - np.floor(h3)
            keep = h3 > density
            if not np.any(keep):
                continue
            cu = (ci + 0.5 + (h1 - 0.5) * 0.55) * cell
            cvv = (cv + 0.5 + (h2 - 0.5) * 0.55) * cell
            d = star2d(u - cu, v - cvv, r_out * (0.8 + 0.45 * h1), r_in * (0.8 + 0.45 * h2),
                       rot=h1 * 6.283)
            out = np.maximum(out, smoothstep(0.012, -0.012, d) * keep)
    return np.clip(out, 0, 1)


# ------------------------------------------------------------------ paint: body
def paint_body(V, F=None):
    """returns (N,3) colours for the body/limb mesh in model space"""
    N = len(V)
    pm = np.stack([np.abs(V[:, 0]), V[:, 1], V[:, 2]], -1)
    x, y, z = pm[:, 0], pm[:, 1], pm[:, 2]

    base = np.tile(hex2rgb(TEAL), (N, 1))
    # fibrous felt tone variation
    n = felt_noise(V, 5.0, 0.7)
    base = mix(base, TEAL_LIGHT, 0.35 * (n - 0.5) + 0.18)
    base = mix(base, TEAL_DARK, 0.40 * smoothstep(0.55, 0.95, 1 - n))
    # airbrushed pink mottling on the felt (cheeks, shoulders, knees)
    mot = felt_noise(V, 3.4, 3.3)
    soft_pink = smoothstep(0.66, 0.84, mot)
    col0 = mix(base, PINK, 0.16 * soft_pink)

    col = col0

    # ---- pink felt spots on the head: explicit patches (like the reference)
    SPOTS = [
        (0.00, -0.55, 3.62, 0.30), (0.42, -0.30, 3.66, 0.19), (-0.40, -0.34, 3.62, 0.17),
        (0.20, -0.72, 3.30, 0.21), (-0.24, -0.66, 3.36, 0.19), (0.72, -0.34, 3.34, 0.16),
        (-0.70, -0.36, 3.30, 0.15), (0.94, 0.02, 3.10, 0.13), (-0.94, 0.02, 3.12, 0.13),
        (0.52, 0.30, 3.56, 0.15), (-0.52, 0.32, 3.54, 0.14), (0.00, 0.42, 3.50, 0.17),
        (0.30, -0.86, 2.98, 0.15), (-0.32, -0.88, 2.96, 0.14), (1.02, -0.30, 2.86, 0.11),
        (-1.04, -0.28, 2.84, 0.11), (0.34, 0.34, 3.10, 0.12), (-0.36, 0.36, 3.08, 0.12),
    ]
    for (sx, sy, sz, sr) in SPOTS:
        dd = np.sqrt((V[:, 0] - sx) ** 2 + (V[:, 1] - sy) ** 2 + (V[:, 2] - sz) ** 2)
        edge = 0.02 + 0.06 * hash3(V, abs(sx) * 7.0)
        col = mix(col, PINK, 0.96 * smoothstep(sr, sr - 0.018, dd))
    # ---- softer random mottling on the cranium
    head_mask = smoothstep(2.60, 2.72, z) * smoothstep(1.36, 1.10, np.abs(V[:, 0]))
    rnd = hash3(np.round(V * 22.0) / 22.0, 3.1)
    spot = smoothstep(0.985, 0.995, rnd)
    # regular pattern of round patches instead of pure noise: use a 3D wobble field
    wob = (np.sin(V[:, 0] * 9.0) * np.sin(V[:, 1] * 8.0 + 1.0) * np.sin(V[:, 2] * 7.5))
    spots = smoothstep(0.62, 0.80, wob) * head_mask
    col = mix(col, PINK, 0.55 * spots)

    # ---- big pink patch on the crown
    crown = smoothstep(3.55, 3.42, z) * smoothstep(0.55, 0.35, np.sqrt(V[:, 0] ** 2 + (V[:, 1] + 0.1) ** 2))
    col = mix(col, PINK_DEEP, 0.9 * crown)

    # ---- cheek blush
    for s in (1, -1):
        d = np.sqrt((V[:, 0] - 0.88 * s) ** 2 + (V[:, 1] + 0.92) ** 2 + (V[:, 2] - 2.28) ** 2)
        col = mix(col, PINK, 0.55 * smoothstep(0.55, 0.18, d))

    # ---- inner ear: pink on the front/inner face of the ear
    ear_zone = smoothstep(1.05, 1.25, x) * smoothstep(2.55, 2.75, z) * smoothstep(0.05, -0.15, y)
    col = mix(col, PINK, 0.9 * ear_zone)

    # ---- cream stars embroidered on arms and legs (cylindrical unwrap around each limb)
    def limb_stars(A, B, r_mean, cell, r_out, r_in, t0=0.12, t1=0.95, seed=0.0):
        A = np.array(A, np.float32); B = np.array(B, np.float32)
        ab = B - A
        L = np.linalg.norm(ab)
        ab = ab / L
        tmp = np.array([0.0, 0.0, 1.0], np.float32)
        if abs(ab @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], np.float32)
        u1 = np.cross(ab, tmp); u1 /= np.linalg.norm(u1)
        u2 = np.cross(ab, u1)
        rel = V - A
        t = rel @ ab / L
        perp = rel - (t[:, None] * L) * ab
        ang = np.arctan2(perp @ u2, perp @ u1)
        # coverage mask: only near this limb
        rad = np.linalg.norm(perp, axis=1)
        on = smoothstep(r_mean + 0.16, r_mean + 0.02, rad) * smoothstep(t0 - 0.05, t0 + 0.08, t) * smoothstep(t1 + 0.08, t1 - 0.05, t)
        return star_field(ang * r_mean + seed, t * L, cell=cell, r_out=r_out, r_in=r_in, density=0.5) * on

    for s_ in (1, -1):
        cov = limb_stars((0.60 * s_, -0.20, 0.78), (1.12 * s_, -1.92, 0.52), 0.46,
                         cell=0.34, r_out=0.10, r_in=0.043, t0=0.16, t1=0.90, seed=0.0 if s_ > 0 else 1.1)
        col = mix(col, CREAM_STAR, 1.0 * cov)
        cov2 = limb_stars((0.88 * s_, -0.16, 1.68), (1.38 * s_, -1.18, 0.82), 0.33,
                          cell=0.30, r_out=0.082, r_in=0.034, t0=0.22, t1=0.92, seed=2.3 if s_ > 0 else 3.7)
        col = mix(col, CREAM_STAR, 0.98 * cov2)

    # ---- paw pads: big pink heel pad + 4 toe pads on the foot fronts
    for s in (1, -1):
        X = V[:, 0] * s
        front = smoothstep(-1.95, -2.10, V[:, 1])
        # main pad
        dpad = np.sqrt(((X - 1.10) / 1.0) ** 2 + ((V[:, 2] - 0.40) / 0.8) ** 2)
        pad = smoothstep(0.42, 0.33, dpad) * front
        col = mix(col, PINK, 0.95 * pad)
        for (tx, tz, tr) in [(0.80, 0.66, 0.155), (1.02, 0.74, 0.165), (1.24, 0.72, 0.16), (1.44, 0.60, 0.14)]:
            dt = np.sqrt((X - tx) ** 2 + ((V[:, 2] - tz) / 1.0) ** 2)
            col = mix(col, PINK_DEEP, 0.9 * smoothstep(tr * 1.25, tr * 0.8, dt) * front)
        # small finger pads on the paw
        dhand = np.sqrt(((X - 1.32) / 1.0) ** 2 + ((V[:, 2] - 0.72) / 0.9) ** 2 + ((V[:, 1] + 1.2) / 1.4) ** 2)
        col = mix(col, PINK, 0.75 * smoothstep(0.30, 0.20, dhand))

    # ---- embroidered eyebrows: small thin stitches just above each eye
    for s_ in (1, -1):
        ex, ez = 0.52 * s_, 3.02
        dx = (V[:, 0] - ex) / 0.26
        dz = (V[:, 2] - ez) / 0.10
        rr = np.sqrt(dx ** 2 + dz ** 2)
        band = smoothstep(0.30, 0.14, np.abs(rr - 0.85))
        front = smoothstep(-0.55, -0.80, V[:, 1])
        col = mix(col, '#5E4126', 0.8 * band * front)

    # ---- pink felt tips on the tentacles (last ~25% of every strand)
    from plush.body import TENTACLES
    for line, radii in TENTACLES:
        for s_ in (1, -1):
            pts = np.array([[p[0] * s_, p[1], p[2]] for p in line], np.float32)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            L = cum[-1]
            dmin = np.full(N, 1e9, np.float32)
            tbest = np.zeros(N, np.float32)
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                ab = b - a
                t = np.clip(((V - a) @ ab) / (ab @ ab), 0, 1)
                d = np.linalg.norm(V - (a + t[:, None] * ab), axis=1)
                closer = d < dmin
                dmin = np.where(closer, d, dmin)
                tbest = np.where(closer, (cum[i] + t * seg[i]) / max(L, 1e-6), tbest)
            near = smoothstep(0.24, 0.15, dmin)
            tip = smoothstep(0.80, 0.90, tbest)
            col = mix(col, PINK, 0.95 * near * tip)

    # ---- whip stitches: rings near the wrists/ankles + ear rims
    dash_field = (0.5 + 0.5 * np.sign(np.sin(np.arctan2(V[:, 1] + 1.5, V[:, 2] - 0.6) * 34)))
    ankle = smoothstep(0.06, 0.02, np.abs(z - 1.02)) * smoothstep(0.55, 0.75, np.abs(V[:, 0]))
    col = mix(col, STITCH, 0.85 * ankle * dash_field)
    dash2 = (0.5 + 0.5 * np.sign(np.sin(np.arctan2(V[:, 1] + 0.85, V[:, 2] - 1.2) * 34)))
    wrist = smoothstep(0.07, 0.02, np.abs(z - 1.12)) * smoothstep(1.15, 1.32, np.abs(V[:, 0]))
    col = mix(col, STITCH, 0.8 * wrist * dash2)
    return col
