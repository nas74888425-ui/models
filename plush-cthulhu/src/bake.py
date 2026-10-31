"""Bake a stylised albedo into vertex colours from the SDF primitive list."""
import numpy as np


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def prim_dist(rec, V):
    """distance from every vertex to every primitive -> (nprim, nvert)"""
    d = []
    for pr in rec:
        kind = pr[0]
        if kind == 'ell':
            c = np.asarray(pr[1], np.float32)
            r = np.asarray(pr[2], np.float32)
            q = (V - c) / r
            d.append((np.sqrt((q * q).sum(1)) - 1.0) * float(r.mean()))
        elif kind == 'sph':
            c = np.asarray(pr[1], np.float32)
            d.append(np.linalg.norm(V - c, axis=1) - pr[2])
        elif kind == 'rc':
            a = np.asarray(pr[1], np.float32)
            b = np.asarray(pr[2], np.float32)
            r1, r2 = pr[3], pr[4]
            ba = b - a
            l2 = float(ba @ ba)
            rr = r1 - r2
            a2 = l2 - rr * rr
            il2 = 1.0 / l2
            pa = V - a
            y = pa @ ba
            z = y - l2
            xv = pa * l2 - y[:, None] * ba
            x2 = (xv * xv).sum(1)
            y2 = y * y * l2
            z2 = z * z * l2
            k = np.sign(rr) * rr * rr * x2
            d1 = np.sqrt(x2 + z2) * il2 - r2
            d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
            d2 = np.sqrt(x2 + y2) * il2 - r1
            out = np.where(np.sign(z) * a2 * z2 > k, d1, d3)
            out = np.where(np.sign(y) * a2 * y2 < k, d2, out)
            d.append(out)
    return np.stack(d, 0)


def noise(p, scale=1.0, seed=0):
    """cheap 3-octave trig noise in [0,1]"""
    x, y, z = p[:, 0] * scale, p[:, 1] * scale, p[:, 2] * scale
    n = (np.sin(x * 2.1 + seed) * np.cos(y * 2.7 - seed * 0.7) * np.sin(z * 1.9 + 0.4)
         + 0.5 * np.sin(x * 4.3 - 0.3) * np.cos(z * 4.1 + seed)
         + 0.25 * np.sin(y * 7.9 + 1.1) * np.cos(x * 8.3))
    return np.clip(0.5 + 0.45 * n, 0, 1)


def bake(V, rec, cols, eps=0.06, eps_wide=0.30, wide_cols=('#8E7FB5', '#9B87C4'),
         spot=0.60, spot_eps=0.12):
    Vm = V.copy()
    Vm[:, 0] = np.abs(Vm[:, 0])        # primitives were recorded on the +x half
    D = prim_dist(rec, Vm)
    C = np.stack([hex2rgb(c) for c in cols], 0)          # (nprim,3)
    m = D.min(0)
    epsv = np.array([eps_wide if c in wide_cols else eps for c in cols], np.float32)[:, None]
    # colour-only spot primitives get a soft edge
    for i, c in enumerate(cols):
        if c == '#7D6FA8':
            epsv[i] = 0.11
    W = np.exp(-(D - m) / epsv)
    W /= W.sum(0, keepdims=True)
    col = W.T @ C
    # mottled spots on the skin: darken where the noise is high
    n = noise(V, 0.55, 0.3)
    skinish = (col[:, 1] > col[:, 2] * 0.85) & (col[:, 1] > col[:, 0] * 1.1)
    s = np.clip((n - spot) / spot_eps, 0, 1)[:, None] * skinish[:, None]
    spot_col = hex2rgb('#4F9DA6')
    col = col * (1 - 0.5 * s) + spot_col * (0.5 * s)
    return col
