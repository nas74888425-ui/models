"""Tiny numpy software rasterizer for fast shape previews (z-buffer + lambert + matcap-ish)."""
import numpy as np
from PIL import Image


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f])  # rows: cam basis


def render(objs, eye, target, up=(0, 0, 1), W=420, H=600, fov=32.0, bg=(0.78, 0.78, 0.79),
           light=(0.25, -0.85, 0.55)):
    R = look_at(eye, target, up)
    eye = np.asarray(eye, float)
    fpx = (H / 2) / np.tan(np.radians(fov) / 2)
    img = np.tile(np.array(bg, np.float32), (H, W, 1))
    zbuf = np.full((H, W), 1e9, np.float32)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    for o in objs:
        V = o['V']
        F = o['F']
        col = np.array(o.get('color', (0.5, 0.8, 0.75)), np.float32)
        vcol = o.get('vcol', None)
        pc = (V - eye) @ R.T                     # camera space, -z forward
        z = -pc[:, 2]
        keep = z > 0.05
        sx = W / 2 + fpx * pc[:, 0] / z
        sy = H / 2 - fpx * pc[:, 1] / z
        tri = F[keep[F].all(1)]
        # face normals in world
        v0, v1, v2 = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        ln = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        n = n / ln
        # shade
        diff = np.clip(n @ L, 0, None)
        ctr = v0.mean(0)
        Vd = np.asarray(eye) - ctr
        Vd = Vd / (np.linalg.norm(Vd) + 1e-9)
        rim = np.clip(1.0 - np.abs(n @ Vd), 0, 1)
        shade = 0.42 + 0.72 * diff + 0.30 * diff ** 8
        if vcol is not None:
            base = vcol[tri].mean(1)
        else:
            base = np.repeat(col[None, :], len(tri), 0)
        fcol = np.clip(base * shade[:, None] + 0.05 * (rim ** 8)[:, None], 0, 1)
        px = np.stack([sx, sy], 1)
        zt = z[tri]
        for t in range(len(tri)):
            a, b, c = px[tri[t, 0]], px[tri[t, 1]], px[tri[t, 2]]
            x0 = int(max(0, np.floor(min(a[0], b[0], c[0]))))
            x1 = int(min(W - 1, np.ceil(max(a[0], b[0], c[0]))))
            y0 = int(max(0, np.floor(min(a[1], b[1], c[1]))))
            y1 = int(min(H - 1, np.ceil(max(a[1], b[1], c[1]))))
            if x1 < x0 or y1 < y0:
                continue
            xs = np.arange(x0, x1 + 1) + 0.5
            ys = np.arange(y0, y1 + 1) + 0.5
            X, Y = np.meshgrid(xs, ys)
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-12:
                continue
            w0 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
            w1 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
            w2 = 1 - w0 - w1
            m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            zz = w0 * zt[t, 0] + w1 * zt[t, 1] + w2 * zt[t, 2]
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            upd = m & (zz < sub)
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = fcol[t]
    img = np.clip(img, 0, 1) ** (1 / 1.0)
    return (img * 255).astype(np.uint8)


def save(arr, path):
    Image.fromarray(arr).save(path)
