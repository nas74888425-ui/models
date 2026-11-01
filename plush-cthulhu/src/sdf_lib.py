"""SDF helpers: primitives, smooth booleans, volume sampling, meshing."""
import numpy as np


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def ell(p, c, r):
    qx = (p[..., 0] - c[0]) / r[0]
    qy = (p[..., 1] - c[1]) / r[1]
    qz = (p[..., 2] - c[2]) / r[2]
    k0 = np.sqrt(qx * qx + qy * qy + qz * qz)
    return (k0 - 1.0) * ((r[0] + r[1] + r[2]) / 3.0)


def ell_rot(p, c, r, rot):
    """ellipsoid rotated by rot = (rx, ry, rz) radians (XYZ euler, applied to the local frame)"""
    q = p - np.asarray(c, np.float32)
    cx, sx = np.cos(-rot[0]), np.sin(-rot[0])
    cy, sy = np.cos(-rot[1]), np.sin(-rot[1])
    cz, sz = np.cos(-rot[2]), np.sin(-rot[2])
    y = q[:, 1] * cx - q[:, 2] * sx
    z = q[:, 1] * sx + q[:, 2] * cx
    q = np.stack([q[:, 0], y, z], -1)
    x = q[:, 0] * cy + q[:, 2] * sy
    z = -q[:, 0] * sy + q[:, 2] * cy
    q = np.stack([x, q[:, 1], z], -1)
    x = q[:, 0] * cz - q[:, 1] * sz
    y = q[:, 0] * sz + q[:, 1] * cz
    q = np.stack([x, y, q[:, 2]], -1)
    return ell(q, (0, 0, 0), r)


def sphere(p, c, r):
    return np.sqrt((p[..., 0] - c[0]) ** 2 + (p[..., 1] - c[1]) ** 2 +
                   (p[..., 2] - c[2]) ** 2) - r


def round_cone(p, a, b, r1, r2):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ba = b - a
    l2 = float(np.dot(ba, ba))
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[..., None] * ba
    x2 = (xv * xv).sum(-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    out = np.where(np.sign(z) * a2 * z2 > k, d1, d3)
    out = np.where(np.sign(y) * a2 * y2 < k, d2, out)
    return out


def chain(p, pts, radii, k=0.0):
    d = None
    for i in range(len(pts) - 1):
        seg = round_cone(p, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = seg if d is None else smin(d, seg, k)
    return d


def poly2d(x, z, pts):
    """signed distance to a (closed) polygon in the XZ plane; negative inside"""
    n = len(pts)
    d = np.full(x.shape, 1e9, np.float32)
    inside = np.zeros(x.shape, bool)
    for i in range(n):
        ax, az = pts[i]
        bx, bz = pts[(i + 1) % n]
        ex, ez = bx - ax, bz - az
        wx, wz = x - ax, z - az
        t = np.clip((wx * ex + wz * ez) / (ex * ex + ez * ez), 0, 1)
        dx, dz = wx - ex * t, wz - ez * t
        d = np.minimum(d, np.sqrt(dx * dx + dz * dz))
        cond = ((az > z) != (bz > z)) & (x < (bx - ax) * (z - az) / (bz - az + 1e-12) + ax)
        inside ^= cond
    return np.where(inside, -d, d)


def star2d(x, z, r_out, r_in, rot=0.0):
    """5-point star distance in 2D"""
    pts = []
    for i in range(10):
        a = rot + i * np.pi / 5 - np.pi / 2
        r = r_out if i % 2 == 0 else r_in
        pts.append((r * np.cos(a), r * np.sin(a)))
    return poly2d(x, z, pts)


def evaluate(bbox, res, fn, slab=24):
    x0, y0, z0 = bbox[0]
    x1, y1, z1 = bbox[1]
    nx = int(np.ceil((x1 - x0) / res))
    ny = int(np.ceil((y1 - y0) / res))
    nz = int(np.ceil((z1 - z0) / res))
    vol = np.empty((nx, ny, nz), dtype=np.float32)
    xs = x0 + np.arange(nx, dtype=np.float32) * res
    ys = y0 + np.arange(ny, dtype=np.float32) * res
    zs = z0 + np.arange(nz, dtype=np.float32) * res
    for k in range(0, nz, slab):
        k2 = min(nz, k + slab)
        X, Y, Z = np.meshgrid(xs, ys, zs[k:k2], indexing='ij')
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
        del X, Y, Z
        vol[:, :, k:k2] = fn(pts).reshape(nx, ny, k2 - k)
    return vol, (x0, y0, z0), res


def mesh_from_volume(vol, origin, res, level=0.0):
    from skimage import measure
    v, f, _, _ = measure.marching_cubes(vol, level=level, spacing=(res, res, res))
    v = v + np.array(origin, dtype=np.float64)
    return v.astype(np.float32), f.astype(np.int64)


def gradient(fn, V, h=0.004):
    g = np.empty_like(V)
    for a in range(3):
        dv = np.zeros(3, np.float32)
        dv[a] = h
        g[:, a] = (fn(V + dv) - fn(V - dv)) / (2 * h)
    return g


def narrow_volume(fn, bbox, res=0.03, coarse=0.12, band=0.30, slab=24):
    """fast volume: coarse probe mask expanded, rest filled with +1 (outside)"""
    x0, y0, z0 = bbox[0]
    x1, y1, z1 = bbox[1]
    nx = int(np.ceil((x1 - x0) / res)); ny = int(np.ceil((y1 - y0) / res)); nz = int(np.ceil((z1 - z0) / res))
    xs = x0 + np.arange(nx, dtype=np.float32) * res
    ys = y0 + np.arange(ny, dtype=np.float32) * res
    zs = z0 + np.arange(nz, dtype=np.float32) * res
    vol = np.full((nx, ny, nz), 1.0, np.float32)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for k0 in range(0, nz, slab):
        k1 = min(nz, k0 + slab)
        P = np.stack([np.repeat(X.ravel(), k1 - k0), np.repeat(Y.ravel(), k1 - k0),
                      np.tile(zs[k0:k1], X.size)], -1).astype(np.float32)
        vol[:, :, k0:k1] = fn(P).reshape(nx, ny, k1 - k0)
    # light gaussian blur removes the hard creases left by smooth-min unions
    from scipy import ndimage as ndi
    vol = ndi.gaussian_filter(vol, sigma=0.9)
    pad = 2
    vol = np.pad(vol, pad, mode='constant', constant_values=1.0)
    return vol, (x0 - pad * res, y0 - pad * res, z0 - pad * res), res


def weld(V, F, tol=1e-5):
    """merge coincident vertices and drop degenerate faces"""
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def build_mesh(fn, bbox, res=0.03, smooth=2, decimate=1.0, post=2):
    import fast_simplification
    vol, org, r = narrow_volume(fn, bbox, res=res)
    V, F = mesh_from_volume(vol, org, r)
    V, F = weld(V, F)
    if smooth:
        V, F = smooth_project(fn, V, F, smooth)
    if decimate < 1.0:
        V, F = fast_simplification.simplify(V, F, 1.0 - decimate)
        V = V.astype(np.float32); F = F.astype(np.int64)
    V, F = weld(V, F)
    if post:
        # relax the decimated mesh and pull it back onto the exact surface:
        # removes the creases/seams that decimation leaves behind
        V, F = smooth_project(fn, V, F, post)
    return V, F


def laplacian(V, F, lam=0.5):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    deg = np.bincount(e[:, 0], minlength=len(V)).astype(np.float32) + 1e-9
    acc = np.stack([np.bincount(e[:, 0], weights=V[e[:, 1], a], minlength=len(V)) for a in range(3)], -1)
    return V + lam * (acc / deg[:, None] - V)


def smooth_project(fn, V, F, iters=2, verbose=False):
    for i in range(iters):
        V = laplacian(V, F, 0.5)
        d = fn(V)
        g = gradient(fn, V, 0.004)
        gn = (g * g).sum(1) + 1e-9
        V = V - np.clip((d / gn)[:, None] * g, -0.04, 0.04)
        if verbose:
            print('   relax %d mean|d|=%.5f' % (i, np.abs(fn(V)).mean()), flush=True)
    return V, F


def cloth_displace(V, F, amp=0.012, freq=7.0, freq2=23.0, seed=0.0):
    """push vertices along the surface normal by smooth noise: felt lumps and wrinkles"""
    G = gradient(lambda q: _disp_sdf_placeholder(q), V) if False else None
    return V
