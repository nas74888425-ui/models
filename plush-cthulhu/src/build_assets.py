"""Build all meshes (model space, 1 unit = 100 px of the reference) with baked vertex colours."""
import sys, time
sys.path.insert(0, '/home/user/plush_cthulhu/src')
import numpy as np
from sdf_lib import build_mesh
import plush.body as B
import plush.details as D
import plush.paint as P
from plush.wings import wing_sdf

OUT = '/home/user/plush_cthulhu/models'
BB_BODY = [(-1.95, -1.95, -0.05), (1.95, 1.30, 4.10)]


def save(name, V, F, C, extra=None):
    d = dict(V=V.astype(np.float32), F=F.astype(np.int64), C=np.clip(C, 0, 1).astype(np.float32))
    if extra:
        d.update(extra)
    np.savez_compressed('%s/%s.npz' % (OUT, name), **d)
    print('  %-8s verts=%6d tris=%6d' % (name, len(V), len(F)), flush=True)


t0 = time.time()
print('body…', flush=True)
V, F = build_mesh(B.body_sdf, BB_BODY, res=0.018, smooth=2, decimate=0.45, post=2)
save('body', V, F, P.paint_body(V))

for side, tag in ((1, 'R'), (-1, 'L')):
    c = B.EYE_C * np.array([side, 1, 1])
    bb = [(c[0] - 0.45, c[1] - 0.45, c[2] - 0.45), (c[0] + 0.45, c[1] + 0.45, c[2] + 0.45)]
    V, F = build_mesh(lambda p, side=side: D.eye_sdf(p, side), bb, res=0.008, smooth=1, decimate=0.35, post=1)
    save('eye' + tag, V, F, D.eye_colors(V, side), {'side': np.array([side])})

print('wings…', flush=True)
V, F = build_mesh(wing_sdf, [(-1.9, 0.0, 1.0), (1.9, 0.9, 2.6)], res=0.016, smooth=2, decimate=0.45, post=2)
save('wings', V, F, D.wing_colors(V))

print('diaper…', flush=True)
V, F = build_mesh(D.diaper_sdf, [(-1.25, -1.45, 0.10), (1.25, 0.95, 1.20)], res=0.014, smooth=2, decimate=0.45, post=2)
save('diaper', V, F, D.diaper_colors(V))

print('done %.1fs' % (time.time() - t0))
