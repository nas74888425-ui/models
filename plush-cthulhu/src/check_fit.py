"""Silhouette fit against the single reference view (orthographic front)."""
import sys
sys.path.insert(0, '/home/user/plush_cthulhu/src')
import numpy as np
from PIL import Image
import trimesh
from scipy import ndimage as ndi

REF = np.array(Image.open('/home/user/plush_cthulhu/previews/mask.png')) > 128


def project(pts, ref_shape, scale, ou, ov):
    img = np.zeros(ref_shape, bool)
    xi = np.round(ou + scale * pts[:, 0]).astype(int)
    yi = np.round(ov - scale * pts[:, 2]).astype(int)
    ok = (xi >= 0) & (xi < ref_shape[1]) & (yi >= 0) & (yi < ref_shape[0])
    img[yi[ok], xi[ok]] = True
    return ndi.binary_fill_holes(ndi.binary_closing(img, np.ones((7, 7))))


def fit(V, F, verbose=True, save=''):
    m = trimesh.Trimesh(V, F, process=False)
    pts, _ = trimesh.sample.sample_surface_even(m, 120000)
    best = (-1, None, None)
    for s in np.arange(92, 109, 2.0):
        for ou in np.arange(300, 400, 6):
            for ov in np.arange(390, 460, 6):
                img = project(pts, REF.shape, s, ou, ov)
                iou = (REF & img).sum() / max(1, (REF | img).sum())
                if iou > best[0]:
                    best = (iou, (s, ou, ov), img)
    iou, (s, ou, ov), img = best
    if verbose:
        print('IoU %.3f   scale=%.0f  origin=(%.0f,%.0f)' % (iou, s, ou, ov))
        for y in range(30, 450, 30):
            a = np.nonzero(REF[y])[0]
            b = np.nonzero(img[y])[0]
            ra = '%3d %3d' % (a.min(), a.max()) if len(a) else '  -   -'
            rb = '%3d %3d' % (b.min(), b.max()) if len(b) else '  -   -'
            print('  y=%3d ref %s | model %s' % (y, ra, rb))
    if save:
        vis = np.zeros(REF.shape + (3,), np.uint8)
        vis[REF & img] = (70, 190, 170)
        vis[REF & ~img] = (230, 80, 60)
        vis[~REF & img] = (60, 90, 230)
        vis[~REF & ~img] = (32, 32, 34)
        Image.fromarray(vis).save(save)
    return iou


if __name__ == '__main__':
    d = np.load(sys.argv[1])
    print(fit(d['V'], d['F'], save='/home/user/plush_cthulhu/previews/diff.png'))
