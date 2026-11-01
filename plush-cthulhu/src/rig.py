"""Armature + skinning for the plush doll.

Bones (model space, Z up, character faces -Y):
  root -> hips -> spine -> chest -> neck -> head -> (earL/R, tent1L..4R, jaw)
  hips -> thighL/R -> shinL/R -> footL/R
  chest -> upperarmL/R -> forearmL/R -> handL/R
  chest -> wingL/R
  head  -> eyelidLidL/R (deform helpers for the eyes), lidL/R plates
"""
import numpy as np
import bpy
from plush.body import _TENT_RAW
from mathutils import Vector, Matrix

# (name, head, tail, parent, deform, connected)
BONES = [
    ('root',        (0, 0, 0.0),       (0, 0, 0.40),    None,       False, False),
    ('hips',        (0, 0, 0.55),      (0, 0, 1.00),    'root',     True,  False),
    ('spine',       (0, 0, 1.00),      (0, 0, 1.45),    'hips',     True,  True),
    ('chest',       (0, 0, 1.45),      (0, 0, 1.85),    'spine',    True,  True),
    ('neck',        (0, 0, 1.85),      (0, 0, 2.10),    'chest',    True,  True),
    ('head',        (0, 0, 2.10),      (0, 0, 3.30),    'neck',     True,  True),
    ('jaw',         (0, -0.70, 2.30),  (0, -1.15, 2.15), 'head',    True,  False),
    # ears (thin plates growing out of the head)
    ('earL',        (0.95, 0.16, 2.70), (1.55, 0.22, 3.40), 'head',  True,  False),
    ('earR',        (-0.95, 0.16, 2.70), (-1.55, 0.22, 3.40), 'head', True,  False),
    # wings (behind the shoulders)
    ('wingL',       (0.55, 0.36, 2.20), (1.65, 0.42, 1.75), 'chest', True,  False),
    ('wingR',       (-0.55, 0.36, 2.20), (-1.65, 0.42, 1.75), 'chest', True, False),
    # arms: shoulder -> elbow -> wrist -> fist resting in front of the thigh
    ('upperarmL',   (0.50, -0.02, 1.90), (0.98, -0.60, 1.52), 'chest', True, False),
    ('forearmL',    (0.98, -0.60, 1.52), (1.02, -0.88, 1.22), 'upperarmL', True, True),
    ('handL',       (1.02, -0.88, 1.22), (1.02, -1.12, 1.10), 'forearmL', True, True),
    ('upperarmR',   (-0.50, -0.02, 1.90), (-0.98, -0.60, 1.52), 'chest', True, False),
    ('forearmR',    (-0.98, -0.60, 1.52), (-1.02, -0.88, 1.22), 'upperarmR', True, True),
    ('handR',       (-1.02, -0.88, 1.22), (-1.02, -1.12, 1.10), 'forearmR', True, True),
    # eyes and eyelids
    ('eyeL',        (0.475, -1.00, 3.12), (0.475, -1.30, 3.12), 'head', True, False),
    ('eyeR',        (-0.475, -1.00, 3.12), (-0.475, -1.30, 3.12), 'head', True, False),
    ('lidL',        (0.475, -1.00, 3.12), (0.475, -1.35, 3.12), 'head', True, False),
    ('lidR',        (-0.475, -1.00, 3.12), (-0.475, -1.35, 3.12), 'head', True, False),
    # legs: thigh -> short shin -> foot, soles facing the camera
    ('thighL',      (0.44, 0.00, 0.62), (1.06, -0.20, 0.52), 'hips', True, False),
    ('shinL',       (1.06, -0.20, 0.52), (1.10, -0.50, 0.44), 'thighL', True, True),
    ('footL',       (1.10, -0.50, 0.44), (1.10, -0.90, 0.40), 'shinL', True, True),
    ('thighR',      (-0.44, 0.00, 0.62), (-1.06, -0.20, 0.52), 'hips', True, False),
    ('shinR',       (-1.06, -0.20, 0.52), (-1.10, -0.50, 0.44), 'thighR', True, True),
    ('footR',       (-1.10, -0.50, 0.44), (-1.10, -0.90, 0.40), 'shinR', True, True),
]

# tentacle bones follow the SDF centre-lines of the 8 tentacles
TENT_BONES = []
for _i, (_pts, _r) in enumerate(_TENT_RAW, start=1):
    for side, tag in ((1, 'L'), (-1, 'R')):
        pts = [(float(x) * side, float(y), float(z)) for (x, y, z) in _pts]
        for j in range(len(pts) - 1):
            name = 'tent%d%s_%d' % (_i, tag, j + 1)
            parent = 'head' if j == 0 else 'tent%d%s_%d' % (_i, tag, j)
            TENT_BONES.append((name, pts[j], pts[j + 1], parent, True, j > 0))
BONES = BONES + TENT_BONES


def create_rig(name='PlushRig'):
    arm = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.edit_bones
    for (nm, h, t, par, deform, conn) in BONES:
        b = eb.new(nm)
        b.head = Vector(h)
        b.tail = Vector(t)
        b.use_deform = deform
        if par:
            b.parent = eb[par]
            b.use_connect = conn
    # rolls: keep things simple and predictable
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = None
    return rig


# ---------------------------------------------------------------- skin weights
def _seg_dist(P, a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _bone_segments(only_deform=True):
    segs = []
    for (nm, h, t, par, deform, conn) in BONES:
        if only_deform and not deform:
            continue
        segs.append((nm, h, t))
    return segs


def smooth_weights(W, F, iters=4, lam=0.55):
    """average weights across mesh edges so the deformation has no hard seams"""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = np.concatenate([e, e[:, ::-1]], 0)
    n = W.shape[0]
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float32) + 1e-6
    for _ in range(iters):
        acc = np.zeros_like(W)
        for b in range(W.shape[1]):
            acc[:, b] = np.bincount(e[:, 0], weights=W[e[:, 1], b], minlength=n)
        acc /= deg[:, None]
        W = W * (1 - lam) + acc * lam
    return W


def skin_part(ob, rig, allowed=None, extra_bias=None, sigma=1.6, smooth=4):
    """envelope skinning: w = exp(-(d/sigma)^2) over the allowed bones"""
    me = ob.data
    P = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get('co', P)
    P = P.reshape(-1, 3)
    segs = [(nm, h, t) for (nm, h, t) in _bone_segments() if (allowed is None or nm in allowed)]
    if not segs:
        return
    D = np.stack([_seg_dist(P, h, t) for (nm, h, t) in segs], 1)     # (N, B)
    if extra_bias:
        for nm, bias in extra_bias.items():
            for i, (snm, _, _) in enumerate(segs):
                if snm == nm:
                    D[:, i] *= bias
    W = np.exp(-(D / sigma) ** 2)
    W /= W.sum(1, keepdims=True)
    if smooth:
        me = ob.data
        me.calc_loop_triangles()
        tris = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get('vertices', tris)
        tris = tris.reshape(-1, 3)
        W = smooth_weights(W, tris.astype(np.int64), smooth)
        W[W < 1e-4] = 0.0
    # keep the 3 strongest influences per vertex (glTF friendly)
    if W.shape[1] > 3:
        idx = np.argsort(-W, 1)[:, 3:]
        np.put_along_axis(W, idx, 0.0, 1)
    W[W < 1e-4] = 0.0
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    W = W / s
    groups = {}
    for i, (nm, _, _) in enumerate(segs):
        groups[nm] = ob.vertex_groups.new(name=nm) if nm not in ob.vertex_groups else ob.vertex_groups[nm]
    for i, (nm, _, _) in enumerate(segs):
        w = W[:, i]
        nz = np.nonzero(w > 0)[0]
        # group by rounded weight to cut the number of API calls
        for q in np.unique(np.round(w[nz], 3)):
            if q <= 0:
                continue
            ids = nz[np.round(w[nz], 3) == q]
            groups[nm].add(ids.tolist(), float(q), 'REPLACE')
    mod = ob.modifiers.new('Armature', 'ARMATURE')
    mod.object = rig
    mod.use_deform_preserve_volume = False
    ob.parent = rig
    ob.matrix_parent_inverse = Matrix.Identity(4)
