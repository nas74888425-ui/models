"""Procedural keyframe animation for the plush doll: idle, laugh, cry, angry.

Each public function writes one Action (30 fps). All values are tuned for the
bone layout in rig.py (Z-up, character faces -Y).
"""
import math
import bpy
from mathutils import Euler

FPS = 30


# ---------------------------------------------------------------- helpers
def clear_pose(rig):
    for pb in rig.pose.bones:
        pb.rotation_mode = 'XYZ'
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)


def new_action(rig, name):
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    rig.animation_data_create()
    rig.animation_data.action = act
    if hasattr(act, 'slots'):
        slot = act.slots.new(id_type='OBJECT', name=name) if len(act.slots) == 0 else act.slots[0]
        rig.animation_data.action_slot = slot
    clear_pose(rig)
    return act


def key(rig, bone, frame, rot=None, loc=None, scale=None, interp='BEZIER'):
    pb = rig.pose.bones[bone]
    pb.rotation_mode = 'XYZ'
    if rot is not None:
        pb.rotation_euler = Euler([math.radians(a) for a in rot], 'XYZ')
        pb.keyframe_insert('rotation_euler', frame=frame)
    if loc is not None:
        pb.location = loc
        pb.keyframe_insert('location', frame=frame)
    if scale is not None:
        pb.scale = scale
        pb.keyframe_insert('scale', frame=frame)


def r(x=0.0, y=0.0, z=0.0):
    return (x, y, z)


def action_fcurves(act):
    """Blender 5.x keeps fcurves inside layer/strip/channelbag; 4.x on the action"""
    fcs = []
    if hasattr(act, 'layers'):
        for layer in act.layers:
            for strip in layer.strips:
                for cb in getattr(strip, 'channelbags', []):
                    fcs.extend(cb.fcurves)
    else:
        fcs = list(act.fcurves)
    return fcs


def finish(act, interp='BEZIER'):
    for fc in action_fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = interp
            kp.handle_left_type = kp.handle_right_type = 'AUTO_CLAMPED'


def bake_sine(rig, act, bone, channel, frames, freq, amp, phase=0.0, axis=None, base=None):
    """write a sampled sine on one channel of one bone"""
    pb = rig.pose.bones[bone]
    pb.rotation_mode = 'XYZ'
    for f in range(1, frames + 1):
        t = (f - 1) / FPS
        v = math.sin(2 * math.pi * freq * t + phase) * amp
        if channel == 'rot':
            cur = list(pb.rotation_euler)
            cur[axis] = math.radians(v + (base[axis] if base else 0.0))
            pb.rotation_euler = cur
            pb.keyframe_insert('rotation_euler', frame=f)
        elif channel == 'loc':
            cur = list(pb.location)
            cur[axis] = v + (base[axis] if base else 0.0)
            pb.location = cur
            pb.keyframe_insert('location', frame=f)
        elif channel == 'scale':
            cur = list(pb.scale)
            cur[axis] = 1.0 + v + (base[axis] if base else 0.0)
            pb.scale = cur
            pb.keyframe_insert('scale', frame=f)


# ---------------------------------------------------------------- idle
def idle(rig, frames=72):
    act = new_action(rig, 'Idle')
    for f in range(1, frames + 1):
        ph = 2 * math.pi * (f - 1) / frames
        breathe = math.sin(ph)
        key(rig, 'spine', f, rot=(1.2 * breathe, 0, 0.6 * math.sin(ph * 0.5)))
        key(rig, 'chest', f, rot=(1.6 * breathe, 0, 0), scale=(1 + 0.012 * breathe, 1 + 0.012 * breathe, 1 + 0.008 * breathe))
        key(rig, 'neck', f, rot=(0.8 * breathe, 0, 0))
        key(rig, 'head', f, rot=(-1.4 * breathe, 0.8 * math.sin(ph * 0.5 + 0.6), 1.6 * math.sin(ph * 0.5)))
        key(rig, 'hips', f, loc=(0, 0.004 * breathe, 0))
        for s, tag in ((1, 'L'), (-1, 'R')):
            key(rig, 'ear%s' % tag, f, rot=(2.5 * math.sin(ph + 0.4 * s), 0, 3.0 * s * math.sin(ph * 0.7)))
            key(rig, 'wing%s' % tag, f, rot=(0, 0, 4.0 * s * math.sin(ph * 0.8)))
            key(rig, 'upperarm%s' % tag, f, rot=(0, 0, 1.5 * s * breathe))
        for i in range(1, 5):
            for s, tag in ((1, 'L'), (-1, 'R')):
                for j in range(1, 4):
                    b = 'tent%d%s_%d' % (i, tag, j)
                    if b in rig.pose.bones:
                        key(rig, b, f, rot=(1.8 * math.sin(ph + i * 0.6 + j * 0.4), 0, 1.6 * math.sin(ph * 0.8 + i)))
    finish(act)
    return act


# ---------------------------------------------------------------- laugh
def laugh(rig, frames=48):
    """giggling: bouncy body, head thrown back, squinting lids, waving tentacles"""
    act = new_action(rig, 'Laugh')
    for f in range(1, frames + 1):
        t = (f - 1) / FPS
        ph = 2 * math.pi * 2.0 * t          # ~2 giggles per second
        bounce = math.sin(ph)
        key(rig, 'hips', f, loc=(0, 0, 0.028 * max(0, bounce)), rot=(1.2 * bounce, 0, 0))
        key(rig, 'spine', f, rot=(2.5 + 2.0 * bounce, 0, 1.2 * math.sin(ph * 0.5)),
            scale=(1 - 0.015 * bounce, 1 + 0.015 * bounce, 1 + 0.015 * bounce))
        key(rig, 'chest', f, rot=(2.0 + 2.5 * bounce, 0, 0),
            scale=(1 + 0.030 * bounce, 1 + 0.026 * bounce, 1 - 0.015 * bounce))
        key(rig, 'neck', f, rot=(-2.5 - 1.5 * bounce, 0, 0))
        key(rig, 'head', f, rot=(-7.0 - 4.0 * bounce, 2.5 * math.sin(ph * 0.5), 4.0 * math.sin(ph * 0.5 + 1.0)))
        key(rig, 'jaw', f, rot=(-3.0 - 2.0 * bounce, 0, 0))
        for s, tag in ((1, 'L'), (-1, 'R')):
            key(rig, 'ear%s' % tag, f, rot=(-6.0 * bounce, 0, 5.0 * s * bounce))
            key(rig, 'wing%s' % tag, f, rot=(0, 0, (10.0 + 8.0 * bounce) * s))
            key(rig, 'upperarm%s' % tag, f, rot=(0, 0, 4.0 * s * bounce))
            key(rig, 'lid%s' % tag, f, scale=(1.0, 1.0, 1.06 + 0.10 * max(0, bounce)))
            key(rig, 'eye%s' % tag, f, scale=(1.0, 1.0, 1.0))
        for i in range(1, 5):
            for s, tag in ((1, 'L'), (-1, 'R')):
                for j in range(1, 4):
                    b = 'tent%d%s_%d' % (i, tag, j)
                    if b in rig.pose.bones:
                        amp = 7.0 + 3.0 * i
                        key(rig, b, f, rot=(-amp * math.sin(ph + i * 0.5 + j * 0.7),
                                            0, 3.0 * math.sin(ph * 0.5 + i + j)))
    finish(act)
    return act


# ---------------------------------------------------------------- cry
def cry(rig, frames=96):
    """sobbing: slumped body, shaking shoulders, rubbing eyes with the paws"""
    act = new_action(rig, 'Cry')
    for f in range(1, frames + 1):
        t = (f - 1) / FPS
        sob = 2 * math.pi * 1.6 * t
        shake = math.sin(sob)
        tremble = math.sin(2 * math.pi * 9.0 * t) * 1.2
        key(rig, 'hips', f, loc=(0, 0, -0.012 + 0.005 * max(0, shake)), rot=(-1.5 - 1.0 * max(0, shake), 0, 0))
        key(rig, 'spine', f, rot=(-4.0 - 1.5 * max(0, shake), tremble * 0.6, 0),
            scale=(1 + 0.02 * shake, 1 + 0.03 * shake, 1 - 0.015 * shake))
        key(rig, 'chest', f, rot=(-4.5 - 2.0 * max(0, shake), tremble, 0))
        key(rig, 'neck', f, rot=(3.5 + 1.0 * max(0, shake), 0, 0))
        key(rig, 'head', f, rot=(7.0 + 2.0 * max(0, shake), tremble * 2.0, 5.0 * math.sin(sob * 0.5)))
        key(rig, 'jaw', f, rot=(-3.0 - 2.0 * max(0, shake), 0, 0))
        for s, tag in ((1, 'L'), (-1, 'R')):
            key(rig, 'ear%s' % tag, f, rot=(7.0 + 3.0 * max(0, shake), 0, -4.0 * s))
            key(rig, 'wing%s' % tag, f, rot=(0, 0, (-6.0 - 4.0 * shake) * s))
            # arms lifted to the face, rubbing
            rub = 10.0 + 6.0 * math.sin(2 * math.pi * 2.6 * t + (0 if s > 0 else math.pi))
            key(rig, 'upperarm%s' % tag, f, rot=(-58.0 - 5.0 * max(0, shake), 0, -18.0 * s))
            key(rig, 'forearm%s' % tag, f, rot=(-52.0 - rub * 0.4, 0, 0))
            key(rig, 'hand%s' % tag, f, rot=(-14.0 - 8.0 * math.sin(2 * math.pi * 2.6 * t), 0, 0))
            # eyes squeezed shut + running tears handled by material/shape
            key(rig, 'lid%s' % tag, f, scale=(1.0, 1.0, 1.95))
            key(rig, 'eye%s' % tag, f, scale=(0.96, 0.96, 0.86))
        for i in range(1, 5):
            for s, tag in ((1, 'L'), (-1, 'R')):
                for j in range(1, 4):
                    b = 'tent%d%s_%d' % (i, tag, j)
                    if b in rig.pose.bones:
                        key(rig, b, f, rot=(4.0 + 3.0 * math.sin(sob + i + j) + tremble,
                                           0, 2.0 * math.sin(sob * 0.7 + i)))
    finish(act)
    return act


# ---------------------------------------------------------------- angry
def angry(rig, frames=60):
    """puffed-up anger: chest out, fists clenched low, quick tremble, red-ish lids"""
    act = new_action(rig, 'Angry')
    for f in range(1, frames + 1):
        t = (f - 1) / FPS
        huff = 2 * math.pi * 2.2 * t
        b = math.sin(huff)
        tremble = math.sin(2 * math.pi * 13.0 * t) * 0.8
        key(rig, 'hips', f, loc=(0, 0, 0.010 * max(0, b)), rot=(-1.8 - 0.8 * max(0, b), 0, 0))
        key(rig, 'spine', f, rot=(-3.0 - 1.2 * max(0, b), tremble, 0),
            scale=(1 + 0.030 * max(0, b), 1 + 0.025 * max(0, b), 1 + 0.01 * max(0, b)))
        key(rig, 'chest', f, rot=(-3.5 - 1.5 * max(0, b), tremble * 1.4, 0),
            scale=(1 + 0.05 * max(0, b), 1 + 0.045 * max(0, b), 1 - 0.012 * max(0, b)))
        key(rig, 'neck', f, rot=(2.0 + 0.8 * max(0, b), 0, 0))
        key(rig, 'head', f, rot=(4.0 + 1.5 * max(0, b), tremble * 2.5, 2.0 * math.sin(huff * 0.5)))
        key(rig, 'jaw', f, rot=(-5.0 - 2.0 * max(0, b), 0, 0))
        for s, tag in ((1, 'L'), (-1, 'R')):
            key(rig, 'ear%s' % tag, f, rot=(-12.0 - 5.0 * b, 0, 8.0 * s))
            key(rig, 'wing%s' % tag, f, rot=(0, 0, (16.0 + 10.0 * max(0, b)) * s))
            key(rig, 'upperarm%s' % tag, f, rot=(12.0 + 3.0 * b, 0, -14.0 * s))
            key(rig, 'forearm%s' % tag, f, rot=(-26.0 - 5.0 * b, 0, 0))
            key(rig, 'hand%s' % tag, f, rot=(10.0, 0, 0), scale=(1.06, 1.06, 1.06))
            key(rig, 'lid%s' % tag, f, scale=(1.0, 1.0, 1.00))
            key(rig, 'eye%s' % tag, f, scale=(1.05, 1.05, 1.05))
        for i in range(1, 5):
            for s, tag in ((1, 'L'), (-1, 'R')):
                for j in range(1, 4):
                    bb = 'tent%d%s_%d' % (i, tag, j)
                    if bb in rig.pose.bones:
                        key(rig, bb, f, rot=(-6.0 - 2.0 * math.sin(huff + i + j) + tremble,
                                             0, -3.0 * s - 2.0 * math.sin(huff * 0.5 + i)))
    finish(act)
    return act


ANIMS = [('Idle', idle), ('Laugh', laugh), ('Cry', cry), ('Angry', angry)]
