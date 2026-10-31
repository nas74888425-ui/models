"""Assemble the plush doll in Blender: meshes, materials, eyelids, armature,
skinning, animations, studio lighting. Saves the .blend and renders stills.

Run:  python3 build_scene.py            (build + save .blend)
      python3 build_scene.py render     (build + render preview stills)
"""
import sys, os, math
sys.path.insert(0, '/home/user/plush_cthulhu/src')
import numpy as np
import bpy
from mathutils import Vector

import materials
import rig as R
import anim as A

OUT = '/home/user/plush_cthulhu'
MODELS = OUT + '/models'


# ---------------------------------------------------------------- mesh loading
def load_npz(name):
    d = np.load('%s/%s.npz' % (MODELS, name))
    return d['V'], d['F'].astype(np.int64), d['C']


def srgb_to_linear(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_object(name, V, F, C, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.validate()
    me.update()
    col = me.color_attributes.new(name='Col', type='FLOAT_COLOR', domain='POINT')
    rgba = np.concatenate([srgb_to_linear(C), np.ones((len(C), 1), np.float32)], 1)
    col.data.foreach_set('color', rgba.ravel())
    me.shade_smooth()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def eyelid_mesh(side, tag, R_eye, lat=22, lon=28):
    """skin-coloured squashed ellipsoid covering the top of the eyeball (the felt lid)"""
    import plush.body as B
    c = B.EYE_C * np.array([side, 1, 1])
    rx, ry, rz = R_eye * 1.06, R_eye * 1.06, R_eye * 0.78
    cz = c[2] + R_eye * 0.34
    V, F = [], []
    for i in range(1, lat):
        th = math.pi * i / lat
        for j in range(lon):
            ph = 2 * math.pi * j / lon
            V.append([c[0] + rx * math.sin(th) * math.cos(ph),
                      c[1] + ry * math.sin(th) * math.sin(ph),
                      cz + rz * math.cos(th)])
    top = len(V); V.append([c[0], c[1], cz + rz])
    bot = len(V); V.append([c[0], c[1], cz - rz])
    ring = lon
    for j in range(ring):
        F.append([top, (j + 1) % ring, j])
    for i in range(lat - 2):
        a0 = i * ring; b0 = (i + 1) * ring
        for j in range(ring):
            j2 = (j + 1) % ring
            F.append([a0 + j, a0 + j2, b0 + j2])
            F.append([a0 + j, b0 + j2, b0 + j])
    base = (lat - 2) * ring
    for j in range(ring):
        F.append([base + j, base + (j + 1) % ring, bot])
    V = np.array(V, np.float32)
    C = np.tile([0.33, 0.58, 0.50], (len(V), 1)).astype(np.float32)
    return make_object('lid' + tag, V, np.array(F, np.int64), C, materials.felt_material())


# ---------------------------------------------------------------- scene
def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = A.FPS
    sc.unit_settings.system = 'METRIC'
    felt = materials.felt_material()
    cotton = materials.cloth_material()
    glass = materials.eye_material()

    import plush.body as B
    R_lid = B.EYE_R * 1.035
    parts = {}
    parts['body'] = make_object('Body', *load_npz('body'), felt)
    parts['wings'] = make_object('Wings', *load_npz('wings'), felt)
    parts['diaper'] = make_object('Diaper', *load_npz('diaper'), cotton)
    parts['navel'] = make_object('Navel', *load_npz('navel'), felt)
    eyeR = make_object('EyeR', *load_npz('eyeR'), glass)
    eyeL = make_object('EyeL', *load_npz('eyeL'), glass)
    parts['eyeR'], parts['eyeL'] = eyeR, eyeL
    lidR = eyelid_mesh(-1, 'R', R_lid)
    lidL = eyelid_mesh(1, 'L', R_lid)
    parts['lidR'], parts['lidL'] = lidR, lidL

    rig = R.create_rig()
    body_bones = {'hips', 'spine', 'chest', 'neck', 'head', 'jaw'}
    for s, tag in ((1, 'L'), (-1, 'R')):
        body_bones |= {'ear' + tag, 'upperarm' + tag, 'forearm' + tag, 'hand' + tag,
                       'thigh' + tag, 'shin' + tag, 'foot' + tag}
        for i in range(1, 5):
            for j in range(1, 4):
                body_bones.add('tent%d%s_%d' % (i, tag, j))
    R.skin_part(parts['body'], rig, allowed=body_bones, sigma=1.5)
    R.skin_part(parts['diaper'], rig, allowed={'hips', 'spine', 'thighL', 'thighR'}, sigma=1.6)
    R.skin_part(parts['wings'], rig, allowed={'chest', 'wingL', 'wingR', 'spine'}, sigma=1.2)
    R.skin_part(parts['navel'], rig, allowed={'spine', 'hips'}, sigma=2.5)
    for tag in ('R', 'L'):
        R.skin_part(parts['eye' + tag], rig, allowed={'eye' + tag}, sigma=5.0)
        R.skin_part(parts['lid' + tag], rig, allowed={'lid' + tag}, sigma=5.0)

    # animations
    acts = {}
    for name, fn in A.ANIMS:
        acts[name] = fn(rig)
        tr = rig.animation_data.nla_tracks.new()
        tr.name = name
        st = tr.strips.new(name, 1, acts[name])
        st.name = name
        rig.animation_data.action = None
    A.clear_pose(rig)
    rig.animation_data.action = None

    # re-parent the eyes/lids to the head bone so they follow rotations rigidly
    for tag in ('R', 'L'):
        for nm in ('eye' + tag, 'lid' + tag):
            ob = parts[nm]
            ob.parent = rig
            ob.parent_type = 'BONE'
            ob.parent_bone = 'head'
            # keep the world transform
            bone = rig.data.bones['head']
            ob.matrix_world = ob.matrix_world
    # contact shadow catcher
    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'Floor'
    m = bpy.data.materials.new('backdrop')
    m.use_nodes = True
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.62, 0.63, 0.62, 1)
    m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.95
    floor.data.materials.append(m)
    setup_lighting()
    add_camera()
    return rig, parts


def setup_lighting():
    sc = bpy.context.scene
    world = bpy.data.worlds.new('W')
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.32, 0.34, 0.36, 1)
    bg.inputs['Strength'].default_value = 0.6
    sc.world = world

    def area(name, loc, rot, size, energy, color=(1, 1, 1)):
        d = bpy.data.lights.new(name, 'AREA')
        d.energy = energy
        d.size = size
        d.color = color
        o = bpy.data.objects.new(name, d)
        o.location = loc
        o.rotation_euler = rot
        bpy.context.collection.objects.link(o)
        return o

    area('key', (-1.6, -2.6, 2.8), (math.radians(58), 0, math.radians(-32)), 2.2, 130, (1.0, 0.97, 0.93))
    area('fill', (2.2, -1.8, 1.6), (math.radians(74), 0, math.radians(52)), 2.6, 55, (0.90, 0.95, 1.0))
    area('rim', (0.8, 2.4, 2.6), (math.radians(-58), 0, math.radians(165)), 1.8, 110, (0.95, 0.97, 1.0))


def add_camera():
    cam_data = bpy.data.cameras.new('cam')
    cam_data.lens = 62
    cam = bpy.data.objects.new('cam', cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    aim(cam, (2.9, -8.2, 3.1), (0, 0, 1.70))
    return cam


def aim(cam, eye, target):
    cam.location = Vector(eye)
    d = Vector(target) - Vector(eye)
    cam.rotation_mode = 'QUATERNION'
    cam.rotation_quaternion = d.to_track_quat('-Z', 'Y')


def render_still(path, res=(760, 900), samples=64, frame=1, action=None):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = -0.2
    sc.frame_set(frame)
    if action:
        rig = bpy.data.objects['PlushRig']
        rig.animation_data.action = bpy.data.actions[action]
        if hasattr(rig.animation_data, 'action_slot'):
            try:
                rig.animation_data.action_slot = rig.animation_data.action.slots[0]
            except Exception:
                pass
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    print('rendered', path, flush=True)


if __name__ == '__main__':
    rig, parts = build()
    bpy.ops.wm.save_as_mainfile(filepath=OUT + '/models/plush_cthulhu.blend')
    print('saved blend')
    if 'render' in sys.argv:
        os.makedirs(OUT + '/renders', exist_ok=True)
        render_still(OUT + '/renders/hero.png', res=(900, 1050), samples=48, frame=1, action='Idle')


def render_anim(name, nframes, every=2, res=(360, 430), samples=14, outdir=None):
    """render a strip of frames of one action for the preview GIFs"""
    sc = bpy.context.scene
    outdir = outdir or (OUT + '/renders/frames')
    os.makedirs(outdir, exist_ok=True)
    rig = bpy.data.objects['PlushRig']
    act = bpy.data.actions[name]
    rig.animation_data.action = act
    if hasattr(rig.animation_data, 'action_slot'):
        try:
            rig.animation_data.action_slot = act.slots[0]
        except Exception:
            pass
    cam = sc.camera
    aim(cam, (2.4, -7.6, 2.9), (0, 0, 1.70))
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.exposure = -0.15
    path = '%s/%s_%%03d.png' % (outdir, name)
    for i, f in enumerate(range(1, nframes + 1, every)):
        sc.frame_set(f)
        sc.render.filepath = path % f
        bpy.ops.render.render(write_still=True)
        print('frame', name, f, flush=True)
