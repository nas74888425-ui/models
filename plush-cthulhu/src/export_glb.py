"""Export the rigged + animated doll to glTF/GLB (embedded), plus a full-poly STL.

The animations are exported as separate named clips: Idle, Laugh, Cry, Angry.
"""
import sys, os
sys.path.insert(0, '/home/user/plush_cthulhu/src')
import bpy
import build_scene as B

OUT = '/home/user/plush_cthulhu'


def export(glb_path=None, blend_path=None):
    sc = bpy.context.scene
    sc.render.fps = 30
    glb = glb_path or (OUT + '/models/plush_cthulhu_animated.glb')
    bpy.ops.object.select_all(action='DESELECT')
    for ob in bpy.data.objects:
        if ob.type in {'MESH', 'ARMATURE'} and ob.name not in ('Floor',):
            ob.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['PlushRig']
    bpy.ops.export_scene.gltf(
        filepath=glb,
        export_format='GLB',
        use_selection=True,
        export_apply=False,
        export_yup=True,
        export_animations=True,
        export_animation_mode='NLA_TRACKS',
        export_nla_strips=True,
        export_force_sampling=True,
        export_frame_range=False,
        export_skins=True,
        export_def_bones=False,
        export_optimize_animation_size=True,
        export_texcoords=False,
        export_normals=True,
        export_vertex_color='ACTIVE',
        export_image_format='NONE',
        export_draco_mesh_compression_enable=False,
        export_all_vertex_colors=False,
        export_materials='EXPORT',
    )
    print('glb written', glb, os.path.getsize(glb) / 1e6, 'MB', flush=True)
    if blend_path:
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print('blend written', blend_path, flush=True)


if __name__ == '__main__':
    B.build()
    export()
