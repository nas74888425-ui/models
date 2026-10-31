"""Materials: felt / wool cloth, glassy eyes, cotton nappy."""
import bpy


def _principled(nt):
    return nt.nodes.get('Principled BSDF')


def felt_material(name='felt', rough=0.93, sheen=0.75, bump=0.85, fibre=0.18):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(nt)
    attr = nt.nodes.new('ShaderNodeVertexColor')
    attr.layer_name = 'Col'
    attr.location = (-820, 260)
    # fibre flecks: multiply the albedo by a fine noise around 1.0
    tex = nt.nodes.new('ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 420.0
    tex.inputs['Detail'].default_value = 8.0
    tex.inputs['Roughness'].default_value = 0.8
    tex.location = (-820, -60)
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (0.72, 0.72, 0.72, 1)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
    ramp.location = (-600, -60)
    nt.links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = fibre
    mix.location = (-330, 160)
    nt.links.new(attr.outputs['Color'], mix.inputs[6])
    nt.links.new(ramp.outputs['Color'], mix.inputs[7])
    nt.links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    # felted surface: two-scale bump
    n2 = nt.nodes.new('ShaderNodeTexNoise')
    n2.inputs['Scale'].default_value = 90.0
    n2.inputs['Detail'].default_value = 14.0
    n2.inputs['Roughness'].default_value = 0.8
    n2.location = (-820, -400)
    bp = nt.nodes.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = bump
    bp.inputs['Distance'].default_value = 0.006
    bp.location = (-430, -400)
    nt.links.new(n2.outputs['Fac'], bp.inputs['Height'])
    n3 = nt.nodes.new('ShaderNodeTexNoise')
    n3.inputs['Scale'].default_value = 620.0
    n3.inputs['Detail'].default_value = 4.0
    n3.location = (-820, -650)
    bp2 = nt.nodes.new('ShaderNodeBump')
    bp2.inputs['Strength'].default_value = bump * 0.55
    bp2.inputs['Distance'].default_value = 0.0015
    bp2.location = (-220, -450)
    nt.links.new(n3.outputs['Fac'], bp2.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], bp2.inputs['Normal'])
    nt.links.new(bp2.outputs['Normal'], bsdf.inputs['Normal'])
    bsdf.inputs['Roughness'].default_value = rough
    if 'Sheen Weight' in bsdf.inputs:
        bsdf.inputs['Sheen Weight'].default_value = sheen
        bsdf.inputs['Sheen Roughness'].default_value = 0.45
        bsdf.inputs['Sheen Tint'].default_value = (1.0, 0.98, 0.96, 1)
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.18
    return mat


def eye_material(name='eye_glass'):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _principled(mat.node_tree)
    attr = mat.node_tree.nodes.new('ShaderNodeVertexColor')
    attr.layer_name = 'Col'
    mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = 0.12
    if 'Coat Weight' in bsdf.inputs:
        bsdf.inputs['Coat Weight'].default_value = 0.9
        bsdf.inputs['Coat Roughness'].default_value = 0.02
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.6
    return mat


def cloth_material(name='cotton'):
    return felt_material(name='cotton', rough=0.95, sheen=0.8, bump=0.7, fibre=0.22)
