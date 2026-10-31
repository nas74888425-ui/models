import sys; sys.path.insert(0,'/home/user/plush_cthulhu/src')
import bpy, build_scene as B, os
from PIL import Image
B.build()
sc=bpy.context.scene
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=10
sc.cycles.use_denoising=True
sc.render.resolution_x, sc.render.resolution_y = 300, 340
sc.view_settings.view_transform='AgX'; sc.view_settings.exposure=-0.15
rig=bpy.data.objects['PlushRig']
out='/home/user/plush_cthulhu/previews/pose'; os.makedirs(out,exist_ok=True)
shots=[]
for name,frames in [('Idle',[1,36]),('Laugh',[7,19]),('Cry',[10,42]),('Angry',[8,26])]:
    act=bpy.data.actions[name]; rig.animation_data.action=act
    try: rig.animation_data.action_slot=act.slots[0]
    except Exception: pass
    for f in frames:
        sc.frame_set(f); p='%s/%s_%03d.png'%(out,name,f); sc.render.filepath=p
        bpy.ops.render.render(write_still=True); shots.append((name,f,p)); print('shot',name,f,flush=True)
ims=[Image.open(p).convert('RGB') for _,_,p in shots]
w,h=ims[0].size
sheet=Image.new('RGB',(w*4,h*2),(26,28,30))
for i,im in enumerate(ims): sheet.paste(im,((i%4)*w,(i//4)*h))
sheet.save('/home/user/plush_cthulhu/previews/poses.png'); print('sheet saved')
