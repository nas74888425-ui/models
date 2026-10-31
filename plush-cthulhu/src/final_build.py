import sys, os
sys.path.insert(0,'/home/user/plush_cthulhu/src')
import bpy
import build_scene as B
import export_glb as E
OUT='/home/user/plush_cthulhu'
B.build()
B.render_still(OUT+'/renders/hero.png', res=(860,1000), samples=44, frame=1, action='Idle')
for tag, act, frame in [('laugh','Laugh',8),('cry','Cry',24),('angry','Angry',14)]:
    B.aim(bpy.context.scene.camera, (2.7,-7.9,3.0), (0,0,1.70))
    B.render_still(OUT+'/renders/pose_%s.png'%tag, res=(700,820), samples=36, frame=frame, action=act)
B.aim(bpy.context.scene.camera, (2.2,-7.4,2.9), (0,0,1.70))
for nm,n,every in [('Laugh',48,4),('Cry',96,8),('Angry',60,5)]:
    B.render_anim(nm, n, every=every, res=(300,340), samples=12)
E.export()
print('ALL DONE', flush=True)
