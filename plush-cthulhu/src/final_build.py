import sys, os
sys.path.insert(0,'/home/user/plush_cthulhu/src')
import bpy
import build_scene as B
import export_glb as E
OUT='/home/user/plush_cthulhu'
B.build()
B.aim(bpy.context.scene.camera, (0.0, -9.4, 3.0), (0, 0, 1.95))
B.render_still(OUT+'/renders/hero.png', res=(820,950), samples=40, frame=1, action='Idle')
for tag, act, frame in [('laugh','Laugh',8),('cry','Cry',24),('angry','Angry',14)]:
    B.aim(bpy.context.scene.camera, (2.4, -9.0, 3.1), (0, 0, 1.95))
    B.render_still(OUT+'/renders/pose_%s.png'%tag, res=(660,760), samples=32, frame=frame, action=act)
B.aim(bpy.context.scene.camera, (2.0, -8.6, 2.9), (0, 0, 1.95))
for nm,n,every in [('Laugh',48,4),('Cry',96,8),('Angry',60,6)]:
    B.render_anim(nm, n, every=every, res=(300,340), samples=14)
E.export()
print('ALL DONE', flush=True)
