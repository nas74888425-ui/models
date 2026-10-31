import sys, time
sys.path.insert(0,'/home/user/plush_cthulhu/src')
import numpy as np, fast_simplification
import plush.body as B
from sdf_lib import build_mesh
from preview import render, save

t0=time.time()
BB=[(-2.3,-2.7,-0.05),(2.3,1.6,3.85)]
V,F=build_mesh(B.body_sdf, BB, res=0.035, smooth=2, decimate=0.45)
print('mesh',V.shape,F.shape,'%.1fs'%(time.time()-t0),flush=True)
Vs,Fs=fast_simplification.simplify(V,F,0.4)
objs=[{'V':Vs,'F':Fs,'color':(0.46,0.68,0.60)}]
for name,(eye,tgt) in {'front':((0,-14,2.4),(0,0,1.9)),'q34':((7,-12,3.6),(0,0,1.8)),'side':((14,-1,2.5),(0,0,1.8))}.items():
    save(render(objs,eye,tgt,W=360,H=300,fov=32,light=(0.25,-0.8,0.55)),'/home/user/plush_cthulhu/previews/body_%s.png'%name)
    print(name,'%.1fs'%(time.time()-t0),flush=True)
np.savez_compressed('/home/user/plush_cthulhu/previews/body_raw.npz',V=V,F=F)
