import sys, time
sys.path.insert(0,'/home/user/plush_cthulhu/src')
import numpy as np
import plush.body as B
from sdf_lib import build_mesh
from check_fit import fit
V,F=build_mesh(B.body_sdf, [(-2.3,-2.7,-0.05),(2.3,1.6,3.85)], res=0.045, smooth=1, decimate=0.5)
print('IoU %.3f'%fit(V,F,verbose=True))
