import sys; sys.path.insert(0,'/home/user/plush_cthulhu/src')
import numpy as np
from sdf_lib import build_mesh
import plush.body as B, plush.paint as P
from plush.details import eye_sdf, eye_colors, wing_colors, diaper_sdf, diaper_colors
from plush.wings import wing_sdf
from compare import shoot, side_by_side
from preview import save as psave

def render_compare(res=0.022, out='/home/user/cmp.png'):
    V,F = build_mesh(B.body_sdf, [(-2.0,-2.0,-0.05),(2.0,1.35,4.10)], res=res, smooth=1, decimate=0.4, post=1)
    objs=[{'V':V,'F':F,'vcol':P.paint_body(V)}]
    for side in (1,-1):
        c = B.EYE_C*np.array([side,1,1]); bb=[(c[0]-.46,c[1]-.46,c[2]-.46),(c[0]+.46,c[1]+.46,c[2]+.46)]
        Ve,Fe = build_mesh(lambda p,s=side: eye_sdf(p,s), bb, res=0.010, smooth=1, decimate=0.4, post=1)
        objs.append({'V':Ve,'F':Fe,'vcol':eye_colors(Ve,side)})
    Vw,Fw = build_mesh(wing_sdf, [(-2.05,0.0,0.9),(2.05,1.0,2.7)], res=0.02, smooth=1, decimate=0.5, post=1)
    objs.append({'V':Vw,'F':Fw,'vcol':wing_colors(Vw)})
    Vd,Fd = build_mesh(diaper_sdf, [(-1.35,-1.55,0.10),(1.35,1.0,1.35)], res=0.018, smooth=1, decimate=0.5, post=1)
    objs.append({'V':Vd,'F':Fd,'vcol':diaper_colors(Vd)})
    psave(shoot(objs,'x'), '/home/user/cmp_render.png')
    return side_by_side('/home/user/cmp_render.png', out)

if __name__ == '__main__':
    print(render_compare())
