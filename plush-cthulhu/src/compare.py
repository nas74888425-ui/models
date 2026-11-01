"""Render the model in the reference's framing and put them side by side."""
import sys, os
sys.path.insert(0, '/home/user/plush_cthulhu/src')
import numpy as np
from PIL import Image
from preview import render, save, look_at

REF = Image.open('/home/user/plush_cthulhu/reference/reference.png').convert('RGB')

def shoot(objs, path, W=706, H=474):
    # reference camera: front, ~10 deg above, framing the doll from y=444 to y=30
    return render(objs, (0.0, -10.5, 2.55), (0.0, 0.0, 2.02), W=W, H=H, fov=27,
                  light=(0.25, -0.9, 0.35), bg=(0.14, 0.14, 0.13))

def side_by_side(img_path, out='/home/user/cmp.png'):
    a = REF.copy()
    b = Image.open(img_path).convert('RGB').resize(a.size, Image.LANCZOS)
    w, h = a.size
    sh = Image.new('RGB', (w * 2 + 8, h), (20, 20, 20))
    sh.paste(a, (0, 0)); sh.paste(b, (w + 8, 0))
    sh.save(out)
    return out
