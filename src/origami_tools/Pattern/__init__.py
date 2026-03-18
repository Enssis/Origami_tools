from .drawn_shapes import DrawnShapes, Folds
from .pattern import Pattern

from origami_tools.Geometry import Line, Circle, Shape, Point, Vec
import numpy as np

def base_flap_pattern(lasercut, h=9, r=1.7, angle=np.pi/6, w=10.0, inv=False):
    base_flap = Pattern("base_flap", lasercut, origin=Point(0, 0))
    
    A_b_flap = Point(0, 0)
    B_b_flap = A_b_flap + Vec.from_angle(angle - np.pi/2, abs(h/np.cos(angle)))
    D_b_flap = A_b_flap + Vec(w, 0)
    C_b_flap = D_b_flap + Vec.from_angle(-angle - np.pi/2, abs(h/np.cos(angle)))

    hole =  Circle(Point(w / 2 , h / 2 ), r)
    shapes : list[Shape] = [Line(A_b_flap, B_b_flap), Line(B_b_flap, C_b_flap), Line(C_b_flap, D_b_flap)]
    base_flap.w_h(w, h)
    base_flap.add_shapes(shapes, outside=True, param="cut")
    base_flap.add_shapes([hole], param="cut", background=False, duplicate=True)
    if inv:
        base_flap.mirror(Line(Point(0, h/2), Point(1, h/2)))
    return base_flap



__all__ = ["Pattern", "DrawnShapes", "Folds", "base_flap_pattern"]