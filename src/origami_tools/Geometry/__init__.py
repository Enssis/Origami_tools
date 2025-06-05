import numpy as np


def merge_close_points(points, tol=1e-6):
    result = []
    for p in points:
        if all((p - q).norm() > tol for q in result):
            result.append(p)
    return result

# application du theoreme d'Al-Kashi
def alkashi(ac, bc, c):
	return np.sqrt(ac**2 + bc**2 - 2 * ac * bc * np.cos(c))

# application du theoreme d'alkashi pour trouver l'angle
def alkashi_angle(ac, bc, ab):
	return np.arccos((ac**2 + bc**2 - ab**2) / (2 * ac * bc))


def skewm(v):
    """Create a skew-symmetric matrix from a vector."""
    if len(v) != 3:
        raise ValueError("Vector must have length 3")
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]], float)


from .vector import Vec, Point, E2X, E2Y, E3X, E3Y, E3Z, O20, O30
from .repere import Repere 
from .plane import Plane
from .line import Line
from .shape import Shape
from .surface import Surface, Polygon, Circle, HoledPolygon, RegularPolygon, Rectangle
from .volume import Volume   

BASE_REPERE2D = Repere.base(2)
BASE_REPERE3D = Repere.base(3)

__all__ = [
    "merge_close_points",
    "alkashi",
    "alkashi_angle",
    "skewm",
    "Vec",
    "Point",
    "Repere",
    "Plane",
    "Line",
    "Shape",
    "Surface",
    "Polygon",
    "Circle",
    "HoledPolygon",
    "RegularPolygon",
    "Rectangle",
    "Volume",
    "E2X",
    "E2Y",
    "E3X",
    "E3Y",
    "E3Z",
    "O20",
    "O30",
    "BASE_REPERE2D",
    "BASE_REPERE3D"
]