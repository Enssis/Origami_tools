# from . import *
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ..Utils._types import Number, Group
from ..Utils._svg_utils import *
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .line import Line
    from .plane import Plane

@dataclass
class Vec(list):
    """ A class representing a vector in 2D or 3D space. """
    x : Number
    y : Number
    z : Number | None = None
    dimension : int = 2

    @classmethod
    def from_2points(cls, p1 : Group | 'Vec', p2 : Group | 'Vec'):
        if len(p1) != len(p2):
            raise ValueError("p1 and p2 must have the same length")
        if len(p1) == 2:
            return cls(p2[0] - p1[0], p2[1] - p1[1])
        elif len(p1) == 3:
            return cls(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        else:
            raise ValueError("p1 and p2 must have length 2 or 3")

    @classmethod
    def from_list(cls, v : Group):
        """Create a vector from a list or tuple."""
        if len(v) == 2:
            return cls(v[0], v[1])
        elif len(v) == 3:
            return cls(v[0], v[1], v[2])
        else:
            raise ValueError("Vector must have length 2 or 3")
    
    def __post_init__(self):
        if self.z is not None:
            self.dimension = 3
        super().__init__([self.x, self.y, self.z] if self.z is not None else [self.x, self.y])

    def __neg__(self):
        if self.dimension == 2:
            return type(self)(-self[0], -self[1])
        else:
            return type(self)(-self[0], -self[1], -self[2])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if self.dimension == 2:
                return type(self)(self[0] * other, self[1] * other)
            else:
                return type(self)(self[0] * other, self[1] * other, self[2] * other)
        elif isinstance(other, Vec):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must have the same dimension")
            return np.dot(self, other)
        else:
            raise TypeError("Expected a scalar or Vec/Point object")
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __matmul__(self, other):
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        
        z = self[0] * other[1] - self[1] * other[0]
        if self.dimension == 2:
            
            return type(self)(0, 0, z)
        else:
            x = self[1] * other[2] - self[2] * other[1]
            y = self[2] * other[0] - self[0] * other[2]
            return type(self)(x, y, z)

    def __add__(self, other):
        if isinstance(other, (Vec, Group)):
            if self.dimension != len(other):
                raise ValueError("Vectors must have the same dimension")
            if self.dimension == 2:
                return type(self)(self[0] + other[0], self[1] + other[1])
            else:
                return type(self)(self[0] + other[0], self[1] + other[1], self[2] + other[2])
        elif isinstance(other, Number):
            if self.dimension == 2:
                return type(self)(self[0] + other, self[1] + other)
            else:
                return type(self)(self[0] + other, self[1] + other, self[2] + other)
        raise TypeError("Expected a Vec object")

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Expected a scalar")
        if self.dimension == 2:
            return type(self)(self[0] / other, self[1] / other)
        else:
            return type(self)(self[0] / other, self[1] / other, self[2] / other)

    def __gt__(self, other):
        if isinstance(other, Vec):
            if self.dimension != other.dimension:
                raise ValueError("Vectors must have the same dimension")
            return self.norm() > other.norm()
        elif isinstance(other, Number):
            if self.dimension == 2:
                return self[0] > other and self[1] > other
            else:
                return self[0] > other and self[1] > other and self[2] > other
        raise TypeError("Expected a Vec object or Number")

    def norm(self):
        return float(np.linalg.norm([self[0], self[1], self[2] if self.dimension == 3 else 0]))
    
    def angle(self, other : 'Vec'):
        """Calculate the angle between two vectors."""
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        return float(np.arctan2((self @ other)[2], self * other))

    def normalize(self):
        norm = self.norm()
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector")
        return type(self).from_list(self / norm)
    
    def normal(self, other : 'Vec'):
        """Calculate the normal vector to the vector."""
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec or Point object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        return Vec.from_list(np.cross(self, other)).normalize()
    
    def in_3D(self):
        """Convert the vector to 3D coordinates."""
        if self.dimension == 2:
            return type(self)(self[0], self[1], 0)
        else:
            return self

    def as_array(self):
        """Convert the point to a numpy array."""
        return np.array([self[0], self[1], self[2]]) if self.dimension == 3 else np.array([self[0], self[1]])

    def copy(self):
        """Create a copy of the vector."""
        return type(self)(self[0], self[1], self[2]) if self.dimension == 3 else type(self)(self[0], self[1])
    
    def distance(self, other):
        """Calculate the distance between two vectors."""
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        return float(np.linalg.norm(self - other))
    
    def __eq__(self, other):
        if not isinstance(other, Vec):
            return False
        return bool(self.distance(other) < 1e-3)

    def __ne__(self, other):
        if not isinstance(other, Vec):
            return True
        return not self.__eq__(other)

    def project(self, other : 'Vec'):
        """Project the vector onto another vector."""
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        u_norm = other.normalize()
        return np.dot(self, u_norm) * other

    def transform(self, mat : Group):
        """Transform the vector using a transformation matrix."""
        if not isinstance(mat, (list, np.ndarray)):
            raise TypeError("Expected a list or numpy array")
        if len(mat) != self.dimension + 1:
            raise ValueError(f"Transformation matrix must have {self.dimension + 1} rows")
        if len(mat[0]) != self.dimension + 1:
            raise ValueError(f"Transformation matrix must have {self.dimension + 1} columns")
        
        vect = np.array([self[0], self[1], 1]) if self.dimension == 2 else np.array([self[0], self[1], self[2], 1])
        transformed_vect = np.dot(mat, vect)
        return type(self).from_list(transformed_vect[:self.dimension])

    def to_roation_matrix(self, angle : Number):
        """Create a rotation matrix for a given angle around a vector."""
        v = self.normalize()
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = v
        return np.array([[cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
                        [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
                        [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]])


    def rotate(self, angle : Number, point : Group | 'Vec' | None = None, axis : Group | 'Vec' | None = None, deg = False):
        """Rotate the coordinate system by a given angle (in radians) around a given axis."""
        if point is None:
            if self.dimension == 2:
                point = Vec(0, 0)
            else:
                point = Vec(0, 0, 0)

        if not isinstance(point, Point):
            point = Vec.from_list(point)
        if deg:
            angle = np.radians(angle)
        
        if not isinstance(axis, Vec):
            if axis is None:
                axis = Vec(0, 0, 1)
            else:
                axis = Vec.from_list(axis)

        if self.dimension == 2:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        else:
            if axis is None:
                axis = Vec(0, 0, 1)
            if len(axis) != self.dimension:
                raise ValueError(f"Rotation axis must have length {self.dimension}")
            rotation_matrix = axis.to_roation_matrix(angle)

        return type(self).from_list(np.dot(rotation_matrix, self - point) + point)
    
    def mirror(self, mir : 'Plane | Line'):
        """Mirror the point across a given axis."""
        from .line import Line
        from .plane import Plane 
        if isinstance(mir, Line) and self.dimension != 2 or isinstance(mir, Plane) and self.dimension != 3:
            raise ValueError(f"Mirror must be in {self.dimension}D")

        if self.dimension == 2:
            if not isinstance(mir, Line):
                raise ValueError("Mirror must be a Line in 2D")
            vect_plane_point = (self - mir.points[0]).project(mir.normal_vect())
            x = self[0] - 2 * vect_plane_point[0]
            y = self[1] - 2 * vect_plane_point[1]
            return type(self).from_list([x, y])
        else:
            if not isinstance(mir, Plane):
                raise ValueError("Mirror must be a Plane in 3D")
            vect_plane_point = (self - mir.point).project(mir.normal)
            x = self[0] - 2 * vect_plane_point[0]
            y = self[1] - 2 * vect_plane_point[1]
            z = self[2] - 2 * vect_plane_point[2]
            return type(self).from_list([x, y, z])
            
    
    def translate(self, v : Group | 'Vec'):
        """Translate the point by a given vector."""
        if len(v) != self.dimension:
            raise ValueError(f"Translation vector must have length {self.dimension}")
        self[0] = v[0] + self[0]
        self[1] = v[1] + self[1]
        if self.dimension == 3:
            self[2] = v[2] + self[2]

    def show(self, ax = None, show = False):
        """Show the point."""
        if self.dimension == 2:
            if ax is None:
                fig, ax = plt.subplots()
            ax.plot(self[0], self[1], 'o')
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self[0], self[1], self[2])
        if show:
            plt.show()
        else:
            return ax


class Point(Vec):
    """ A class representing a Point in 2D or 3D space. """

    @classmethod
    def from_homogeneous(cls, v : Group):
        """Create a point from homogeneous coordinates."""
        if len(v) not in (3, 4):
            raise ValueError(f"Homogeneous coordinates must have length 3 or 4, not {len(v)}")
        if v[-1] == 0:
            return None
        if len(v) == 3:
            return cls(v[0] / v[2], v[1] / v[2])
        else:
            return cls(v[0] / v[3], v[1] / v[3], v[2] / v[3])

    def __str__(self):
        if self.dimension == 2:
            return f"Point({self[0]:.{3}f}, {self[1]:.{3}f})"
        else:
            return f"Point({self[0]:.{3}f}, {self[1]:.{3}f}, {self[2]:.{3}f})"

    def as_homogeneous(self):
        """Convert the vector to a homogeneous coordinate."""
        if self.dimension == 2:
            return np.array([self[0], self[1], 1])
        else:
            return np.array([self[0], self[1], self[2], 1])


E3X = Vec(1, 0, 0)
E3Y = Vec(0, 1, 0)
E3Z = Vec(0, 0, 1)

E2X = Vec(1, 0)
E2Y = Vec(0, 1)

O20 = Point(0,0)
O30 = Point(0,0,0)



