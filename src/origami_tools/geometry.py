from typing import Sequence
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from .utils._types import Number, Group

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
                    [-v[1], v[0], 0]])


def normalize(vector):
    """Normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm

def axe_angle_to_mat_rotation(v : 'Vec', angle):
    """Create a rotation matrix for a given angle around a vector."""
    v = normalize(v)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    ux, uy, uz = v
    return np.array([[cos_angle + ux**2 * (1 - cos_angle), ux * uy * (1 - cos_angle) - uz * sin_angle, ux * uz * (1 - cos_angle) + uy * sin_angle],
                     [uy * ux * (1 - cos_angle) + uz * sin_angle, cos_angle + uy**2 * (1 - cos_angle), uy * uz * (1 - cos_angle) - ux * sin_angle],
                     [uz * ux * (1 - cos_angle) - uy * sin_angle, uz * uy * (1 - cos_angle) + ux * sin_angle, cos_angle + uz**2 * (1 - cos_angle)]])



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
        if isinstance(other, Vec):
            if self.dimension != other.dimension:
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
        return float(np.linalg.norm(self))
    
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
        return type(self)(self[0], self[1], self[2]) if len(self) == 3 else type(self)(self[0], self[1])
    
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
        return bool(self.distance(other) < 1e-6)

    def project(self, other):
        """Project the vector onto another vector."""
        if not isinstance(other, Vec):
            raise TypeError("Expected a Vec object")
        if self.dimension != other.dimension:
            raise ValueError("Vectors must have the same dimension")
        u_norm = normalize(other)
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
            rotation_matrix = axe_angle_to_mat_rotation(axis, angle)

        return type(self).from_list(np.dot(rotation_matrix, self - point) + point)
    
    def mirror(self, mir : 'Plane | Line'):
        """Mirror the point across a given axis."""
        
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


E3X = Vec(1, 0, 0)
E3Y = Vec(0, 1, 0)
E3Z = Vec(0, 0, 1)

E2X = Vec(1, 0)
E2Y = Vec(0, 1)

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


O20 = Point(0,0)
O30 = Point(0,0,0)        

@dataclass
class Plane():
    """ A class representing a plane in 3D space. """
    point : Point
    normal : Vec

    def __post_init__(self):
        self.normal = self.normal.normalize()

    @classmethod
    def from_points(cls, p1 : Point, p2 : Point, p3 : Point):
        """Create a plane from three points."""
        if not isinstance(p1, Point) or not isinstance(p2, Point) or not isinstance(p3, Point):
            raise ValueError("p1, p2 and p3 must be Point objects")
        if p1.dimension != 3 or p2.dimension != 3 or p3.dimension != 3:
            raise ValueError("p1, p2 and p3 must be in 3D space")
        
        normal = Vec.from_2points(p1, p2) @ Vec.from_2points(p1, p3)
        return cls(p1, normal)

    def homogeneous(self):
        """Convert the plane to homogeneous coordinates."""
        return np.array([self.normal[0], self.normal[1], self.normal[2], -self.normal @ self.point])



@dataclass
class Repere():
    """ A class representing a coordinate system in 2D or 3D space. """
    origin : Point 
    axis : list[Vec] 
    dimension : int = 2

    def __post_init__(self):
        if self.dimension != self.origin.dimension:
            self.dimension = self.origin.dimension
        
        if self.dimension == 2:
            if len(self.axis) != 2:
                raise ValueError("Repere must have 2 axis in 2D")
            if (self.axis[0] @ self.axis[1]).norm() == 0:
                raise ValueError("axis[0] and axis[1] must not be colinear")
        elif self.dimension == 3:
            if len(self.axis) != 3:
                raise ValueError("Repere must have 3 axis in 3D")
            if (self.axis[0] @ self.axis[1]).norm() == 0 or (self.axis[0] @ self.axis[2]).norm() == 0 or (self.axis[1] @ self.axis[2]).norm() == 0:
                raise ValueError("axis[0], axis[1] and axis[2] must not be colinear")

    @classmethod
    def base(cls, dimension : int = 2):
        """Create a base coordinate system."""
        if dimension == 2:
            return cls(O20, [E2X, E2Y])
        elif dimension == 3:
            return cls(O30, [E3X, E3Y, E3Z])
        else:
            raise ValueError("Repere must be of dimension 2 or 3")

    def __repr__(self):
        if self.dimension == 2:
            return f"Repere(origin = {self.origin}, x = {self.axis[0]}, y = {self.axis[1]})"
        else:
            return f"Repere(origin = {self.origin}, x = {self.axis[0]}, y = {self.axis[1]}, z = {self.axis[2]})"
        
    def __eq__(self, other):
        if not isinstance(other, Repere):
            return False
        if self.dimension != other.dimension:
            return False
        if self.dimension == 2:
            return (self.origin == other.origin and
                    self.axis[0] == other.axis[0] and
                    self.axis[1] == other.axis[1])
        else:
            return (self.origin == other.origin and
                    self.axis[0] == other.axis[0] and
                    self.axis[1] == other.axis[1] and
                    self.axis[2] == other.axis[2])
        
    def __ne__(self, other):
        if not isinstance(other, Repere):
            return True
        return not self.__eq__(other)

    def copy(self):
        if self.dimension == 2:
            return Repere(self.origin, [self.axis[0], self.axis[1]])
        else:
            return Repere(self.origin, [self.axis[0], self.axis[1], self.axis[2]])

    def translate(self, d : Vec):
        """Translate the coordinate system by a given distance in x and y directions."""
        if len(d) != self.dimension:
            raise ValueError(f"Translation vector must have length {self.dimension}")
        
        self.origin = self.origin + d
    
    def rotate(self, angle : Number, axis : Vec = E3Z, deg = False):
        """Rotate the coordinate system by a given angle (in radians) around a given axis."""
        if axis.dimension != 3:
            raise ValueError(f"Rotation axis must have length 3")
        
        if deg:
            angle = np.radians(angle)
        
        if self.dimension == 2:
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])
        else:
            rotation_matrix = axe_angle_to_mat_rotation(axis, angle)

        for i in range(self.dimension):
            self.axis[i] = np.dot(rotation_matrix, self.axis[i])

    def rotation_mat(self, other : 'None | Repere' = None):
        if other is None:
            other = Repere.base(dimension=self.dimension)

        if not isinstance(other, Repere):
            raise TypeError("Expected a Repere object")
        if self.dimension != other.dimension:
            raise ValueError("Repere dimensions must match")

        vec_mat = np.vstack([self.axis[0], self.axis[1], self.axis[2]]) if self.dimension == 3 else np.hstack([self.axis[0], self.axis[1]])
        vec_mat2 = np.vstack([other.axis[0], other.axis[1], other.axis[2]]) if other.dimension == 3 else np.hstack([other.axis[0], other.axis[1]])
        return vec_mat @ np.linalg.inv(vec_mat2)
    
    def transformation_mat(self, repere = None):
        if repere is None:
            repere = Repere.base(self.dimension)

        if not isinstance(repere, Repere):
            raise TypeError("Expected a Repere object")
        if self.dimension != repere.dimension:
            raise ValueError("Repere dimensions must match")
        
        translation = self.origin - repere.origin
        rotation = self.rotation_mat(repere)

        transformation_matrix = np.eye(self.dimension + 1)
        transformation_matrix[:self.dimension, :self.dimension] = rotation
        transformation_matrix[:self.dimension, self.dimension] = translation
        
        return transformation_matrix

BASE_REPERE2D = Repere.base(2)
BASE_REPERE3D = Repere.base(3)

@dataclass
class Shape():
    """ A class representing a shape defined by a list of points. """
    points : list[Point]

    def __post_init__(self):
        if len(self.points) < 1:
            raise ValueError("At least a point is required to define a shape")
        self.dimension = self.points[0].dimension
        for i in range(len(self.points)):
            if self.points[i].dimension != self.dimension:
                raise ValueError("All points must have the same dimension")
            self.points[i] = self.points[i].copy()
    
    def as_json(self):
        """Convert the shape to a JSON serializable format."""
        return {
            "type": type(self).__name__,
            "points": [point for point in self.points]
        }

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.points[index]
        else:
            raise TypeError("Index must be an integer or a slice")

    def __str__(self):
        if self.dimension == 2:
            return f"Shape with {len(self.points)} points in 2D: {self.points}\n"
        else:
            return f"Shape with {len(self.points)} points in 3D: {self.points}\n"

    def __len__(self):
        return len(self.points)

    def __add__(self, other):
        """Add another shape to this shape."""
        if isinstance(other, Shape):
            if self.dimension != other.dimension:
                raise ValueError("Shapes must have the same dimension")
            return Shape(self.points + other.points)
        elif isinstance(other, Vec):
            n_shape = self.copy()
            n_shape.translate(other)
            return n_shape
        
        raise TypeError("Expected a Shape or vec object")

    def get_mirror(self, plane : 'Plane | Line'):
        n_shape = self.copy()
        n_shape.mirror(plane)
        return n_shape

    def mirror(self, plane : 'Plane | Line'):
        pts = []
        for i in range(len(self)):
            pts.append(self.points[i].mirror(plane))
        self.points = pts
    
    def rotate(self, angle : Number, point : Point | Group | None = None, axis : Point | Group | None = None, deg = False):
        for i in range(len(self)):
            self.points[i] = self.points[i].rotate(angle, point, axis, deg)
    
    def translate(self, v : list):
        for i in range(len(self.points)):
            self.points[i].translate(v)
    
    def scale(self, factor):
        print("Scale not implemented for this geometry type")

    def perimeter(self):
        perimeter = 0
        for i in range(len(self) - 1):
                perimeter += self.points[i].distance(self.points[i + 1])
        return perimeter
    
    def copy(self):
        points = [point.copy() for point in self.points]
        return Shape(points)
    
    def center(self):
        """Calculate the center of the shape."""
        if self.dimension == 2:
            x = sum(point[0] for point in self.points) / len(self.points)
            y = sum(point[1] for point in self.points) / len(self.points)
            return Point(x, y)
        else:
            x = sum(point[0] for point in self.points) / len(self.points)
            y = sum(point[1] for point in self.points) / len(self.points)
            z = sum(point[2] for point in self.points) / len(self.points)
            return Point(x, y, z)

    def repr_3D(self):
        """give the list of point to represent the shape in 3D"""
        if self.dimension != 3:
            raise ValueError("Shape must be in 3D")
        
        x = np.array(self.points[:][0])
        y = np.array(self.points[:][1])
        z = np.array(self.points[:][2])
        return (x, y, z)
    
    def show(self, ax = None, show = False):
        """Show the surface."""
        nb_pts = len(self)
        if self.dimension == 2:
            if ax is None:
                fig, ax = plt.subplots()
            plt.plot([self[i % nb_pts][0] for i in range(nb_pts + 1)], [self[i % nb_pts][1] for i in range(nb_pts + 1)])
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            ax.plot([self[i % nb_pts][0] for i in range(nb_pts + 1)], [self[i % nb_pts][1] for i in range(nb_pts + 1)], [self[i % nb_pts][2] for i in range(nb_pts + 1)])
        if show:
            plt.show()
        else :
            return ax
        
    def transform(self, mat : Group):
        """Transform the shape using a transformation matrix."""
        if not isinstance(mat, (list, np.ndarray)):
            raise TypeError("Expected a list or numpy array")
        if len(mat) != self.dimension + 1:
            raise ValueError(f"Transformation matrix must have {self.dimension + 1} rows")
        if len(mat[0]) != self.dimension + 1:
            raise ValueError(f"Transformation matrix must have {self.dimension + 1} columns")
        
        for i in range(len(self.points)):
            self.points[i] = self.points[i].transform(mat)


@dataclass
class Line(Shape):
    """ A class representing a line defined by two points. """
    dashed : bool = False
    dash_length : float = 6
    dash_ratio : float = 0.5
        
    def __init__(self, point1 : Point, point2 : Point, dashed : bool = False, dash_length : float = 6, dash_ratio : float = 0.5):
        super().__init__([point1, point2])
        if self.dimension != point1.dimension or self.dimension != point2.dimension:
            raise ValueError("Points must have the same dimension")

    def __str__(self):
        if self.dimension == 2:
            return f"Line from {self.points[0]} to {self.points[1]} in 2D\n"
        else:
            return f"Line from {self.points[0]} to {self.points[1]} in 3D\n"

    def __repr__(self):
        if self.dimension == 2:
            return f"Line({self.points[0]}, {self.points[1]}) \n"
        else:
            return f"Line({self.points[0]}, {self.points[1]}) \n"

    @classmethod
    def from_dir(cls, point : Point, direction : Vec):
        """Create a line from a point and a direction vector."""
        if not isinstance(direction, Vec):
            raise TypeError("Expected a Vec object")
        if point.dimension != direction.dimension:
            raise ValueError("Point and direction must have the same dimension")
        
        if point.dimension == 2:
            return cls(point, point + direction)
        else:
            return cls(point, point + direction)

    @classmethod
    def from_angle(cls, point : Point, angle : Number, length : Number):
        """Create a line from a point and an angle."""
        
        x = point[0] + length * np.cos(angle)
        y = point[1] + length * np.sin(angle)
        return cls(point, Point(x, y))

    @classmethod
    def from_point_normal(cls, point : Point, normal : Vec, length : Number = 5):
        """Create a line from a point and a normal vector."""
        if not isinstance(normal, Vec):
            raise TypeError("Expected a Vec object")
        if point.dimension != normal.dimension:
            raise ValueError("Point and normal must have the same dimension")
        
        if point.dimension == 2:
            return cls(point, Point(point[0] + length * normal[1], point[1] + length * normal[0]))
        else:
            raise ValueError("Normal vector must be in 2D space")


    def as_homogeneous(self):
        p1 = self.points[0].as_homogeneous()
        p2 = self.points[1].as_homogeneous()
        if self.dimension != 2:
            return np.block([p1[3] * p2[:3] - p2[3] * p1[:3], np.cross(p1[0:3], p2[0:3])])
        return skewm(p1) @ p2 

    def has_point(self, point : Point):
        """Check if a point is on the line."""
        if not isinstance(point, Point):
            raise TypeError("Expected a Point object")
        if self.dimension != point.dimension:
            raise ValueError("Point and line must have the same dimension")
        
        if self.dimension == 2:
            return not(point[0] < min(self.points[0][0], self.points[1][0]) or point[0] > max(self.points[0][0], self.points[1][0]) or point[1] < min(self.points[0][1], self.points[1][1]) or point[1] > max(self.points[0][1], self.points[1][1]))
        else:
            return not(point[0] < min(self.points[0][0], self.points[1][0]) or point[0] > max(self.points[0][0], self.points[1][0]) or point[1] < min(self.points[0][1], self.points[1][1]) or point[1] > max(self.points[0][1], self.points[1][1]) or point[2] < min(self.points[0][2], self.points[1][2]) or point[2] > max(self.points[0][2], self.points[1][2]))

    def intersection(self, other, limit = True):
        """Calculate the intersection of two lines."""
        if not isinstance(other, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != other.dimension:
            raise ValueError("Lines must have the same dimension")
        
        if self.dimension == 2:
            if (self.direction_vect() @ other.direction_vect()).norm() == 0:
                if self.has_point(other.points[0]):
                    return other.points[0]
                elif self.has_point(other.points[1]):
                    return other.points[1]
                else:
                    return None
            inter = Point.from_homogeneous(skewm(self.as_homogeneous()) @ other.as_homogeneous())
            if inter is None:
                return None
            if limit:
                if not(self.has_point(inter) and other.has_point(inter)):
                    return None
            return inter
        else:
            L1 = self.as_homogeneous()
            L2 = other.as_homogeneous()
            l1 = L1[:3]
            l1p = L1[3:]
            l2 = L2[:3]
            l2p = L2[3:]

            if abs(np.dot(l1, l2p) + np.dot(l2, l1p)) > 1e-3:

                print("Lines don't intersect", np.dot(l1, l2p) + np.dot(l2, l1p))
                return
            
            n = np.cross(l1, l2)
            (v, v4) = (np.cross(n, l1), np.dot(n, l1p))

            return Point.from_homogeneous(np.block([-v4 * l2 + np.cross(v, l2p), np.dot(v, l2)]))
        
    # Find the lines which are the nearest of a line around a point
    def find_lines_near_point(self, point : Point, lines : list):
        nb_lines = len(lines)
        if nb_lines == 0:
            return [-1, -1]
        if nb_lines == 1:
            return [0, 0]
        
        angles = [np.pi * 2, np.pi * 2]
        lines_near = [-1, -1]
        for i in range(nb_lines):
            line2 = lines[i]
            angles_lines = self.angles_with(line2, point)
            if angles_lines is None:
                continue
            if angles_lines[0] < angles[0]:
                angles[0] = angles_lines[0]
                lines_near[0] = i
            if angles_lines[1] < angles[1]:
                angles[1] = angles_lines[1]
                lines_near[1] = i
        return lines_near
    
    def offset_intersect_lines(self, lines, pt : Point, d, L):
        # get the first line on the left and right 
        pts = []
        near_lines = self.find_lines_near_point(pt, [line[0] for line in lines])
        if near_lines[0] > -1:

            normal = self.normal_vect().normalize()
            # create the deplacement vector for the line
            dpl = [normal * v for v in (np.array([d + L/2, L/2, -d - L/2, -L/2]))] 

            # create the deplacement vector for the intersected lines
            o_l = [lines[near_lines[0]], lines[near_lines[1]]]

            if pt.distance(o_l[0][0][0]) < 1e-5:
                l1 = Line(pt, o_l[0][0][1])
            elif pt.distance(o_l[0][0][1]) < 1e-5:
                l1 = Line(pt, o_l[0][0][0])
            else:
                l1 = Line(pt, o_l[0][0][1]) if (Vec.from_2points(pt, o_l[0][0][0]) * normal) > 0 else Line(pt, o_l[0][0][0])

            if pt.distance(o_l[1][0][0]) < 1e-5:
                l2 = Line(pt, o_l[1][0][1])
            elif pt.distance(o_l[1][0][1]) < 1e-5:
                l2 = Line(pt, o_l[1][0][0])
            else:
                l2 = Line(pt, o_l[1][0][0]) if Vec.from_2points(pt, o_l[1][0][0]) * normal > 0 else Line(pt, o_l[1][0][1])
                
            v1 = l1.normal_vect().normalize() 
            v2 = l2.normal_vect().normalize()
            
            o_dpl = [ v1 * -( o_l[0][3] /2 + o_l[0][2]), v1 * -(o_l[0][3]/2), 
                    v2 * (o_l[0][3]/2 + o_l[1][2]), v2 * (o_l[0][3]/2) ]
            for k in range(2):
                ps = []
                for l in range(2):
                    ldp1 = self.copy()
                    ldp1.translate(dpl[k * 2 + l])
                    ldp2 = o_l[k][0].copy()
                    ldp2.translate(o_dpl[k * 2 + int(l ^ (o_l[k][1] == self.points[1]))])
                    p1 = ldp1.intersection(ldp2, False)
                    ps.append(p1)
                pts.append(ps)
        return pts

    def angles_with(self, other, intersection : Point | None = None):
        """Calculate the angle between two lines."""
        if not isinstance(other, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != other.dimension:
            raise ValueError("Lines must have the same dimension")
        
        if self.dimension == 2:
            # get the sin of the angle btween the twolines (||a x b|| / (||a||||b||) = |sin(theta)|)
            d1 = self.direction_vect()
            d2 = other.direction_vect()
            angle = d1.angle(d2)
            if intersection is None:
                return [angle, np.pi * 2 - angle]
            else:
                start = intersection.distance(self.points[0]) < 1e-6
                end = intersection.distance(self.points[1]) < 1e-6
                other_start = intersection.distance(other.points[0]) < 1e-6
                other_end = intersection.distance(other.points[1]) < 1e-6 

                if (start or end) and (other_start or other_end):
                    if start and other_start or end and other_end:
                        if angle > 0:
                            return [angle, np.pi * 2 - angle]
                        else:
                            return [np.pi * 2 + angle, - angle]
                    else:
                        if angle > 0:
                            return [angle + np.pi, np.pi - angle]
                        else:
                            return [np.pi + angle, np.pi - angle]
                else:
                    #TODO à refaire
                    print("Angle not implemented for this case")    
        else:
            raise ValueError("Angle not implemented for 3D lines")

    def length(self):
        return self.points[0].distance(self.points[1])
    
    def midpoint(self):
        return self.points[0] + (self.points[1] - self.points[0]) / 2 
    
    def slope(self):
        """Calculate the slope of the line."""
        if self.points[0][0] == self.points[1][0]:
            raise ValueError("Slope is undefined for vertical lines")
        return (self.points[1][1] - self.points[0][1]) / (self.points[1][0] - self.points[0][0])

    def direction_vect(self):
        """Calculate the direction vector of the line."""
        vect = Vec.from_2points(self.points[1], self.points[0])
        return vect.normalize() 
    
    def normal_vect(self):
        """Calculate the normal vector of the line."""
        vect = self.direction_vect()
        return Vec(-vect[1], vect[0])


    def get_line_dashed(self, dash_length : Number | None = None, dash_ratio : Number | None = None):
        """Get a dashed line representation of the line.
            start at the middle of the dash"""
        if dash_length is None:
            dash_length = self.dash_length
        if dash_ratio is None:
            dash_ratio = self.dash_ratio
        if dash_length <= 0:
            raise ValueError("Dash length must be positive")
        if dash_ratio <= 0 or dash_ratio >= 1:
            raise ValueError("Dash ratio must be between 0 and 1")
        
        line_length = self.points[0].distance(self.points[1])
        n_dashes = int(line_length / dash_length) + 1
        if line_length % dash_length > 0:
            n_dashes += 1
        dashes = []
        dir_line = Vec.from_2points(self.points[0], self.points[1]).normalize()
        for i in range(n_dashes):
            start = self.points[0] + dir_line * (i - 0.5 * dash_ratio) * dash_length
            if i == 0:
                start = self.points[0]
                end = start + dir_line * dash_length * dash_ratio * (0.5)
            elif i == n_dashes - 1:
                end = self.points[1]
            else:
                end = start + dir_line * dash_length * dash_ratio

            dashes.append(Line(start, end))
        return dashes

    def is_dashed(self):
        """Check if the line is dashed."""
        return self.dashed
    
    def thicken_to_rect(self, thickness : float | int):
        """Thicken the line by a given thickness."""
        if thickness <= 0:
            raise ValueError("Thickness must be positive")
        
        direct = self.normal_vect().normalize() * thickness
        p1 = self.points[0] + direct
        p2 = self.points[1] + direct
        p3 = self.points[1] - direct
        p4 = self.points[0] - direct
        
        return Polygon([p1, p2, p3, p4])
    
    def copy(self):
        """Create a copy of the line."""
        return Line(self.points[0].copy(), self.points[1].copy(), self.dashed, self.dash_length, self.dash_ratio)


@dataclass
class Surface(Shape):

    def __post_init__(self):
        super().__post_init__()
    
    def copy(self):
        """Create a copy of the surface."""
        return Surface([point.copy() for point in self.points])
    
    def in_3D(self):
        if self.dimension == 2:
            for i in range(len(self)):
                self.points[i] = Point(self.points[i][0], self.points[i][1], 0)
            self.dimension = 3
        return self
    
    def in_2D(self):
        if self.dimension == 3:
            normal = self.normal_vect()
            dir1 = (E3Y @ normal).normalize()
            if dir1.norm() == 0:
                dir1 = (normal @ E3Z).normalize()
            dir2 = (normal @ dir1).normalize()
            for i in range(len(self)):
                x = dir1 * self.points[i]
                y = dir2 * self.points[i]
                if isinstance(x, Vec) or isinstance(y, Vec):
                    raise ValueError("Cannot convert to 2D")
                self.points[i] = Point(x, y)
            self.dimension = 2
        return self

    def offset(self, d : float | int | list | tuple):
        """Offset the surface with one or differents values for each line"""
        if not isinstance(d, (list, tuple)):
            d = [d] * (len(self) - 1)
        if len(d) != len(self) - 1:
            raise ValueError("Offset must have the same length as the surface")
        
        plane_norm = self.normal_vect()
        lines = []
        surf = self.in_3D()

        for i in range(len(self) - 1):
            line = Line(surf.points[i], surf.points[i + 1])
            line_dir = line.direction_vect()
            depl = (plane_norm @ line_dir.in_3D()).normalize() * d[i]
            line.translate(depl)
            lines.append(line)
        pts = [lines[i].intersection(lines[(i + 1) % len(lines)]) for i in range(len(lines))]
        return Polygon(pts)

    def extrude(self, e : int | float,  symm = True ,positif = True, direction : Vec | None = None):
        """Extrude the surface to create a volume."""
        surfaces = [self.copy(), self.copy()]
        if self.dimension == 2:
            surfaces[0] = surfaces[0].in_3D()
            surfaces[1] = surfaces[1].in_3D()

        if direction is None:
            direction = self.normal_vect()
        
        dir_ext = direction * e * (1 if positif else -1)
        if symm:
            dir_ext = dir_ext / 2
            surfaces[0].translate(0-dir_ext)
        surfaces[1].translate(dir_ext)
        
        for i in range(len(self)):
            p1 = surfaces[0][i].copy()
            p2 = surfaces[0][(i + 1) % len(surfaces[0])].copy()
            p3 = surfaces[1][i].copy()
            p4 = surfaces[1][(i + 1) % len(surfaces[0])].copy()
            surfaces.append(Surface([p1, p2, p4, p3]))
        return Volume(surfaces)

    def triangulate(self):
        """Triangulate the surface."""
        triangles = []
        for i in range(1, len(self) - 2):
            triangles.append(Surface([self[0], self[i], self[i + 1]]))
        return triangles

    def edges(self):
        """Get the edges of the surface."""
        edges = []
        for i in range(len(self) - 1):
            edges.append(Line(self[i], self[(i + 1) % len(self)]))
        return edges
    
    def normal_vect(self):
        """Calculate the normal vector of the polygon."""

        a = Vec.from_2points(self[0], self[1])
        b = Vec.from_2points(self[0], self[2])
        return (a @ b).normalize()

    def get_lines(self):
        """Get the lines of the surface."""
        lines = []
        for i in range(len(self) - 1):
            lines.append(Line(self[i], self[(i + 1) % len(self)]))
        return lines
    


@dataclass
class Circle(Surface):
    """ A class representing a circle defined by its center and radius. """
    radius : Number
    normal : Vec | None = None 

    def __init__(self, center : Point, radius : Number, normal : Vec | None = None):
        if not isinstance(center, Point):
            raise TypeError("Center must be a Point object")
        if center.dimension != 2 and (normal is None or not isinstance(normal, Vec)):
            raise ValueError("Normal vector must be provided for 3D circles")
        
        if center.dimension == 2:
            super().__init__([center])
        
        self.radius = radius
        self.normal = normal

        super().__post_init__()

    @classmethod
    def from_center_and_point(cls, center : Point, point_on_circle : Point):
        """Create a circle from its center and a point on the circle."""
        if not isinstance(center, Point) or not isinstance(point_on_circle, Point):
            raise ValueError("center and point_on_circle must be Point objects")
        if center.dimension != 2 or point_on_circle.dimension != 2:
            raise ValueError("center and point_on_circle must be in 2D space")
        
        radius = center.distance(point_on_circle)
        return cls(center, radius)
    
    @classmethod
    def from_diameter(cls, point1 : Point, point2 : Point):
        """Create a circle from two points on the diameter."""
        if not isinstance(point1, Point) or not isinstance(point2, Point):
            raise ValueError("point1 and point2 must be Point objects")
        if point1.dimension != 2 or point2.dimension != 2:
            raise ValueError("point1 and point2 must be in 2D space")
        
        center = (point1 + point2) / 2
        radius = center.distance(point1)
        return cls(center, radius)

    def __repr__(self):
        return f"Circle({self[0]}, {self.radius})"
    
    def copy(self):
        """Create a copy of the circle."""
        return Circle(self[0].copy(), self.radius)


@dataclass
class Rectangle(Surface):
    """ A class representing a rectangle defined by its top left corner and its width and height. """
    width : Number = 0
    height : Number = 0

    def __post_init__(self):
        if not isinstance(self[0], Point) :
            raise TypeError("Rectangle points must be Point objects")
        if len(self) != 2:
            raise ValueError("Rectangle must have two points")
        if self.width < 0 or self.height < 0:
            raise ValueError("Width and height must be positive")
        
        if self[0][0] > self[1][0]:
            self[0][0], self[1][0] = self[1][0], self[0][0]
        if self[0][1] > self[1][1]:
            self[0][1], self[1][1] = self[1][1], self[0][1]
        
        self.width = abs(self[0][0] - self[1][0])
        self.height = abs(self[0][1] - self[1][1])

        super().__post_init__()

    @classmethod
    def from_points(cls, point1 : Point, point2 : Point):
        """Create a rectangle from two points."""
        if not isinstance(point1, Point) or not isinstance(point2, Point):
            raise ValueError("point1 and point2 must be Point objects")
        if point1.dimension != 2 or point2.dimension != 2:
            raise ValueError("point1 and point2 must be in 2D space")
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        return cls([point1, point2], width, height)
    
    @classmethod
    def from_cs(cls, corner : Point, width : Number, height : Number):
        """Create a rectangle from its top left corner and its width and height."""
        if not isinstance(corner, Point):
            raise ValueError("corner must be a Point object")
        if corner.dimension != 2:
            raise ValueError("corner must be in 2D space")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        return cls([corner, Point(corner[0] + width, corner[1] + height)], width, height)

    def __repr__(self):
        return f"Rectangle : top left : {self[0][0], self[0][1]} ; bottom right : {self[1][0], self[1][1]} ; width : {self.width} ; height : {self.height}"
                

    def area(self):
        return self.width * self.height
    
    def copy(self):
        """Create a copy of the rectangle."""
        return Rectangle([self[0].copy(), self[1].copy()], self.width, self.height)
    
    def mirror(self, plane):
        super().mirror(plane)
        if self[0][0] > self[1][0]:
            self[0][0], self[1][0] = self[1][0], self[0][0]
        if self[0][1] > self[1][1]:
            self[0][1], self[1][1] = self[1][1], self[0][1]


@dataclass
class Polygon(Surface):
    """ A class representing a polygon defined by its vertices. """
    n_points : int = 0


    def __post_init__(self):
        if len(self) < 3:
            raise ValueError("Polygon must have at least 3 points")
        # Close the polygon by adding the first point at the end
        
        self.n_points = len(self)

        super().__post_init__()

    def __repr__(self):
        return f"Polygon({super().__repr__()})"
    
    def copy(self):
        """Create a copy of the polygon."""
        return Polygon([point.copy() for point in self])

@dataclass
class RegularPolygon(Polygon):
    """ A class representing a regular polygon defined by its center, radius, and number of points. """
    plane : Plane | None = None
    radius : Number = 0
    rotation : Number = 0

    @classmethod
    def from_center_and_radius(cls, center : Point, radius : Number, n_points : int, rotation : Number = 0, plane : Plane | None = None):
        """Create a regular polygon from its center, radius, and number of points."""
        if not isinstance(center, Point):
            raise ValueError("center must be a Point object")
        if center.dimension != 2 and plane is None:
            raise ValueError("center must be in 2D space or plane must be specified for 3D polygons")
        angle = 2 * np.pi / n_points 
        points = []

        if center.dimension == 2:
            for i in range(n_points):
                x = center[0] + radius * np.cos(i * angle + rotation)
                y = center[1] + radius * np.sin(i * angle + rotation)
                points.append(Point(x, y))
        else :
            if plane is None:
                raise ValueError("Plane must be specified for 3D polygons")
            points.append(Vec.from_2points(center, plane.point).normalize() * radius + center)
            rot = axe_angle_to_mat_rotation(plane.normal, rotation)
            for i in range(n_points):
                points.append(np.dot(rot, points[-1] - center) + center)
        
        return cls(points, n_points, plane, radius, rotation)
    

    def copy(self):
        """Create a copy of the regular polygon."""
        return RegularPolygon(self.points.copy(), self.n_points, self.plane, self.radius, self.rotation)
    
@dataclass    
class Volume():
    """ A class representing a volume defined by a list of surfaces. """
    surfaces : Sequence[Surface] 
    n_surfaces : int = 0

    def __post_init__(self):
        if len(self) < 1:
            raise ValueError("Volume must have at least one surface")
        for i in range(len(self)):
            if self[i].dimension != 3:
                raise ValueError("All surfaces must be in 3D")
        
        self.n_surfaces = len(self)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.surfaces[index]
        else:
            raise TypeError("Index must be an integer or a slice")
    
    def __len__(self):
        return len(self.surfaces)

    def copy(self):
        """Create a copy of the volume."""
        return Volume([point.copy() for point in self])
    

    def mesh_3D(self):
        """Create a 3D mesh representation of the volume."""
        
        surfaces = []
        for surface in self:
            surfaces.extend(surface.triangulate())
        volume_mesh = mesh.Mesh(np.zeros(len(surfaces), dtype=mesh.Mesh.dtype))
        for i, surface in enumerate(surfaces):
            for j in range(3):
                volume_mesh.vectors[i][j] = surface[j].as_array() # type: ignore
        
        return volume_mesh
    
    def __str__(self):
        """String representation of the volume."""
        pres = ""
        for surface in self:
            pres += str(surface) + "\n"
        return pres

    def wireframe(self):
        """Create a wireframe representation of the volume."""
        lines = []
        for surface in self:
            lines.extend(surface.edges())
        return lines
    
    def __add__(self, other):
        """add two volumes or surfaces together."""
        if not isinstance(other, (Volume, Surface)):
            raise ValueError("Other must be a Volume object")
        

        surfaces = []
        for surface in self:
            surfaces.append(surface.copy())
        if isinstance(other, Surface):
            surfaces.append(other.copy())
        else:
            for surface in other:
                surfaces.append(surface.copy())
        return Volume(surfaces)
    
    def show(self, save = False, path = None):
        ax = plt.figure().add_subplot(111, projection='3d')
        
        volume_mesh = self.mesh_3D()

        poly = Poly3DCollection(volume_mesh.vectors, alpha=0.5) # type: ignore
        poly.set_edgecolor('0')
        # Auto scale to the mesh size
        scale = volume_mesh.points.flatten() # type: ignore
        scalex = []
        scaley = []
        scalez = []
        for i in range(int(len(scale) / 3)):
            scalex.append(scale[i * 3])
            scaley.append(scale[i * 3 + 1])
            scalez.append(scale[i * 3 + 2])
        ax.auto_scale_xyz(scalex, scaley, scalez) # type: ignore
        ax.add_collection3d(poly) # type: ignore
        ax.set_aspect('equal')

        ax.set_title('3D Volume')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore
        if save:
            if path is None:
                path = 'volume.png'
            plt.savefig(path)
            plt.close()
            # print("Volume saved to volume.png")
        else:
            plt.show() 

    def translate(self, v : Vec):
        """Translate the volume by a given vector."""
        if not isinstance(v, (Point, Vec)):
            raise ValueError("Translation vector must be a Point object")
        
        for i in range(len(self)):
            self[i].translate(v) 
    
    def save(self, filename : str):
        """Save the volume to a file."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        if not filename.endswith('.stl'):
            filename += '.stl'
        
        volume_mesh = self.mesh_3D()
        volume_mesh.save(filename) # type: ignore
        print(f"Volume saved to {filename}")

@dataclass
class Parallelepiped(Volume):
    pass