from .vector import *

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
            rotation_matrix = axis.to_roation_matrix(angle)

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
