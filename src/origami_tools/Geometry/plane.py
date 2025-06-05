from .vector import *

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
