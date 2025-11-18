from .vector import *
if TYPE_CHECKING:
    from .line import Line


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
        
    def __setitem__(self, index, value):
        if isinstance(index, int):
            if not isinstance(value, Point):
                raise TypeError("Value must be a Point object")
            if value.dimension != self.dimension:
                raise ValueError("Point must have the same dimension as the shape")
            self.points[index] = value.copy()
        else:
            raise TypeError("Index must be an integer")

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
                plt.plot([self[i % nb_pts][0] for i in range(nb_pts + 1)], [self[i % nb_pts][1] for i in range(nb_pts + 1)])
            else :
                fig, ax = plt.subplots()
                ax.plot([self[i % nb_pts][0] for i in range(nb_pts + 1)], [self[i % nb_pts][1] for i in range(nb_pts + 1)])
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

    def change_direction(self):
        """Change the direction of the shape."""
        if self.dimension == 2:
            self.points.reverse()
        else:
            raise ValueError("Change direction is only implemented for 2D shapes")
    
    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)) -> svg.Element:
        path_svg = []
        path_svg.append(svg.M(mm_to_px(self.points[0][0]), mm_to_px(self.points[0][1])))
        for point in self.points[1:]:	
            p = point + origin	
            p = point + origin	
            path_svg.append(svg.L(mm_to_px(p[0]), mm_to_px(p[1])))
        return svg.Path(d=path_svg, stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)