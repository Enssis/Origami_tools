from .shape import *
from .line import Line
from .repere import Repere
from .plane import Plane

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
    
    def in_2D(self, repere : Repere | None = None):
        if self.dimension == 3:
            normal = self.normal_vect()
            dir1 = (E3Y @ normal).normalize() if repere is None else repere.axis[0]
            if dir1.norm() == 0:
                dir1 = (normal @ E3Z).normalize()
            dir2 = (normal @ dir1).normalize() if repere is None else repere.axis[1]
            for i in range(len(self)):
                x = dir1 * self.points[i]
                y = dir2 * self.points[i]
                if isinstance(x, Vec) or isinstance(y, Vec):
                    raise ValueError("Cannot convert to 2D")
                self.points[i] = Point(x, y)
            self.dimension = 2
        return self

    def get_dashed(self, dash_length : Number | None = None, dash_ratio : Number | None = None):
        """Get a dashed representation of the surface."""
        if dash_length is None:
            dash_length = 6
        if dash_ratio is None:
            dash_ratio = 0.5
        
        lines = self.get_lines()
        dashed_lines = []
        for line in lines:
            dashed_lines.extend(line.get_line_dashed(dash_length, dash_ratio))
        return dashed_lines

    def offset(self, d: float | int | list | tuple):
        """Offset the surface by displacing each edge perpendicularly within the plane.
        Positive values shrink the shape, negative values expand it.
        """
        # Convert scalar to per-edge offset list
        if not isinstance(d, (list, tuple)):
            d = [d] * (len(self) - 1)
        if len(d) != len(self) - 1:
            raise ValueError("Offset must match the number of edges")

        # Ensure computation in 3D space
        surf = self.in_3D()
        plane_norm = surf.normal_vect()

        lines: list[Line] = []

        for i in range(len(surf) - 1):
            line = Line(surf.points[i], surf.points[i + 1])
            line_dir = line.direction_vect()

            # Compute perpendicular vector in plane
            perp_dir = (plane_norm @ line_dir).normalize() * d[i]
            line.translate(perp_dir)
            lines.append(line)

        # Intersect each pair of consecutive lines to get new corners
        pts = []
        for i in range(len(lines)):
            inter = lines[i].intersection(lines[(i + 1) % len(lines)], False)
            if inter is not None:
                pts.append(inter)

        return type(self)(pts)


    def extrude(self, e : int | float,  symm = True ,positif = True, direction : Vec | None = None):
        """Extrude the surface to create a volume."""

        from .volume import Volume

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
    
    def normal_vect(self) -> Vec:
        """Compute an average normal vector for a polygon.
        
        Works for convex and non-convex polygons, assuming approximate planarity.
        """
        if len(self) < 3:
            raise ValueError("Polygon must have at least 3 points")

        normal = Vec(0, 0, 0)
        for i in range(len(self) - 2):
            a = Vec.from_2points(self[i], self[i + 1])
            b = Vec.from_2points(self[i + 1], self[i + 2])
            n = a @ b  # Cross product
            normal = normal +  n
        if normal.norm() == 0:
            raise ValueError("Degenerate polygon with zero normal vector")

        return normal.normalize()


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

    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
        """Convert the circle to an SVG path."""
        center = self[0] + origin
        return svg.Circle(cx=svg.Length(center[0], "mm"), cy=svg.Length(center[1], "mm"), r=svg.Length(self.radius, "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)

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

    def get_corners(self):
        """Get the corners of the rectangle."""
        if self.dimension != 2:
            raise ValueError("Rectangle must be in 2D")
        return [
            self[0],
            Point(self[1][0], self[0][1]),
            self[1],
            Point(self[0][0], self[1][1])
        ]
    
    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
        """Convert the rectangle to an SVG path."""
        top_left = self[0] + origin
        return svg.Rect(x=svg.Length(top_left[0], "mm"), y=svg.Length(top_left[1], "mm"), width=svg.Length(self.width, "mm"), height=svg.Length(self.height, "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)

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
        return super().__repr__()
    
    def copy(self):
        """Create a copy of the polygon."""
        return Polygon([point.copy() for point in self])

    def cut(self, line: Line, decals: list[Number] | None = None):
        """
        Cut the polygon by a given line.

        Args:
            line (Line): The cutting line.
            decals (list[Number] | None): Optional offset values for each polygon edge.
                Expected length: number of polygon edges (len(self)) + 1 for the new edge.

        Returns:
            tuple: 
                - List of Polygon objects created by the cut.
                - List of decal lists corresponding to each polygon.
        """
        n_edges = len(self)

        # Check decal length consistency
        if decals is not None:
            if len(decals) != n_edges:
                raise ValueError("decals must have the same length as the number of polygon edges + 1")

        # Build list of points with inserted intersections
        j = 1
        pts = [self[0].copy()]
        intersections : list[int] = []
        insert_decals = [decals[0]] if not decals is None else [] # decals.copy() if decals is not None else []

        for i in range(n_edges - 1):
            seg = Line(self[i], self[i + 1])
            inter = line.intersection(seg)
            if inter is not None:
                if inter == self[i + 1]:
                    intersections.append(j)
                elif inter != self[i]:
                    pts.append(inter)
                    intersections.append(j)
                    if decals is not None:
                        # Insert matching decal value at the new edge
                        insert_decals.append(decals[(i) % (n_edges - 1)])
                    j += 1
            
            if decals is not None:
                insert_decals.append(decals[(i + 1) % (n_edges - 1)])
            # No intersection, just copy the point
            pts.append(self[i + 1].copy())
            j += 1
        if decals is not None:
            insert_decals.append(decals[-1])

        # If no intersection or odd number → invalid cut
        if len(intersections) % 2 != 0 or len(intersections) == 0:
            return [self.copy()], [decals[:-1]] if decals is not None else []

        # Build resulting sub-polygons
        start = intersections[0]
        polygons = []
        new_pts = [pts[start].copy()]
        n_decals = [[insert_decals[start]] if decals is not None else []]
        n_poly = 0

        for i in range(j - 1):
            index = (i + start) % (j - 1) + 1
            new_pts.append(pts[index].copy())

            if decals is not None:
                n_decals[n_poly].append(insert_decals[index])

            if index in intersections:
                # Close current polygon
                new_pts.append(new_pts[0].copy())
                polygons.append(Polygon(new_pts).in_2D())
                new_pts = [pts[index].copy()]

                if decals is not None:
                    # The last inserted decal is the one for the added segment
                    n_decals[n_poly][-1] = insert_decals[-1]
                n_poly += 1
                n_decals.append(
                    [insert_decals[index]] if decals is not None else []
                )

        return polygons, n_decals

    
    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
        """Convert the polygon to an SVG path."""
        pts = []
        for point in self:
            p = point + origin
            pts.append([mm_to_px(p[0]), mm_to_px(p[1])])
        return svg.Polygon(points=pts, stroke=color, stroke_opacity=opacity, fill=fill, stroke_width=width)
    

@dataclass
class HoledPolygon(Surface):
    outside : Polygon
    holes : list[Polygon]

    def __post_init__(self):
        if len(self) < 3:
            raise ValueError("Polygon must have at least 3 points")
        # Close the polygon by adding the first point at the end

        super().__post_init__()

    def __str__(self):
        return f"HoledPolygon(outside :{self.outside}, holes : {self.holes})"

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """Create a copy of the holed polygon."""
        return HoledPolygon(self.points.copy(), self.outside.copy(), [hole.copy() for hole in self.holes])

    @classmethod
    def from_polygons(cls, outside : Polygon, holes : list[Polygon]):
        """Create a holed polygon from an outside polygon and a list of holes."""
        if not isinstance(outside, Polygon):
            raise TypeError("outside must be a Polygon object")
        if not all(isinstance(hole, Polygon) for hole in holes):
            raise TypeError("holes must be a list of Polygon objects")
        if len(holes) < 1:
            raise ValueError("HoledPolygon must have at least one hole")
        pts = outside.points.copy()
        for hole in holes:
            pts.extend(hole.points.copy())
            hole = hole.copy()
        return cls(pts, outside.copy(), holes)
    
    def update(self):
        n = self.outside.n_points
        self.outside = Polygon(self.points[:n])
        holes = []
        for hole in self.holes:
            holes.append(Polygon(self.points[n:n + hole.n_points]))
            n += hole.n_points
        self.holes = holes


    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
        """Convert the holed polygon to an SVG path."""
        path_svg = []
        # self.update()
        polygons = [self.outside] + self.holes
        for polygon in polygons:
            if polygon[0] != polygon[-1] :
                polygon.points.append(polygon[0])
            path_svg.append(svg.M(mm_to_px(polygon[0][0] + origin[0]), mm_to_px(polygon[0][1] + origin[1])))
            for point in polygon.points[1:]:	
                p = point + origin	
                p = point + origin	
                path_svg.append(svg.L(mm_to_px(p[0]), mm_to_px(p[1])))
        return svg.Path(d=path_svg, stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)
    
    



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
            rot = plane.normal.to_roation_matrix(rotation)
            for i in range(n_points):
                points.append(np.dot(rot, points[-1] - center) + center)
        
        return cls(points, n_points, plane, radius, rotation)
    

    def copy(self):
        """Create a copy of the regular polygon."""
        return RegularPolygon(self.points.copy(), self.n_points, self.plane, self.radius, self.rotation)
    