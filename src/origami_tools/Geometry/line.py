from dataclasses import field
from .vector import *
from .shape import Shape
from . import skewm

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

    def __eq__(self, other):
        """Check if two lines are equal."""
        if not isinstance(other, Line):
            return False
        if self.dimension != other.dimension:
            return False
        return (self.points[0] == other.points[0] and self.points[1] == other.points[1]) or \
               (self.points[0] == other.points[1] and self.points[1] == other.points[0])

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

    def has_point(self, point: Point) -> bool:
        """
        Check if a given point lies on the line segment.
        
        Conditions:
        - Must be within the bounding box of the segment (with tolerance).
        - Must be colinear with the segment's direction vector.
        """
        if not isinstance(point, Point):
            raise TypeError("Expected a Point object")
        if self.dimension != point.dimension:
            raise ValueError("Point and line must have the same dimension")

        tol = 1e-6  # Numerical tolerance for floating point comparisons

        # Bounding box check
        for i in range(self.dimension):
            if not (min(self.points[0][i], self.points[1][i]) - tol <= point[i] <= max(self.points[0][i], self.points[1][i]) + tol):
                return False

        # Colinearity check: vector from A to point must be colinear with direction vector
        v = self.points[1] - self.points[0]
        u = point - self.points[0]

        if self.dimension == 2:
            # 2D cross product (z-component)
            cross = v[0] * u[1] - v[1] * u[0]
            return abs(cross) < tol
        elif self.dimension == 3:
            # 3D cross product should be near zero vector
            cross = v @ u 
            return cross.norm() < tol
        else:
            raise NotImplementedError("has_point not implemented for dimension > 3")


    def intersection(self, other, limit = True):
        """Calculate the intersection of two lines."""
        if not isinstance(other, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != other.dimension:
            raise ValueError("Lines must have the same dimension")
        

        if self.dimension == 2:
            if (self.direction_vect() @ other.direction_vect()).norm() < 1e-3:
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
                return None
            
            n = np.cross(l1, l2)
            (v, v4) = (np.cross(n, l1), np.dot(n, l1p))

            return Point.from_homogeneous(np.block([-v4 * l2 + np.cross(v, l2p), np.dot(v, l2)]))
        
    def shape_intersection(self, other : 'Shape') -> list[Point]:
        """Calculate the intersection of a line with a shape."""
        if not isinstance(other, Shape):
            raise TypeError("Expected a Shape object")
        if self.dimension != other.dimension:
            raise ValueError("Line and shape must have the same dimension")
        
        intersections = []
        for i in range(len(other) - 1):
            line2 = Line(other.points[i], other.points[i + 1])
            inter = self.intersection(line2, limit=False)
            if inter is not None:
                intersections.append(inter)
        
        return intersections

    # Find the lines which are the nearest of a line around a point
    def find_lines_near_point(self, point : Point, lines : list):
        """Find the two lines that are nearest to a point.
        Returns the indices of the two nearest lines."""

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
                    v2 * (o_l[1][3]/2 + o_l[1][2]), v2 * (o_l[1][3]/2) ]
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

    def angles_with(self, other, intersection: Point | None = None) -> list[Number]:
        """
        Compute the two possible angles between self and another Line.
        
        If an intersection point is provided, the direction vectors are aligned accordingly
        so that angles are oriented from the intersection. This helps determine
        the 'left' and 'right' angular difference around the intersection.

        Returns two angles in radians, always in the range [0, 2π).
        """
        if not isinstance(other, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != other.dimension:
            raise ValueError("Lines must have the same dimension")

        if self.dimension == 2:
            # Get direction vectors
            d1 = self.direction_vect()
            d2 = other.direction_vect()

            # If no intersection info: return both possible unsigned angles
            if intersection is None:
                angle = d1.angle(d2)
                return [angle, (2 * np.pi - angle) % (2 * np.pi)]

            # Align direction vectors so they "emanate" from the intersection point
            if intersection == self.points[1]:
                d1 = -d1
            # elif intersection != self.points[0]:
            #     raise ValueError("Intersection point does not belong to the line")

            if intersection == other.points[1]:
                d2 = -d2
            # elif intersection != other.points[0]:
            #     raise ValueError("Intersection point does not belong to the other line")

            # Compute the signed angle from d1 to d2
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            dot = d1 * d2
            angle = float(np.arctan2(cross, dot) % (2 * np.pi))

            # Return angle and its supplementary (i.e., the reflex angle)
            return [angle, (2 * np.pi - angle) % (2 * np.pi)]
        else:
            raise ValueError("Angle computation not implemented for 3D lines")


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


    def get_line_dashed(self, dash_length: Number | None = None, dash_ratio: Number | None = None):
        """
        Generate a dashed version of the line, starting with a visible dash,
        and including dashes at both endpoints even if partial.
        """
        if dash_length is None:
            dash_length = self.dash_length
        if dash_ratio is None:
            dash_ratio = self.dash_ratio

        if dash_length <= 0:
            raise ValueError("Dash length must be positive")
        if not (0 < dash_ratio < 1):
            raise ValueError("Dash ratio must be strictly between 0 and 1")

        p0, p1 = self.points
        total_length = p0.distance(p1)
        dir_vec = Vec.from_2points(p0, p1).normalize()

        dash_size = dash_length * dash_ratio
        gap_size = dash_length * (1 - dash_ratio)
        pattern_size = dash_size + gap_size

        n_patterns = int((total_length) // pattern_size)

        dashes = []
        start_dec = (total_length - (n_patterns * pattern_size + gap_size) ) / 2
        cursor = p0 + dir_vec * (start_dec + gap_size)
        if start_dec > 0.2:
            dashes.append(Line(p0, cursor - dir_vec * gap_size))

        for _ in range(n_patterns):
            dash_start = cursor
            dash_end = dash_start + dir_vec * dash_size
            if p0.distance(dash_start) >= total_length:
                break
            if dash_end.distance(p0) > total_length:
                dash_end = p1

            dash = Line(dash_start, dash_end)
            if dash.length() > 0.5:  # Avoid zero-length dashes
                dashes.append(dash)

            cursor = dash_end + dir_vec * gap_size
            if cursor.distance(p0) >= total_length:
                break

        # Force final dash if not aligned exactly at end
        if dashes:
            last_end = dashes[-1].points[1] + gap_size * dir_vec
            if Vec.from_2points(last_end, p1) * dir_vec > 0 : 
                if last_end.distance(p1) > 0.2:
                    if last_end.distance(p1) < dash_size:
                        dashes.append(Line(last_end, p1))

        return dashes

    def is_dashed(self):
        """Check if the line is dashed."""
        return self.dashed
    
    def thicken_to_rect(self, thickness : float | int):
        """Thicken the line by a given thickness."""
        if thickness <= 0:
            raise ValueError("Thickness must be positive")
        
        from .surface import Polygon

        direct = self.normal_vect().normalize() * thickness
        p1 = self.points[0] + direct
        p2 = self.points[1] + direct
        p3 = self.points[1] - direct
        p4 = self.points[0] - direct
        
        return Polygon([p1, p2, p3, p4])
    
    def copy(self):
        """Create a copy of the line."""
        return Line(self.points[0].copy(), self.points[1].copy(), self.dashed, self.dash_length, self.dash_ratio)

    def as_svg(self, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
        start = self.points[0] + origin
        end = self.points[1] + origin
        if self.is_dashed():
            dash_full = self.dash_ratio * self.dash_length
            dash_empty = self.dash_length - dash_full
            return svg.Line(x1=svg.Length(start[0], "mm"), y1=svg.Length(start[1], "mm"), x2=svg.Length(end[0], "mm"), y2=svg.Length(end[1], "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, stroke_dasharray=[dash_full, dash_empty])
        return svg.Line(x1=svg.Length(start[0], "mm"), y1=svg.Length(start[1], "mm"), x2=svg.Length(end[0], "mm"), y2=svg.Length(end[1], "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width)
    
    def coefficients(self):
        """Return the equation of the line in the form Ax + By + C = 0."""
        if self.dimension != 2:
            raise NotImplementedError("Equation not implemented for dimensions other than 2")
        
        A = self.points[1][1] - self.points[0][1]
        B = self.points[0][0] - self.points[1][0]
        C = A * self.points[0][0] + B * self.points[0][1]
        
        return A, B, -C
        

@dataclass
class MultiLine(Line):
    break_points : list[Point] = field(default_factory=list)

    @classmethod
    def from_lines(cls, lines: list["Line"]):
        """
        Create a MultiLine from a list of Line objects.
        The lines are connected if their endpoints match.

        Parameters:
            lines (list[Line]): List of Line objects to be connected.

        Returns:
            MultiLine or None if the connection fails.
        """
        if not lines:
            raise ValueError("List of lines cannot be empty.")

        # Extract list of point sequences from each line
        segments = [line.points for line in lines]

        # Track original indices of the lines to preserve connection order
        order = [[i] for i in range(len(segments))]

        while len(segments) > 1:
            for i in range(1, len(segments)):
                # Case 1: segments[i].start == segments[0].start → reverse segment 0
                if segments[i][0] == segments[0][0]:
                    segments[0] = segments[0][::-1][:-1] + segments[i]
                    order[0] = order[0][::-1] + order[i]
                    break

                # Case 2: segments[i].end == segments[0].start → prepend i to 0
                elif segments[i][-1] == segments[0][0]:
                    segments[0] = segments[i] + segments[0][1:]
                    order[0] = order[i] + order[0]
                    break

                # Case 3: segments[i].start == segments[0].end → append i to 0
                elif segments[i][0] == segments[0][-1]:
                    segments[0] = segments[0][:-1] + segments[i]
                    order[0] = order[0] + order[i]
                    break

                # Case 4: segments[i].end == segments[0].end → reverse segment 0 and append i
                elif segments[i][-1] == segments[0][-1]:
                    segments[0] = segments[i] + segments[0][::-1][1:]
                    order[0] = order[i] + order[0][::-1]
                    break

            else:
                # No connection found → cannot merge further
                print("No connection found for segment:", segments[0])
                print("Remaining segments:", segments)
                return None, None

            # Remove the used segment and its order
            segments.pop(i)
            order.pop(i)

        # At this point, segments[0] contains the full connected polyline

        return cls([segments[0][0], segments[0][-1]], dashed=lines[0].dashed, dash_length=lines[0].dash_length, dash_ratio=lines[0].dash_ratio, break_points=segments[0][1:-1]), order[0]

        
    def intersection(self, other, limit = True):
        """Calculate the intersection with a line"""
        points = [self[0]] + self.break_points + [self[-1]]
        for i in range(len(points) - 1, 0, -1):
            line = Line(points[i - 1], points[i])
            inter = line.intersection(other, limit=limit)
            if inter is not None:
                return inter
        return None
        
    def show(self, ax = None, show = False):
        """Show the surface."""
        points = [self[0]] + self.break_points + [self[-1]]
        nb_pts = len(points)
        if self.dimension == 2:
            if ax is None:
                fig, ax = plt.subplots()
            plt.plot([points[i % nb_pts][0] for i in range(nb_pts + 1)], [points[i % nb_pts][1] for i in range(nb_pts + 1)])
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            ax.plot([points[i % nb_pts][0] for i in range(nb_pts + 1)], [points[i % nb_pts][1] for i in range(nb_pts + 1)], [points[i % nb_pts][2] for i in range(nb_pts + 1)])
        if show:
            plt.show()
        else :
            return ax
        

    def direction_vect(self):
        """Calculate the direction vector of the line."""
                    
        vect = Vec.from_2points(self.points[0], self.break_points[0])
        for i in range(len(self.break_points)):
            if i == len(self.break_points) - 1:
                vect += Vec.from_2points(self.break_points[i], self.points[1])
            else:
                vect += Vec.from_2points(self.break_points[i], self.break_points[i + 1])
        return vect.normalize()
    
    def add_line(self, line: Line):
        """Add a line to the MultiLine. return -1 if the line is added at the end, 0 if it is added at the beginning."""
        if not isinstance(line, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != line.dimension:
            raise ValueError("Line must have the same dimension as MultiLine")
        
        # Check if the line can be added
        if self.points[-1] == line.points[0]:
            self.break_points.append(line.points[0])
            self.points[-1] = line.points[1]
            return -1
        elif self.points[0] == line.points[1]:
            self.break_points.insert(0, line.points[1])
            self.points[0] = line.points[0]
            return 0
        elif self.points[0] == line.points[0]:
            self.break_points.insert(0, line.points[0])
            self.points[0] = line.points[1]
            return 0
        elif self.points[-1] == line.points[1]:
            self.break_points.append(line.points[1])
            self.points[-1] = line.points[0]
            return -1
        else:
            raise ValueError("Line does not connect to MultiLine")
        
    def __str__(self):
        
        return f"MultiLine from {self.points[0]} to {self.points[1]} with {self.break_points} break points in {self.dimension}D\n"
    
    def __repr__(self):
        return f"MultiLine({self.points[0]}, {self.points[1]}, break_points={self.break_points}) \n"
    
    def get_lines(self):
        """Get the list of lines that compose the MultiLine."""
        lines = []
        points = [self[0]] + self.break_points + [self[-1]]
        for i in range(len(points) - 1):
            lines.append(Line(points[i], points[i + 1], self.dashed, self.dash_length, self.dash_ratio))
        return lines
    
    def has_line(self, line: Line):
        """Check if the MultiLine contains a line."""
        if not isinstance(line, Line):
            raise TypeError("Expected a Line object")
        if self.dimension != line.dimension:
            raise ValueError("Line must have the same dimension as MultiLine")
        
        points = [self[0]] + self.break_points + [self[-1]]
        for i in range(len(points) - 1):
            if Line(points[i], points[i + 1]) == line:
                return True
        return False