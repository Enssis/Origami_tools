import svg
from ..geometry import Point, Circle, Rectangle, Line, Polygon, Shape
from ._types import Number

def mm_str(value):
	""" 
		Transform a value in mm to a string with "mm" at the end
	"""
	return "{:.2f}mm".format(value) 

def px_to_mm(px):
	""" 
		Transform a value in px to mm
	"""
	return px / 96 * 25.4

def mm_to_px(mm):
	"""
		Transform a value in mm to px
	"""
	return mm / 25.4 * 96

def svg_circle_from_circle(circle : Circle, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
	"""
		Transform a circle in a SVG circle
	"""
	center = circle[0] + origin
	return svg.Circle(cx=svg.Length(center[0], "mm"), cy=svg.Length(center[1], "mm"), r=svg.Length(circle.radius, "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill) 

def svg_rect_from_rectangle(rectangle : Rectangle, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
	"""
		Transform a rectangle in a SVG rectangle
	"""
	top_left = rectangle[0] + origin
	return svg.Rect(x=svg.Length(top_left[0], "mm"), y=svg.Length(top_left[1], "mm"), width=svg.Length(rectangle.width, "mm"), height=svg.Length(rectangle.height, "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)

def svg_polygon_from_polygon(polygon : Polygon, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
	"""
		Transform a polygon in a SVG polygon
	"""
	pts = []
	for point in polygon:
		p = point + origin
		pts.append([mm_to_px(p[0]), mm_to_px(p[1])])
	return svg.Polygon(points=pts, stroke=color, stroke_opacity=opacity, fill=fill, stroke_width=width)

def svg_path_from_shape(path : Shape, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
	"""
		Transform a path in a SVG path
	"""
	path_svg = []
	path_svg.append(svg.M(mm_to_px(path[0][0]), mm_to_px(path[0][1])))
	for point in path[1:]:	
		p = point + origin	
		path_svg.append(svg.L(mm_to_px(p[0]), mm_to_px(p[1])))
	return svg.Path(d=path_svg, stroke=color, stroke_opacity=opacity, stroke_width=width, fill=fill)


def svg_shape_from_shape(shape : Shape, color="black", opacity : Number =1, width : Number =1, fill="none", origin=Point(0, 0)):
	"""
		Transform a shape in a SVG shape
	"""
	# transform a circle in a circle for SVG
	if isinstance(shape, Circle):
		return svg_circle_from_circle(shape, color=color, opacity=opacity, width=width, fill=fill, origin=origin)
	
	# transform a rectangle in a rectangle for SVG
	elif isinstance(shape, Rectangle):
			return svg_rect_from_rectangle(shape, color=color, opacity=opacity, width=width, fill=fill, origin=origin)
	
	# transform a line in a line for SVG
	elif isinstance(shape, Line):
		return svg_line_from_line(shape, color=color, opacity=opacity, width=width)
	
	# transform a polygon in a polygon for SVG
	elif isinstance(shape, Polygon):
		return svg_polygon_from_polygon(shape, color=color, opacity=opacity, width=width, fill=fill, origin=origin)
	
	# transform a path in a path for SVG
	else :
		return svg_path_from_shape(shape, color=color, opacity=opacity, width=width, fill=fill, origin=origin)


# transforme une ligne décrite par son point de départ et son point d'arriver en ligne pour la librairie SVG
def svg_line_from_line(line : Line, color="black", opacity : Number =1, width : Number =1, origin=Point(0, 0)):
	start = line[0] + origin
	end = line[1] + origin
	if line.is_dashed():
		dash_full = line.dash_ratio * line.dash_length
		dash_empty = line.dash_length - dash_full
		return svg.Line(x1=svg.Length(start[0], "mm"), y1=svg.Length(start[1], "mm"), x2=svg.Length(end[0], "mm"), y2=svg.Length(end[1], "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width, stroke_dasharray=[dash_full, dash_empty])
	return svg.Line(x1=svg.Length(start[0], "mm"), y1=svg.Length(start[1], "mm"), x2=svg.Length(end[0], "mm"), y2=svg.Length(end[1], "mm"), stroke=color, stroke_opacity=opacity, stroke_width=width)


# transforme une chaine de caractère en texte pour SVG
def svg_text_from_text(text, x, y, font_size=10, color="black", opacity=1, text_anchor : None | str="start"):
	t_anchor = None
	if text_anchor is not None:
		if text_anchor == "start":
			t_anchor = "start"
		elif text_anchor == "middle":
			t_anchor = "middle"
		elif text_anchor == "end":
			t_anchor = "end"
		else:
			print(f"Erreur : {text_anchor} n'est pas un ancre de texte valide.")
			return None
	return svg.Text(x=svg.Length(x, "mm"), y=svg.Length(y, "mm"), text=text, font_size=svg.Length(font_size, "mm"), fill=color, fill_opacity=opacity, text_anchor=t_anchor)

def save_svg(svg, path):
	with open(path, "w") as f:
		f.write(svg.as_str())
	print(f"Fichier SVG sauvegardé dans {path}")

def rgb_to_hex(rgb):
	# Convertit une couleur
	r, g, b = rgb[:-1].split('(')[1].split(",")
	r = int(r.strip())
	g = int(g.strip())
	b = int(b.strip())
	return "#{:02x}{:02x}{:02x}".format(r, g, b)

def hex_to_rgb(hex):
	# Convertit une couleur
	r = int(hex[1:3], 16)
	g = int(hex[3:5], 16)
	b = int(hex[5:7], 16)
	return f"rgb({r},{g},{b})"

