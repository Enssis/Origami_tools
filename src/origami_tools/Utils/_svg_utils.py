import svg
# from ..geometry import Point, Circle, Rectangle, Line, Polygon, Shape
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


def simplifed_hex(hexcolor):
	"""
	Simplify a hex color.
	"""
	if len(hexcolor) < 7:
		raise ValueError("Hex color must be at least 7 characters long")
	if len(hexcolor) != 7 or hexcolor[0] != '#':
		raise ValueError("Hex color must be in the format #RRGGBB")
	
	return "#" + hexcolor[1] + hexcolor[3] + hexcolor[5]
	
def hsv_to_hex(h, s, v):
    """
    Convert HSV to HEX color format.
    """
    h = float(h)
    s = float(s)
    v = float(v)

    if s == 0.0:
        r = g = b = int(v * 255)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    i = int(h * 6.0)  # Assume h is [0,1]
    f = (h * 6.0) - i
    p = int(v * (1.0 - s) * 255)
    q = int(v * (1.0 - f * s) * 255)
    t = int(v * (1.0 - (1.0 - f) * s) * 255)
    v = int(v * 255)

    i %= 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return "#{:02x}{:02x}{:02x}".format(r, g, b)