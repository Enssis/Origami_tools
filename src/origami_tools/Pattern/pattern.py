import json
from typing import List
from svglib.svglib import svg2rlg
import svg
from IPython.display import SVG, display
import numpy as np
import io
from reportlab.graphics import renderPDF


from ..Utils._types import Number
from ..LaserParams import ParamList
from ..Geometry import Point, Line, Vec, Plane, E2X, Shape, Surface, BASE_REPERE3D, HoledPolygon, Circle, Polygon, Arc
from .drawn_shapes import DrawnShapes, Folds

class Pattern:
	"""
	Class representing a crease pattern for origami or similar applications.
	Attributes:
		name (str): Name of the pattern.
		laser_cut (ParamList): Laser cutting parameters.
		origin (Point): Origin point of the pattern on the screen.
	"""
	def __init__(self, name="", param_list=None, origin=Point(0, 0)):
		"""
		Initializes a new Pattern instance.
		Args:
			name (str): Name of the pattern.
			laser_cut (ParamList, optional): Laser cutting parameters. Defaults to a new ParamList instance.
			origin (Point, optional): Origin point of the pattern on the screen. Defaults to Point(0, 0).
		"""

		if param_list is None:
			self.param_list = ParamList()
		else:
			self.param_list = param_list
		
		self._shapes = []
		self._texts = []

		self.origin = origin

		self.width = 100
		self.height = 100
		self.__canvas = None

		self.name = name
		if self.name == "":
			self.name = "pattern"

		self.__shapes_id = []

	def __str__(self):
		shapes = ""
		for shape in self._shapes:
			shapes += "\t" + str(shape) + "\n"

		return f"Pattenr : name={self.name},\n laser_cut={self.param_list},\n shapes=\n{shapes},\n texts={self._texts},\n origin ={self.origin}, width={self.width}, height={self.height}"
	
	def __repr__(self):
		return self.__str__()

	def as_json(self):
		"""
			renvoie le patron sous forme de json
		"""
		return {
			"name": self.name,
			"laser_cut": self.param_list.as_json(),
			"shapes": [shape.as_json() for shape in self._shapes],
			"texts": self._texts,
			"origin": [self.origin[0], self.origin[1]],
			"width": self.width,
			"height": self.height
		}

	def save_json(self, path, name=""):
		if name == "":
			name = self.name
		if path[-1] != "/":
			path += "/"
		save_dir = path + name + ".json"

		json.dump(self.as_json(), open(save_dir, "w"), indent=4)
		print(f"Fichier JSON sauvegardé dans {save_dir}")

	def w_h(self, width, height):
		self.width = width
		self.height = height
	
	def set_origin(self, origin : Point):
		self.origin = origin

	def reset(self): 
		self._shapes = []
		self._texts = []
		self.__shapes_id = []

		self.__canvas = None

	def copy(self, origin=None):
		
		if origin is None:
			origin = self.origin
		patron = Pattern(self.name, param_list=self.param_list, origin=origin)
		for shape in self._shapes:
			if isinstance(shape, DrawnShapes):
				patron.add_drawn_shapes(shape.copy())
			elif isinstance(shape, Folds):
				patron.add_folds(shape.shapes, fold_type=shape.fold_type, fold_value=shape.fold_value, param=shape.param)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
		patron._texts = self._texts.copy()
		patron.w_h(self.width, self.height)
		return patron

	def shape2svg(self, laser_cut : ParamList, all_visible=False):
		elems = []
		self._shapes = sorted(self._shapes, key=lambda x: x.z_offset if isinstance(x, DrawnShapes) else 0)
		for shape in self._shapes :
			if isinstance(shape, DrawnShapes):
				elems += laser_cut.fab_shapes(shape.shapes, param=shape.param, background=shape.background, outline=shape.outline, origin=self.origin, all_visible=all_visible) 
				
		return elems

	def texts2svg(self, laser_cut):
		elems = []
		for text in self._texts :
			text_coord = [text[0][0] , text[0][1] + self.origin[0] , text[0][2] + self.origin[1]]
			elems.append(laser_cut.fab_text(text_coord[0], text_coord[1], text_coord[2], param=text[1], font_size=text[2], text_anchor=text[3]))
		return elems

	def add_folds(self, lines, fold_type="n", fold_value : Number = 0, param : str | None =None, outside=False, duplicate = False, z_offset : Number = 0):
		"""
			ajoute des lignes au patron \n
			lines : liste de lignes à ajouter \n
				ligne : [x0, y0, x1, y1] ou Line (A, B) \n
			fold_type : type de fold \n
				"m" : pli montagne \n
				"v" : pli vallée \n
				"n" : pas de pli \n
			fold_value : valeur du fold totalement foldé] \n
			param : nom du paramètre de découpe \n
		"""
		nlines = []
		if param is None:
			if fold_type == "m" or fold_type == "v":
				param = "fold_cut"
			else:
				param = "cut"
		if isinstance(lines, Line):
			lines = [lines]
		for line in lines:
			if not isinstance(line, Line):
				if len(line) == 4 :
					nlines.append(Line(Point(line[0], line[1]), Point(line[2], line[3])))
				elif len(line) == 2 :
					nlines.append(Line(line[0], line[1]))
			else :
				nlines.append(line.copy())
		f_id = Folds.create_id(param, fold_type, fold_value, duplicate, z_offset)
		if f_id in self.__shapes_id:
			index = self.__shapes_id.index(f_id)
			self._shapes[index].shapes += nlines
		else:
			self._shapes.append(Folds(nlines, param=param, outside=outside, z_offset=z_offset, fold_type=fold_type, duplicate=duplicate, fold_value=fold_value))
			self.__shapes_id.append(f_id)


	def add_texts(self, texts, param : str | None =None, font_size=10, text_anchor="start"):
		if param is None:
			param = "text"
		for text in texts:
			self._texts.append([text, param, font_size,text_anchor])
	

	def add_shapes(self, shapes : Shape | List[Shape], param : str | None = None, background=False, outline=True, outside=False, duplicate = False, z_offset : Number = 0):
		"""
			add shapes to the patron \n
			shapes : list of shapes to add \n
				shape : Line(), Circle(), Rectangle(), Polygon(), Shape() , HoledPolygon() \n
			param : name of the parameter to use \n
			background : if True, the shape is filled \n
			outline : if True, the shape is outlined \n
		"""
		if isinstance(shapes, Shape):
			shapes = [shapes]
		else :
			t_shapes = []
			for shape in shapes:
				if isinstance(shape, Shape):
					t_shapes.append(shape.copy())
				else:
					print(f"Erreur : {shape} n'est pas une forme.")
			shapes = t_shapes
		if param is None:
			param = "cut"
		s_id = DrawnShapes.create_id(param, background, outline, outside, duplicate, z_offset)
		if s_id in self.__shapes_id:
			index = self.__shapes_id.index(s_id)
			self._shapes[index].shapes += shapes
		else:
			self._shapes.append(DrawnShapes(shapes.copy(), param=param, background=background, outline=outline, outside=outside, duplicate=duplicate, z_offset=z_offset))
			self.__shapes_id.append(s_id)
	
	def add_drawn_shapes(self, drawn_shapes : DrawnShapes):
		"""
			add drawn shapes to the patron \n
			drawn_shapes : DrawnShapes() \n
		"""
		s_id = drawn_shapes.id()
		if s_id in self.__shapes_id:
			index = self.__shapes_id.index(s_id)
			self._shapes[index].shapes += drawn_shapes.shapes
		else:
			self._shapes.append(drawn_shapes)
			self.__shapes_id.append(s_id)


	def thicken_lines(self, l, param="", background=True, outline=False):
		thickened = []
		for shape in self._shapes:
			if isinstance(shape, Folds):
				if shape.fold_type != "n":
					thickened += shape.thicken_to_rect(l)
		self.add_drawn_shapes(DrawnShapes(thickened, param=param, background=background, outline=outline))

	def mirror(self, plane : Plane | Line):
		for shape in self._shapes:
			if isinstance(shape, DrawnShapes):
				shape.mirror(plane)

	def show(self, repr="", recreate = True, all_visible = False):
		"""
			affiche le patron dans un notebook Jupyter \n
			repr : "cut" pour la découpe, "patron" pour le patron, "decal" pour le patron avec les plis, "lasercut" pour le patron découpé
		"""
		if repr != "":
			if repr == "cut":
				self.create(all_visible=all_visible)
			elif repr == "patron":
				self.create_pattern()
			elif repr == "decal":
				self.create_patron_decal(False, 0)
			elif repr == "lasercut":
				patron = self.create_lasercut_patron()
				patron.create()
				self.__canvas = patron.__canvas
		elif recreate:
			self.create_pattern()

		if self.__canvas is None:
			self.create_pattern()


		display(SVG(data=self.__canvas.as_str())) # type: ignore


	def save_SVG(self, path, name=""):
		if name == "":
			name = self.name
		if self.__canvas is None:
			self.create()
		if path[-1] != "/":
			path += "/"
		save_dir = path + name + ".svg"

		with open(save_dir, "w") as f:
			f.write(self.__canvas.as_str()) # type: ignore
		print(f"Fichier SVG sauvegardé dans {save_dir}")

	def save_PDF(self, path, name=""):
		if name == "":
			name = self.name
		if self.__canvas is None:
			self.create()
		if path[-1] != "/":
			path += "/"
		save_dir = path + name + ".pdf"


		svg_io = io.StringIO(self.__canvas.as_str()) # type: ignore
		drawing = svg2rlg(svg_io) 
		renderPDF.drawToFile(drawing, save_dir) # type: ignore

		print(f"Fichier PDF sauvegardé dans {save_dir}")

	def translate(self, v):
		"""
			move the patron \n
			v : vector to move \n
		"""
		for shape in self._shapes:
			if isinstance(shape, DrawnShapes):
				shape.translate(v)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
				raise ValueError(f"Erreur : {shape} n'est pas une forme.")
		#self.origin = self.origin + v

	def rotate(self, angle, center=None):
		"""
			rotate the patron \n
			angle : angle in degrees \n
			center : center of rotation, if None, the center of the patron is used \n
		"""
		if center is None:
			center = Point(self.width/2, self.height/2)
		for shape in self._shapes:
			if isinstance(shape, DrawnShapes):
				shape.rotate(angle, center)
			else:
				raise ValueError(f"Erreur : {shape} n'est pas une forme.")

	def __add__(self, other):
		if not isinstance(other, Pattern):
			return NotImplemented
		patron = self.copy()
		other = other.copy()
		depl = Vec.from_2points(patron.origin, other.origin)
		other.translate(depl)
		for shape in other._shapes:
			if isinstance(shape, DrawnShapes):
				patron.add_drawn_shapes(shape.copy())
			elif isinstance(shape, Folds):
				patron.add_folds(shape.shapes, fold_type=shape.fold_type, fold_value=shape.fold_value, param=shape.param)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
		patron._texts = patron._texts + other._texts
		patron.w_h(max(patron.width, other.width) + depl[0], max(patron.height, other.height) + depl[1])
		return patron

	def create(self, all_visible=False):
		svg_elements = []
		svg_elements += self.shape2svg(self.param_list, all_visible=all_visible)
		svg_elements += self.texts2svg(self.param_list)


		self.__canvas = svg.SVG(
			width=svg.Length(self.width + 2 *self.origin[0], "mm"),
			height=svg.Length(self.height + 2 * self.origin[1], "mm"),
			elements=svg_elements, 
		)

	def get_duplicate(self, param="", param_suffix=""):
		"""
			renvoie un patron dupliqué
		"""
		patron = self.copy()
		patron.name += "_duplicate"
		patron.reset()
		patron._shapes = []
		for shape in self._shapes:
			if isinstance(shape, DrawnShapes):
				if shape.duplicate:
					dr_shape = shape.copy()
					if param != "":
						dr_shape.param = param
					elif param_suffix != "":
						dr_shape.param += param_suffix
					patron.add_drawn_shapes(dr_shape)
		return patron

	def create_patron_decal(self, montain, e, z_offset : Number = 1, param = ""):
		name = self.name + "_decal_" + ("mountain" if montain else "valley")
		new_patron = Pattern(name, param_list=self.param_list, origin=self.origin)

		original_lines = []
		offset_instructions = []
		fixed_lines = []


		# Étape 1 — Séparer les lignes décalables et les autres
		for shape in self._shapes:
			if isinstance(shape, Folds):
				if not ((montain ^ (shape.fold_type == "v")) or shape.fold_type == "n" or shape.fold_value == 0 or e == 0):
					d = 2 * e / np.tan(shape.fold_value / 2)
					for line in shape.shapes:
						if isinstance(line, Line):
							original_lines.append(line)
							offset_instructions.append((line, d, shape))
				else:
					# On ajoute directement les plis non concernés
					new_patron.add_folds(shape.shapes, param=shape.param if param == "" else param, duplicate=shape.duplicate, z_offset=z_offset)
					fixed_lines.extend(shape.shapes)  # pour intersections plus complètes si utile
			else:
				all_lines = []
				for s in shape.shapes:
					if isinstance(s, Line):
						all_lines.append(s.copy())
					elif isinstance(s, Surface):
						if isinstance(s, Circle):
							new_patron.add_shapes(
								[s.copy()],
								param=shape.param,
								background=shape.background,
								outline=shape.outline,
								outside=shape.outside,
								duplicate=shape.duplicate,
								z_offset=z_offset
							)
						else :
							all_lines += s.get_lines()
				fixed_lines.extend(all_lines)
				new_patron.add_shapes(
						all_lines,
						param=shape.param if param == "" else param,
						background=shape.background,
						outline=shape.outline,
						outside=shape.outside,
						duplicate=shape.duplicate,
						z_offset=z_offset
					)

		# Étape 2 — Construire la table d'intersections sur lignes originales
		intersections_map = {i: [[], []] for i in range(len(offset_instructions))}
		for i, (line_i, _, _) in enumerate(offset_instructions):
			for j in range(i + 1, len(offset_instructions)):
				line_j = offset_instructions[j][0]
				inter = line_i.intersection(line_j, limit=True)
				if inter is None:
					continue
				if inter == line_i[0]:
					intersections_map[i][0].append(j)
				elif inter == line_i[1]:
					intersections_map[i][1].append(j)
				if inter == line_j[0]:
					intersections_map[j][0].append(i)
				elif inter == line_j[1]:
					intersections_map[j][1].append(i)
			for j, line in enumerate(fixed_lines):
				inter = line_i.intersection(line, limit=True)
				# print(f"Inter {line_i} with {line} : {inter}")
				if inter is not None:
					if inter == line_i[0]:
						intersections_map[i][0].append(- j - 1)
					elif inter == line_i[1]:
						intersections_map[i][1].append(- j - 1)
			for j in range(2):
				inter_lines = []
				for index in intersections_map[i][j]:
					if index >= 0:
						line = offset_instructions[index][0]
					else:
						line = fixed_lines[-index - 1]
					
					angle = line_i.angles_with(line, line_i[j])[0]
					if len(inter_lines) == 0:
						inter_lines = [(index, angle), (index, angle)]
					else:
						if angle < inter_lines[0][1]:
							inter_lines[0] = (index, angle)
						elif angle > inter_lines[1][1]:
							inter_lines[1] = (index, angle)
				intersections_map[i][j] = [inter_lines[j][0], inter_lines[j - 1][0]]
		# Étape 3 — Créer et tronquer les lignes décalées
		for i, (line, d, shape) in enumerate(offset_instructions):
			normal = line.normal_vect().normalize() * d / 2
			for sign in [+1, -1]:
				offset_line = line.copy()
				offset_line.translate(sign * normal)
				# Liste des autres lignes décalées
				for j in range(2):
					if len(intersections_map[i][j]) != 2:
						continue
					index  = intersections_map[i][j][int(sign / 2 + 0.5)]
					# print(index)
					# print(line, sign, j)
					if index < 0:
						# Intersection avec une ligne fixe
						other_line = fixed_lines[-index - 1]
						d2 = 0
					else:
						# Intersection avec une autre ligne décalée
						other_line = offset_instructions[index][0]
						d2 = offset_instructions[index][1] / 2
					if other_line[0] == line[0] or other_line[1] == line[1]:
						d2 = -d2

					# On décale la ligne fixe ou décalée
					other_offset = other_line.copy()
					other_offset.translate(sign * other_line.normal_vect().normalize() * d2)

					inter = offset_line.intersection(other_offset, limit=False)
					if inter is not None:
						offset_line[j] = inter
					else:
						print(f"Erreur : pas d'intersection entre {offset_line} et {other_offset} pour la ligne {i}, index {index}, sign {sign}, j {j}")

				# print(f"cut_points for line {i}: {len(cut_points)}, cut_points={cut_points}")

				new_patron.add_shapes(
						[offset_line],
						param=shape.param if param == "" else param,
						background=shape.background,
						outline=shape.outline,
						outside=shape.outside,
						duplicate=shape.duplicate,
						z_offset=z_offset
					)


		new_patron._texts = self._texts.copy()
		new_patron.w_h(self.width, self.height)
		return new_patron




	def create_patron_offset(self, e: Number = 0, k : Number = 0.5, full=False, param="", z_offset : Number = 0) -> "Pattern":
		"""
		Generate an offset pattern (patron) for adhesive or cutting.
		
		Args:
			mountain (bool): If True, treat folds as mountain folds, otherwise valley.
			e (Number): Thickness or extrusion amount (for fold-based offsets).
			L (Number): Offset length for all shapes.
			full (bool): If True, generate a full offset shape with cut-out holes.
			param (str): Parameter name for resulting shapes.
		
		Returns:
			Patron: The new offset pattern.
		"""
		shapes = []
		n_patron = Pattern(self.name + "_offset", param_list=self.param_list, origin=self.origin)
		n_patron.w_h(self.width, self.height)

		

		outside_shape : DrawnShapes | None = None				

		# Default offset value
		
		if full :
			param = "adhesive" if param == "" else param
		else:
			param = "fold_cut" if param == "" else param

		for drawnshape in self._shapes:
			d = 0 if k == 0 else -0.25
			# Handle outer contour (cut line)
			if drawnshape.outside:
				if outside_shape is None:
					outside_shape = drawnshape.copy()
				else:
					outside_shape.shapes += drawnshape.shapes
				continue
				

			# Handle fold lines (with optional thickness-based offset)
			if isinstance(drawnshape, Folds):
				if drawnshape.fold_type != "n":  # Skip neutral folds
					d = -e / np.tan(drawnshape.fold_value / 2) - k / 2
			
			# Process individual shapes
			for shape in drawnshape.shapes:
				if isinstance(shape, Circle):
					n_patron.add_shapes([shape.copy()],
										param=param,
										background=drawnshape.background,
										outline=drawnshape.outline,
										outside=drawnshape.outside,
										z_offset=z_offset)
				else:
					shapes.append([
						shape.copy(),                # Shape
						drawnshape.param,            # Parameter
						drawnshape.background,       # Background flag
						drawnshape.outline,          # Outline flag
						drawnshape.outside,          # Outside flag
						[d] * (len(shape) - 1)       # Offset distances per edge
					])

		outside_poly = None
		if outside_shape is not None:
			d = 0 if k == 0 else -0.25
			outside_poly = outside_shape.to_polygon()
			# outside_poly.show()
			outside_shape.shapes = [outside_poly.copy()]
			
			shapes.append([
						outside_poly.copy(),                # Shape
						outside_shape.param,            # Parameter
						outside_shape.background,       # Background flag
						outside_shape.outline,          # Outline flag
						outside_shape.outside,          # Outside flag
						[d] * (len(outside_poly) - 1)       # Offset distances per edge
					])

			dep = 0 if k == 0 else 0.25
			outside_poly = outside_poly.copy().offset(dep).in_2D(BASE_REPERE3D)
			if not full and k > 0:
				n_patron.add_drawn_shapes(DrawnShapes(
					[outside_poly],
					param=param,
					background=outside_shape.background,
					outline=outside_shape.outline,
					outside=True,
					z_offset=z_offset
				))


		# Step 2: Cutting polygons by lines
		# outside_poly.show() if outside_poly is not None else None
		i = 0
		while i < len(shapes):
			current_shape = shapes[i][0]
			if not isinstance(current_shape, Line):
				i += 1
				continue

			for j in range(len(shapes)):
				if isinstance(shapes[j][0], Line):
					continue

				poly = shapes[j][0]
				if not isinstance(poly, Polygon):
					poly = poly.to_polygon()

				# Combine offsets from both shapes
				offset_values = shapes[j][5] + shapes[i][5]
				# print(f"Offsets: before : {shapes[j][5]} + {shapes[i][5]} = {offset_values}")
				# Perform the cut
				sub_polygons, sub_offsets = poly.cut(current_shape, offset_values)
				# print(f"Cutting {j} : {poly} with {current_shape} resulted in {len(sub_polygons)} sub-polygons.")
				# print(f"Offsets: before : {shapes[j][5]} + {shapes[i][5]} = {offset_values} ; after {sub_offsets}")
				# print(f"sub_polygons: {sub_polygons}\n")

				if len(sub_polygons) == 1:
					continue
				
				# Replace first part in place, append the others
				shapes[j][0] = sub_polygons[0]
				shapes[j][5] = sub_offsets[0]
				for k in range(1, len(sub_polygons)):
					shapes.append([
						sub_polygons[k], shapes[j][1], shapes[j][2], shapes[j][3], shapes[j][4], sub_offsets[k]
					])

			i += 1

		# Step 3: Apply offset and render shapes
		holes = []
		for shape in shapes:
			if isinstance(shape[0], Line) or len(shape[0]) < 4:
				continue  # Skip invalid or trivial polygons
			
			# print(f"Processing shape {shape[0]} with offsets {shape[5]}")
			offset_shape = shape[0].offset(shape[5])
			offset_shape.in_2D(BASE_REPERE3D)

			if full:
				offset_shape.change_direction()  # Ensure valid hole orientation
				holes.append(offset_shape)
			else:
				n_patron.add_shapes(offset_shape,
									param=param if param else "fold_cut",
									background=shape[2],
									outline=shape[3],
									outside=shape[4],
									z_offset=z_offset)

		# Step 4: Create final holed polygon if requested
		if full:
			if outside_poly is None:
				raise ValueError("No outer boundary defined for full offset.")

			holed = HoledPolygon.from_polygons(outside_poly, holes)
			n_patron.add_shapes([holed],
								param=param if param else "adhesive",
								background=True,
								outline=False,
								outside=shapes[0][4],
								z_offset=z_offset)

		return n_patron


	def create_patron_offset3(self, e : Number =0, L : Number =2, param="fold_cut", closed = True):
		
		lines_list = []

		# list all lines in the same table with the type of fold and d 
		for shape in self._shapes:
			if isinstance(shape, Folds):
				d = e * np.tan(shape.fold_value/2) if shape.fold_type != "n" else 0
				for line in shape.shapes:
					if isinstance(line, Line):
						lines_list.append([line.copy(), shape.fold_type, d, L if shape.fold_type != "n" else 0])
			elif isinstance(shape, DrawnShapes):
				for line in shape.shapes:
					if isinstance(line, Line):
						lines_list.append([line.copy(), "n", 0, 0])
		
		# list all offseted lines for each side
		lines = [[], []]
		for i in range(len(lines_list)):
			# get the line and its normal vector
			line = lines_list[i][0]
			# gets intersected_lines for the start point, the end point 
			intersect_lines = [[], []]
			# and middle point for each side
			middle_lines = []
			d = lines_list[i][2]
			L = lines_list[i][3]

			# search for intersection
			for j in range(len(lines_list)):
				# if the line is the same, we skip it
				if j == i:
					continue
				line2 = lines_list[j][0]
				# get the intersection between the two lines
				inter = line.intersection(line2)
				if inter is not None:
					# if the intersection is a limit point we add the line to the intersected_lines
					if inter.distance(line[0]) < 1e-5:
						intersect_lines[0].append(lines_list[j])
					elif inter.distance(line[1]) < 1e-5:
						intersect_lines[1].append(lines_list[j])
					else:
						# if the intersection is not a limit point, we add it to the middle_lines
						pts = [pt[1] for pt in middle_lines]
						if inter in pts:
							middle_lines[pts.index(inter)].append(lines_list[j])
						else :
							start_dist = inter.distance(line[0])
							for k in range(len(pts)):
								if start_dist < pts[k].distance(line[0]):
									middle_lines.insert(k, [[lines_list[j]], inter])
									break
							else:
								middle_lines.append([[lines_list[j]], inter])
			
			# pts start and end
			# pts left right
			# pts side 1, side 2 
			pts = [[],[]]

			for j in range(2):
				# if there are intersected lines
				l0 = Line(line[j], line[1^j])
				normal = l0.normal_vect().normalize()
				
				if len(intersect_lines[j]) > 0:
					
					pts[j].extend(l0.offset_intersect_lines(intersect_lines[j], line[j], d, L))

					
				else :
					pt_left_s1 = line[j] + normal * (L/2 + d)
					pt_right_s1 = line[j] - normal * (L/2 + d)
					pt_left_s2 = line[j] + normal * (L/2)
					pt_right_s2 = line[j] - normal * (L/2)
					pts[j].append([pt_left_s1, pt_left_s2])
					pts[j].append([pt_right_s1, pt_right_s2])
					if closed:
						lines[0].append(Line(pt_left_s1.copy(), pt_right_s1.copy()))
						lines[1].append(Line(pt_left_s2.copy(), pt_right_s2.copy()))


			if len(middle_lines) == 0:
				lines[0].append(Line(pts[0][0][0], pts[1][1][0]))
				lines[0].append(Line(pts[0][1][0], pts[1][0][0]))

				lines[1].append(Line(pts[0][0][1], pts[1][1][1]))
				lines[1].append(Line(pts[0][1][1], pts[1][0][1]))
			else:
				pts_list = [pts[0]]
				for j in range(len(middle_lines)):
					inter = middle_lines[j][1]
					intersect_lines = middle_lines[j][0]
					line0 = Line(inter, line[0])
					line1 = Line(inter, line[1])
					pts_list.append(line0.offset_intersect_lines(intersect_lines + [[line1, lines_list[i][1], d, lines_list[i][3]]], inter, d, lines_list[i][3]))
					pts_list.append(line1.offset_intersect_lines(intersect_lines + [[line0, lines_list[i][1], d, lines_list[i][3]]], inter, d, lines_list[i][3]))
				pts_list.append(pts[1])
				for j in range(0, len(pts_list), 2):

					lines[0].append(Line(pts_list[j][0][0], pts_list[j + 1][1][0]))
					lines[0].append(Line(pts_list[j][1][0], pts_list[j + 1][0][0]))

					lines[1].append(Line(pts_list[j][0][1], pts_list[j + 1][1][1]))
					lines[1].append(Line(pts_list[j][1][1], pts_list[j + 1][0][1]))
				pass
		
		patron = Pattern(self.name + "_martyr", param_list=self.param_list, origin=self.origin)
		patron2 = patron.copy()
		patron.add_folds(lines[1], param=param)
		patron2.add_folds(lines[0], param=param)
		patron2.mirror(Line(Point(self.width + 2.5, 0), Point(self.width + 2.5, 5)))

		patron += patron2
		patron.w_h(self.width  * 2 + 5 + self.origin[0], self.height + self.origin[1])

		return patron

	def create_lasercut_patron(self, asym=False, e : Number = 0, k : Number=0.5, adhesif_param ="adhesive", mirror=True, montain=True, cut_param=""):
		"""
			return two patrons on one \n
			asym : if True, the two patrons are asymetric \n
			e : thickness of the fold \n
			l : thickness of the removed adhesive \n
			adhesif_param : name of the parameter to use for the adhesive \n
			mirror : if True, the patron is mirrored \n
			montain : if True, the patron is a mountain fold \n
		"""
		patron = self.copy()
		if not asym:
			e = 0
		patron_offset_m = patron.create_patron_offset(e, k, True, adhesif_param)
		patron_offset_mcut = patron.create_patron_decal(montain, e, param=cut_param)

		if mirror:
			w_mirror = patron.width + 2.5
			mirror_line = Line(Point(w_mirror, 0), Point(w_mirror, 5))
			patron.mirror(mirror_line)
			patron.w_h(w_mirror * 2, patron.height)
			patron_offset_v = patron.create_patron_offset(e, k, True, adhesif_param)
			patron_offset_vcut = patron.create_patron_decal(not montain, e, param=cut_param)
			full_patron = patron_offset_m + patron_offset_mcut + patron_offset_v + patron_offset_vcut
		else:
			full_patron = patron_offset_m + patron_offset_mcut


		full_patron.name = self.name + "_lasercut"
		return full_patron

	def create_lasercut_martyr(self, asym=False, e=0, L=2):
		
		lines = [[], []]
		for shape in self._shapes:
			if isinstance(shape, Folds):
				d = 0
				if shape.fold_type != "n":
					d = e * np.tan(shape.fold_value/2) 
				for line in shape.shapes:
					if isinstance(line, Line):
						normal = line.normal_vect().normalize()
						new_l1 = line.copy()	
						new_l2 = line.copy()
						new_l1.translate(normal * L/2)
						new_l2.translate(0-normal * L/2)
						i = 0 if shape.fold_type == "m" else 1
						lines[i].append(new_l1)
						lines[i].append(new_l2)

						if asym == True:
							new_l1 = line.copy()	
							new_l2 = line.copy()
							new_l1.translate(normal * (L/2 + d))
							new_l2.translate(0-normal * (L/2 + d))
							lines[i ^ 1].append(new_l1)
							lines[i ^ 1].append(new_l2)
						else:
							lines[i ^1].append(new_l1.copy())
							lines[i ^1].append(new_l2.copy())

		patron = Pattern(self.name + "_martyr", param_list=self.param_list, origin=self.origin)
		patron2 = patron.copy()
		patron.add_folds(lines[0], param="fold_cut")
		patron2.add_folds(lines[1], param="fold_cut")
		patron2.mirror(Plane(Point(self.width + 2.5, 0), E2X))

		patron += patron2
		patron.w_h(self.width  * 2 + 5 + self.origin[0], self.height + self.origin[1])

		return patron

	def cut_half(self, side : int = 1):
		"""
			cut the patron in half \n
			side : 0 for left, 1 for right \n
		"""
		if side not in [0, 1]:
			raise ValueError("side must be 0 or 1")
		patron = self.copy()
		middle_x = self.width / 2
		cut_line = Line(Point(middle_x, 0), Point(middle_x, self.height))
		mult = -1 if side == 0 else 1
		for shape in patron._shapes:
			shapes = []
			for s in shape.shapes:
				if isinstance(s, Line):
					if mult * s[0][0] < mult * middle_x:
						s[0][0] = middle_x
					if mult * s[1][0] < mult * middle_x:
						s[1][0] = middle_x
					if s[0][0] == s[1][0]:
						continue
				elif isinstance(s, Circle):
					inter_pts = s.line_intersection(cut_line)
					if inter_pts is None or len(inter_pts) == 1:
						if mult * s[0][0] < mult * middle_x:
							continue
					
					else: 
						if inter_pts[0][1] > inter_pts[1][1]:
							up_point = inter_pts[0] 
							down_point = inter_pts[1]
						else:
							up_point = inter_pts[1] 
							down_point = inter_pts[0]
						if side == 1:
							s = Arc([s[0], down_point, up_point], s.radius)
						else:
							s = Arc([s[0], up_point, down_point], s.radius)


				else :
					s = s.to_polygon()
					if isinstance(s, Polygon):
						polygons = s.cut(cut_line)
						if len(polygons) == 1:
							if mult * polygons[0][0][0] < mult * middle_x:
								continue
						else :
							s = s.cut(cut_line)[side]
				shapes.append(s)
			shape.shapes = shapes
		return patron

	def create_pattern(self):
		svg_elements = []
		contour_col = "black"
		# couleur moutain, couleur valley
		pli_col = ["red", "blue"]
		width = 2
		self._shapes = sorted(self._shapes, key=lambda x: x.z_offset if isinstance(x, DrawnShapes) else 0)
		for shape in self._shapes :
			if isinstance(shape, Folds):
				i = 0 if shape.fold_type == "m" else 1
				if shape.fold_type == "n":
					svg_elements += shape.to_svg(color=contour_col, opacity=1, width=width, origin=self.origin)
				else:
					svg_elements += shape.to_svg(color=pli_col[i], opacity=1.0 - shape.fold_value / np.pi, width=width, origin=self.origin)
			elif isinstance(shape, DrawnShapes):
				svg_elements += shape.to_svg(color=contour_col, opacity=1, width=width, fill="black" if shape.background else "none", origin=self.origin)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
				continue

		self.__canvas = svg.SVG(
			width=svg.Length(self.width + 2 * self.origin[0], "mm"),
			height=svg.Length(self.height + 2 * self.origin[1], "mm"),
			elements=svg_elements, 
		)
