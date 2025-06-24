
from dataclasses import dataclass
from typing import Sequence


from ..Utils._types import Number
from ..Geometry import Point, Line, Plane, Shape, Polygon, Circle, RegularPolygon, Rectangle

@dataclass
class DrawnShapes():
	"""
		Classe représentant un patron dessiné, qui peut contenir plusieurs formes.
		Les formes peuvent être des lignes, des polygones, des cercles, etc.
		:param shapes: liste de formes à dessiner
		:param param: nom du paramètre à utiliser pour le dessin
		:param background: si True, le patron est dessiné en arrière-plan
		:param outline: si True, le patron est dessiné avec un contour
		:param outside: si True, la forme est dessinée à l'extérieur de la forme, sinon à l'intérieur
		:param z_offset: décalage en z pour le dessin, utilisé pour l'ordre des patrons
	"""
	shapes: Sequence[Shape]  # Liste des formes à dessiner
	param: str  # Nom du paramètre à utiliser pour le dessin
	background: bool = False  # Si True, le patron est dessiné en arrière-plan
	outline: bool = True  # Si True, le patron est dessiné avec un contour
	outside: bool = False  # Si True, la forme est dessinée à l'extérieur de la forme, sinon à l'intérieur
	duplicate: bool = False  # Si True, la forme sera dupliquée lors du dessin
	z_offset: Number = 0  # Décalage en z pour le dessin, utilisé pour l'ordre des patrons 

	def __str__(self):
		return f"DrawnShape {self.id()}: {self.shapes}, param={self.param}, background={self.background}, outline={self.outline}, outside={self.outside}, z_offset={self.z_offset}"
	
	def __repr__(self):
		return self.__str__()

	@staticmethod
	def create_id(param_name, background, outline, outside, duplicate, z_offset : Number = 0): 
		return f"{param_name}_{int(background)}_{int(outline)}_{int(outside)}_{int(duplicate)}_{int(z_offset * 100) / 100}"

	def as_json(self):
		"""
			renvoie le patron sous forme de json
		"""
		return {
			"id": self.id(),
			"shapes": [shape.as_json() for shape in self.shapes],
			"param": self.param,
			"background": self.background,
			"outline": self.outline,
			"outside": self.outside,
			"duplicate": self.duplicate,
			"z_offset": self.z_offset
		}

	def id(self):
		"""
			renvoie l'id du patron
		"""
		return DrawnShapes.create_id(self.param, self.background, self.outline, self.outside, self.duplicate ,self.z_offset)

	def mirror(self, plane : Plane | Line):
		for shape in self.shapes:
			if isinstance(shape, Shape):
				shape.mirror(plane)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")


	def rotate(self, angle, center):
		for shape in self.shapes:
			if isinstance(shape, Shape):
				shape.rotate(angle, center)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")


	def translate(self, v):
		for shape in self.shapes:
			if isinstance(shape, Shape):
				shape.translate(v)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")

	def copy(self):
		"""
			copie le patron
		"""
		new_shapes = []
		for shape in self.shapes:
			new_shapes.append(shape.copy())
		return DrawnShapes(new_shapes, param=self.param, background=self.background, outline=self.outline, outside=self.outside, duplicate=self.duplicate, z_offset=self.z_offset)
	
	def to_svg(self, color="black", opacity=1, width=1, fill=None, origin=Point(0, 0)):
		"""
			Transform the shapes in the SVG shapes
		"""
		svg_shapes = []
		if fill is None:
			if self.background:
				fill = color
			else:
				fill = "none"
		if not self.outline:
			width = 0

		for shape in self.shapes:
			svg_shapes.append(shape.as_svg(color=color, opacity=opacity, width=width, fill=fill, origin=origin))

		return svg_shapes
	
	def to_polygon(self):
		if len(self.shapes) == 1:
			if isinstance(self.shapes[0], Polygon):
				return self.shapes[0]
			if isinstance(self.shapes[0], Line):
				raise ValueError("Cannot convert a line to a polygon")
			if isinstance(self.shapes[0], Circle):
				return RegularPolygon.from_center_and_radius(self.shapes[0][0], self.shapes[0].radius, 20)
			if isinstance(self.shapes[0], Rectangle):
				return Polygon(self.shapes[0].get_corners())
		elif len(self.shapes) > 1:
			if not isinstance(self.shapes[0], Line):
				raise ValueError("Cannot convert multiple shapes to a polygon")
			pts = [self.shapes[0][0].copy(), self.shapes[0][1].copy()]
			lines = self.shapes[1:]
			done = []
			for _ in range(1, len(self.shapes)):
				for j in range(len(lines)):
					if j in done:
						continue
					shape = lines[j].copy()
					if not isinstance(shape, Line):
						print(f"Erreur : {shape} n'est pas une ligne.")
						continue
					if shape[0] == pts[-1]:
						pts.append(shape[1])
					elif shape[1] == pts[-1]:
						pts.append(shape[0])
					else:
						continue
					done.append(j)
		return Polygon(pts) # type: ignore


@dataclass			
class Folds(DrawnShapes):
	"""
		Classe représentant un pli dans un patron, qui peut contenir plusieurs lignes.
		:param lines: liste de lignes à plier
		:param param: nom du paramètre à utiliser pour le pli
		:param fold_type: type de pli
			"m" : pli montagne \n
			"v" : pli vallée \n
			"n" : pas de pli \n
		:param fold_value: valeur maximale du pli
		:param outside: si True, le pli est dessiné à l'extérieur de la forme, sinon à l'intérieur
		:param z_offset: décalage en z pour le pli, utilisé pour l'ordre des plis
	"""
	fold_type : str = "n" # Type de pli, "m" pour montagne, "v" pour vallée, "n" pour pas de pli
	fold_value : Number = 0  # Valeur maximale du pli

	def as_json(self):
		"""
			renvoie le pli sous forme de json
		"""
		d = super().as_json()
		d["fold_type"] = self.fold_type
		d["fold_value"] = self.fold_value
		d["id"] = self.id()
		return d

	def __str__(self):
		if self.fold_type == "m":
			fold = "montain"
		elif self.fold_type == "v":
			fold = "valley"
		else:
			fold = "no fold"
		return f"Pli {self.id()}: {super().__str__()}, {fold}, fold_value={self.fold_value}"
	
	def __repr__(self):
		return self.__str__()
	
	@staticmethod
	def create_id(param_name, fold_type, fold_value, duplicate, z_offset : Number = 0): # type: ignore
		return f"{param_name}_{fold_type}_{int(fold_value * 100) / 100}_{int(duplicate)}_{int(z_offset * 100) / 100}"

	def id(self):
		return Folds.create_id(self.param, self.fold_type, self.fold_value, self.duplicate, self.z_offset)

	def copy(self):
		"""
			copie le pli
		"""
		new_shapes = []
		for shape in self.shapes:
			new_shapes.append(shape.copy())
		return Folds(new_shapes, param=self.param, fold_type=self.fold_type, fold_value=self.fold_value, outside=self.outside, duplicate=self.duplicate, z_offset=self.z_offset)
	
	def thicken_to_rect(self, l):
		"""
			renvoie un rectangle épais de l'épaisseur l
		"""
		thickened = []
		for shape in self.shapes:
			if isinstance(shape, Line):
				thickened.append(shape.thicken_to_rect(l))
			else:
				print(f"Erreur : {shape} n'est pas une ligne.")
		return thickened
	
	def to_svg(self, color="black", opacity : Number =1, width : Number = 1, origin=Point(0, 0)): # type: ignore
		lines = []
		for line in self.shapes:
			if isinstance(line, Line):
				lines.append(line.as_svg(color=color, opacity=opacity, width=width, origin=origin))
			else:
				print(f"Erreur : {line} n'est pas une ligne.")
		return lines

@dataclass
class FoldLine(Line):
	type: str = "n"  # "m" for mountain, "v" for valley, "n" for no fold
	L : Number = 0  # length of the fold, only used for mountain and valley folds