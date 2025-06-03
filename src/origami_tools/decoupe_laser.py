from typing import List, Sequence, overload
import svg
from IPython.display import SVG, display
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import io
import os
import prettytable as pt 
import json

from origami_tools import get_material_path
from .geometry import *
from .utils._svg_utils import *
# =========== Classe Patron ===========


LASER_SAVE_PATH = get_material_path()

class LaserParam:
	def __init__(self, color, name="", ep : Number =0.2, full=True, dash_length : Number = 6, dash_full_ratio : float = 0.5, power : Number =80, speed : Number =4, passe : int=1):
		self.color = color
		self.ep = ep
		self.full = full
		self.dash_length = dash_length
		self.dash_full_ratio = dash_full_ratio
		self.power = power
		self.speed = speed
		self.passe = passe
		if name == "":
			self.name = "Laser_param_" + str(power) + "_" + str(speed) + "_" + color
		else:
			self.name = name
	
	def __str__(self):
		return f"{self.name} : color={self.color}, ep={self.ep}, full={self.full}, dash_length={self.dash_length}, dash_full_ratio={self.dash_full_ratio}, power={self.power}, speed={self.speed}, passe={self.passe}, \n"
	
	def __repr__(self):
		return self.__str__()

	def as_json(self):
		"""
			renvoie le paramètre sous forme de json
		"""
		return {
			"name": self.name,
			"color": self.color,
			"ep": self.ep,
			"full": self.full,
			"dash_length": self.dash_length,
			"dash_full_ratio": self.dash_full_ratio,
			"power": self.power,
			"speed": self.speed,
			"passe": self.passe
		}

	def as_csv(self):
		return f"{self.name},{self.color},{self.ep},{self.full},{self.dash_length},{self.dash_full_ratio},{self.power},{self.speed},{self.passe}\n"

	def id(self):
		id_name = f"{self.passe}p{self.power}p{self.speed}v"
		if self.full:
			return "f" + id_name
		else:
			return f"d{self.dash_length}l{self.dash_full_ratio}r" + id_name 

	@staticmethod
	def default_cut():
		return LaserParam("red", "def_cut")
	
	@staticmethod
	def default_text():
		return LaserParam("black", "def_text")
	
	def copy(self):
		"""
			copie le paramètre
		"""
		return LaserParam(self.color, name=self.name, ep=self.ep, full=self.full, dash_length=self.dash_length, dash_full_ratio=self.dash_full_ratio, power=self.power, speed=self.speed, passe=self.passe)
	
	@staticmethod
	def default_dash():
		return LaserParam("red", name="def_dash", full=False, ep=0.5)

	"""
		sauvegarde le paramètre dans un fichier
		profile : nom du profil dans lequel on veut sauvegarder le paramètre
		overwrite : si True, écrase le paramètre existant dans le fichier
	"""
	def save(self, profile, overwrite=False, dir_path = None):
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		with open(path, "r") as f:
			lines = f.readlines()
			for line in lines:
				if line.startswith(self.name):
					if overwrite:
						print(f"Le paramètre {self.name} existe déjà dans le fichier de sauvegarde. Il sera écrasé.")
						line = self.__str__()
						break
					else:
						print(f"Le paramètre {self.name} existe déjà dans le fichier de sauvegarde.")
						return False
			else :
				lines.append(self.__str__())
		
		text = ""
		for line in lines:
			text += line
		with open(path, "w") as f:
			f.write(text)
		
		print(f"Fichier de sauvegarde des paramètres laser dans {path}")
		return True
	
	@staticmethod
	def load(name, profile, dir_path=None):
		"""
			charge le paramètre depuis un fichier de sauvegarde \n
			name : nom du paramètre à charger \n
			profile : nom du profil dans lequel on veut charger le paramètre \n
		"""
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		with open(path, "r") as f:
			lines = f.readlines()[1:]
			for line in lines:
				if line.startswith(name):
					return LaserParam.load_from_csv(line)
		print(f"Le paramètre {name} n'existe pas dans le fichier de sauvegarde {path}.")
		return LaserParam.default_cut()

	@staticmethod
	def load_from_csv(param_str):
		param = param_str[:-1].split(",")
		name, color, ep, full, dash, dash_full_ratio, power, speed, passe = param 
		return LaserParam(color, name=name, ep=float(ep), full=full == "True", dash_length=float(dash), dash_full_ratio=float(dash_full_ratio), power=int(power), speed=float(speed), passe=int(passe))

	@staticmethod
	def load_from_str(param_str):
		name = param_str.split(":")[0].strip()
		param = param_str.split(":")[1].split(",")
		color = param[0].split("=")[1].strip()
		ep = float(param[1].split("=")[1].strip())
		full = param[2].split("=")[1].strip() == "True"
		dash_length = float(param[3].split("=")[1].strip())
		dash_full_ratio = float(param[4].split("=")[1].strip())
		power = int(param[5].split("=")[1].strip())
		speed = float(param[6].split("=")[1].strip())
		passe = int(param[7].split("=")[1].strip())

		return LaserParam(color, name=name, ep=ep, full=full, dash_length=dash_length, dash_full_ratio=dash_full_ratio, power=power, speed=speed, passe=passe)


# =========== Classe LaserCut ===========
def laser_cut_colors(hex = True):
	if hex:
		return ['#000000', 
			'#ff0000',
			'#0000ff',
			'#336699',
			'#00ffff',
			'#00ff00',
			'#009933',
			'#006633',
			'#999933',
			'#996633',
			'#663300',
			'#660066',
			'#9900cc',
			'#ff00ff',
			'#ff6600',
			'#ffff00']
	
	return ["rgb(0,0,0)",
		"rgb(255,0,0)",
		"rgb(0,0,255)",
		"rgb(51,102,153)",
		"rgb(0,255,255)",
		"rgb(0,255,0)",
		"rgb(0,153,51)",
		"rgb(0,102,51)",
		"rgb(153,153,51)",
		"rgb(153,102,51)",
		"rgb(102,51,0)",
		"rgb(102,0,102)",
		"rgb(153,0,204)",
		"rgb(255,0,255)",
		"rgb(255,102,0)",
		"rgb(255,255,0)",
		]

def get_lasercut_color_num(color, hex = True):
	colors = laser_cut_colors(hex)
	if color in colors:
		return colors.index(color) + 1
	else:
		print(f"Erreur : la couleur {color} n'est pas dans la liste des couleurs laser.")
		return -1

@dataclass
class LaserCut: 

	def __init__(self, params : list[LaserParam] = [], default_cut = None, default_text=None, profile="default"):
		if len(params) == 0:
			lc = LaserCut.load_from_profile(profile)
			if lc is None:
				print(f"Le profil {profile} n'existe pas.")
				return
			params = list(lc.params.values())
		self.params = {}
		for param in params:
			if isinstance(param, LaserParam):
				self.params[param.name] = param
			else:
				print(f"Le paramètre {param} n'est pas un LaserParam.")
		if default_cut is None:
			self.default_cut = params[1]
		else:
			self.default_cut = default_cut
		if default_text is None:
			self.default_text = params[0]
		else:
			self.default_text = default_text
		self.profile = profile
		self.names = list(self.params.keys())

	def __str__(self):
		desc = f"LaserCut {self.profile}:\n"
		desc += f"  default_cut : {self.default_cut.name} \n"
		desc += f"  default_text : {self.default_text.name}\n"
		desc += "  params :\n"
		for name, param in self.params.items():
			desc += f"  {name} : {param}"

		return desc

	def copy(self):
		"""
			copie le laser cut
		"""
		params = []
		for param in self.params.values():
			params.append(param.copy())
		return LaserCut(params=params, default_cut=self.default_cut, default_text=self.default_text, profile=self.profile)

	def __repr__(self):
		return self.__str__()

	def as_json(self):
		"""
			renvoie le laser cut sous forme de json
		"""
		return {
			"profile": self.profile,
			"default_cut": self.default_cut.as_json(),
			"default_text": self.default_text.as_json(),
			"params": [param.as_json() for param in self.params.values()]
		}

	def get_param_num(self, n):
		""" 
			renvoie le paramètre numero n
		"""
		if n < len(self.params):
			return list(self.params.values())[n]
		else:
			print(f"Le paramètre {n} n'existe pas dans ce profile.")
			return None

	"""
		sauvegarde le profil dans un fichier
		overwrite_file : si True, écrase le fichier existant
		overwrite_param : si True, écrase le paramètre existant dans le fichier
	"""
	def save_profile(self, overwrite_file=False, overwrite_param=False, dir_path=None):
		# sauvegarde le profil dans un fichier
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + self.profile + ".csv"
		
		# si le fichier n'existe pas, on le crée
		file = ""
		if not os.path.exists(path):
			os.mknod(path)
			overwrite_file = True
		# on recupère tout les noms de paramètre
		param_names = list(self.params.keys())

		# on ouvre le fichier et on lit les lignes
		if not overwrite_file:
			with open(path, "r") as f:
				lines = f.readlines()
				for line in lines:
					# on vérifie si la ligne commence par un des noms de paramètre
					for key in param_names:
						if line.startswith(key):
							# si le paramètre existe déjà, on le remplace par le nouveau si on a l'option overwrite
							# sinon on ne fait rien
							if overwrite_param:
								print(f"Le parametre {key} existe déjà dans le fichier de sauvegarde. Il sera écrasé.")
								line = self.params[key].as_csv()					
							else:
								print(f"Le profil {self.profile} existe déjà dans le fichier de sauvegarde.")
							
							# on supprime le nom du paramètre de la liste des noms a tester
							param_names.remove(key)
							break
					# si on a trouvé tout les noms de paramètre, on sort de la boucle
					if len(param_names) == 0:
						break
				# si on a pas trouvé tout les noms de paramètre, on les ajoute à la fin du fichier
				else :
					for key in param_names:
						lines.append(self.params[key].as_csv())
			
			# on écrit le nouveau contenu du fichier
			for line in lines:
				file += line
		else :
			file = "name,color,ep,full,dash_length,dash_full_ratio,power,speed,passe\n"
			for key in self.params.keys():
				file += self.params[key].as_csv()

		with open(path, "w") as f:
			f.write(file)
		
		print(f"Fichier de sauvegarde des paramètres laser dans {path}")

	@staticmethod
	def load_from_profile(profile, dir_path=None):
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		if not os.path.exists(path):
			print(f"Le profil {profile} n'existe pas.")
			return None
		with open(path, "r") as f:
			lines = f.readlines()[1:]
			params = []
			for line in lines:
				param = LaserParam.load_from_csv(line)
				if param is not None:
					params.append(param)
			if len(params) > 0:
				return LaserCut(params=params, profile=profile)
				
		print(f"Le profil {profile} n'existe pas dans le fichier de sauvegarde {path}.")
		return None

	def fab_shapes(self, shapes : Sequence[Shape] | List[Shape], param="", background=False, outline=True, origin=Point(0, 0)):
		"""
			return a list of shapes for the laser cut \n
			shapes : list of shapes to cut \n
				shape : Line(), Circle(), Rectangle(), Polygon(), Shape() \n
			param : name of the parameter to use \n
			background : if True, the shape is filled \n
			outline : if True, the shape is outlined \n
		"""
		if param in self.names:
			param = self.params[param]
		else: 
			param = self.default_cut

		if outline:
			ep = param.ep
		else:
			ep = 0

		if background:
			fill = param.color
		else:
			fill = "none"

		fab_shapes = []
		for shape in shapes:
			if isinstance(shape, Line):
				if not param.full:
					lines = shape.get_line_dashed(param.dash_length, param.dash_full_ratio)
					fab_shapes += [line.as_svg(param.color, opacity=1, width=param.ep, origin=origin) for line in lines]
				else:
					fab_shapes.append(shape.as_svg(param.color, opacity=1, width=param.ep, origin=origin))
			else:
				if not param.full and isinstance(shape, Surface):
					lines = shape.get_dashed(param.dash_length, param.dash_full_ratio)
					fab_shapes += [line.as_svg(param.color, opacity=1, width=param.ep, origin=origin) for line in lines]
				else:
					fab_shapes.append(shape.as_svg(color=param.color, opacity=1, width=ep, fill=fill, origin=origin))
		return fab_shapes

	# retourne un texte pour la gravure
	def fab_text(self, text, x, y, font_size=10, text_anchor="start", param=""):
		if param in self.names:
			param = self.params[param]
		else: 
			param = self.default_text

		return svg_text_from_text(text, x, y, font_size=font_size, color=param.color, text_anchor=text_anchor)

	@staticmethod
	def table_from_profile(profile, dir_path=None):
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		with open(path) as fp:
			mytable = pt.from_csv(fp)
		return mytable

	def show_param(self):
		# Affiche le tableau des paramètres
		table = pt.PrettyTable()
		table.field_names = ["Nom", "Couleur", "Couleur num", "Epaisseur", "Plein", "Longueur trait", "Ratio plein/pointillé", "Puissance", "Vitesse", "Passe"]
		for param in self.params.values():
			table.add_row([param.name, param.color, str(get_lasercut_color_num(param.color)), param.ep, param.full, param.dash_length, param.dash_full_ratio, param.power, param.speed, param.passe])
		print(table)

	def show_cut_param(self):
		# Affiche le tableau des paramètres
		table = pt.PrettyTable()
		table.field_names = [ "Couleur", "Couleur num", "Puissance", "Vitesse", "Passe"]
		c_nums = []
		for param in self.params.values():
			c_num = get_lasercut_color_num(param.color)
			if c_num in c_nums:
				continue
			table.add_row([param.color, c_num, param.power, param.speed, param.passe])
			c_nums.append(c_num)
		table.sortby = "Couleur num"
		print(table)

# =========== Classe Patron ===========

class DrawnShapes():
	def __init__(self, shapes : Sequence[Shape], param = None, background=False, outline=True, outside=False):
		if param is None:
			raise ValueError("Le paramètre de dessin est obligatoire")
		self.shapes = shapes
		self.param = param
		self.background = background
		self.outline = outline
		self.outside = outside  # True if the shape is on the outside of the shape, False if it is on the interior

	def __str__(self):
		return f"DrawnShape {self.id()}: {self.shapes}, param={self.param}, background={self.background}, outline={self.outline}"
	
	def __repr__(self):
		return self.__str__()

	@staticmethod
	def create_id(param_name, background, outline, outside):
		return f"{param_name}_{int(background)}_{int(outline)}_{int(outside)}"

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
			"outside": self.outside
		}

	def id(self):
		"""
			renvoie l'id du patron
		"""
		return DrawnShapes.create_id(self.param, self.background, self.outline, self.outside)

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
		return DrawnShapes(new_shapes, param=self.param, background=self.background, outline=self.outline, outside=self.outside)
	
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
			pts = [self.shapes[0][0], self.shapes[0][1]]
			lines = self.shapes[1:]
			done = []
			for _ in range(1, len(self.shapes)):
				for j in range(len(lines)):
					if j in done:
						continue
					shape = lines[j]
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


			
class Folds(DrawnShapes):
	def __init__(self, lines : Sequence[Line], param = None, fold_type="n", fold_value : Number =0, outside = False):
		"""
			create a fold \n
			lines : list of lines to fold \n
				line : [x0, y0, x1, y1] ou Line (A, B) \n
			param : name of the parameter to use \n
			fold_type : type of fold \n
				"m" : mountain fold \n
				"v" : valley fold \n
				"n" : no fold \n
			fold_value : max value of the fold \n
		"""
		
		super().__init__(shapes=lines, param=param, outside=outside)
		self.fold_type = fold_type
		self.fold_value = fold_value

	def as_json(self):
		"""
			renvoie le pli sous forme de json
		"""
		return {
			"id": self.id(),
			"shapes": [shape.as_json() for shape in self.shapes],
			"param": self.param,
			"fold_type": self.fold_type,
			"fold_value": self.fold_value
		}

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
	def create_id(param_name, fold_type, fold_value): # type: ignore
		return f"{param_name}_{fold_type}_{int(fold_value * 100) / 100}"

	def id(self):
		return Folds.create_id(self.param, self.fold_type, self.fold_value)

	def copy(self):
		"""
			copie le pli
		"""
		new_shapes = []
		for shape in self.shapes:
			new_shapes.append(shape.copy())
		return Folds(new_shapes, param=self.param, fold_type=self.fold_type, fold_value=self.fold_value)
	
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

class Patron:

	def __init__(self, name="", laser_cut=None, size = 1, origin=Point(0, 0)):
		self.name = name
		if laser_cut is None:
			self.laser_cut = LaserCut()
		else:
			self.laser_cut = laser_cut
		self.shapes = []
		self.texts = []

		self.origin = origin

		self.size = size

		self.width = 100
		self.height = 100
		self.canvas = None

		if self.name == "":
			self.name = "patron"

		self.shapes_id = []

	def __str__(self):
		shapes = ""
		for shape in self.shapes:
			shapes += "\t" + str(shape) + "\n"

		return f"Patron : name={self.name},\n laser_cut={self.laser_cut},\n shapes=\n{shapes},\n texts={self.texts},\n origin ={self.origin}, size={self.size}, width={self.width}, height={self.height}"
	
	def __repr__(self):
		return self.__str__()

	def as_json(self):
		"""
			renvoie le patron sous forme de json
		"""
		return {
			"name": self.name,
			"laser_cut": self.laser_cut.as_json(),
			"shapes": [shape.as_json() for shape in self.shapes],
			"texts": self.texts,
			"origin": [self.origin[0], self.origin[1]],
			"size": self.size,
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
	
	def set_origin(self, origin):
		self.width = self.width + origin[0] - self.origin[0]
		self.height = self.height + origin[1] - self.origin[1]
		self.origin = origin

	def reset(self): 
		self.shapes = []
		self.texts = []

		self.canvas = None

	def copy(self, origin=None):
		
		if origin is None:
			origin = self.origin
		patron = Patron(self.name, laser_cut=self.laser_cut, size=self.size, origin=origin)
		for shape in self.shapes:
			if isinstance(shape, DrawnShapes):
				patron.add_drawn_shapes(shape.copy())
			elif isinstance(shape, Folds):
				patron.add_folds(shape.shapes, fold_type=shape.fold_type, fold_value=shape.fold_value, param=shape.param)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
		patron.shapes_id = self.shapes_id.copy()
		patron.texts = self.texts.copy()
		patron.w_h(self.width, self.height)
		return patron

	def shape2svg(self, laser_cut : LaserCut):
		elems = []
		for shape in self.shapes :
			if isinstance(shape, DrawnShapes):
				if isinstance(shape, Folds):
					elems += laser_cut.fab_shapes(shape.shapes, param=shape.param, background=shape.background, outline=shape.outline, origin=self.origin) 
				else:
					elems = laser_cut.fab_shapes(shape.shapes, param=shape.param, background=shape.background, outline=shape.outline, origin=self.origin) + elems
		return elems

	def texts2svg(self, laser_cut):
		elems = []
		for text in self.texts :
			text_coord = [text[0][0] , text[0][1] * self.size + self.origin[0] , text[0][2] * self.size + self.origin[1]]
			elems.append(laser_cut.fab_text(text_coord[0], text_coord[1], text_coord[2], param=text[1], font_size=text[2], text_anchor=text[3]))
		return elems

	def add_folds(self, lines, fold_type="n", fold_value : Number = 0, param : str | None =None, outside=False):
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
				param = "def_pli_cut"
			else:
				param = "def_cut"
		for line in lines:
			if not isinstance(line, Line):
				if len(line) == 4 :
					nlines.append(Line(Point(line[0], line[1]), Point(line[2], line[3])))
				elif len(line) == 2 :
					nlines.append(Line(line[0], line[1]))
			else :
				nlines.append(line.copy())
		f_id = Folds.create_id(param, fold_type, fold_value)
		if f_id in self.shapes_id:
			index = self.shapes_id.index(f_id)
			self.shapes[index].shapes += nlines
		else:
			self.shapes.append(Folds(nlines, param=param, fold_type=fold_type, fold_value=fold_value, outside=outside))
			self.shapes_id.append(f_id)


	def add_texts(self, texts, param : str | None =None, font_size=10, text_anchor="start"):
		if param is None:
			param = "def_text"
		for text in texts:
			self.texts.append([text, param, font_size,text_anchor])
	

	def add_shapes(self, shapes : Shape | List[Shape], param : str | None = None, background=False, outline=True, outside=False):
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
			param = "def_cut"
		s_id = DrawnShapes.create_id(param, background, outline, outside)
		if s_id in self.shapes_id:
			index = self.shapes_id.index(s_id)
			self.shapes[index].shapes += shapes
		else:
			self.shapes.append(DrawnShapes(shapes.copy(), param=param, background=background, outline=outline, outside=outside))
			self.shapes_id.append(s_id)
	
	def add_drawn_shapes(self, drawn_shapes : DrawnShapes):
		"""
			add drawn shapes to the patron \n
			drawn_shapes : DrawnShapes() \n
		"""
		s_id = drawn_shapes.id()
		if s_id in self.shapes_id:
			index = self.shapes_id.index(s_id)
			self.shapes[index].shapes += drawn_shapes.shapes
		else:
			self.shapes.append(drawn_shapes)
			self.shapes_id.append(s_id)


	def thicken_lines(self, l, param="", background=True, outline=False):
		thickened = []
		for shape in self.shapes:
			if isinstance(shape, Folds):
				if shape.fold_type != "n":
					thickened += shape.thicken_to_rect(l)
		self.add_drawn_shapes(DrawnShapes(thickened, param=param, background=background, outline=outline))

	def mirror(self, plane : Plane | Line):
		for shape in self.shapes:
			if isinstance(shape, DrawnShapes):
				shape.mirror(plane)

	def show(self, repr=""):
		"""
			affiche le patron dans un notebook Jupyter \n
			repr : "cut" pour la découpe, "patron" pour le patron, "decal" pour le patron avec les plis, "lasercut" pour le patron découpé
		"""
		if repr != "":
			if repr == "cut":
				self.create()
			elif repr == "patron":
				self.create_pattern()
			elif repr == "decal":
				self.create_patron_decal(False, 0)
			elif repr == "lasercut":
				patron = self.create_lasercut_patron()
				patron.create()
				self.canvas = patron.canvas
		if self.canvas is None:
			self.create_pattern()


		display(SVG(data=self.canvas.as_str())) # type: ignore


	def save_SVG(self, path, name=""):
		if name == "":
			name = self.name
		if self.canvas is None:
			self.create()
		if path[-1] != "/":
			path += "/"
		save_dir = path + name + ".svg"

		with open(save_dir, "w") as f:
			f.write(self.canvas.as_str()) # type: ignore
		print(f"Fichier SVG sauvegardé dans {save_dir}")

	def save_PDF(self, path, name=""):
		if name == "":
			name = self.name
		if self.canvas is None:
			self.create()
		if path[-1] != "/":
			path += "/"
		save_dir = path + name + ".pdf"


		svg_io = io.StringIO(self.canvas.as_str()) # type: ignore
		drawing = svg2rlg(svg_io) 
		renderPDF.drawToFile(drawing, save_dir) # type: ignore

		print(f"Fichier PDF sauvegardé dans {save_dir}")

	def translate(self, v):
		"""
			move the patron \n
			v : vector to move \n
		"""
		for shape in self.shapes:
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
		for shape in self.shapes:
			if isinstance(shape, DrawnShapes):
				shape.rotate(angle, center)
			else:
				raise ValueError(f"Erreur : {shape} n'est pas une forme.")

	def __add__(self, other):
		if not isinstance(other, Patron):
			return NotImplemented
		patron = self.copy()
		other = other.copy()
		depl = Vec.from_2points(patron.origin, other.origin)
		other.translate(depl)
		for shape in other.shapes:
			if isinstance(shape, DrawnShapes):
				patron.add_drawn_shapes(shape.copy())
			elif isinstance(shape, Folds):
				patron.add_folds(shape.shapes, fold_type=shape.fold_type, fold_value=shape.fold_value, param=shape.param)
			else:
				print(f"Erreur : {shape} n'est pas une forme.")
		patron.texts = patron.texts + other.texts
		patron.w_h(max(patron.width, other.width) + depl[0], max(patron.height, other.height) + depl[1])
		return patron

	def create(self):
		svg_elements = []
		svg_elements += self.shape2svg(self.laser_cut)
		svg_elements += self.texts2svg(self.laser_cut)


		self.canvas = svg.SVG(
			width=svg.Length(self.width + 2 *self.origin[0], "mm"),
			height=svg.Length(self.height + 2 * self.origin[1], "mm"),
			elements=svg_elements, 
		)

	def create_patron_decal(self, montain, e):
		name = self.name + "_decal_" + ("mountain" if montain else "valley")
		new_patron = Patron(name, laser_cut=self.laser_cut, size=self.size, origin=self.origin)

		original_lines = []
		offset_instructions = []
		fixed_lines = []


		# Étape 1 — Séparer les lignes décalables et les autres
		for shape in self.shapes:
			if isinstance(shape, Folds):
				if not ((montain ^ (shape.fold_type == "v")) or shape.fold_type == "n"):
					d = e * np.tan(shape.fold_value / 2)
					for line in shape.shapes:
						if isinstance(line, Line):
							original_lines.append(line)
							offset_instructions.append((line, d, shape))
				else:
					# On ajoute directement les plis non concernés
					new_patron.add_folds(shape.shapes, param=shape.param)
					fixed_lines.extend(shape.shapes)  # pour intersections plus complètes si utile
			else:
				all_lines = []
				for s in shape.shapes:
					if isinstance(s, Line):
						all_lines.append(s.copy())
					elif isinstance(s, Surface):
						all_lines += s.get_lines()
				fixed_lines.extend(all_lines)
				new_patron.add_shapes(
						all_lines,
						param=shape.param,
						background=shape.background,
						outline=shape.outline,
						outside=shape.outside
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
			normal = line.normal_vect().normalize() * d
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
						d2 = offset_instructions[index][1]
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
						param=shape.param,
						background=shape.background,
						outline=shape.outline,
						outside=shape.outside
					)


		new_patron.texts = self.texts.copy()
		new_patron.w_h(self.width, self.height)
		return new_patron




	def create_patron_offset(self, mountain = False, e : Number = 0, L : Number = 2, full = False, param=""):
		shapes = []
		n_patron = Patron(self.name + "_offset", laser_cut=self.laser_cut, size=self.size, origin=self.origin)
		n_patron.w_h(self.width, self.height)
		outside_poly = None
		for drawnshape in self.shapes:
			d = 0 if L == 0 else 0.25
			if drawnshape.outside:
				# if the shape is outside, we offset it by 0.5
				outside_poly = drawnshape.to_polygon()
				drawnshape.shapes = [outside_poly.copy()]
				dep = 0 if L == 0 else -0.25
				outside_poly = outside_poly.copy().offset(dep).in_2D(BASE_REPERE3D)
				if not full and L > 0:
					n_patron.add_drawn_shapes(DrawnShapes([outside_poly], param=drawnshape.param, background=drawnshape.background, outline=drawnshape.outline, outside=True))
			if isinstance(drawnshape, Folds):
				if drawnshape.fold_type != "n":
					if mountain ^ (drawnshape.fold_type == "v"):
						d = e * np.tan(drawnshape.fold_value/2) + L
					else:
						d = L
			for shape in drawnshape.shapes:	
				shapes.append([shape, drawnshape.param, drawnshape.background, drawnshape.outline, drawnshape.outside, [d] * (len(shape) - 1)])

		nb_shapes = len(shapes)
		i = 0
		while i < nb_shapes:
			nb_test = nb_shapes
			shape = shapes[i][0]
			if isinstance(shape, Line):
				for j in range(nb_test):
					if isinstance(shapes[j][0], Line):
						continue
					poly = shapes[j][0]
					if not isinstance(shapes[j][0], Polygon):
						poly = shapes[j][0].to_polygon()
					dec = shapes[j][5] + shapes[i][5]
					pols, decals = poly.cut(shape, dec)

					if len(pols) == 1:
						# if the polygon is not cut, we continue
						continue
					shapes[j][0] = pols[0]
					shapes[j][5] = decals[0]
					for k in range(1, len(pols)):
						shapes.append([pols[k], shapes[j][1], shapes[j][2], shapes[j][3], shapes[j][4], decals[k]])
						nb_shapes += 1
			i += 1
		holes = []
		for shape in shapes:
			if isinstance(shape[0], Line) or len(shape[0]) < 4:
				continue
			# print(shape[0])
			poly_shape = shape[0].offset(shape[5])
			poly_shape.in_2D(BASE_REPERE3D)
			if full:
				poly_shape.change_direction()
				holes.append(poly_shape)
			else:
				if param == "":
					param = "def_pli_cut"
				n_patron.add_shapes(poly_shape, param=param, background=shape[2], outline=shape[3], outside=shape[4])		
		
		if full:
			if outside_poly is None :
				raise ValueError("no outside defined")
			holed_poly = HoledPolygon.from_polygons(outside_poly, holes)
			if param == "":
				param = "def_adhesif_remove"
			n_patron.add_shapes([holed_poly], param=param, background=True, outline=shapes[0][3], outside=shapes[0][4])

		return n_patron

	def create_patron_offset3(self, e : Number =0, L : Number =2, param="def_pli_cut", closed = True):
		
		lines_list = []

		# list all lines in the same table with the type of fold and d 
		for shape in self.shapes:
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
		
		patron = Patron(self.name + "_martyr", laser_cut=self.laser_cut, size=self.size, origin=self.origin)
		patron2 = patron.copy()
		patron.add_folds(lines[1], param=param)
		patron2.add_folds(lines[0], param=param)
		patron2.mirror(Line(Point(self.width + 2.5, 0), Point(self.width + 2.5, 5)))

		patron += patron2
		patron.w_h(self.width  * 2 + 5 + self.origin[0], self.height + self.origin[1])

		return patron

	def create_lasercut_patron(self, asym=False, e : Number =0, L : Number=2, adhesif_param ="def_adhesif_remove"):
		"""
			return two patrons on one \n
			asym : if True, the two patrons are asymetric \n
			e : thickness of the fold \n
			l : thickness of the removed adhesive \n
			adhesif_param : name of the parameter to use for the adhesive \n
		"""
		patrons = [self.copy(), self.copy()]

		# if asym is False or the thickness of the plate is null, we create the two patrons with the same fold
		if not asym or e == 0:
			if L != 0:
				patrons[0].thicken_lines(L / 2, param=adhesif_param)
				patrons[1].thicken_lines(L / 2, param=adhesif_param)
		else:
			# else we create the two patrons with lines to remove the material surplus
			patrons[0] = self.create_patron_decal(True, e)
			patrons[1] = self.create_patron_decal(False, e)
			if L != 0:
				adh_rem_p = [[],[]]
				for shape in self.shapes:
					if isinstance(shape, Folds):
						if shape.fold_type == "n":
							adh_rem_p[0] += shape.shapes
							adh_rem_p[1] += shape.shapes
						else :
							d = e * np.tan(shape.fold_value/2)
							lines_small = shape.thicken_to_rect(L/2)
							lines_big = shape.thicken_to_rect(L/2 + d)
							i = 0 if shape.fold_type == "m" else 1
							adh_rem_p[i] += lines_small
							adh_rem_p[i ^ 1] += lines_big
					elif isinstance(shape, DrawnShapes):
						adh_rem_p[0] += shape.shapes
						adh_rem_p[1] += shape.shapes

				patrons[0].add_shapes(adh_rem_p[0], param=adhesif_param, background=True, outline=False)
				patrons[1].add_shapes(adh_rem_p[1], param=adhesif_param, background=True, outline=False)
		patrons[1].mirror(Line(Point(self.width + 2.5, 0), Point(self.width + 2.5, 5)))

		patron = patrons[0] + patrons[1]
		patron.w_h(self.width * 2 + 5, self.height)
		patron.name = self.name + "_lasercut"
		return patron

	def create_lasercut_martyr(self, asym=False, e=0, L=2):
		
		lines = [[], []]
		for shape in self.shapes:
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

		patron = Patron(self.name + "_martyr", laser_cut=self.laser_cut, size=self.size, origin=self.origin)
		patron2 = patron.copy()
		patron.add_folds(lines[0], param="def_pli_cut")
		patron2.add_folds(lines[1], param="def_pli_cut")
		patron2.mirror(Plane(Point(self.width + 2.5, 0), E2X))

		patron += patron2
		patron.w_h(self.width  * 2 + 5 + self.origin[0], self.height + self.origin[1])

		return patron

	def create_pattern(self):
		svg_elements = []
		contour_col = "black"
		# couleur moutain, couleur valley
		pli_col = ["red", "blue"]
		width = 2

		for shape in self.shapes :
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

		self.canvas = svg.SVG(
			width=svg.Length(self.width + 2 * self.origin[0], "mm"),
			height=svg.Length(self.height + 2 * self.origin[1], "mm"),
			elements=svg_elements, 
		)
