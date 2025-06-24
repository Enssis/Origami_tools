from dataclasses import dataclass
import os
from typing import List, Sequence
import prettytable as pt 

from ..Utils._types import Number
from ..Utils._svg_utils import svg_text_from_text
from ..Geometry import Shape, Line, Surface, Point, Circle
from . import LASER_SAVE_PATH, get_lasercut_color_num

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
	def from_id(id):
		"""
			renvoie un paramètre à partir de son id
			id : id du paramètre
		"""
		if id.startswith("f"):
			full = True
			id = id[1:]
		else:
			full = False
			id = id[1:]
		
		passe, power, speed = id.split("p")
		passe = int(passe)
		power = int(power[:-1])
		speed = float(speed[:-1])
		
		return LaserParam("red", name=f"Laser_param_{power}_{speed}_red", ep=0.2, full=full, power=power, speed=speed, passe=passe)

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
				if not param.full and isinstance(shape, Surface) and not isinstance(shape, Circle):
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