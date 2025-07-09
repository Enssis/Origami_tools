from dataclasses import dataclass
import os
from typing import List, Sequence
import prettytable as pt 
import json

from ..Utils._types import Number
from ..Utils._svg_utils import svg_text_from_text, simplifed_hex
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
			return f"d{self.dash_length}l{int(self.dash_full_ratio) * 100}r" + id_name 

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


class ListTemplate(dict[str, str]):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.__dict__ = self

	def __str__(self):
		text = "ListTemplate:\n"
		for key, value in self.items():
			text += f"  {key} : {value}\n"

		return text

	def __repr__(self):
		return self.__str__()

	def as_json(self):
		return self.__dict__

	def copy(self) -> "ListTemplate":
		return ListTemplate(**self.__dict__.copy())

	def save(self, name, dir_path=None):
		"""
			sauvegarde le template dans un fichier
			name : nom du template
			dir_path : chemin du dossier dans lequel on veut sauvegarder le template
		"""
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + name + ".json"
		json.dump(self.as_json(), open(path, "w"), indent=4)
		print(f"Template {name} sauvegardé dans {path}")

	@staticmethod
	def load(name, dir_path=None):
		"""
			charge le template depuis un fichier
			name : nom du template à charger
			dir_path : chemin du dossier dans lequel on veut charger le template
		"""
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + name + ".json"
		if not os.path.exists(path):
			print(f"Le template {name} n'existe pas dans le fichier de sauvegarde {path}.")
			return None
		with open(path, "r") as f:
			data = json.load(f)
			return ListTemplate(**data)
		
	def contain_list(self, key : list[str]) -> bool:
		"""
			renvoie True si le template contient la liste de clés
			key : liste de clés à tester
		"""
		for k in key:
			if k not in self.keys():
				return False
		return True

	@staticmethod
	def default_template():
		"""
			renvoie un template par défaut
		"""
		return ListTemplate(
			cut = "default_cut",
			text = "default_text",
		)

class ParamList: 
	"""
	Class to manage a list of laser parameters.
	"""

	def __init__(self, params : dict[str, LaserParam] = {}, template : ListTemplate = ListTemplate(), profile : str = "default"):
		self.params = params
		self.template = template
		self.profile = profile
		if self.params == {}:
			pl = ParamList.load_from_profile(self.profile)
			if pl is None:
				print(f"Le profil {self.profile} n'existe pas.")
				return
			self.params = pl.params
			if self.template == ListTemplate():
				self.template = pl.template

		self.names = list(self.params.keys())

		for value in self.template.values():
			if value not in self.names: 
				raise ValueError(f"{value} does not exist in the params list.")


	
	@classmethod
	def from_list(cls, params : list[LaserParam], template : ListTemplate = ListTemplate() , profile="default"):
		"""
		Initializes a ParamList from a list of LaserParam objects.
		params : list of LaserParam objects
		template : optional, a dictionary or ListTemplate for additional parameters
		profile : name of the profile for this ParamList
		"""
		params_dict : dict[str, LaserParam] = {param.name: param for param in params if isinstance(param, LaserParam)}

		return cls(params=params_dict, template=template, profile=profile)


	def __str__(self):
		desc = f"LaserCut {self.profile}:\n"
		desc += f"  template : {self.template} \n"
		desc += "  params :\n"
		for name, param in self.params.items():
			desc += f"  {name} : {param}"

		return desc

	def copy(self):
		"""
			copie le laser cut
		"""
		params = {name: param.copy() for name, param in self.params.items()}
		template = self.template.copy()
		return ParamList(params=params, template=template, profile=self.profile)

	def __repr__(self):
		return self.__str__()

	def as_json(self):
		"""
			renvoie le laser cut sous forme de json
		"""
		return {
			"profile": self.profile,
			"template": self.template.as_json() if isinstance(self.template, ListTemplate) else self.template,
			"params": [param.as_json() for param in self.params.values()]
		}

	def get_param_num(self, n) -> LaserParam:
		""" 
			renvoie le paramètre numero n
		"""
		if n < len(self.params):
			return list(self.params.values())[n]
		else:
			print(f"Le paramètre {n} n'existe pas dans ce profile.")
			return LaserParam.default_cut()

	def get_param(self, name) -> LaserParam:
		"""
			renvoie le paramètre avec le nom name
			name : nom du paramètre
		"""
		if name in self.names:
			return self.params[name]
		else:
			print(f"Le paramètre {name} n'existe pas dans ce profile.")
			return LaserParam.default_cut()

	def save_profile(self, overwrite_file=False, overwrite_param=False, dir_path=None):
		"""
		sauvegarde le profil dans un fichier
		overwrite_file : si True, écrase le fichier existant
		overwrite_param : si True, écrase le paramètre existant dans le fichier
		"""
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
								print(f"Le parametre {key} existe déjà dans le fichier de sauvegarde.")
							
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
		
		# on sauvegarde le template dans un fichier
		self.template.save(self.profile + "_template", dir_path=dir_path)

		print(f"Fichier de sauvegarde des paramètres laser dans {path}")

	@staticmethod
	def load_from_profile(profile, dir_path=None, template_name=None):
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		if not os.path.exists(path):
			print(f"Le profil {profile} n'existe pas.")
			return None
		# on charge le template si il est spécifié
		if template_name is None:
			template_name = profile + "_template"
		template = ListTemplate.load(template_name, dir_path=dir_path)
		if template is None:
			template = ListTemplate()
			# raise ValueError(f"Le template {template_name} n'existe pas.")
				
		with open(path, "r") as f:
			lines = f.readlines()[1:]
			params = []
			for line in lines:
				param = LaserParam.load_from_csv(line)
				if param is not None:
					params.append(param)
			if len(params) > 0:
				return ParamList.from_list(params=params, template=template, profile=profile)
				
		print(f"Le profil {profile} n'existe pas dans le fichier de sauvegarde {path}.")
		return None

	def fab_shapes(self, shapes : Sequence[Shape] | List[Shape], param="", background=False, outline=True, origin=Point(0, 0), all_visible=False):
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
			if not self.template.contain_list([param]):
				raise ValueError(f"The template {self.template} does not contain the key {param}.")
			name = self.template[param]
			param = self.params[name]

		if outline:
			ep = param.ep if not all_visible or param.ep >= 1 else 1
		else:
			ep = 0

		if background:
			fill = param.color
		else:
			fill = "none"

		fab_shapes = []
		width = param.ep if not all_visible or param.ep >= 1 else 1
		for shape in shapes:
			if isinstance(shape, Line):
				if not param.full:
					lines = shape.get_line_dashed(param.dash_length, param.dash_full_ratio)
					fab_shapes += [line.as_svg(param.color, opacity=1, width=width, origin=origin) for line in lines]
				else:
					fab_shapes.append(shape.as_svg(param.color, opacity=1, width=width, origin=origin))
			else:
				if not param.full and isinstance(shape, Surface) and not isinstance(shape, Circle):
					lines = shape.get_dashed(param.dash_length, param.dash_full_ratio)
					fab_shapes += [line.as_svg(param.color, opacity=1, width=width, origin=origin) for line in lines]
				else:
					fab_shapes.append(shape.as_svg(color=param.color, opacity=1, width=ep, fill=fill, origin=origin))
		return fab_shapes

	# retourne un texte pour la gravure
	def fab_text(self, text, x, y, font_size=10, text_anchor="start", param=""):
		if param in self.names:
			param = self.params[param]
		else: 
			if not self.template.contain_list(["text"]):
				raise ValueError("The template does not contain the key 'text'.")
			name = self.template["text"]
			param = self.params[name]

		return svg_text_from_text(text, x, y, font_size=font_size, color=param.color, text_anchor=text_anchor)

	@staticmethod
	def table_from_profile(profile, dir_path=None):
		if dir_path is None:
			dir_path = LASER_SAVE_PATH
		path = dir_path + profile + ".csv"
		with open(path) as fp:
			mytable = pt.from_csv(fp)
		return mytable
	
	@staticmethod
	def update_names(profile,dir_path=None):
		pl = ParamList.load_from_profile(profile, dir_path)
		if pl is None:
			print(f"Le profil {profile} n'existe pas.")
			return
		for key, value in pl.params.items():
			value.name = value.id() + "_" + simplifed_hex(value.color)

		pl.params = {key: value for key, value in pl.params.items() if isinstance(value, LaserParam)}
		pl.names = list(pl.params.keys())
		pl.save_profile(overwrite_file=True, overwrite_param=True, dir_path=dir_path)

	def show_param(self, markdown=False):
		# Affiche le tableau des paramètres
		table = pt.PrettyTable()
		if markdown:
			table.set_style(pt.TableStyle.MARKDOWN)

		table.field_names = ["Template_name","Nom", "Couleur", "Couleur num", "Epaisseur", "Plein", "Longueur trait", "Ratio plein/pointillé", "Puissance", "Vitesse", "Passe"]
		for param in self.params.values():
			template_name = ""
			for key, value in self.template.items():
				if value == param.name:
					template_name = key
					break
			color = "`" + param.color + "`" if markdown else param.color
			table.add_row([template_name, param.name, color, str(get_lasercut_color_num(param.color)), param.ep, param.full, param.dash_length, param.dash_full_ratio, param.power, param.speed, param.passe])
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

	def add_param(self, param : LaserParam):
		"""
			ajoute un paramètre au laser cut
			param : LaserParam
		"""
		if isinstance(param, LaserParam):
			self.params[param.name] = param
			self.names.append(param.name)
			print(f"Le paramètre {param.name} a été ajouté au laser cut {self.profile}.")
		else:
			print(f"Le paramètre {param} n'est pas un LaserParam.")