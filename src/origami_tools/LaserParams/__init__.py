from origami_tools import get_material_path

LASER_SAVE_PATH = get_material_path()

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

from .laser_cut import ParamList, LaserParam, ListTemplate

__all__ = [
    "laser_cut_colors",
    "get_lasercut_color_num",
    "ParamList",
    "LaserParam",
	"ListTemplate",
]