import os
import json

def _get_config():
    with open(os.path.dirname(__file__) + "/config.json", "r") as f:
        return json.load(f)

def _set_config(config : dict):
    with open(os.path.dirname(__file__) + "/config.json", "w") as f:
        json.dump(config, f, indent=4)

def reset_config():
    """
        reset le config
    """
    config = {
        "material_path": ""
    }
    _set_config(config)

def get_material_path():
    """
        get le path du material
    """
    config = _get_config()
    material_path = config["material_path"]
    if material_path == "":
        material_path = os.path.dirname(__file__) + "/material_profiles/"
        config["material_path"] = material_path
        _set_config(config)
    if not os.path.exists(material_path):
        raise FileNotFoundError(f"Le path {material_path} n'existe pas")
    return material_path

def set_material_path(path):
    """
        change le path du material
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le path {path} n'existe pas")
    config = _get_config()
    config["material_path"] = path
    _set_config(config)

def get_materials_default():
    return os.path.dirname(__file__) + "/material_profiles/default.csv"

from .decoupe_laser import *
from .geometry import *

# __all__ = [
#     "geometry", 
#     "decoupe_laser", 
# ]
