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
        "material_path": "",
        "origami_path": "",
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
    if path[-1] != "/":
        path += "/"
    config["material_path"] = path
    print(f"The path for materials file is now {path}")
    _set_config(config)

def get_materials_default():
    return os.path.dirname(__file__) + "/material_profiles/default.csv"

def get_origami_dir_default():
    return os.path.dirname(__file__) + "/Origamis/origami_saves/"

def get_origami_dir():
    """
        get the path of origami directory
    """
    config = _get_config()
    origami_path = config["origami_path"]
    if origami_path == "":
        origami_path = get_origami_dir_default()
        config["origami_path"] = origami_path
        _set_config(config)
    if not os.path.exists(origami_path):
        raise FileNotFoundError(f"Le path {origami_path} n'existe pas")
    return origami_path

def set_origami_dir(path):
    """
        change the directory path for origami
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le path {path} n'existe pas")
    config = _get_config()
    if path[-1] != "/":
        path += "/"
    config["origami_path"] = path
    print(f"The path for origami file is now {path}")
    _set_config(config)



# from .decoupe_laser import *
from . import Geometry, LaserCut, Patron
__all__ = [
    "Geometry", 
    "LaserCut",
    "Patron", 
]
