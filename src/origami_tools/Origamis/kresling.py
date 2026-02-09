from dataclasses import dataclass
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from ..Geometry import *
from ..Utils._types import Number
from .. import get_origami_dir
from ..Pattern import Pattern
from ..LaserParams import ParamList


def rotation_vect_and_point(vec, pt, angle):
    """
    Give the rotation matrix for a rotation of angle around the vector vec and point pt
    """

    a, b, c = pt
    u, v, w = vec / np.linalg.norm(vec)

    ca = np.cos(angle)
    sa = np.sin(angle)

    return np.array([[u**2 + (v**2 + w**2) * ca,
					u*v*(1-ca) - w*sa,
					u*w*(1-ca) + v*sa,
					(a * (v ** 2 + w**2) - u*(b * v + c * w))*(1 - ca) + (b * w - c * v) * sa],
                    [u*v*(1-ca) + w*sa,
					v**2 + (u**2 + w**2) * ca,
					v*w*(1-ca) - u*sa,
					(b * (u ** 2 + w**2) - v*(a * u + c * w))*(1 - ca) + (c * u - a * w) * sa],
                    [u*w*(1-ca) - v*sa,
					v*w*(1-ca) + u*sa,
					w**2 + (u**2 + v**2) * ca,
					(c * (u ** 2 + v**2) - w*(a * u + b * v))*(1 - ca) + (a * v - b * u) * sa],
                    [0, 0, 0, 1]])

ORIGAMI_DIR_PATH = get_origami_dir()


@dataclass
class TDK:
    """
    Class for a Kresling tower
    """
    n : int
    l : Number
    b : Number
    r : Number
    h1 : Number
    h2 : Number
    m : Number = 1
    a : Number = 0
    r_p : Number = 0
    nb_stable : int = 2
    name : str = ""


    def __post_init__(self):
        if self.name == "":
            self.name = f"TDK_{self.n}_{int(self.l * 10) / 10}_{int(self.b * 10) / 10}_{int(self.r * 10) / 10}_{int(self.m * 100) / 100}"

        if self.l ** 2 - 4 * self.r ** 2 < 0 :
            self.h_min = 0
        else: 
            self.h_min = np.sqrt(self.l ** 2 - 4 * self.r ** 2)
        
        if self.a == 0:
            self.a = 2 * self.r * np.sin(np.pi / self.n)

        if self.r_p == 0:
            self.r_p = self.b * np.sin(alkashi_angle(self.a, self.b, self.l))


        temp_acos = np.arccos((self.l**2 - self.b**2)/(4*self.m*self.r**2*np.sin(np.pi/self.n)))
        self.phi_1 = np.pi/2 - temp_acos
        self.phi_2 = np.pi/2 + temp_acos


        self.patron = None
        self.attache = None
        self.volume = None

    @classmethod
    def from_eta(cls, l, b, n, r, m=1, name=""):
        """
        Create a Kresling tower from eta
        :param l: length of the long side
        :param b: length of the short side
        :param n: number of folds
        :param r: radius of the folds
        :param m: ration of radius up and down
        :param name: name of the tower
        :return: Kresling tower
        """

        a = 2 * r * np.sin(np.pi / n)
        r_p = np.sqrt(b**2 - ((l**2-a**2-b**2)/(2*a))**2)
        x0 = r * np.cos(np.pi / n)
        xC = np.sqrt(m**2*r**2 - ((l**2 - b**2) / (2 * a))**2)
        if np.abs(l**2 - b**2) > 2 * m*r** 2 * np.sin(np.pi / n):
            print("La tour n'a pas de position stable")
            return cls(n, l, b, r, 0, 0, m, a, r_p, 0, name)
        
        h2 = np.sqrt(r_p**2 -(x0 - xC)**2)

        if (x0 + xC)**2 > r_p**2:
            h1 = 0
            nb_stable = 1
        else :
            h1 = np.sqrt(r_p**2 - (x0 + xC)**2)
            nb_stable = 2

        return cls(n, l, b, r, h1, h2, m, a, r_p, nb_stable, name) 
    
    @classmethod
    def from_n_r_h(cls, n, r, h1, h2, nb_stables=2, m=1, name=""):
        """
        Create a Kresling tower from n, r, h1, h2
        :param n: number of folds
        :param r: radius of the folds
        :param h1: height of the first floor
        :param h2: height of the second floor
        :param nb_stables: number of stable positions
        :param name: name of the tower
        :return: Kresling tower
        """
        if h1 > h2:
            h1, h2 = h2, h1

        if h2**2 - h1**1 > 4 * m*r**2 * np.cos(np.pi / n):
            raise(ValueError("Les valeurs de h1 et h2 ne sont pas compatibles avec la construction d'une tour de Kresling"))

        base = (h2 ** 2 + h1 ** 2) / 2 + (m + 1) * r ** 2
        change = np.tan(np.pi / n) * np.sqrt(4 * m**2* r ** 4 * np.cos(np.pi / n) ** 2 - ((h2 ** 2 - h1 ** 2) / 2) ** 2)
        l = np.sqrt(base + change)
        b = np.sqrt(base - change)
        
        if h1 <= 0:
            h1 = 0
            nb_stables = 1

        return cls(n, l, b, r, h1, h2, m, nb_stable=nb_stables, name=name) # type: ignore

    ## TODO : add m
    @classmethod
    def from_dh_dphi_n_r(cls, dh, dphi, n, r, name=""):
        """
        Create a Kresling tower from dh, dphi, n, r
        :param dh: difference of height between the two stable positions
        :param dphi: difference of angle between the two stable positions
        :param n: number of folds
        :param r: radius of the folds
        :param name: name of the tower
        :return: Kresling tower
        """
        h2_h1_sq = r**2 * np.cos(np.pi/n) * 4 * np.sin(dphi/2) # h2**2 - h1**2
        h1 = (h2_h1_sq - dh**2) / (2 * dh)
        h2 = h1 + dh

        return cls.from_n_r_h(n, r, h1, h2, name=name) # type: ignore

## TODO : add m
    @classmethod
    def from_dict(cls, d, name=""):
        """
        Create a Kresling tower from a dictionary
        :param d: dictionary with the parameters of the tower
        :return: Kresling tower
        """
        return cls(d["n"], d["l"], d["b"], d["r"], d["h1"], d["h2"], d["a"], d["r_p"], d["nb_stable"]) # type: ignore

## TODO : add m
    # representation de l'objet en chaine de caracteres
    def __str__(self):
        return f"{self.name} : n={self.n}, l={self.l}, b={self.b}, r={self.r}, h1={self.h1}, h2={self.h2}"
    
    ## TODO : add m
    def __repr__(self):
        return self.__str__()


    # calcul de phi_l et phi_b (equations J. Berre)
    def john_phi_l(self, h):
        if self.m != 1:
            raise NotImplementedError("La formule de J. Berre pour phi_l n'est pas compatible avec m != 1")
        return 2 * np.arcsin(np.sqrt(self.l**2 - h**2)/ (2 * self.r)) - np.pi / self.n

    def john_phi_b(self, h):
        if self.m != 1:
            raise NotImplementedError("La formule de J. Berre pour phi_l n'est pas compatible avec m != 1")
        return 2 * np.arcsin(np.sqrt(self.b**2 - h**2)/ (2 * self.r)) + np.pi / self.n

    # calcul de phi (equation J. Berre)
    def john_phi(self, h) -> float:
        if self.m != 1:
            raise NotImplementedError("La formule de J. Berre pour phi_l n'est pas compatible avec m != 1")
        if self.h1 <= h and self.h2 >= h:
            return self.john_phi_l(h)
        else:
            return self.john_phi_b(h)
    
    # calcul de phi_l et phi_b (equations perso)
    def phi_l(self,h):
        return np.arccos((h**2 - self.l**2 + self.r**2 *(self.m**2 + 1))/(2*self.m*self.r**2)) - np.pi/self.n

    def phi_b(self,h):
        return np.arccos((h**2 - self.b**2 + self.r**2 *(self.m**2 + 1))/(2*self.m*self.r**2)) + np.pi/self.n

    # calcul de phi (equation perso)
    def phi(self, h) -> float:
        if self.h1 <= h and self.h2 >= h:
            return self.phi_l(h)
        else:
            return self.phi_b(h)


    def h(self, phi):

        gamma = np.pi / self.n
        if phi <= self.phi_1 and phi >= self.phi_2:
            return np.sqrt(self.b**2 - self.r**2*(self.m**2 + 1 - 2 * self.m * np.cos(phi - gamma)))
        return np.sqrt(self.l**2 - self.r**2*(self.m**2 + 1 - 2 * self.m * np.cos(phi + gamma)))

    def l_compressed(self, h):
        return np.sqrt(h**2 + 2 * self.r**2 * (1 - np.cos(self.phi(h) + np.pi/self.n)))
    
    def b_compressed(self, h):
        return np.sqrt(h**2 + 2 * self.r**2 * (1 - np.cos(self.phi(h) - np.pi/self.n)))

    # parametres des tours fabriquees par J. Berre
    @staticmethod
    def tours_john(i : int) : 
        tours = [TDK(8, 202.2, 174.6, 90, 106.3, 166.3), 
                TDK(8, 174.2, 139.1, 90, 41.5, 124.7),
                TDK(8, 183.5, 148.5, 90, 80.0, 130.0),
                TDK(12, 210.2, 186.5, 100, 106.3, 166.3)]
        return tours[i]
    
    def as_dict(self):
        return {
            "n": self.n,
            "l": self.l,
            "b": self.b,
            "r": self.r,
            "h1": self.h1,
            "h2": self.h2,
            "a": self.a,
            "r_p": self.r_p,
            "nb_stable": self.nb_stable
        }

    # sauvegarde la tour dans un fichier
    def save(self, path = "", overwrite = False):
        if path == "":
            path = ORIGAMI_DIR_PATH + "saved_towers.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le path {path} n'existe pas")
        if not path.endswith(".json"):
            raise ValueError(f"Le path {path} n'est pas un fichier json")

        with open(path, "r") as f:
            towers = json.load(f)
            if self.name in towers.keys() and not overwrite:
                print("Le nom existe deja")
                return None
        
        towers[self.name] = self.as_dict()

        with open(path, "w") as f:
            json.dump(towers, f, indent=4)
    
    # load a tour from the saved tours
    @staticmethod
    def load(name, path = ""):
        """
        Load a Kresling tower from a json file
        :param name: name of the tower
        :param path: path of the json file
        :return: Kresling tower
        """
        if path == "":
            path = ORIGAMI_DIR_PATH + "saved_towers.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le path {path} n'existe pas")
        if not path.endswith(".json"):
            raise ValueError(f"Le path {path} n'est pas un fichier json")
        
        with open(path, "r") as f:
            towers = json.load(f)
            if name not in towers.keys():
                raise NameError("La tour n'existe pas")
            return TDK.from_dict(towers[name], name)
        
    @staticmethod
    def names(path = ""):
        """
        Get the names of the saved towers in the file path
        :return: list of names
        """
        if path == "":
            path = ORIGAMI_DIR_PATH + "saved_towers.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le path {path} n'existe pas")
        if not path.endswith(".json"):
            raise ValueError(f"Le path {path} n'est pas un fichier txt")
        
        with open(path, "r") as f:
            towers = json.load(f)
            return towers.keys()
        
    def surface(self):
        l = self.a * (self.n - 1) + np.sqrt(self.l**2 - self.r_p**2) 
        return f"{self.r_p} * {l} = {self.r_p * l} mm²"


    # calcul des angles des plis montagnes, vallees et rho 
    def angle_pli_montagne(self, h):
        pin = np.pi / self.n
        n_N = np.array([self.a * h, 0, -self.a * self.r * (np.cos(self.phi(h)) - np.cos(pin))])

        n_P = 2 * self.r * np.sin(pin) * np.array([-h * np.cos(self.phi(h) + pin), -h * np.sin(self.phi(h) + pin), self.r * (np.cos(pin) - np.cos(self.phi(h)))])

        return np.arccos(np.dot(n_N, n_P) / (np.linalg.norm(n_N) * np.linalg.norm(n_P)))
    


    @staticmethod
    def ang_mountain_phi_stat(phi, n, r, l, b):
        gamma = np.pi/n
        c_phi = np.cos(phi)
        c_gamma = np.cos(gamma)

        temp_acos = np.arccos((l**2 - b**2)/(4*r**2*np.sin(np.pi/n)))
        phi_1 = np.pi/2 - temp_acos
        phi_2 = np.pi/2 + temp_acos


        if phi <= phi_1 and phi >= phi_2:
            h = np.sqrt(l**2 - 4 * r**2 * np.sin((phi + np.pi/n)/2)**2)
        else:
            h = np.sqrt(b**2 - 4 * r**2 * np.sin((phi - np.pi/n)/2)**2)

        a = (r*(c_phi - c_gamma))**2
        c_theta = (a-h**2*np.cos(gamma + phi))/(h**2 + a)

        return np.arccos(c_theta)


    def ang_mountain_phi(self, phi):
        return self.ang_mountain_phi_stat(phi, self.n, self.r, self.l, self.b)
    
    def angle_pli_vallee(self, h):
        pin = np.pi / self.n

        n_P = 2 * self.r * np.sin(pin) * np.array([-h * np.cos(self.phi(h) + pin), -h * np.sin(self.phi(h) + pin), self.r * (np.cos(pin) - np.cos(self.phi(h)))])

        n_Q = np.array([- self.r * h * (np.sin(3 * pin) - np.sin(pin)), self.r * h * (np.cos(3 * pin) - np.cos(pin)), self.r ** 2 * (2 * np.sin(pin) * np.cos(self.phi(h)) - np.sin(2 * pin))])
        
        return np.pi - np.arccos(np.dot(n_P, n_Q) / (np.linalg.norm(n_Q) * np.linalg.norm(n_P)))

    def angle_pli_rho(self, h):
        pin = np.pi / self.n

        n_N = np.array([self.a * h, 0, -self.a * self.r * (np.cos(self.phi(h)) - np.cos(pin))])

        n_B = np.array([0, 0, 1])
        return np.arccos(np.dot(n_N, n_B) / (np.linalg.norm(n_N) * np.linalg.norm(n_B)))

    def delta_b(self, phi):
        return np.sqrt(self.l**2 - 4 * self.r**2 * np.sin(np.pi/self.n) * np.sin(phi)) - self.b

    def max_delta_b(self):
        return np.sqrt(self.l**2 - 4 * self.r**2 * np.sin(np.pi/self.n)) - self.b
    
    def delta_phi(self):
        return 2 * np.arcsin((self.h2**2 - self.h1**2) / (4 * self.r**2 * np.cos(np.pi/self.n)))

    ### =============== Creation du patron =============== ###

    def create_motif_base(self, laser_cut, origin=Point(0, 0)):
        motif = Pattern(self.name, laser_cut, origin=origin)
        angle_UVW = alkashi_angle(self.l, self.b, self.a)
        angle_VUW = alkashi_angle(self.l, self.a, self.b)
        R0 = Point(0,0)
        line_l = Line.from_angle(R0, angle_VUW, self.l)
        line_a = Line(R0, Point(self.a, 0))
        line_b = Line.from_angle(Point(self.a, 0), angle_UVW + angle_VUW, self.b)

        motif.add_folds([line_l], "v", self.angle_pli_montagne(self.h1))
        motif.add_folds([line_a], "n", 0)
        motif.add_folds([line_b], "m", self.angle_pli_vallee(self.h1))
        return motif 

    def create_patron_simu(self, h):
        if self.patron is None:
            self.create_patron(h)
        self.patron.create_pattern() # type: ignore

    
    def create_patron(self, h : Number =0, attache : Pattern | None = None, param_list : ParamList | None = None, closing : int = 2, side : int = 0, closing_type : int = 1, outside_param="fold_cut") -> Pattern:
        """
        Create the patron of the Kresling tower
        :param h: height of the tower, if 0, use h1
        :param attache: Patron to attach to the tower, if None, no attachment. Must be the first one on the top left of the patron and be open on the down side
        :param laser_cut: ParamList object to use for the patron, if None, use the default list
        :param closing: number of the closing method
        :param side: 0 => no sides diff, 1 => side1, 2 => side2
        :param closing_type: 1 => closing at half pane; 2 => closigne at half 2 panes 
        :return: Patron object
        """


        if h == 0: #default height
            h = self.h1

        w = 5 + self.a/2 if (closing == 2 and side == 1) else 5 + self.a if (closing == 2 and side == 2) else 5 # margin for attache depending on closing type

        self.patron = Pattern(self.name, param_list=param_list, origin=Point(w, 5))
        
        if not self.patron.param_list.template.contain_list([outside_param, "fold_cut"]): # type: ignore
            raise ValueError(f"Not all parameters of {[outside_param, "fold_cut"]} is in the template list {self.patron.param_list.template.__str__()}")
        
        if isinstance(attache, Pattern): # if attache is given, prepare it's place on the patron
            attache.origin = self.patron.origin.copy() 
            self.patron.origin[1] += attache.height

        dec = np.sqrt(self.b ** 2 - self.r_p ** 2)

        angle_v = self.angle_pli_vallee(h)
        angle_m = self.angle_pli_montagne(h)

        long = self.a * self.n
        phi_delta = alkashi_angle(self.l, self.b, self.a)
        phi1 = np.arcsin(self.r_p/self.l)
        phi2 = phi_delta + phi1


        b_lines = [Line.from_angle(Point(0, 0), phi2, self.b)]
        l_lines = [Line.from_angle(Point(0, 0), phi1, self.l)]

        depl = Vec(self.a, 0)

        for i in range(1, self.n):
            b_line = b_lines[-1].copy()
            b_lines.append(Line(b_line[0] + depl, b_line[1] + depl))

            l_line = l_lines[-1].copy()
            l_lines.append(Line(l_line[0] + depl, l_line[1] + depl))
        
        last_bline = b_lines[-1].copy()
        last_bline.translate(depl)
        b_lines.append(last_bline)

        self.patron.add_folds(b_lines[1:-1], "m", angle_m)
        self.patron.add_folds(l_lines, "v", angle_v)
        
        if closing == 2 and side == 1:
            p2 = b_lines[0][1].copy()
            A = Point(0,0)
            if attache is not None:
                A = Point(-self.a/2, -attache.height)
                half_attache = attache.cut_half()
                half_attache.translate(-depl)
                self.patron += half_attache
            limit_lines : list[Shape] = [] 
            if closing_type == 2:
                p2.translate(-depl/2)
                if attache is None:
                    limit_lines.append(Line.from_dir(p2, depl / 2))
                else:
                    half_attache = attache.cut_half(0)
                    half_attache.rotate(np.pi)
                    half_attache.translate(Vec(dec - self.a, self.r_p + attache.height))
                    limit_lines.append(Line.from_dir(p2, Vec(0, half_attache.height)))
                    self.patron.add_shapes([Line.from_dir(Point(0,0), Vec(-half_attache.width / 2, 0))], param="fold_cut")
                    self.patron += half_attache
            limit_lines.extend([Line(A, Point(- self.a/2, 0)), Line(Point(-self.a/2, 0), p2)])
            self.patron.add_shapes(limit_lines, outside=True, param=outside_param)
            self.patron.add_folds([b_lines[0]], "m", angle_m)
        else:
            if not (closing == 2 and side == 2 and closing_type == 1):
                self.patron.add_folds([b_lines[0]], "m", angle_m, outside=True, param=outside_param)
        
        if closing == 2 and side == 2:
            p1 = Point(self.a * (self.n - 1), 0)
            p2 = l_lines[-1][1].copy()
            
            if closing_type == 2:
                p2.translate(depl/2)
                p1.translate(depl)
                self.patron.add_folds([b_lines[-1]], "m", angle_m)

            else :
                l_line = l_lines[0].copy()
                self.patron.add_shapes([Line(l_line[0] - depl, l_line[1] - depl)], outside=True, param=outside_param)

            
            limit_lines : list[Shape] = [Line(p1 + depl / 2, p2)]
            if attache is not None:
                p3 = Point(self.a * (self.n - 1) + self.a / 2, -attache.height)
                half_attache = attache.cut_half(0)
                half_attache.translate(Vec((self. n - 1) * self.a, 0))
                if closing_type == 2:
                    half_attache.translate(depl)
                    p3.translate(depl)
                    half_attache2 = attache.cut_half()
                    half_attache2.rotate(np.pi)
                    half_attache2.translate(Vec(dec + self.a * self.n, self.r_p + attache.height))
                    self.patron += half_attache2
                    self.patron.add_shapes([Line.from_dir(p2, Vec(- half_attache2.width / 2, 0))], param="fold_cut")
                    limit_lines.append(Line.from_dir(p2, Vec(0, half_attache.height)))

                limit_lines.append(Line(p3, p1 + depl / 2))
                self.patron += half_attache
            else : 
                limit_lines.append(Line.from_dir(p1, depl / 2))
            self.patron.add_shapes(limit_lines, outside=True, param=outside_param)
            if closing_type != 2:
                self.patron.add_folds([b_lines[0]], "m", angle_m)
        else :
            self.patron.add_folds([b_lines[-1]], "m", angle_m, outside=True, param=outside_param)


        if attache is None:
            self.patron.add_shapes([Line(Point(0, 0), Point(long, 0))], outside=True, param=outside_param)
            self.patron.add_shapes([Line(Point(dec, self.r_p), Point(dec + long, self.r_p))], outside=True, param=outside_param)
        else:
            dual_attache = attache.copy()
            dual_attache.rotate(np.pi)
            dual_attache.translate(Vec(dec, self.r_p + attache.height))
            if closing == 2 and side == 2 and closing_type == 1:
                dual_attache.translate(Vec(self.a, 0))
                dual_attache += attache.copy()
                dual_attache.translate(Vec(-self.a, 0))
            else :
                dual_attache += attache.copy()

            rho = self.angle_pli_rho(h)
            up_folds = []
            down_folds = []
            for i in range(self.n):
                self.patron += dual_attache.copy()
                dual_attache.translate(Vec(self.a, 0))
                up_folds.append(Line(Point(i * self.a, 0), Point((i + 1) * self.a, 0)))
                up_folds.append(Line(Point(dec + i * self.a, self.r_p), Point(dec + (i + 1) * self.a, self.r_p)))
            if closing == 2 and side == 2 and closing_type == 1:
                up_folds[-1] = Line(Point(-self.a, 0), Point(0, 0))
            self.patron.add_folds(up_folds, "m", rho + np.pi/2, duplicate=True, param="fold_horizontal")
            # self.patron.add_folds(down_folds, "v", rho + np.pi/2, duplicate=True, param="fold_horizontal")
            
                
        
        self.patron.w_h((self.n + 1) * self.a, self.r_p)
        return self.patron

    
    def show(self):
        if self.patron is None:
            self.create_patron()
        self.patron.show() # type: ignore


    def create_3D(self, h_rep : Number | None = None, type : int = 0, fold_pc : Number = 100, ep_tot : Number = 0, side : int=0, h_constr : Number | None = None, e_pp = 0.5, rotation : Number = 0, chiral : bool = False):
        """
        type :
            0 => les plis ne sont pas des bonnes longuers (utilisation de phi)
            1 => le panneau n'est pas déformé (utilisation de rho)
            2 => le panneau est déformé comme 1 (utilisation des plis vallés / montagnes) 
            3 => l et b ne sont pas reliés au sommet V (TODO)
        fold_pc : pourcentage de pliage (pour le type 2)
        h_rep : hauteur de la tour représentée 
        h_constr : hauteur de la tour pour laquelle on calcule les plis 
        side : 0 => pas d'asymetrie, 1 => asymetrie montagne, 2 => asymetrie vallee
        ep_tot : epaisseur pour le calcul (e)
        """
        if h_rep is None:
            h_rep = self.h1
        if h_constr is None:
            h_constr = h_rep

        gamma = np.pi / self.n
        phi = self.phi(h_rep)
        x0 = self.r * np.cos(gamma)
        U = Point(x0, - self.r * np.sin(gamma), 0)
        W = Point(x0, self.r * np.sin(gamma), 0)
        if type == 1:
            rho = self.angle_pli_rho(h_rep)
            V = Point(x0 - self.r_p * np.cos(rho), self.a / 2 + np.sqrt(self.b ** 2 - self.r_p **2), self.r_p * np.sin(rho))
        else :
            V = Point(self.r * np.cos(phi), self.r * np.sin(phi), h_rep) 

        # rotation matrix for starting position

        verts = [Polygon([U, V, W, U])]
        if type == 2:
            angle_m_pc = (np.pi - self.angle_pli_montagne(h_rep)) * fold_pc / 100
            angle_v_pc = (np.pi - self.angle_pli_vallee(h_rep)) * fold_pc / 100

            for i in range(2 * self.n - 1):
                n_poly = verts[-1].copy()
                v_norm = n_poly.normal_vect()
                # get the middle point of the edge
                middle = Point.from_list(n_poly[1] + n_poly[0]) / 2 if i % 2 == 1 else Point.from_list(n_poly[1] + n_poly[2]) / 2

                n_poly.rotate(np.pi, middle, v_norm)
                direc = v_norm * (ep_tot - e_pp) / 2 if ep_tot != 0 else Vec(0, 0, 0) 
                if i % 2 == 0:
                    n_poly.rotate(-angle_m_pc, n_poly[2] - direc, n_poly[1] - n_poly[2])
                else:
                    n_poly.rotate(-angle_v_pc, n_poly[1] + direc, n_poly[1] - n_poly[0])

                verts.append(n_poly)
            if side != 0 and ep_tot != 0:
                angle_m = self.angle_pli_montagne(h_constr)
                angle_v = self.angle_pli_vallee(h_constr)
                # angle_rho = self.angle_pli_rho(h)
                d_mont = ep_tot / np.tan(angle_m/2) if side == 1 else 0
                d_val = ep_tot / np.tan(angle_v/2) if side == 2 else 0
                for i in range(len(verts)):
                    verts[i] = verts[i].offset([-d_val, -d_mont, 0])

        else :
            verts = []
            # asymetrie
            if side != 0 and ep_tot != 0:
                angle_m = self.angle_pli_montagne(h_constr)
                angle_v = self.angle_pli_vallee(h_constr)
                # angle_rho = self.angle_pli_rho(h)
                d_mont = ep_tot / np.tan(angle_m/2) if side == 1 else 0
                d_val = ep_tot / np.tan(angle_v/2) if side == 2 else 0
                pol = Polygon([U, V, W, U]).offset([-d_val, -d_mont, 0])
                U = pol[0]
                V = pol[1]
                W = pol[2]


            # transformation matrix
            mat_rot = np.array([[np.cos(2 * gamma), -np.sin(2 * gamma), 0, 0],
                                [np.sin(2 * gamma), np.cos(2 * gamma), 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]])
            mat_up_rot = np.array([[np.cos(phi + gamma), np.sin(phi + gamma), 0, 0],
                                [np.sin(phi + gamma), -np.cos(phi + gamma), 0, 0],
                                [0, 0, -1, h_rep], [0, 0, 0, 1]])

            U2 = U.copy().transform(mat_up_rot)
            V2 = V.copy().transform(mat_up_rot)
            W2 = W.copy().transform(mat_up_rot)

            for _ in range(self.n):
                U = U.transform(mat_rot)
                V = V.transform(mat_rot)
                W = W.transform(mat_rot)
                verts.append(Polygon([U, V, W]))
                U2 = U2.transform(mat_rot)
                V2 = V2.transform(mat_rot)
                W2 = W2.transform(mat_rot)
                verts.append(Polygon([U2, V2, W2]))
            
        if chiral:
            for pol in verts:
                pol = pol.mirror(Plane(Point(0, 0, 0), Vec(0, 1, 0)))
        if rotation != 0:
            for pol in verts:
                pol = pol.rotate(rotation, Point(0, 0, 0), Vec(0, 0, 1))

        return verts  

    def create_volume(self, h_rep : Number | None = None, constr_type : int =0, fold_pc : Number =100, ep : Number =0, decal : Number=0, ep_tot : Number =0, side : int =0, h_constr : Number | None = None, flat = False, rotation : Number = 0, chiral : bool = False):
        if h_rep is None:
            h_rep = self.h1
        if h_constr is None:
            h_constr = h_rep
        verts = self.create_3D(h_rep, constr_type, fold_pc, ep_tot - ep, side, h_constr, ep, rotation, chiral)
        if ep == 0 or flat:
            volume = Volume(verts)
        else:
            volume = verts[0].copy().extrude(ep)
            if decal != 0:
                normal = verts[0].normal_vect()
                normal.z = 0
                volume.translate(normal * decal)
            for i in range(len(verts)):
                plaque_volume = verts[i].copy().extrude(ep)
                if decal != 0:
                    normal = verts[i].normal_vect()
                    normal.z = 0
                    plaque_volume.translate(normal * decal)
                volume = volume + plaque_volume
        return volume

    def show_3D(self, h : Number | None = None, constr_type=0, fold_pc=100, ep=0.5, show_type=0, recreate = False, camera_pos= [45, 0, 25], rotation : Number= 0, chiral : bool = False):
        """
        type : 
            O => les plis ne sont pas des bonnes longuers (utilisation de phi)
            1 => le panneau n'est pas déformé (utilisation de rho)
            2 => le panneau est déformé comme 1 (utilisation des plis vallés / montagnes) 
            3 => l et b ne sont pas reliés au sommet V (TODO)

        fold_pc : pourcentage de pliage (pour le type 2)        
        """
        if h is None:
            h = self.h2

        if self.volume is None or recreate:
            self.volume = self.create_volume(h, constr_type, fold_pc, ep, rotation=rotation, chiral=chiral) 
        volume_mesh = self.volume.mesh_3D()


        ax = plt.figure().add_subplot(111, projection='3d')
        poly = Poly3DCollection(volume_mesh.vectors, alpha=0.5) # type: ignore
        poly.set_edgecolor('0')
        # Auto scale to the mesh size
        scale = volume_mesh.points.flatten() # type: ignore
        ax.auto_scale_xyz(scale, scale, scale) # type: ignore

        if fold_pc == 100 and show_type == 0:
            theta = np.linspace(0, 2 * np.pi, 50)
            x = np.cos(theta) * self.r
            y = np.sin(theta) * self.r 
            z = np.zeros_like(theta)
            ax.plot(x, y, z, label='parametric curve')
            z2 = z + h
            ax.plot(x, y, z2, label='parametric curve')

        ax.add_collection3d(poly) # type: ignore
        ax.set_aspect('equal')

        ax.set_title(self.name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore

        ax.azim = camera_pos[0] # type: ignore
        ax.dist = camera_pos[1] # type: ignore
        ax.elev = camera_pos[2] # type: ignore

        plt.show()  


    def save_stl(self, path, name="", h=None, constr_type=0, fold_pc=100, ep=0.5, recreate = False, rotation : Number = 0):
        if h is None:
            h = self.h1
        if self.volume is None or recreate:
            self.volume = self.create_volume(h, constr_type, fold_pc, ep, rotation=rotation) 
        volume_mesh = self.volume.mesh_3D()
        if name == "":
            name = self.name
        volume_mesh.save(path + f"{name}.stl") # type: ignore
        print(f"Saved {self.name} to {path + f'{name}.stl'}")


        
class MultiStoriesKresling:

    def __init__(self, towers : list[TDK], chiralities : list[bool] = [False], name : str =""):
        self.towers = towers
        self.chiralities = chiralities
        if len(towers) != len(chiralities):
            if len(chiralities) == 1:
                self.chiralities = chiralities * len(towers)
            else:
                raise ValueError("The number of towers must be equal to the number of chiralities or 1")
        self.name = name if name != "" else "MultiStoriesKresling_" + towers[0].name

        self.patron = None
        self.volume = None

    def create_volume(self, h_rep : list[Number] | None = None, constr_type : int =0, fold_pc : Number =100, ep : Number =0, decal : Number=0, ep_tot : Number =0, side : int =0, h_constr : Number | None = None, flat = False, base_rotation : Number = 0):
        volume = Volume.place_holder()
        if h_rep is not None and len(h_rep) != len(self.towers):
            raise ValueError("The number of heights must be equal to the number of towers or 1")
        
        rotation = base_rotation
        height = 0

        for i, tower in enumerate(self.towers):
            if h_rep is None:
                h = tower.h2
            else:
                h = h_rep[i]
            tower_volume = tower.create_volume(h, constr_type, fold_pc, ep, decal, ep_tot, side, h_constr, flat, rotation, self.chiralities[i])
            sens = -1 if self.chiralities[i] else 1
            rotation += (tower.phi(h) + np.pi / tower.n) * sens
            tower_volume.translate(Vec(0, 0, height))
            height += h
            if volume is None:
                volume = tower_volume
            else:
                volume = volume + tower_volume

        self.volume = volume    
        return volume
    
    def show_3D(self, h : list[Number] | None = None, constr_type=0, fold_pc=100, ep=0.5, show_type=0, recreate = False, camera_pos= [45, 0, 25], base_rotation : Number = 0, height_repr = None):
        """
        type : 
            O => les plis ne sont pas des bonnes longuers (utilisation de phi)
            1 => le panneau n'est pas déformé (utilisation de rho)
            2 => le panneau est déformé comme 1 (utilisation des plis vallés / montagnes) 
            3 => l et b ne sont pas reliés au sommet V (TODO)

        fold_pc : pourcentage de pliage (pour le type 2)        
        """

        if self.volume is None or recreate:
            self.volume = self.create_volume(h, constr_type, fold_pc, ep, base_rotation=base_rotation) 
        volume_mesh = self.volume.mesh_3D()


        ax = plt.figure().add_subplot(111, projection='3d')
        poly = Poly3DCollection(volume_mesh.vectors, alpha=0.5) # type: ignore
        poly.set_edgecolor('0')
        # Auto scale to the mesh size
        scale = volume_mesh.points.flatten() # type: ignore
        ax.auto_scale_xyz(scale, scale, scale) # type: ignore
        # height rep, set z axis of the graph to it
        if height_repr is not None:
            ax.set_zlim(0, height_repr)

        # if fold_pc == 100 and show_type == 0:
        #     height = 0
        #     for i, tower in enumerate(self.towers):
        #         if h is None:
        #             height += tower.h2
        #         else:
        #             height += h[i]
        #     theta = np.linspace(0, 2 * np.pi, 50)
        #     x = np.cos(theta) * self.towers[0].r
        #     y = np.sin(theta) * self.towers[-1].r 
        #     z = np.zeros_like(theta)
        #     ax.plot(x, y, z, label='parametric curve')
        #     z2 = z + height
        #     ax.plot(x, y, z2, label='parametric curve')

        ax.add_collection3d(poly) # type: ignore
        ax.set_aspect('equal')

        ax.set_title(self.name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore

        ax.azim = camera_pos[0] # type: ignore
        ax.dist = camera_pos[1] # type: ignore
        ax.elev = camera_pos[2] # type: ignore

        plt.show()  