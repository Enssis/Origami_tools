from dataclasses import dataclass
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize


from ..Geometry import *
from ..Utils._types import Number
from ..Utils._svg_utils import hsv_to_hex
from ..Utils import rad2deg, deg2rad, min_search_grad, add_arrow_to_line2D
from .. import get_origami_dir
from ..Pattern import Pattern
from ..LaserParams import ParamList
from .__init__ import general_U_rot, general_U_trac


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
    color : str = "cyan"
    U_raid = [general_U_rot(), general_U_trac()]


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
        self.phi_1 = np.pi/2 + temp_acos
        self.phi_2 = np.pi/2 - temp_acos


        self.patron = None
        self.attache = None
        self.volume = None

    @classmethod
    def from_eta(cls, l, b, n, r, m=1.0, name=""):
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
        xC2 = m**2*r**2 - ((l**2 - b**2) / (2 * a))**2
        if xC2 < 0:
            print("La tour n'a pas de position stable")
            return cls(n, l, b, r, 0, 0, m, a, r_p, 0, name)
        xC = np.sqrt(xC2)
        

        h2 = np.sqrt(r_p**2 -(x0 - xC)**2)

        if (x0 + xC)**2 > r_p**2:
            h1 = 0
            nb_stable = 1
        else :
            h1 = np.sqrt(r_p**2 - (x0 + xC)**2)
            nb_stable = 2

        return cls(n, l, b, r, h1, h2, m, a, r_p, nb_stable, name) 
    
    @classmethod
    def from_n_r_h(cls, n, r, h1, h2, nb_stables=2, m=1.0, name=""):
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

        base = (h2 ** 2 + h1 ** 2) / 2 + (m**2 + 1) * r ** 2
        change = np.tan(np.pi / n) * np.sqrt(4 * m**2* r**4 * np.cos(np.pi / n)**2 - ((h2 ** 2 - h1 ** 2) / 2) ** 2)
        l = np.sqrt(base + change)
        b = np.sqrt(base - change)
        
        if h1 <= 0:
            h1 = 0
            nb_stables = 1

        return cls(n, l, b, r, h1, h2, m, nb_stable=nb_stables, name=name) # type: ignore


    @classmethod
    def from_dh_dphi_n_r(cls, dh, dphi, n, r, m=1.0, name=""):
        """
        Create a Kresling tower from dh, dphi, n, r
        :param dh: difference of height between the two stable positions
        :param dphi: difference of angle between the two stable positions
        :param n: number of folds
        :param r: radius of the folds
        :param name: name of the tower
        :return: Kresling tower
        """

        h2_h1_sq = m* r**2 * np.cos(np.pi/n) * 4 * np.sin(dphi/2) # h2**2 - h1**2
        h1 = (h2_h1_sq - dh**2) / (2 * dh)
        h2 = h1 + dh

        return cls.from_n_r_h(n, r, h1, h2, m=m, name=name) # type: ignore

    @classmethod
    def from_dict(cls, d, name=""):
        """
        Create a Kresling tower from a dictionary
        :param d: dictionary with the parameters of the tower
        :return: Kresling tower
        """
        return cls(d["n"], d["l"], d["b"], d["r"], d["h1"], d["h2"], d["m"], d["a"], d["r_p"], d["nb_stable"], name, d["color"]) # type: ignore


    # representation de l'objet en chaine de caracteres
    def __str__(self):
        return f"{self.name} : n={self.n}, l={self.l}, b={self.b}, r={self.r}, h1={self.h1}, h2={self.h2}, m={self.m}"
    

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
        a = (h**2 - self.b**2 + self.r**2 *(self.m**2 + 1))/(2*self.m*self.r**2)
        if a < -1 or a > 1:
            print("No solution for phi_b, returning 0 or pi for h =", h, self.name, "h1 =", self.h1, "h2 =", self.h2)
            if a < -1:
                return 0
            else:
                return np.pi
        return np.arccos(a) + np.pi/self.n

    # calcul de phi (equation perso)
    def phi(self, h) -> float:
        if self.h1 <= h and self.h2 >= h:
            return self.phi_l(h)
        else:
            return self.phi_b(h)


    def h(self, phi):

        gamma = np.pi / self.n
        if phi <= self.phi_1 and phi >= self.phi_2:
            return np.sqrt(self.l**2 - self.r**2*(self.m**2 + 1 - 2 * self.m * np.cos(phi + gamma)))
        A = self.b**2 - self.r**2*(self.m**2 + 1 - 2 * self.m * np.cos(- phi + gamma))
        if A < 0:
            print("No solution for h, returning max or min height for phi =", rad2deg(phi))
            if phi < self.phi_2:
                return 0
            else:
                return self.b
        return np.sqrt(A)


    # def l_compressed(self, h):
    #     return np.sqrt(h**2 + 2 * self.r**2 * (1 - np.cos(self.phi(h) + np.pi/self.n)))

    # def b_compressed(self, h):
    #     return np.sqrt(h**2 + 2 * self.r**2 * (1 - np.cos(self.phi(h) - np.pi/self.n)))




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
            "m": self.m,
            "a": self.a,
            "r_p": self.r_p,
            "nb_stable": self.nb_stable, 
            "color": self.color
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

    ## TODO : add m    
    def surface(self):
        l = self.a * (self.n - 1) + np.sqrt(self.l**2 - self.r_p**2) 
        return f"{self.r_p} * {l} = {self.r_p * l} mm²"



    ## TODO : add m
    # ------------- calcul des angles des plis montagnes, vallees et rho et des longueurs instantanées des plis -------------- 


    def l_dep(self, phi, h):
        gamma = np.pi / self.n
        return np.sqrt(self.r**2 * (self.m**2 + 1 - 2*self.m*np.cos(gamma + phi)) + h**2)

    def deriv_l_h(self, phi, h):
        return 2 * h / self.l_dep(phi, h)

    
    def b_dep(self, phi, h):
        gamma = np.pi / self.n
        return np.sqrt(self.r**2 * (self.m**2 + 1 - 2*self.m*np.cos(gamma - phi)) + h**2)

    def deriv_b_h(self, phi, h):
        return 2 * h / self.b_dep(phi, h)

    
    def theta_a(self, phi, h):
        """ Angle entre base et panneau ABC"""
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(phi)-np.cos(gamma))

        return np.arccos(-rcos/(np.sqrt(h**2 + rcos**2)))
    
    def deriv_theta_a_h(self, phi, h):
        gamma = np.pi/self.n
        return h * self.m * self.r * np.sin(phi) / (h **2 + self.r**2*(self.m * np.cos(phi)-np.cos(gamma))**2)


    def theta_ah(self, phi, h):
        """ Angle entre haut et panneau BCD"""
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        return np.arccos(rcos/(np.sqrt(h**2 + rcos**2)))
    
    def deriv_theta_ah_h(self, phi, h):
        gamma = np.pi/self.n
        return - h * self.r * np.sin(phi) / (h **2 + self.r**2*(self.m * np.cos(gamma)-np.cos(phi))**2)


    def theta_b(self, phi, h):
        """Angle montagne """
        gamma = np.pi/self.n
        rmgamma = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        rmphi = self.r * (self.m * np.cos(phi)-np.cos(gamma))
        h2 = h**2
        return np.arccos(-(h2*np.cos(phi + gamma) + rmgamma * rmphi)/(np.sqrt((h2 + rmgamma**2)*(h2 + rmphi**2))))

    def deriv_theta_b_h(self, phi, h):
        gamma = np.pi/self.n
        a1 = self.m * np.cos(gamma) - np.cos(phi)
        a2 = self.m * np.cos(phi) - np.cos(gamma)
        cpg = np.cos(phi + gamma)
        spg = np.sin(phi + gamma)
        r = self.r
        numer = 3 * h**4 * cpg + h ** 2 + r ** 2 * ((a1 ** 2 + a2 ** 2) * cpg + 4 * a1 * a2) + a1 * a2 * cpg
        denom = (h**2 + a1**2 * r ** 2) * (h**2 + a2**2 * r ** 2) * np.sqrt(h**2 * spg**2 + r**2*(a1**2 + a2**2 - 2 * a1 * a2 * cpg))
        return -2  * numer / denom
    

    def theta_l(self, phi, h):
        """Angle valley """
        gamma = np.pi/self.n
        rmgamma = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        rmphi = self.r * (self.m * np.cos(phi)-np.cos(gamma))
        h2 = h**2
        return np.arccos(-(h2*np.cos(phi - gamma) + rmgamma * rmphi)/(np.sqrt((h2 + rmgamma**2)*(h2 + rmphi**2))))

    def deriv_theta_l_h(self, phi, h):
        gamma = np.pi/self.n
        a1 = self.m * np.cos(gamma) - np.cos(phi)
        a2 = self.m * np.cos(phi) - np.cos(gamma)
        cpg = np.cos(phi - gamma)
        spg = np.sin(phi - gamma)
        r = self.r
        numer = 3 * h**4 * cpg + h ** 2 + r ** 2 * ((a1 ** 2 + a2 ** 2) * cpg + 4 * a1 * a2) + a1 * a2 * cpg
        denom = (h**2 + a1**2 * r ** 2) * (h**2 + a2**2 * r ** 2) * np.sqrt(h**2 * spg**2 + r**2*(a1**2 + a2**2 - 2 * a1 * a2 * cpg))
        return -2  * numer / denom

    #----------- Energy and forces --------------

    # def U_tot(tdk, phi, h, U_t, U_r):
    #     dl = tdk.l_dep(phi, h) - tdk.l
    #     db = tdk.b_dep(phi, h) - tdk.b
    #     dtheta_a = tdk.theta_a(phi, h) - np.pi/2
    #     dtheta_ah = tdk.theta_ah(phi, h) - np.pi/2
    #     dtheta_b = tdk.theta_b(phi, h) - np.pi
    #     dtheta_l = tdk.theta_l(phi, h) - np.pi
    #     n = tdk.n
    #     return n * (U_t(dl) / tdk.l + U_t(db) / tdk.b + U_r(dtheta_a) * tdk.a + U_r(dtheta_ah) * tdk.a * tdk.m + U_r(dtheta_b) * tdk.b + U_r(dtheta_l) * tdk.l)

    def U(self, phi, h):
        dl = self.l_dep(phi, h) - self.l
        db = self.b_dep(phi, h) - self.b
        dtheta_a = self.theta_a(phi, h) - np.pi/2
        dtheta_ah = self.theta_ah(phi, h) - np.pi/2
        dtheta_b = self.theta_b(phi, h) - np.pi
        dtheta_l = self.theta_l(phi, h) - np.pi
        n = self.n
        # print(f"dl : {dl}, db : {db}, dtheta_a : {dtheta_a}, dtheta_ah : {dtheta_ah}, dtheta_b : {dtheta_b}, dtheta_l : {dtheta_l}")
        U_r = self.U_raid[0]
        U_t = self.U_raid[1]

        return n * (U_t(dl) / self.l + U_t(db) / self.b + U_r(dtheta_a) * self.a + U_r(dtheta_ah) * self.a * self.m + U_r(dtheta_b) * self.b + U_r(dtheta_l) * self.l)

    def set_U_raid(self, U_rot, U_trac):
        self.U_raid = [U_rot, U_trac]















    ## TODO : add m
    def angle_pli_montagne(self, h):
        pin = np.pi / self.n
        n_N = np.array([self.a * h, 0, -self.a * self.r * (np.cos(self.phi(h)) - np.cos(pin))])

        n_P = 2 * self.r * np.sin(pin) * np.array([-h * np.cos(self.phi(h) + pin), -h * np.sin(self.phi(h) + pin), self.r * (np.cos(pin) - np.cos(self.phi(h)))])

        return np.arccos(np.dot(n_N, n_P) / (np.linalg.norm(n_N) * np.linalg.norm(n_P)))
    

    ## TODO : add m
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

    ## TODO : add m
    def ang_mountain_phi(self, phi):
        return self.ang_mountain_phi_stat(phi, self.n, self.r, self.l, self.b)
    
    ## TODO : add m    
    def angle_pli_vallee(self, h):
        pin = np.pi / self.n

        n_P = 2 * self.r * np.sin(pin) * np.array([-h * np.cos(self.phi(h) + pin), -h * np.sin(self.phi(h) + pin), self.r * (np.cos(pin) - np.cos(self.phi(h)))])

        n_Q = np.array([- self.r * h * (np.sin(3 * pin) - np.sin(pin)), self.r * h * (np.cos(3 * pin) - np.cos(pin)), self.r ** 2 * (2 * np.sin(pin) * np.cos(self.phi(h)) - np.sin(2 * pin))])
        
        return np.pi - np.arccos(np.dot(n_P, n_Q) / (np.linalg.norm(n_Q) * np.linalg.norm(n_P)))

    ## TODO : add m
    def angle_pli_rho(self, h):
        pin = np.pi / self.n

        n_N = np.array([self.a * h, 0, -self.a * self.r * (np.cos(self.phi(h)) - np.cos(pin))])

        n_B = np.array([0, 0, 1])
        return np.arccos(np.dot(n_N, n_B) / (np.linalg.norm(n_N) * np.linalg.norm(n_B)))

    ## TODO : add m
    def delta_b(self, phi):
        return np.sqrt(self.l**2 - 4 * self.r**2 * np.sin(np.pi/self.n) * np.sin(phi)) - self.b

    ## TODO : add m
    def max_delta_b(self):
        return np.sqrt(self.l**2 - 4 * self.r**2 * np.sin(np.pi/self.n)) - self.b

    ## TODO : add m   
    def delta_phi(self):
        return 2 * np.arcsin((self.h2**2 - self.h1**2) / (4 * self.r**2 * np.cos(np.pi/self.n)))

    ### =============== Creation du patron =============== ###
    ## TODO : add m
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

    ## TODO : add m   
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
        if np.isnan(angle_m) :
            angle_m = 0
        if np.isnan(angle_v) :
            angle_v = 0

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

    ## TODO : add m
    def create_3D(self, h_rep : Number | None = None, phi_rep : Number | None = None, type : int = 0, fold_pc : Number = 100, ep_tot : Number = 0, side : int=0, h_constr : Number | None = None, e_pp = 0.5, rotation : Number = 0, chiral : bool = False):
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
        # ------ Value initialisation -----
        if h_rep is None:
            h_rep = self.h1
        if h_constr is None:
            h_constr = h_rep

        gamma = np.pi / self.n
        if phi_rep is None:
            phi = self.phi(h_rep)
        else:
            phi = phi_rep
        x0 = self.r * np.cos(gamma)



        # ----- Calculates vertices of firsts panels ABC and BCD  -----
        A = Point(x0, - self.r * np.sin(gamma), 0)
        B = Point(x0, self.r * np.sin(gamma), 0)
        if type == 1:
            rho = self.angle_pli_rho(h_rep)
            C = Point(x0 - self.r_p * np.cos(rho), self.a / 2 + np.sqrt(self.b ** 2 - self.r_p **2), self.r_p * np.sin(rho))
        else :
            C = Point(self.r * np.cos(phi) * self.m, self.r * np.sin(phi) * self.m, h_rep) 
        D = C.rotate(2 * gamma, Point(0, 0, 0), Vec(0, 0, 1))

        # rotation matrix for starting position

        verts = [Polygon([A,B,C]), Polygon([B,C,D])]

        # Calculation with angles between panels
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
                pol = Polygon([A, B, C, A]).offset([-d_val, -d_mont, 0])
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

            U2 = A.copy().transform(mat_up_rot)
            V2 = B.copy().transform(mat_up_rot)
            W2 = C.copy().transform(mat_up_rot)

            for _ in range(self.n):
                A = A.transform(mat_rot)
                B = B.transform(mat_rot)
                C = C.transform(mat_rot)
                verts.append(Polygon([A, B, C]))
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

    ## TODO : add m
    def create_volume(self, h_rep : Number | None = None, phi_rep : Number | None = None, constr_type : int =0, fold_pc : Number =100, ep : Number =0, decal : Number=0, ep_tot : Number =0, side : int =0, h_constr : Number | None = None, rotation : Number = 0, chiral : bool = False):
        
        if h_rep is None:
            h_rep = self.h1
        # 
        if h_constr is None:
            h_constr = h_rep
        
        if phi_rep is None:
            phi_rep = self.phi(h_rep)

        # print(f"Creating volume with h_rep : {h_rep}, phi_rep : {phi_rep}, constr_type : {constr_type}, fold_pc : {fold_pc}, ep : {ep}, decal : {decal}, ep_tot : {ep_tot}, side : {side}, h_constr : {h_constr}, flat : {flat}, rotation : {rotation}, chiral : {chiral}")

        verts = self.create_3D(h_rep, phi_rep, constr_type, fold_pc, ep_tot - ep, side, h_constr, ep, rotation, chiral)

        # Add thickness to the volume
        if ep == 0:
            volume = Volume(verts)
        else:
            volume = verts[0].copy().extrude(ep)
            if decal != 0:
                normal = verts[0].normal_vect()
                normal.z = 0
                volume.translate(normal * decal)
            for i in range(1, len(verts)):
                plaque_volume = verts[i].copy().extrude(ep)
                if decal != 0:
                    normal = verts[i].normal_vect()
                    normal.z = 0
                    plaque_volume.translate(normal * decal)
                volume = volume + plaque_volume
        return volume


    def show_3D(self, h : Number | None = None, phi : Number | None = None, constr_type=0, fold_pc=100, ep=0.0, show_circles=False, recreate = True, camera_pos= [45, 0, 25], rotation : Number= 0, translation : Number = 0, chiral : bool = False, ax = None, show = True):
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
            self.volume = self.create_volume(h, phi, constr_type, fold_pc, ep, rotation=rotation, chiral=chiral)
        
        if translation != 0:
            self.volume.translate(Vec(0, 0, translation))

        volume_mesh = self.volume.mesh_3D()
        if ax is None:
            ax = plt.figure().add_subplot(111, projection='3d')
            scale = volume_mesh.points.flatten() # type: ignore
            ax.auto_scale_xyz(scale, scale, scale) # type: ignore

        poly = Poly3DCollection(volume_mesh.vectors, alpha=0.5) # type: ignore
        poly.set_edgecolor('0')
        poly.set_facecolor(self.color)
        
        # optionally show the circles of the top and bottom faces
        if fold_pc == 100 and show_circles:
            theta = np.linspace(0, 2 * np.pi, 50)
            x_d = np.cos(theta) * self.r
            y_d = np.sin(theta) * self.r 
            z_d = np.zeros_like(theta)
            ax.plot(x_d, y_d, z_d, label='Down circle')
            z_up = z_d + h
            x_up = x_d * self.m
            y_up = y_d * self.m
            ax.plot(x_up, y_up, z_up, label='Up circle')


        ax.add_collection3d(poly) # type: ignore
        ax.set_aspect('equal')

        ax.set_title(self.name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore

        ax.azim = camera_pos[0] # type: ignore
        ax.dist = camera_pos[1] # type: ignore
        ax.elev = camera_pos[2] # type: ignore

        if show:
            plt.show()
        else :
            return ax


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











## TODO : add m and inversion haut/bas        
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
        self.volumes = None

    def __str__(self) -> str:
        return f"MultiStoriesKresling with {len(self.towers)} towers, chiralities : {self.chiralities}, name : {self.name}. \n Towers : \n" + "\n".join([str(tower) for tower in self.towers])
    
    def __repr__(self) -> str:
        return self.__str__()

    def as_dict(self):
        return {
            "name": self.name,
            "towers": [tower.as_dict() for tower in self.towers],
            "chiralities": self.chiralities
        }

    @staticmethod
    def from_dict(d):
        towers = [TDK.from_dict(tower_d) for tower_d in d["towers"]]
        chiralities = d["chiralities"]
        return MultiStoriesKresling(towers, chiralities, d["name"])

    # sauvegarde la multi tour dans un fichier
    def save(self, path = "", overwrite = False):
        if path == "":
            path = ORIGAMI_DIR_PATH + "saved_multi_towers.json"
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

    @staticmethod
    def load(name, path = ""):
        if path == "":
            path = ORIGAMI_DIR_PATH + "saved_multi_towers.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le path {path} n'existe pas")
        if not path.endswith(".json"):
            raise ValueError(f"Le path {path} n'est pas un fichier json")

        with open(path, "r") as f:
            towers = json.load(f)
            if name not in towers.keys():
                raise ValueError(f"Le nom {name} n'existe pas dans le fichier {path}")
            return MultiStoriesKresling.from_dict(towers[name])

    def create_volumes(self, h_rep : list[Number] | None = None, phi_rep : list[Number] | None = None, constr_type : int =0, fold_pc : Number =100, ep : Number =0, decal : Number=0, ep_tot : Number =0, side : int =0, h_constr : Number | None = None, base_rotation : Number = 0):
        volumes = []
        if h_rep is not None and len(h_rep) != len(self.towers):
            raise ValueError("The number of heights must be equal to the number of towers or 1")
        
        rotation = base_rotation
        height = 0

        for i, tower in enumerate(self.towers):
            if h_rep is None:
                h = tower.h2
            else:
                h = h_rep[i]
            if phi_rep is None or len(phi_rep) != len(self.towers):
                phi = tower.phi(h)
            else:
                phi = phi_rep[i]

            tower_volume = tower.create_volume(h, phi, constr_type, fold_pc, ep, decal, ep_tot, side, h_constr, rotation, self.chiralities[i])
            sens = -1 if self.chiralities[i] else 1
            rotation += (phi + np.pi / tower.n) * sens
            tower_volume.translate(Vec(0, 0, height))
            height += h
            volumes.append(tower_volume)

        self.volumes = volumes    
        return volumes
    
    def show_3D(self, h : list[Number] | None = None, phi : list[Number] | None = None, constr_type=0, fold_pc=100, ep=0, show_type=0, recreate = True, camera_pos= [45, 0, 25], base_rotation : Number = 0, height_repr = None):
        """
        type : 
            O => les plis ne sont pas des bonnes longuers (utilisation de phi)
            1 => le panneau n'est pas déformé (utilisation de rho)
            2 => le panneau est déformé comme 1 (utilisation des plis vallés / montagnes) 
            3 => l et b ne sont pas reliés au sommet V (TODO)

        fold_pc : pourcentage de pliage (pour le type 2)        
        """

        if self.volumes is None or recreate:
            self.volumes = self.create_volumes(h, phi, constr_type, fold_pc, ep, base_rotation=base_rotation) 
        volumes_mesh = [volume.mesh_3D() for volume in self.volumes]


        ax = plt.figure().add_subplot(111, projection='3d')
        polys = [Poly3DCollection(volume_mesh.vectors, alpha=0.5) for volume_mesh in volumes_mesh] # type: ignore
        
        # Auto scale to the mesh size
        scale = volumes_mesh[-1].points.flatten() # type: ignore
        ax.auto_scale_xyz(scale, scale, scale) # type: ignore
        # height rep, set z axis of the graph to it
        if height_repr is not None:
            ax.set_zlim(0, height_repr)
        else :
            if h is None:
                h = [tower.h2 for tower in self.towers]
            ax.set_zlim(0, sum(h) + 5)

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

        for i, poly in enumerate(polys):
            poly.set_edgecolor('0')
            poly.set_facecolor(self.towers[i].color)
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

    def phis_config(self, pos : list[int]):
        """pos : list of booleans of size n, 1 for high position, 0 for low position"""
        if len(pos) != len(self.towers):
            raise ValueError("The number of positions must be equal to the number of towers")
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        return [tower.phi_2 if pos[i] else tower.phi_1 for i, tower in enumerate(self.towers)] * sens

    def phis_full_extended(self):
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        return [tower.phi_2 for tower in self.towers] * sens

    def phis_full_contracted(self):
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        return [tower.phi_1 for tower in self.towers] * sens

    def min_phi_stable(self):
        phi = 0
        for i, tower in enumerate(self.towers):
            if self.chiralities[i]:
                phi -= tower.phi_1
            else:
                phi += tower.phi_2  
        return phi
    
    def max_phi_stable(self):
        phi = 0
        for i, tower in enumerate(self.towers):
            if self.chiralities[i]:
                phi -= tower.phi_2
            else:
                phi += tower.phi_1  
        return phi
    
    def max_height_stable(self):
        return sum([tower.h2 for tower in self.towers])

    def min_height_stable(self):
        return sum([tower.h1 for tower in self.towers])


    def U(self, phi : list[Number], h : list[Number]):
        if len(phi) != len(self.towers) or len(h) != len(self.towers):
            raise ValueError("The number of towers must be equal to the number of phi, h")
        U_tot = 0
        for i in range(len(self.towers)):
            tower = self.towers[i]
            # print(f"phi[{i}] = {rad2deg(phi[i])}, h[{i}] = {h[i]}")
            U_tot += tower.U(phi[i], h[i])
        return U_tot

    def U_simp_phi(self, phi_tot, x):
        # print(phi_tot, x)
        phi0 = abs(phi_tot - np.sum(x)) % (2*np.pi)
        # print("phi0 :", [phi0], "x :", x)
        phis = [phi0] 
        phis.extend(np.array(x))
        phis = [abs(phi) % (2 * np.pi) for phi in phis]
        # print("phis :", phis)
        hs = [tower.h(abs(phi)) for tower, phi in zip(self.towers, phis)]

        # print("phis :", phis, "hs :", hs)
        return self.U(phis, hs)
    
    def U_simp_h(self, h_tot, x):
        # print(phi_tot, x)
        h0 = abs(h_tot - np.sum(x))
        # print("phi0 :", [phi0], "x :", x)
        hs = [h0] 
        hs.extend(np.array(x))
        # print("phis :", phis)
        phis = [tower.phi(h) for tower, h in zip(self.towers, hs)]

        # print("phis :", phis, "hs :", hs)
        return self.U(phis, hs)

    def U_phi_tot(self, x, phi_tot):
        """"x : list of phi then h without the first phi which is deduced from phi_tot and the other phis"""
        hs = [h for h in x[len(self.towers)-1:]]
        phis = [phi if phi > 0 else -phi for phi in x[:len(self.towers)-1]]
        phis = [abs(phi_tot - np.sum(phis)) % (2*np.pi)] + phis
        return self.U(phis, hs)

    def polygons_from_pos(self, pos):
        volumes = self.create_volumes(pos[1], pos[0])
        volumes_mesh = [volume.mesh_3D() for volume in volumes]
        return [volume_mesh.vectors for volume_mesh in volumes_mesh]


    def show_tower_movement(self, x_ax, curve, segments : list[list[Number]] | None = None, start_ind=0, save=False, path="", name="", graph_intermediates=False, graph_3D=True, animated=True, rotation = True):
        """curve : list of (phis, hs) for each point of the curve"""

        if len(curve) != len(x_ax):
            raise ValueError("The number of points in the curve must be equal to the number of points in the x axis")

        # sens for the movement of the towers, if chiral, the movement is inverted
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])

        # initiate the number of graphs to displays
        ncol = 1 + int(graph_intermediates) + int(graph_3D)

        fig = plt.figure(figsize=(16, 10), dpi=100)

        # -------------------- Graph with the 3D representation of the tower -------------------------
        if graph_3D:
            ax_3D = fig.add_subplot(1, ncol, 1, projection="3d")

            # set the camera pos
            camera_pos = [45, 0, 25]
            ax_3D.azim = camera_pos[0]
            ax_3D.dist = camera_pos[1] # type: ignore
            ax_3D.elev = camera_pos[2]

            # Setting the Axes properties
            ax_3D.set_xlabel('X')
            ax_3D.set_ylabel('Y')
            ax_3D.set_zlabel('Z')
            
            ax_3D.set(xlim3d=(-self.towers[0].r, self.towers[0].r), xlabel='X')
            ax_3D.set(ylim3d=(-self.towers[0].r, self.towers[0].r), ylabel='Y')
            ax_3D.set(zlim3d=(0, self.max_height_stable() * 1.1), zlabel='Z')
            ax_3D.set_xticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))
            ax_3D.set_yticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))
            ax_3D.set_aspect('equal')

        # Store polygon collection references
        poly_collections = []


        # ------------------  Graphs with energy and total height as a function of phi -------------------

        energy_ax = fig.add_subplot(2, ncol, int(graph_3D) + 1)

        kinematic_ax = fig.add_subplot(2, ncol, int(graph_3D) + 1 + ncol)

        if rotation:
            energy_ax.set_xlabel("phi (degrees)")
            energy_ax.set_ylabel("U(phi, h(phi))")
            
            kinematic_ax.set_xlabel("phi (degrees)")
            kinematic_ax.set_ylabel("h total(mm)")
        else:
            energy_ax.set_xlabel("h total (mm)")
            energy_ax.set_ylabel("U(phi(h), h)")
            
            kinematic_ax.set_xlabel("h (mm)")
            kinematic_ax.set_ylabel("phi total (deg)")

        x_ax = rad2deg(np.array(x_ax)) if rotation else np.array(x_ax)
        
        ordened_curve = np.concatenate((curve[-start_ind:], curve[:-start_ind])) if start_ind != 0 else np.array(curve)
        # change colors for segments
        if segments is not None:
            ind_s = 0
            colors = [hsv_to_hex(i / len(segments), 1, 1) for i in range(len(segments))]

            for i in range(len(segments)):
                x = rad2deg(np.array(segments[i])) if rotation else np.array(segments[i])
                label = "energy segment [" + str(int(rad2deg(segments[i][0]))) + "° ; " + str(int(rad2deg(segments[i][-1]))) + "°]" if rotation else "energy segment [" + str(int(segments[i][0])) + " ; " + str(int(segments[i][-1])) + "]"
                
                energy_line = energy_ax.plot(x, [self.U(curve[(ind_s + j - start_ind) % len(curve)][0] * sens, curve[(ind_s + j - start_ind) % len(curve)][1]) for j in range(len(segments[i]))], color=colors[i],label=label)
                
                y = [sum(np.array(curve[(ind_s + j - start_ind) % len(curve)][1])) for j in range(len(segments[i]))] if rotation else [rad2deg(sum(np.array(curve[(ind_s + j - start_ind) % len(curve)][0]))) for j in range(len(segments[i]))]
                kinematic_ax.plot(x, y, color=colors[i],label="height segment [" + str(int(rad2deg(segments[i][0]))) + "° ; " + str(int(rad2deg(segments[i][-1]))) + "°]")

                add_arrow_to_line2D(energy_ax, energy_line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(x_ax))), arrowsize=1.5)
                ind_s += len(segments[i])
                # print(ind_s)

        else:
            line = energy_ax.plot(x_ax, [self.U(ordened_curve[i][0] * sens, ordened_curve[i][1]) for i in range(len(x_ax))], "b", label="energy")
            add_arrow_to_line2D(energy_ax, line, arrow_locs=[0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 0.91],arrowsize=1.5)
            y = [sum(c[1]) for c in ordened_curve] if rotation else [rad2deg(sum(c[0])) for c in ordened_curve]
            kinematic_ax.plot(x_ax, [sum(c[1]) for c in ordened_curve], "b",label="height")


        point_energy = energy_ax.plot([], [], "ro", label="current position")[0]
        point_kinematic = kinematic_ax.plot([], [], "ro", label="current position")[0]

        energy_ax.legend()




        # --------------------- Graphs with phi and h of each tower as a function of x_tot -------------------------

        if graph_intermediates:
            kinematic_phi_ax = fig.add_subplot(2, 3, 3)
            kinematic_phi_ax.set_ylabel("phi (degrees)")

            kinematic_h_ax = fig.add_subplot(2, 3, 6)
            kinematic_h_ax.set_ylabel("h (mm)")

            if rotation:
                kinematic_phi_ax.set_xlabel("phi total (degrees)")
                kinematic_h_ax.set_xlabel("phi total (degrees)")
            else:
                kinematic_phi_ax.set_xlabel("h total (mm)")
                kinematic_h_ax.set_xlabel("h total (mm)")

            for i in range(len(curve[0][0])):
                kinematic_phi_ax.plot(x_ax, [rad2deg(c[0][i]) for c in ordened_curve], label="phi " + str(i+1))
                kinematic_h_ax.plot(x_ax, [c[1][i] for c in ordened_curve], label="h " + str(i+1))
            
            points_phi = kinematic_phi_ax.plot([], [], "ro", label="current position")[0]
            points_h = kinematic_h_ax.plot([], [], "ro", label="current position")[0]

            kinematic_phi_ax.legend()
            kinematic_h_ax.legend()
                

        # -------------------- Animation function -------------------------
        def animate(frame):
            #  ======== 3D animation =========
            if graph_3D: 
                # Clear previous collections
                for poly in poly_collections:
                    poly.remove()
                poly_collections.clear()
            
                # Get new polygon data for current frame
                current_pos = ordened_curve[frame]  # Assuming 0 for second parameter
                polygon_data = self.polygons_from_pos(current_pos)
            
                # Create and add new polygons
                for i, polygon in enumerate(polygon_data):
                    poly = Poly3DCollection(polygon, alpha=0.5, linewidths=1, edgecolors='k')
                    poly.set_edgecolor('0')
                    poly.set_facecolor(self.towers[i].color)
                    ax_3D.add_collection3d(poly) # type: ignore
                    poly_collections.append(poly)
            

            # ========= 2D animation of the energy and height graphs =========

            # Update the point position in the energy graph
            point_energy.set_data([x_ax[frame]], [self.U(np.array(ordened_curve[frame][0]* sens), ordened_curve[frame][1])])  # type: ignore
            point_kinematic.set_data([x_ax[frame]], [sum(ordened_curve[frame][1] if rotation else rad2deg(ordened_curve[frame][0]))])  

            points = [point_energy, point_kinematic]

            # ======== 2D animation of the phi and h graphs for each tower ========

            if graph_intermediates:
                points_phi.set_data([x_ax[frame]] * len(self.towers), [rad2deg(ordened_curve[frame][0])]) # type: ignore
                points_h.set_data([x_ax[frame]] * len(self.towers), [ordened_curve[frame][1]]) # type: ignore
                points += [points_phi, points_h] # type: ignore

            return poly_collections + points

        

        # if abs(x_ax[0] - x_ax[-1]) > deg2rad(0.1) and rotation or abs(x_ax[0] - x_ax[-1]) > 0.5 and not rotation:
        #     frames = [ i if i < len(curve) else len(curve) - (i % len(curve) + 1) for i in range(len(curve) * 2)]
        # else:
        frames = range(len(curve))
        # Create animation
        if animated :
            anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=True)
        
        if save:
            if name == "":
                name = self.name
            if path == "":
                path = ORIGAMI_DIR_PATH + "/animations/"
            if animated:
                anim.save(path + f"animation_{name}.gif", writer='pillow', fps=20) # type: ignore
                print(f"Saved animation to {path + f'animation_{name}.gif'}")
            else :
                plt.savefig(path + f"animation_{name}.svg") # type: ignore
                print(f"Saved graph to {path + f'animation_{name}.svg'}")
        else:
            plt.show()



    def movement3D_rot(self, phi, start_pos : list[Number] | None = None):
        # phi = np.linspace(self.min_phi_stable() - deg2rad(10), self.max_phi_stable() + deg2rad(10), 100)
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        # chiral_plus = sum(sens)
        # start = [tower.phi_1 if chiral else tower.phi_2 for tower, chiral in zip(self.towers, self.chiralities)]
        
        if start_pos is None:
            start_pos = [tower.phi_2 for tower in self.towers] * sens
        start = start_pos[1:]

        phi_0 = sum(start_pos)
        diff = abs(phi[0] - phi[1])
        ind = 0
        for i, p in enumerate(phi):
            if abs(p - phi_0) < diff:
                ind = i
                break 
        else :
            raise ValueError("phi_0 must be in the range of phi")
        phi = np.concatenate((phi[ind:], phi[:ind]))


        # print(start)

        curve = []
        for phi_i in phi:
            # print("phi =", rad2deg(phi_i))
            # phi_t = minimize(lambda x : self.U_simp_phi(phi_i, x), start, bounds=[(0, 2*np.pi)] * (len(self.towers) - 1)).x # graph_x_deriv=False, anim_graph=False, graph_limits=[0, np.pi]) 
            phi_t = min_search_grad(start, lambda x : self.U_simp_phi(phi_i, x)) # graph_x_deriv=False, anim_graph=False, graph_limits=[0, np.pi]) 
            phi_t = [abs(phi_) % (2 * np.pi ) for phi_ in phi_t]
            start = phi_t * sens[1:]
            phi0 = abs(phi_i - np.sum(start)) % (2*np.pi)
            phis = [phi0] + phi_t
            # print("phis :", [rad2deg(phi) for phi in phis], 'phi_i :', rad2deg(phi_i), 'start :', [rad2deg(s) for s in start])
            hs = [tower.h(abs(phi)) for tower, phi in zip(self.towers, phis)]
            curve.append([phis, hs])

        
        return curve, ind


    def movement3D_trans(self, h, start_pos : list[Number] | None = None):
        # phi = np.linspace(self.min_phi_stable() - deg2rad(10), self.max_phi_stable() + deg2rad(10), 100)
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        # chiral_plus = sum(sens)
        # start = [tower.phi_1 if chiral else tower.phi_2 for tower, chiral in zip(self.towers, self.chiralities)]
        
        if start_pos is None:
            start_pos = [tower.h2 for tower in self.towers]
        start = start_pos[1:]

        h_0 = sum(start_pos)
        diff = abs(h[0] - h[1])
        ind = 0
        for i, p in enumerate(h):
            if abs(p - h_0) < diff:
                ind = i
                break 
        else :
            raise ValueError(f"h_0 : {h_0} must be in the range of h")
        h = np.concatenate((h[ind:], h[:ind]))


        # print(start)

        curve = []
        for h_i in h:
            # print("phi =", rad2deg(phi_i))
            # phi_t = minimize(lambda x : self.U_simp_phi(phi_i, x), start, bounds=[(0, 2*np.pi)] * (len(self.towers) - 1)).x # graph_x_deriv=False, anim_graph=False, graph_limits=[0, np.pi]) 
            # start = min_search_grad(start, lambda x : self.U_simp_h(h_i, x)) # graph_x_deriv=False, anim_graph=False, graph_limits=[0, np.pi]) 
            start = minimize(lambda x : self.U_simp_h(h_i, x), start, bounds=[(0, tower.b) for tower in self.towers[1:]]).x 
            h0 = abs(h_i - np.sum(start))
            hs =  np.concatenate(([h0], start))
            # print("hs :", hs, 'h_i :', h_i, 'start :', start, h0)
            # print("phis :", [rad2deg(phi) for phi in phis], 'phi_i :', rad2deg(phi_i), 'start :', [rad2deg(s) for s in start])
            phis = [tower.phi(h) for tower, h in zip(self.towers, hs)] * sens
            curve.append([phis, hs])

        
        return curve, ind



    def movement3D_rot_full(self, phi, start_pos : list[Number] = [], segments : list[list[Number]] | None = None,save=False, path="", name=""):
        """start_pos : list of phi followed by list of h, for the starting position of the animation, if None, it will be the full extended position."""
        # phi = np.linspace(self.min_phi_stable() - deg2rad(10), self.max_phi_stable() + deg2rad(10), 100)
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        # chiral_plus = sum(sens)
        # start = [tower.phi_1 if chiral else tower.phi_2 for tower, chiral in zip(self.towers, self.chiralities)]
        
        if start_pos == []:
            start_pos = [tower.phi_2 for tower in self.towers] * sens 
            start_pos = np.concatenate((start_pos, [tower.h2 for tower in self.towers])) #type: ignore
        start = start_pos[1:]
        phi_0 = sum(start_pos[:len(self.towers)])

        diff = abs(phi[0] - phi[1])
        ind = 0
        for i, p in enumerate(phi):
            if abs(p - phi_0) < diff:
                ind = i
                break 
        else :
            raise ValueError("phi_0 must be in the range of phi")
        phi = np.concatenate((phi[ind:], phi[:ind]))


        # print(start)

        curve = []

        bounds = [(0, np.pi)] * (len(self.towers) - 1) + [(0, tower.h2) for tower in self.towers]

        for phi_i in phi:
            # find phi and h values that minimize U for a given total rotation phi_i 
            minimized_values = minimize(self.U_phi_tot, start, bounds=bounds, args=(phi_i,)).x # graph_x_deriv=False, anim_graph=False, graph_limits=[0, np.pi]) 
            
            #get the finded h and phi values
            hs = minimized_values[len(self.towers)-1:]
            phis = minimized_values[:len(self.towers)-1]

            # Add the first value of phi to have the full list of phi values, deduced from phi_i and the other phis
            phis = np.concatenate(([abs(phi_i - np.sum(phis)) % (2*np.pi)], phis)) * sens

            curve.append([phis, hs])

        
        print(curve)

        plt.plot(phi, [sum(c[1]) for c in curve], label="height")

        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(1, 2, 1, projection="3d")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        camera_pos = [45, 0, 25]
        ax.azim = camera_pos[0]
        ax.dist = camera_pos[1] # type: ignore
        ax.elev = camera_pos[2]

        # Setting the Axes properties
        ax.set(xlim3d=(-self.towers[0].r, self.towers[0].r), xlabel='X')
        ax.set(ylim3d=(-self.towers[0].r, self.towers[0].r), ylabel='Y')
        ax.set(zlim3d=(0, self.max_height_stable() + 5), zlabel='Z')
        ax.set_xticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))
        ax.set_yticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))

        def polygons_from_pos(pos):
            volumes = self.create_volumes(pos[1], pos[0])
            volumes_mesh = [volume.mesh_3D() for volume in volumes]
            return [volume_mesh.vectors for volume_mesh in volumes_mesh]

        # Store polygon collection references
        poly_collections = []

        ax2 = fig.add_subplot(1, 2, 2)

        
        if segments is not None:
            ind_s = 0
            colors = [hsv_to_hex(i / len(segments), 1, 1) for i in range(len(segments))]
            # print(colors)
            for i in range(len(segments)):
                # print(f"Segment {i + 1} : phi in [{rad2deg(segments[i][0])}, {rad2deg(segments[i][-1])}]")
                line = ax2.plot(rad2deg(np.array(segments[i])), [self.U_phi_tot(phi_i, np.array(curve[(ind_s + j - ind) % len(curve)][0][1:]) * sens[1:]) for j, phi_i in enumerate(segments[i])], color=colors[i],label="energy segment " + str(i + 1))
                add_arrow_to_line2D(ax2, line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(phi))), arrowsize=1.5)
                ind_s += len(segments[i])
                # print(ind_s)

        else:
            line = ax2.plot(rad2deg(phi), [self.U(curve[i][0] * sens, curve[i][1]) for i in range(len(phi))], "b", label="energy")
            add_arrow_to_line2D(ax2, line, arrow_locs=[0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 0.91],arrowsize=1.5)



        # point = ax2.plot([rad2deg(phi[0])], [self.U_simp_phi(phi[0], np.array(curve[0][0][1:])) * sens[1:]], "ro", label="current position")[0]
        point = ax2.plot([], [], "ro", label="current position")[0]


        def animate(frame):
            # Clear previous collections
            for poly in poly_collections:
                poly.remove()
            poly_collections.clear()
            
            # Get new polygon data for current frame
            current_pos = curve[frame]  # Assuming 0 for second parameter
            polygon_data = polygons_from_pos(current_pos)
            
            # Create and add new polygons
            for i, polygon in enumerate(polygon_data):
                poly = Poly3DCollection(polygon, alpha=0.5, linewidths=1, edgecolors='k')
                poly.set_edgecolor('0')
                poly.set_facecolor(self.towers[i].color)
                ax.add_collection3d(poly)
                poly_collections.append(poly)
            

            # Update the point position in the energy graph
            point.set_data([rad2deg(phi[frame])], [self.U_simp_phi(phi[frame], np.array(curve[frame][0][1:]) * sens[1:]) ])

            return poly_collections + [point]

        ax.set_aspect('equal')
        if abs(phi[0] - phi[-1]) > deg2rad(10):
            frames = [ i if i < len(curve) else len(curve) - (i % len(curve) + 1) for i in range(len(curve) * 2)]
        else:
            frames = range(len(curve))
        # Create animation
        anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=True)


        ax2.legend()
        ax2.set_xlabel("phi (degrees)")
        ax2.set_ylabel("U(phi, h(phi))")


        if save:
            if name == "":
                name = self.name
            if path == "":
                path = ORIGAMI_DIR_PATH + "/animations/"
            anim.save(path + f"{name}_rot_animation.gif", writer='pillow', fps=20)
            print(f"Saved animation to {path + f'{name}_rot_animation.gif'}")
        else:
            plt.show()