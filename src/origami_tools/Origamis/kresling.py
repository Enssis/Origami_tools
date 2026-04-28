from dataclasses import dataclass
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from scipy.optimize import minimize, minimize_scalar 

from ..Geometry import *
from ..Utils._types import Number
from ..Utils._svg_utils import hsv_to_hex
from ..Utils import min_search, rad2deg, deg2rad, min_search_grad, add_arrow_to_line2D, hex_to_hsv, rgb_vals_to_hex
from .. import get_origami_dir
from ..Pattern import Pattern, base_flap_pattern
from ..LaserParams import ParamList
from .__init__ import general_U_rot, general_U_trac, general_dU_trac, general_dU_rot


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
    k_comp : Number = 1
    k_tor : Number = 0.0000178 * 0.6
    U_raid = [general_U_rot, general_U_trac]
    dU_raid = [general_dU_rot, general_dU_trac]

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

        if self.color == "cyan":
            self.color = rgb_vals_to_hex((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        temp_acos = np.arccos((self.l**2 - self.b**2)/(4*self.m*self.r**2*np.sin(np.pi/self.n)))
        self.phi_1 = np.pi/2 + temp_acos
        self.phi_2 = np.pi/2 - temp_acos

        self.verbose = False
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
        return cls(d["n"], d["l"], d["b"], d["r"], d["h1"], d["h2"], d["m"], d["a"], d["r_p"], d["nb_stable"], name, d["color"]) 


    @classmethod
    def random(cls, range_n=(3, 20), range_dh=(0, 20), range_dphi=(np.pi/2, 3*np.pi/2), range_r=(5, 100), range_m=(0.2, 1), name="", seed=None):
        """
        Create a random Kresling tower
        :param range_n: range of the number of folds
        :param range_dh: range of the difference of height between the two stable positions
        :param range_dphi: range of the difference of angle between the two stable positions
        :param range_r: range of the radius of the folds
        :param name: name of the tower
        :return: Kresling tower
        """
        rng = np.random.default_rng(seed)
        if range_r[0] is None or range_r[1] is None:
            range_r = (5, 100)
        if range_dh[0] is None or range_dh[1] is None:
            range_dh = (0, 20)
        if range_dphi[0] is None or range_dphi[1] is None:
            range_dphi = (np.pi/2, 3*np.pi/2)
        if range_n[0] is None or range_n[1] is None:
            range_n = (3, 20)
        if range_m[0] is None or range_m[1] is None:
            range_m = (0.2, 1)
        
        for _ in range(100): 
            n = rng.integers(range_n[0], range_n[1] + 1)
            dh = rng.uniform(range_dh[0], range_dh[1])
            dphi = rng.uniform(range_dphi[0], range_dphi[1])
            r = rng.uniform(range_r[0], range_r[1])
            m = rng.uniform(range_m[0], range_m[1])

            h2_h1_sq = m* r**2 * np.cos(np.pi/n) * 4 * np.sin(dphi/2) # h2**2 - h1**2
            h1 = (h2_h1_sq - dh**2) / (2 * dh)
            h2 = h1 + dh

            if h2**2 - h1**1 < 4 * m*r**2 * np.cos(np.pi / n):
                return cls.from_n_r_h(n, r, h1, h2, m=m, name=name) # type: ignore
        else:
            raise ValueError("Could not generate a random Kresling tower with the given parameters")

    # representation de l'objet en chaine de caracteres
    def __str__(self):
        return f"{self.name} : n={self.n}, l={self.l}, b={self.b}, r={self.r}, h1={self.h1}, h2={self.h2}, m={self.m}"
    

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return TDK(self.n, self.l, self.b, self.r, self.h1, self.h2, self.m, self.a, self.r_p, self.nb_stable, self.name + "_copy", self.color, self.k_comp, self.k_tor)

    def set_k_raid(self, k_tor, k_comp):
        self.k_tor = k_tor
        self.k_comp = k_comp



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
    def phi_l(self, h):
        return np.arccos((h**2 - self.l**2 + self.r**2 *(self.m**2 + 1))/(2*self.m*self.r**2)) - np.pi/self.n

    def phi_b(self, h):
        a = (h**2 - self.b**2 + self.r**2 *(self.m**2 + 1))/(2*self.m*self.r**2)
        if a < -1 or a > 1:
            if self.verbose :
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

    def phi_num(self, h):
        # return min_search(self.phi(h), lambda phi : self.U(phi, h))
        return minimize_scalar(lambda phi : self.U(phi, h), self.phi(h), bounds=(0, 2 * np.pi)).x # type: ignore


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

    def h_num(self, phi):
        return minimize_scalar(lambda h : self.U(phi, h), self.h(phi), bounds=(0, self.b)).x #type: ignore



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
        print(f"Tour {self.name} sauvegardee dans {path}")

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
        return h / self.l_dep(phi, h)
    
    def deriv_l_phi(self, phi, h):
        gamma = np.pi / self.n
        return self.r**2 * self.m * np.sin(gamma + phi) / self.l_dep(phi, h)

    
    def b_dep(self, phi, h):
        gamma = np.pi / self.n
        return np.sqrt(self.r**2 * (self.m**2 + 1 - 2*self.m*np.cos(gamma - phi)) + h**2)

    def deriv_b_h(self, phi, h):
        return h / self.b_dep(phi, h)
    
    def deriv_b_phi(self, phi, h):
        gamma = np.pi / self.n
        return - self.r**2 * self.m * np.sin(gamma - phi) / self.b_dep(phi, h)

    
    def theta_a(self, phi, h):
        """ Angle entre base et panneau ABC"""
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(phi)-np.cos(gamma))

        return np.arccos(-rcos/(np.sqrt(h**2 + rcos**2)))
    
    def deriv_theta_a_h(self, phi, h):
        gamma = np.pi/self.n
        rcospg = self.r * (self.m * np.cos(phi)-np.cos(gamma))
        return -rcospg / (h**2 + rcospg**2)

    def deriv_theta_a_phi(self, phi, h):
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(gamma) - np.cos(phi))
        return h * self.r * self.m * np.sin(phi) / (h **2 + rcos**2)



    def theta_ah(self, phi, h):
        """ Angle entre haut et panneau BCD"""
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        return np.arccos(rcos/(np.sqrt(h**2 + rcos**2)))
    
    def deriv_theta_ah_h(self, phi, h):
        gamma = np.pi/self.n
        rcosgp = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        return rcosgp / (h**2 + rcosgp**2)

    def deriv_theta_ah_phi(self, phi, h):
        gamma = np.pi/self.n
        rcos = self.r * (self.m * np.cos(gamma) - np.cos(phi))
        return h * self.r * np.sin(phi) / (h **2 + rcos**2)

    def theta_b(self, phi, h):
        """Angle montagne """
        gamma = np.pi/self.n
        rmgamma = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        rmphi = self.r * (self.m * np.cos(phi)-np.cos(gamma))
        h2 = h**2
        A = -(h2*np.cos(phi + gamma) + rmgamma * rmphi)/(np.sqrt((h2 + rmgamma**2)*(h2 + rmphi**2)))
        # if A < -1 or A > 1:
        #     print("No solution for theta_b, returning 0 or pi for phi =", rad2deg(phi), "and h =", h, self.name, "h1 =", self.h1, "h2 =", self.h2)
        #     if A < -1:
        #         return 0
        #     else:
        #         return np.pi
        return np.arccos(A) * np.sign(np.sin(phi + gamma))

    def deriv_theta_b_h(self, phi, h):
        gamma = np.pi/self.n
        cp = np.cos(phi)
        cg = np.cos(gamma)
        r = self.r
        m = self.m
        h2 = h**2
        r2 = r**2
        cpg = np.cos(phi + gamma)
        spg = np.sin(phi + gamma)
        a1 = m * cg - cp
        a2 = m * cp - cg

        b1 = h2 * cpg + r2 * a1 * a2
        b1_deriv = 2 * h * cpg

        b21 = h2 + a1**2 * r2
        b22 = h2 + a2**2 * r2
        b2 = b22 * b21
        b2_deriv = 2 * h * (2 * h2 + r2 * (a1**2 + a2**2))

        denom = 1 - b1**2 / b2
        denom = np.sqrt(denom) 

        deriv = (2 * b2 * b1_deriv - b1 * b2_deriv) / (2 * denom * b2 * np.sqrt(b2))
        
        return deriv
    
    def deriv_theta_b_phi(self, phi, h):
        gamma = np.pi/self.n
        cp = np.cos(phi)
        sp = np.sin(phi)
        cg = np.cos(gamma)
        r = self.r
        m = self.m
        h2 = h**2
        r2 = r**2
        cpg = np.cos(phi + gamma)
        spg = np.sin(phi + gamma)
        a1 = m * cg - cp
        a2 = m * cp - cg

        denom = h2 ** 2 * spg ** 2 + r2 * h2 * ((cg ** 2 + cp**2) * (1 + m**2 + 2 * m * cpg) - cg * cp * (4 * m + 2 * cpg * (1 + m**2)))
        denom = np .sqrt(denom) 

        b1 = h2 * cpg + r2 * a1 * a2
        b1_deriv = - (h2 * cpg + r2 * sp * (2 * m * cp + (1 + m**2) * cg))

        b21 = h2 + a1**2 * r2
        b22 = h2 + a2**2 * r2
        b2 = b22 * b21
        b2_deriv = 2 * r2 * sp * (h2 * (a1 + m * (cg - cp)) + r2 * (a1**3 + a2**2 * m * (cg - cp)))

        return - (b1_deriv / np.sqrt(b2) - 0.5 * b1 * b2_deriv / (np.sqrt(b2)* b2)) / denom




    def theta_l(self, phi, h):
        """Angle valley """
        gamma = np.pi/self.n
        rmgamma = self.r * (self.m * np.cos(gamma)-np.cos(phi))
        rmphi = self.r * (self.m * np.cos(phi)-np.cos(gamma))
        h2 = h**2
        A = -(h2*np.cos(phi - gamma) + rmgamma * rmphi)/(np.sqrt((h2 + rmgamma**2)*(h2 + rmphi**2)))
        return (np.arccos(A)-np.pi) * np.sign(np.sin(phi - gamma)) + np.pi

    def deriv_theta_l_h(self, phi, h):
        gamma = np.pi/self.n
        cp = np.cos(phi)
        cg = np.cos(gamma)
        r = self.r
        m = self.m
        h2 = h**2
        r2 = r**2
        cpg_neg = np.cos(phi - gamma)
        a1 = m * cg - cp
        a2 = m * cp - cg

        b1 = h2 * cpg_neg + r2 * a1 * a2
        b1_deriv = 2 * h * cpg_neg

        b21 = h2 + a1**2 * r2
        b22 = h2 + a2**2 * r2
        b2 = b22 * b21
        b2_deriv = 2 * h * (2 * h2 + r2 * (a1**2 + a2**2))

        denom = 1 - b1**2 / b2
        denom = np.sqrt(denom) 

        deriv = (2 * b2 * b1_deriv - b1 * b2_deriv) / (2 * denom * b2 * np.sqrt(b2))
        
        return deriv

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
        dtheta_a = np.pi/2 - self.theta_a(phi, h)
        dtheta_ah = np.pi/2 - self.theta_ah(phi, h) 
        dtheta_b = np.pi - self.theta_b(phi, h) 
        dtheta_l = np.pi - self.theta_l(phi, h) 
        n = self.n
        # print(f"dl : {dl}, db : {db}, dtheta_a : {dtheta_a}, dtheta_ah : {dtheta_ah}, dtheta_b : {dtheta_b}, dtheta_l : {dtheta_l}")
        U_r = self.U_raid[0](self.k_tor)
        U_t = self.U_raid[1](self.k_comp)

        return n * (U_t(dl) / self.l + U_t(db) / self.b + U_r(dtheta_a) * self.a + U_r(dtheta_ah) * self.a * self.m + U_r(dtheta_b) * self.b + U_r(dtheta_l) * self.l)

    def deriv_U_h(self, phi, h):

        dl = self.l_dep(phi, h) - self.l
        dl_dh = self.deriv_l_h(phi, h)
        db = self.b_dep(phi, h) - self.b
        db_dh = self.deriv_b_h(phi, h)
        dth_a = np.pi/2 - self.theta_a(phi, h)
        dth_a_dh = self.deriv_theta_a_h(phi, h)
        dth_ah = np.pi/2 - self.theta_ah(phi, h)
        dth_ah_dh = self.deriv_theta_ah_h(phi, h)
        dth_b = np.pi - self.theta_b(phi, h)
        dth_b_dh = self.deriv_theta_b_h(phi, h)
        dth_l = np.pi - self.theta_l(phi, h)
        dth_l_dh = self.deriv_theta_l_h(phi, h)

        n = self.n
        dU_r = self.dU_raid[0](self.k_tor)
        dU_t = self.dU_raid[1](self.k_comp)
        return n * (dU_t(dl, dl_dh) / self.l + dU_t(db, db_dh) / self.b - dU_r(dth_a, dth_a_dh) * self.a - dU_r(dth_ah, dth_ah_dh) * self.a * self.m - dU_r(dth_b, dth_b_dh) * self.b - dU_r(dth_l, dth_l_dh) * self.l)


    def deriv_U_phi(self, phi, h):
        dl = self.l_dep(phi, h) - self.l
        dl_dphi = self.deriv_l_phi(phi, h)
        db = self.b_dep(phi, h) - self.b
        db_dphi = self.deriv_b_phi(phi, h)

        n = self.n
        dU_t = self.dU_raid[1](self.k_comp)

        return n * (dU_t(dl, dl_dphi) / self.l + dU_t(db, db_dphi) / self.b)



    def set_U_raid(self, U_rot, U_trac):
        self.U_raid = [U_rot, U_trac]


    def find_curve(self, x_values, rotation = False, graph = False):
        curve = []
        for x in x_values:
            if rotation :
                h = minimize_scalar(lambda h : self.U(x, h), self.h(x), bounds=(0, self.b)).x # type: ignore
                curve.append([x, h])
            else:
                phi = minimize_scalar(lambda phi : self.U(phi, x), self.phi(x), bounds=(0, 2 * np.pi)).x # type: ignore
                curve.append([phi, x])

        curve = np.array(curve)
        if graph:
            plt.plot(curve[:, 0], curve[:, 1])
            plt.xlabel("phi (rad)" if not rotation else "h (mm)")
            plt.ylabel("h (mm)" if not rotation else "phi (rad)")
            plt.title(f"Deplacement curve for {self.name}")
            plt.grid()
            plt.show()

        return curve



    def show_graph(self, type : str, height_abscisse = True):
        y_label = ""
        if type == "energy":
            y_label = "U"
            if height_abscisse:
                x_values = np.linspace(0, self.b, 100)
            else:
                x_values = np.linspace(0, np.pi, 100)

            curve = self.find_curve(x_values , rotation=not height_abscisse)
            
            plt.plot(x_values if height_abscisse else rad2deg(x_values), self.U(curve[:, 0], curve[:, 1]))
            plt.xlabel("phi (rad)" if not height_abscisse else "h (mm)")
            plt.ylabel(y_label)
            plt.title(f"Energy curve for {self.name}")
            plt.grid()
            plt.show()






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


    def create_patron_simplified(self, param_list : ParamList | None = None, flap : str | Pattern = "none", base_flaps : str | list[Pattern] = "none"):
        """
            flap : "none" => no flap, "default" => default flap, Pattern => pattern of the flap to add on each extreme of the pattern
        """


        #----------------- Initialisations of the parameters ---------------- 
        
        if param_list is None:
            param_list = ParamList()

        #verify the presence of the parameters in the template list
        if not param_list.template.contain_list(["fold_cut"]): # type: ignore
            raise ValueError(f"Not all parameters of {["fold_cut"]} is in the template list {param_list.template.__str__()}")

        # initial margin for the pattern
        w_margin = 5
        h_margin = 5

        # -------------- Initialisation of the values for the creation of the folds --------------
        alpha = alkashi_angle(self.l, self.a, self.b) # angle betweenm valley fold and base a 
        beta = alkashi_angle(self.l, self.b, self.m * self.a) # angle between mountain fold and valley fold 
        rho = alkashi_angle(self.b, self.a, self.l) # angle between mountain fold and base a
        dec_ang = 0 # angle between base a and horizontal

        # Lists of the folds to create
        down_line = []
        up_line = []
        mountain_line = []
        valley_line = []

        # starting point
        A0 = Point(0, 0)
        D0 = A0 + Vec.from_angle(beta + alpha + dec_ang, self.b)
        B = A0
        C = Point(0, 0)

        # width anf height of the pattern
        width = 0
        height = 0

        # -------------- Creation of the folds --------------
        for i in range(self.n):

            # compute the points of the folds
            A = B
            B = A + Vec.from_angle(dec_ang, self.a)
            C = A + Vec.from_angle(alpha + dec_ang, self.l)
            D = A + Vec.from_angle(beta + alpha + dec_ang, self.b)

            # add the folds to the lists
            down_line.append(Line(A, B))
            up_line.append(Line(D, C))
            mountain_line.append(Line(A, D))
            valley_line.append(Line(A, C))

            #compute width and height
            width = max(width, B[0], C[0], D[0])
            height = max(height, B[1], C[1], D[1])

            # update the angle for the next iteration
            dec_ang = (np.pi - (alpha + beta + rho))*(i+1)
        
        mountain_line.append(Line(B, C))

        # ------------- Creation of the pattern --------------     


        # Create the patron with the given parameters
        self.patron = Pattern(self.name, param_list=param_list, origin=Point(0, 0))

        # add the folds to the patron
        self.patron.add_folds(down_line + up_line, "n", 0)
        self.patron.add_folds(mountain_line, "m", 0)
        self.patron.add_folds(valley_line, "v", 0)

        


        # ------------- Creation of the flaps --------------

        # ------------- Closing flaps --------------

        # if flap is "default", create a default flap with the same height as the fold where it's attached and a width of a/3, with a cut fold on the outside of the pattern
        if isinstance(flap, str) and flap == "default":
            flap = Pattern(param_list=param_list)
            w = self.a/3
            angle = deg2rad(20)
            flap.w_h(w, self.b)
            A_flap = Point(w, 0)
            B_flap = A_flap - Vec.from_angle(-angle, w)
            D_flap = A_flap + Vec(0, self.b/2)
            C_flap = D_flap - Vec.from_angle(angle, w)
            flap.add_shapes([Line(A_flap, B_flap), Line(B_flap, C_flap), Line(C_flap, D_flap)], outside=True, param="cut")

        if isinstance(flap, Pattern):
            w_margin += flap.width
            # add the flap on the left side
            flap_left = flap.copy()
            flap_left.translate(Vec(-flap_left.width, 0))
            flap_left.rotate(Vec(0, 1).angle(Vec.from_2points(A0, D0)), Point(0, 0))
            flap_left.translate(A0)
            self.patron += flap_left

            # add the flap on the right side
            flap_right = flap.copy()
            flap_right.translate(Vec(-flap_right.width, 0))
            flap_right.rotate(np.pi + Vec(0, 1).angle(Vec.from_2points(B, C)), Point(0, 0))
            flap_right.translate(C)
            self.patron += flap_right

        # TODO ------------- Base flaps --------------
        
        angle = (1 - 3 / (2 * self.n)) * np.pi

        # b_flap = base_flap_pattern(param_list, angle=angle, w=self.a * self.m)
        # t_flap = base_flap_pattern(param_list, angle=angle, w=self.a, inv=True)


        # ------------- Finish creation of the pattern --------------

        # update the size of the pattern
        self.patron.w_h(width + 10, height + 10)
        self.patron.origin = Point(w_margin, h_margin)

        return self.patron


    


    ## TODO : add m
    ## TODO : replace closing and closing_type by string to understand what they do   
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

        # ---------------- Initialisations of the parameters ----------------

        # Set h for asymetric patrons (h is the height where there should be touching between adjacent panels)
        if h == 0: 
            h = self.h1

        # margin for attache depending on closing type
        w = 5 + self.a/2 if (closing == 2 and side == 1) else 5 + self.a if (closing == 2 and side == 2) else 5 

        # Create the patron with the given parameters
        self.patron = Pattern(self.name, param_list=param_list, origin=Point(w, 5))
        
        #verify the presence of the parameters in the template list
        if not self.patron.param_list.template.contain_list([outside_param, "fold_cut"]): # type: ignore
            raise ValueError(f"Not all parameters of {[outside_param, "fold_cut"]} is in the template list {self.patron.param_list.template.__str__()}")
        
        # if attache is given, prepare it's place on the patron
        if isinstance(attache, Pattern): 
            attache.origin = self.patron.origin.copy() 
            self.patron.origin[1] += attache.height


        # ---------------------- Calculate the placement of the folds ----------------------


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

    ## TODO : add m for certains 
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
            rho = self.angle_pli_rho(h_rep) # TODO : add m
            C = Point(x0 - self.r_p * np.cos(rho), self.a / 2 + np.sqrt(self.b ** 2 - self.r_p **2), self.r_p * np.sin(rho))
        else :
            C = Point(self.r * np.cos(phi) * self.m, self.r * np.sin(phi) * self.m, h_rep) 
        if type == 2:
            D = C.copy()
            D.translate(self.m * Vec.from_2points(A, B))
            angle_m_pc = (np.pi - self.angle_pli_montagne(h_rep)) * fold_pc / 100
            D.rotate(angle_m_pc, C, Vec.from_2points(B, C))
        else :
            D = C.rotate(2 * gamma, Point(0, 0, 0), Vec(0, 0, 1))

        # rotation matrix for starting position


        # Calculation with angles between panels
        if type == 2:
            verts = [Polygon([A,B,C]), Polygon([B,C,D])]
            angle_m_pc = (np.pi - self.angle_pli_montagne(h_rep)) * fold_pc / 100
            angle_v_pc = (np.pi - self.angle_pli_vallee(h_rep)) * fold_pc / 100

            # TODO !!
            for i in range(self.n - 1):
                # get panels and their norms
                n_poly_P = verts[-2].copy()
                n_poly_Q = verts[-1].copy()
                v_norm_P = n_poly_P.normal_vect()
                v_norm_Q = n_poly_Q.normal_vect()

                # get the middle point of the edge
                middle = Point.from_list(n_poly_P[1] + n_poly_P[0]) / 2 if i % 2 == 1 else Point.from_list(n_poly_P[1] + n_poly_P[2]) / 2

                n_poly_P.rotate(np.pi, middle, v_norm_P)
                direc = v_norm_P * (ep_tot - e_pp) / 2 if ep_tot != 0 else Vec(0, 0, 0) 
                if i % 2 == 0:
                    n_poly_P.rotate(-angle_m_pc, n_poly_P[2] - direc, n_poly_P[1] - n_poly_P[2])
                else:
                    n_poly_P.rotate(-angle_v_pc, n_poly_P[1] + direc, n_poly_P[1] - n_poly_P[0])

                verts.append(n_poly_P)
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

            # if asymetrie, offset the panel to reduce it's size
            # TODO for m
            if side != 0 and ep_tot != 0:
                angle_m = self.angle_pli_montagne(h_constr)
                angle_v = self.angle_pli_vallee(h_constr)

                d_mont = ep_tot / np.tan(angle_m/2) if side == 1 else 0
                d_val = ep_tot / np.tan(angle_v/2) if side == 2 else 0
                pol = Polygon([A, B, C, A]).offset([-d_val, -d_mont, 0])
                A = pol[0]
                B = pol[1]
                C = pol[2]


            # transformation matrix
            mat_rot = np.array([[np.cos(2 * gamma), -np.sin(2 * gamma), 0, 0],
                                [np.sin(2 * gamma), np.cos(2 * gamma), 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]])
            # mat_up_rot = np.array([[np.cos(phi + gamma), np.sin(phi + gamma), 0, 0],
            #                     [np.sin(phi + gamma), -np.cos(phi + gamma), 0, 0],
            #                     [0, 0, -1, h_rep], [0, 0, 0, 1]])

            # U2 = A.copy().transform(mat_up_rot)
            # V2 = B.copy().transform(mat_up_rot)
            # W2 = C.copy().transform(mat_up_rot)

            B2 = B.copy()
            C2 = C.copy()

            verts.append(Polygon([A, B, C]))
            verts.append(Polygon([B2, C2, D]))

            for _ in range(self.n - 1):
                A = A.transform(mat_rot)
                B = B.transform(mat_rot)
                C = C.transform(mat_rot)
                verts.append(Polygon([A, B, C]))
                B2 = B2.transform(mat_rot)
                C2 = C2.transform(mat_rot)
                D = D.transform(mat_rot)
                verts.append(Polygon([B2, C2, D]))
            
        if chiral:
            for pol in verts:
                pol = pol.mirror(Plane(Point(0, 0, 0), Vec(0, 1, 0)))
        if rotation != 0:
            for pol in verts:
                pol = pol.rotate(rotation, Point(0, 0, 0), Vec(0, 0, 1))

        return verts  


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


    def show_3D(self, h : Number | None = None, phi : Number | None = None, constr_type=0, fold_pc=100, ep=0.0, show_circles=False, recreate = True, camera_pos= [45, 0, 25], rotation : Number= 0, translation : Number = 0, chiral : bool = False, ax = None, show = True, inversed : bool = False):
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
            if inversed:
                self.volume.rotate(np.pi, Point(0, 0, h/2), Vec(0, 1, 0))
        
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











## TODO : add inversion haut/bas        
class MultiStoriesKresling:

    def __init__(self, towers : list[TDK], chiralities : list[bool] = [False], inversed : list[bool] = [False], name : str =""):
        # chiralities : if the tower turn clockwise or anticlockwise
        # inversed : if the tower is inversed (up and down are swapped for the 3D view)
        self.towers = towers
        self.chiralities = chiralities
        self.inversed = inversed
        if len(towers) != len(chiralities):
            if len(chiralities) == 1:
                self.chiralities = chiralities * len(towers)
            else:
                raise ValueError("The number of towers must be equal to the number of chiralities or 1")
        if len(towers) != len(inversed):
            if len(inversed) == 1:
                self.inversed = inversed * len(towers)
            else:
                raise ValueError("The number of towers must be equal to the number of inversed or 1")
        
        if name =="":
            n = self.towers[0].n
            
            self.name =  "TDK_Tower_" 
            self.name += "n" + (str(n) if all(tower.n == n for tower in towers) else "diff") + "_"
            for i, tower in enumerate(towers):
                self.name += f"h{int(tower.h1*100)/100}-{int(tower.h2*100)/100}_r{tower.r}"
                self.name += "C" if self.chiralities[i] else "A"
                if i != len(towers) - 1:
                    self.name += "_"
        else :
            self.name = name 

        self.patron = None
        self.volumes = None

    def __str__(self) -> str:
        return f"MultiStoriesKresling with {len(self.towers)} towers, chiralities : {self.chiralities}, inversed : {self.inversed} , name : {self.name}. \n Towers : \n" + "\n".join([str(tower) for tower in self.towers])
    
    def __repr__(self) -> str:
        return self.__str__()

    def as_dict(self):
        return {
            "name": self.name,
            "towers": [tower.as_dict() for tower in self.towers],
            "chiralities": self.chiralities,
            "inversed": self.inversed
        }


    def __getitem__(self, key) -> TDK:
        return self.towers[key]

    @staticmethod
    def from_dict(d):
        towers = [TDK.from_dict(tower_d) for tower_d in d["towers"]]
        chiralities = d["chiralities"]
        inversed = d["inversed"]
        return MultiStoriesKresling(towers, chiralities, inversed, d["name"])

    def copy(self):
        towers = [tower.copy() for tower in self.towers]
        chiralities = self.chiralities.copy()
        inversed = self.inversed.copy()
        return MultiStoriesKresling(towers, chiralities, inversed, self.name + "_copy")

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

            tower_volume = tower.create_volume(h, phi, constr_type, fold_pc, ep, decal, ep_tot, side, h_constr, rotation if not self.inversed[i] else rotation  - phi + np.pi / tower.n , self.chiralities[i])
            sens = -1 if self.chiralities[i] else 1
            if self.inversed[i]:
                tower_volume.rotate(np.pi, Point(0, 0, height + h/2), Vec(1, 0, 0))
                # tower_volume.rotate(phi/2 * sens, Point(0, 0, height + h/2), Vec(0, 0, 1))
                # rotation += phi/2 * sens
            rotation += (phi + np.pi / tower.n) * sens
            tower_volume.translate(Vec(0, 0, height if not self.inversed[i] else -height))
            height += h
            volumes.append(tower_volume)

        self.volumes = volumes    
        return volumes
    
    def show_3D(self, h : list[Number] | None = None, phi : list[Number] | None = None, constr_type=0, fold_pc=100, ep=0, show_type=0, recreate = True, camera_pos= [45, 0, 25], base_rotation : Number = 0, height_repr = None, save_svg = False, save_path = "", save_name = ""):
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
            ax.set_zlim(0, sum(h))

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

        if save_svg:
            if save_path == "":
                save_path = ORIGAMI_DIR_PATH + "saved_multi_towers_svg/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_name == "":
                save_name = self.name
            plt.savefig(save_path + f"{save_name}.svg")

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

    def dU_dh(self, phi : list[Number], h : list[Number]):
        if len(phi) != len(self.towers) or len(h) != len(self.towers):
            raise ValueError("The number of towers must be equal to the number of phi, h")
        dU_dh_tot = 0
        for i in range(len(self.towers)):
            tower = self.towers[i]
            dU_dh_tot += tower.deriv_U_h(phi[i], h[i])
        return dU_dh_tot
    
    def dU_dphi(self, phi : list[Number], h : list[Number]):
        if len(phi) != len(self.towers) or len(h) != len(self.towers):
            raise ValueError("The number of towers must be equal to the number of phi, h")
        dU_dphi_tot = 0
        for i in range(len(self.towers)):
            tower = self.towers[i]
            dU_dphi_tot += tower.deriv_U_phi(phi[i], h[i])
        return dU_dphi_tot



    def U_simp_phi(self, phi_tot, x):
        # print(phi_tot, x)
        phi0 = abs(phi_tot - np.sum(x)) % (2*np.pi)
        # print("phi0 :", [phi0], "x :", x)
        phis = [phi0] 
        phis.extend(np.array(x))
        phis = [phi % (2 * np.pi) for phi in phis]
        # print("phis :", phis)
        hs = [tower.h(phi) for tower, phi in zip(self.towers, phis)]

        # print("phis :", phis, "hs :", hs)
        return self.U(phis, hs)
    
    def U_simp_h(self, h_tot, x):
        h0 = h_tot - np.sum(x)
        hs = [h0] 
        hs.extend(np.array(x))

        U_tot = 0
        for i, h in enumerate(hs):
            if h < 0.0:
                return np.inf
            U_tot += minimize_scalar(lambda phi : self.towers[i].U(phi, h), self.towers[i].phi(h), bounds=(0, 2 * np.pi)).fun # type: ignore
        return U_tot

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


    def show_tower_movement(self, x_ax, curve, segments : list[list[Number]] | None = None, start_ind=0, save=False, path="", name="", animated=True, rotation = True, camera_pos_3D = [45, 0, 5], height3Dgraph = 2, **organisation):
        """curve : list of (phis, hs) for each point of the curve;
            organisation : pos in the window of the graph wanted. Possibles graphs : energy, kinematic, force, raideur, assemb3D, angle, height
        
        """

        if len(curve) != len(x_ax):
            raise ValueError("The number of points in the curve must be equal to the number of points in the x axis")

        if segments is not None and len(segments) == 1:
            segments = None

        #  ------------ Placement gestion ----------------
        if organisation == {}:
            organisation = {"assemb3D" : (0, 0),  "energy" : (1, 0), "kinematic" : (1, 1), "angle" : (2, 0)}
        ncol = 0
        nrow = 0

        graphs = organisation.keys()
        
        for val in organisation.values() :
            if val[0] + 1 > ncol :
                ncol = val[0] + 1
            if val[1] + 1 > nrow :
                nrow = val[1] + 1 

        # if nrow < height3Dgraph and "assemb3D" in graphs:
        #     nrow = height3Dgraph
            

        # -------------------- Values Initialisation ------------------- 
        # sens for the movement of the towers, if chiral, the movement is inverted
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])
        ordened_curve = np.concatenate((curve[-start_ind:], curve[:-start_ind])) if start_ind != 0 else np.array(curve)

        if segments is not None:
            color_values = [0.6 + 0.4 *i / (len(segments) - 1) for i in range(len(segments))]
        else :
            color_values = [1]

        x_label = r"$\varphi_{total}$ (deg)" if rotation else r"$h_{total}$ (mm)"
        x_ax = rad2deg(np.array(x_ax)) if rotation else np.array(x_ax)

        graph_infos = {
            "energy" : {"ylabel" : r"$U(\varphi, h_{num}(\varphi))$" if rotation else r"$U(\varphi_{num}(h), h)$", "title" : "Energy of the system", "function" : lambda x, y : self.U(x * sens, y), "multi_line" : False, "colors": [hsv_to_hex(270/360, 1, color_value) for color_value in color_values], "arrows" : True},
            
            "kinematic" : {"ylabel" : r"$\varphi_{total}$ (deg)" if not rotation else r"$h_{total}$ (mm)", "title" : f"Relation {"$h(\\varphi)$" if rotation else "$\\varphi(h)$"} of the system", "function" : lambda x, y : sum(y) if rotation else sum(x), "multi_line" : False, "colors": [hsv_to_hex(270/360, 1, color_value) for color_value in color_values], "arrows" : True},

            "force" : {"ylabel" : r"$\frac{dU}{dh}(\varphi, h_{num}(\varphi))$" if rotation else r"$\frac{dU}{dh}(\varphi_{num}(h), h)$", "title" : "Force of the system", "function" : lambda x, y : self.dU_dphi(x * sens, y) if rotation else self.dU_dh(x, y), "multi_line" : False, "colors": [hsv_to_hex(270/360, 1, color_value) for color_value in color_values], "arrows" : True},

            "angle" : {"ylabel" : r"$\varphi$ (deg)", "title" : "Angle of the towers", "function" : [(lambda i : (lambda x, y : x[i]))(j) for j in range(len(self.towers))] , "multi_line" : True, "colors": [[hsv_to_hex(hex_to_hsv(tower.color)[0], 1, color_value) for color_value in color_values] for tower in self.towers], "arrows" : False},

            "height" : {"ylabel" : r"$h$ (mm)", "title" : "Height of the towers", "function" : [(lambda i : (lambda x, y : y[i]))(j) for j in range(len(self.towers))] , "multi_line" : True, "colors": [[hsv_to_hex(hex_to_hsv(tower.color)[0], 1, color_value) for color_value in color_values] for tower in self.towers], "arrows" : False},
        }


        assemb3D = "assemb3D" in graphs
        if assemb3D:
            graphs = [graph for graph in graphs if graph != "assemb3D"]

        points = {}

        fig = plt.figure(figsize=(ncol * 5, min(nrow * 4, 12)), dpi=100)

        # -------------------- Graph with the 3D representation of the tower -------------------------
        if assemb3D:
            position = organisation["assemb3D"]
            ax_3D = plt.subplot2grid((nrow, ncol), position, rowspan=min(height3Dgraph, nrow), projection="3d")

            # set the camera pos
            camera_pos = camera_pos_3D
            ax_3D.azim = camera_pos[0] # type: ignore
            ax_3D.dist = camera_pos[1] # type: ignore
            ax_3D.elev = camera_pos[2] # type: ignore

            # Setting the Axes properties
            ax_3D.set_xlabel('X')
            ax_3D.set_ylabel('Y')
            ax_3D.set_zlabel('Z') # type: ignore
            
            ax_3D.set(xlim3d=(-self.towers[0].r, self.towers[0].r), xlabel='X')
            ax_3D.set(ylim3d=(-self.towers[0].r, self.towers[0].r), ylabel='Y')
            ax_3D.set(zlim3d=(0, self.max_height_stable() * 1.1), zlabel='Z')
            ax_3D.set_xticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))
            ax_3D.set_yticks(np.linspace(-self.towers[0].r, self.towers[0].r, 5))
            ax_3D.set_aspect('equal')

            legend_elements = [Patch(facecolor=self.towers[i].color, edgecolor=None, label=f'Tower {i+1}') for i in range(len(self.towers))]
            ax_3D.legend(handles=legend_elements, loc="upper left")

        # Store polygon collection references
        poly_collections = []

        # --------------------- Graphs --------------------------
        axes = {}
        # Set the axes and their position in the window according to the organisation given in argument
        for graph in graphs:
            position = organisation[graph]
            axes[graph] = fig.add_subplot(nrow, ncol , position[0] + 1 + ncol * position[1])
            
        # Plot the curves on the graphs, if segments are given, plot each segment with a different color and add a label with the value of phi or h for the segment
        if segments is not None:
            ind_s = 0
            for i in range(len(segments)):
                x = rad2deg(np.array(segments[i])) if rotation else np.array(segments[i])
                label = f"$\\varphi$ from {str(int(rad2deg(segments[i][0])))}° to {str(int(rad2deg(segments[i][-1])))}°" if rotation else f"$h$ from {str(int(segments[i][0]*100)/100)} to {str(int(segments[i][-1]*100)/100)}"
                for graph in graphs:
                    if graph_infos[graph]["multi_line"]:
                        for j in range(len(self.towers)):
                            y = [graph_infos[graph]["function"][j](curve[(ind_s + k - start_ind) % len(curve)][0] * sens, curve[(ind_s + k - start_ind) % len(curve)][1]) for k in range(len(segments[i]))]
                            line = axes[graph].plot(x, y, color=graph_infos[graph]["colors"][j][i], label=label + f" - Tower {j+1}")
                            if graph_infos[graph]["arrows"]:
                                add_arrow_to_line2D(axes[graph], line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(x_ax))), arrowsize=1.5)
                    else :
                        y = [graph_infos[graph]["function"](curve[(ind_s + j - start_ind) % len(curve)][0] * sens, curve[(ind_s + j - start_ind) % len(curve)][1]) for j in range(len(segments[i]))]
                        line = axes[graph].plot(x, y, color=graph_infos[graph]["colors"][i], label=label)
                        if graph_infos[graph]["arrows"]:
                            add_arrow_to_line2D(axes[graph], line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(x_ax))), arrowsize=1.5)
                ind_s += len(segments[i])
        else :
            for graph in graphs:
                if graph_infos[graph]["multi_line"]:
                        for j in range(len(self.towers)):
                            y = [graph_infos[graph]["function"][j](ordened_curve[i][0] * sens, ordened_curve[i][1]) for i in range(len(x_ax))]
                            line = axes[graph].plot(x_ax, y, color=graph_infos[graph]["colors"][j][0], label=f"Tower {j+1}")
                            if graph_infos[graph]["arrows"]:
                                add_arrow_to_line2D(axes[graph], line, arrow_locs=[0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 0.91], arrowsize=1.5)
                else :
                    y = [graph_infos[graph]["function"](ordened_curve[i][0] * sens, ordened_curve[i][1]) for i in range(len(x_ax))]
                    line = axes[graph].plot(x_ax, y, color=graph_infos[graph]["colors"][0], label=graph)
                    if graph_infos[graph]["arrows"]:
                        add_arrow_to_line2D(axes[graph], line, arrow_locs=[0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 0.91], arrowsize=1.5)
        


        for graph in graphs:
            # Create the point that will follow the movement of the towers on the graph
            points[graph] = axes[graph].plot([], [], "ro", label="current position")[0]

            # add all the decoration around the graphs
            axes[graph].set_xlabel(x_label)
            axes[graph].set_ylabel(graph_infos[graph]["ylabel"])
            axes[graph].set_title(graph_infos[graph]["title"])
            axes[graph].grid()
            axes[graph].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
            axes[graph].legend()



        
        # if segments is not None:
        #     ind_s = 0
        #     colors = [hsv_to_hex(i / len(segments), 1, 1) for i in range(len(segments))]

        #     for i in range(len(segments)):
        #         x = rad2deg(np.array(segments[i])) if rotation else np.array(segments[i])
        #         label = f"energy segment for $\\varphi$ from {str(int(rad2deg(segments[i][0])))}° to {str(int(rad2deg(segments[i][-1])))}°" if rotation else f"energy segment for $h$ from {str(int(segments[i][0]*100)/100)} to {str(int(segments[i][-1]*100)/100)}"
        #         y_energy = [self.U(curve[(ind_s + j - start_ind) % len(curve)][0] * sens, curve[(ind_s + j - start_ind) % len(curve)][1]) for j in range(len(segments[i]))]
        #         energy_line = energy_ax.plot(x, y_energy, color=colors[i],label=label)
                
        #         energy_bottom = min(energy_bottom, min(y_energy))
        #         energy_top = max(energy_top, max(y_energy))


        #         y = [sum(np.array(curve[(ind_s + j - start_ind) % len(curve)][1])) for j in range(len(segments[i]))] if rotation else [rad2deg(sum(np.array(curve[(ind_s + j - start_ind) % len(curve)][0]))) for j in range(len(segments[i]))]

        #         label_kinematic = f"{"height" if rotation else "angle"} for {"$\\varphi$" if rotation else "$h$"} from {str(int(rad2deg(segments[i][0])) if rotation else int(segments[i][0]*100)/100)}{"°" if rotation else ""} to {str(int(rad2deg(segments[i][-1]))if rotation else int(segments[i][-1]*100)/100)}{"°" if rotation else ""}"
        #         # kinematic_line = kinematic_ax.plot(x, y, color=colors[i],label=label_kinematic)

        #         add_arrow_to_line2D(energy_ax, energy_line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(x_ax))), arrowsize=1.5)
        #         # add_arrow_to_line2D(kinematic_ax, kinematic_line, arrow_locs=np.linspace(0, 1, int(10 * len(segments[0]) / len(x_ax))), arrowsize=1.5)
        #         ind_s += len(segments[i])
                

        # else:
        #     y_energy = [self.U(ordened_curve[i][0] * sens, ordened_curve[i][1]) for i in range(len(x_ax))]
        #     line = energy_ax.plot(x_ax, y_energy, "b", label="energy")
        #     add_arrow_to_line2D(energy_ax, line, arrow_locs=[0.11, 0.21, 0.31, 0.41, 0.61, 0.71, 0.81, 0.91],arrowsize=1.5)
        #     y = [sum(c[1]) for c in ordened_curve] if rotation else [rad2deg(sum(c[0])) for c in ordened_curve]
        #     # kinematic_ax.plot(x_ax, y, "b",label="height" if rotation else "angle")
        #     energy_bottom = min(y_energy)
        #     energy_top = max(y_energy)


        
        # point_kinematic = kinematic_ax.plot([], [], "ro", label="current position")[0]

        
        

        # kinematic_ax.legend()
        # kinematic_ax.set_title(f"Relation {"$h(\\varphi)$" if rotation else "$\\varphi(h)$"} of the system")




        # --------------------- Graphs with phi and h of each tower as a function of x_tot -------------------------

        # if graph_intermediates:
        #     # kinematic_phi_ax = fig.add_subplot(2, 3, 3)
        #     # kinematic_phi_ax.set_ylabel("$\\varphi$ (degrees)")

        #     kinematic_h_ax = fig.add_subplot(2, 3 - 1, 6 - 2)
        #     kinematic_h_ax.set_ylabel("$h$ (mm)")

        #     if rotation:
        #         # kinematic_phi_ax.set_xlabel(r"\varphi_{total} (degrees)")
        #         kinematic_h_ax.set_xlabel(r"\varphi_{total} (degrees)")
        #     else:
        #         # kinematic_phi_ax.set_xlabel(r"$h_{total}$ (mm)")
        #         kinematic_h_ax.set_xlabel(r"$h_{total}$ (mm)")

        #     for i in range(len(curve[0][0])):
        #         # kinematic_phi_ax.plot(x_ax, [rad2deg(c[0][i]) for c in ordened_curve], label="$\\varphi$ tower " + str(i+1), color=self.towers[i].color)
        #         kinematic_h_ax.plot(x_ax, [c[1][i] for c in ordened_curve], label="$h$ tower " + str(i+1), color = self.towers[i].color)
            
        #     # points_phi = kinematic_phi_ax.plot([], [], "ro", label="current position")[0]
        #     points_h = kinematic_h_ax.plot([], [], "ro", label="current position")[0]

        #     # kinematic_phi_ax.set_title("$\\varphi$ of each tower")
        #     # kinematic_phi_ax.legend()
        #     kinematic_h_ax.set_title("$h$ of each tower")
        #     kinematic_h_ax.legend()
                

        # -------------------- Force graphs -------------------------

        # if force_show:
        #     force_ax = fig.add_subplot(2, ncol, ncol + 2)
        #     force_ax.set_xlabel(r"$\varphi_{total}$ (deg)" if rotation else r"$h_{total}$ (mm)")
        #     force_ax.set_ylabel("Force")

            

        #     force_ax.plot(x_ax, [self.dU_dh(c[0] * sens, c[1]) for c in ordened_curve], "b", label="force")




        #     point_force = force_ax.plot([], [], "ro", label="current position")[0]

        #     force_ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
        #     force_ax.grid()
        #     force_ax.set_title("Force applied depending on the position")
        #     force_ax.legend()


        # -------------------- Animation function -------------------------
        def animate(frame):
            #  ======== 3D animation =========
            if assemb3D: 
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
            
            animated_points = []
            # ========= 2D animation of the energy and height graphs =========

            # Update the point position in the energy graph

            for graph in graphs:
                if graph != "assemb3D":
                    if graph_infos[graph]["multi_line"]:
                        points[graph].set_data([x_ax[frame]] * len(self.towers), [graph_infos[graph]["function"][i](ordened_curve[frame][0] * sens, ordened_curve[frame][1]) for i in range(len(self.towers))])  
                    else :
                        points[graph].set_data([x_ax[frame]], [graph_infos[graph]["function"](ordened_curve[frame][0] * sens, ordened_curve[frame][1])])  # type: ignore
                        animated_points.append(points[graph])


            # if "energy" in graphs:
            #     point_energy.set_data([x_ax[frame]], [self.U(np.array(ordened_curve[frame][0]* sens), ordened_curve[frame][1])])  # type: ignore
            #     # point_kinematic.set_data([x_ax[frame]], [sum(ordened_curve[frame][1] if rotation else rad2deg(ordened_curve[frame][0]))])  

            #     points = [point_energy] # type: ignore
            # points += [point_kinematic]

            # ======== 2D animation of the phi and h graphs for each tower ========

            # if graph_intermediates:
            #     # points_phi.set_data([x_ax[frame]] * len(self.towers), [rad2deg(ordened_curve[frame][0])]) # type: ignore
            #     points_h.set_data([x_ax[frame]] * len(self.towers), [ordened_curve[frame][1]]) # type: ignore
            #     # points += [points_phi] # type: ignore
            #     points += [points_h] # type: ignore

            # # ======== 2D animation of the force graph ========
            # if force_show:
            #     point_force.set_data([x_ax[frame]], [self.dU_dh(np.array(ordened_curve[frame][0] * sens), ordened_curve[frame][1])])  # type: ignore
            #     points += [point_force] # type: ignore

            return poly_collections + animated_points

        

        # if abs(x_ax[0] - x_ax[-1]) > deg2rad(0.1) and rotation or abs(x_ax[0] - x_ax[-1]) > 0.5 and not rotation:
        #     frames = [ i if i < len(curve) else len(curve) - (i % len(curve) + 1) for i in range(len(curve) * 2)]
        # else:
        step = max(1, len(curve) // 200)  # Adjust the step to have at most 200 frames
        frames = range(0, len(curve), step)
        # Create animation
        if animated :
            anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=False, repeat=True)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        if save:
            if name == "":
                name = self.name
            if path == "":
                path = ORIGAMI_DIR_PATH + "/animations/"
            elif path[-1] != "\\" and path[-1] != "/":
                path += "/"

            if animated:
                anim.save(path + f"animation_{name}.gif", writer='pillow', fps=20) # type: ignore
                print(f"Saved animation to {path + f'animation_{name}.gif'}")
            else :
                plt.savefig(path + f"animation_{name}.svg") # type: ignore
                print(f"Saved graph to {path + f'animation_{name}.svg'}")
            plt.close()
            plt.clf()
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


        # ---------- initialisation ----------

        # create an array of 1 and -1 depending on the chirality of the towers
        sens = np.array([-1.0 if chiral else 1.0 for chiral in self.chiralities])

        # initial position for the optimization, if not given as argument, we take the fully extended position for all the towers
        if start_pos is None:
            start_pos = [tower.h2 for tower in self.towers]
        start = start_pos[1:]

        # find the index of the position in h that is the closest to the initial position of the towers, to start the movement from there
        h_0 = sum(start_pos)
        # print("h_0 :", h_0, "start_pos: ", start_pos)
        diff = abs(h[0] - h[1])
        ind = 0 # index of the position in h that is the closest to h_0
        for i, p in enumerate(h):
            if abs(p - h_0) < diff:
                ind = i
                break 
        else :
            raise ValueError(f"h_0 : {h_0} must be in the range of h")
        h = np.concatenate((h[ind:], h[:ind]))


        # ------------- Loop to find the height of each tower for each position of the movement -------------
        curve = []
        for h_i in h:

            # If there are only 2 towers, we can use a minimisation for a function with one variable, otherwise, we need to use a minimisation for more than one variable and we need to add a constraint to ensure that the sum of the heights of the towers is equal to h_i
            if len(self.towers) == 2:
                # start = [minimize_scalar(lambda x : self.U_simp_h(h_i, [x]), start, bounds=(0, self.towers[1].b)).x] # type: ignore
                start = [min_search(start[0], lambda x : self.U_simp_h(h_i, [x]), tol = 1e-8, eps=1e-5)]
            else:
                cons = ({'type': 'ineq', 'fun': lambda x:  h_i - sum(x)})
                start = minimize(lambda x : self.U_simp_h(h_i, x), start, bounds=[(0, tower.b) for tower in self.towers[1:]], constraints=cons, method="SLSQP", jac="2-point").x
            
            h0 = h_i - np.sum(start)
            hs =  np.concatenate(([h0], start))

            # we find the phi of each tower corresponding to the height found, and we add it to the curve
            phis = [tower.phi_num(h) for tower, h in zip(self.towers, hs)] * sens
            # print("hs :", hs, "phis :", [rad2deg(phi) for phi in phis], 'h_i :', h_i, 'start :', start)
            curve.append([phis, hs])

        
        return curve, ind
