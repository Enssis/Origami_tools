from .kresling import *

k=1.0

def general_U_rot(k=k*5*10**-5):
    def U_rot(dtheta):
        return 1/2 * k * (dtheta)**2
    return U_rot

def general_U_trac(k=k):
    k_l = k  # stiffness for translationnal energy
    # if dl > 0:
    #     k_l *= 10*dl
    def U_trac(dl):
        return 1/2 * k_l * (dl)**2
    return U_trac

def general_dU_trac(k=k):
    k_l = k  # stiffness for translationnal energy
    # if dl > 0:
    #     k_l *= 10*dl
    def U_trac(dl, derivl):
        return k_l * dl * derivl
    return U_trac


def general_dU_rot(k=k*5*10**-5):
    def U_trac(dtheta, dervi_theta):
        return k * dtheta * dervi_theta
    return U_trac


__all__ = [
    "general_U_rot",
    "general_U_trac",
    "general_dU_trac",
    "TDK",
    "MultiStoriesKresling"
]