import numpy as np
import math as m



def calc_refl_trans_12(n,theta):
    """
    Calculate reflection and transmission coefficients for interface 1-2.

    Args:
        n (float): Refractive index of medium 2 relative to medium 1.
        theta (float): Angle of incidence in degrees.

    Returns:
        tuple: Reflection coefficient (r12) and transmission coefficient (t12).
    """

    a = np.cos(theta * m.pi / 180.0)
    b = np.sin(theta * m.pi / 180.0)

    r12s_nom = a - np.sqrt(n**2 - b**2)
    r12s_denom = a + np.sqrt(n**2 - b**2)
    r12s = np.abs(r12s_nom/r12s_denom)**2

    r12p_nom = (n**2 * a) - np.sqrt(n ** 2 - b ** 2)
    r12p_denom =  (n**2 * a) + np.sqrt(n ** 2 - b ** 2)
    r12p = np.abs(r12p_nom / r12p_denom)**2

    r12 = 1/2 * (r12s + r12p)

    t12 = 1 - r12

    return r12,t12


def calc_refl_trans_21(n):
    """
    Calculate reflection and transmission coefficients for interface 2-1.

    Args:
        n (float): Refractive index of medium 2 relative to medium 1.

    Returns:
        tuple: Reflection coefficient (r21) and transmission coefficient (t21).
    """



    r12_prime = (3 * n ** 2 + 2 * n + 1) / (3 * (n + 1) ** 2) - \
                (2 * n ** 3 * (n ** 2 + 2 * n - 1)) / ((n ** 2 + 1) ** 2 * (n ** 2 - 1)) + \
                (n ** 2 * (n ** 2 + 1) / (n ** 2 - 1) ** 2) * np.log(n) - \
                (n ** 2 * (n ** 2 - 1) ** 2 / (n ** 2 + 1) ** 2) * np.log(n * (n + 1) / (n - 1))


    r21 = 1 - 1 / (n ** 2) * (1 - r12_prime)

    t21 = 1 - r21

    return r21,t21

def calc_refl(alpha,
              L,
              n,
              theta,
              r_dry,
              epsilon):
    """
    Calculate the final reflectance.

    Args:
        alpha (float): Absorption coefficient.
        L (float): Equivalent water thickness.
        n (float): Refractive index of medium 2 relative to medium 1.
        theta (float): Angle of incidence in degrees.
        r_dry (float): Reflectance of the dry sample.
        epsilon (float): Fraction of wet surface area.

    Returns:
        float: Final reflectance (r_final).
    """

    Tw = m.exp(-alpha * L)

    r12,t12 = calc_refl_trans_12(n,theta)
    r21,t21 = calc_refl_trans_21(n)

    r_wet = r12 + ((t12 * t21 * r_dry * Tw ** 2) / (1 - (r21 * r_dry * Tw ** 2)))

    r_final = epsilon * r_wet + (1 - epsilon) * r_dry

    return r_final





