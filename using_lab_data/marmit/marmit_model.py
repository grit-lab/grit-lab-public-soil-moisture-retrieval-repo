import numpy as np
import math as m

def calc_refl_trans_12(n,theta):
    """
        Calculate reflection and transmission coefficients for layer 1 to 2.

        Args:
            n (float): Refractive index.
            theta (float): Incidence angle.

        Returns:
            tuple: A tuple containing:
                - r12 (float): Reflection coefficient from layer 1 to 2, set to 0 following the MARMIT model.
                - t12 (float): Transmission coefficient from layer 1 to 2.
        """

### In the MARMIT model, the term r12 can be ignored in laboratory data because the diffuse radiation is negligible for
#laboratory measurements. This approach is based on the work of Bablet et al., 2018.#####

    r12 = 0

    t12 = 1 - r12

    return r12,t12


def calc_refl_trans_21(n):

    """
    Calculate reflection and transmission coefficients for layer 2 to 1.

    Args:
        n (float): Refractive index.

    Returns:
        tuple: A tuple containing:
            - r21 (float): Reflection coefficient from layer 2 to 1.
            - t21 (float): transmission coefficient from layer 2 to 1.
    """

    r12_prime = (3 * n ** 2 + 2 * n + 1) / (3 * (n + 1) ** 2) - \
                (2 * n ** 3 * (n ** 2 + 2 * n - 1)) / ((n ** 2 + 1) * (n ** 2 - 1)) + \
                ((n ** 2 * (n ** 2 + 1)) / (n ** 2 - 1) ** 2) * m.log(n) - \
                ((n ** 2 * (n ** 2 - 1) ** 2) / (n ** 2 + 1) ** 2 * m.log((n * (n + 1)) / (n - 1)))

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
    Calculate reflectance.

    Args:
        alpha (numpy array): Absorption coefficient.
        L (float): Equivalent water thickness.
        n (numpy array): Refractive index.
        theta (float): Solar zenith angle.
        r_dry (numpy array): Dry soil reflectance.
        epsilon (float): Surface coverage fraction of water.

    Returns:
        float: Predicted wet soil reflectance.
    """

    Tw = m.exp(-alpha * L)

    r12,t12 = calc_refl_trans_12(n,theta)
    r21,t21 = calc_refl_trans_21(n)

    r_wet = r12 + ((t12 * t21 * r_dry * Tw ** 2) / (1 - (r21 * r_dry * Tw ** 2)))

    r_final = epsilon * r_wet + (1 - epsilon) * r_dry

    return r_final





