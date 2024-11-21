import numpy as np


def calc_porosity_factor(fillFactor):
    """
    Calculates the porosity factor.

    Args:
        fillFactor (float): Fill factor of the porous medium.

    Returns:
        float: Porosity factor.
    """
    fillExp = fillFactor ** (2.0 / 3.0)
    K = -np.log(1.0 - 1.209 * fillExp) / (1.209 * fillExp)
    return K




def hapke(w, x, K):
    """
    Calculates H function using 1% IMSA Model.
    Args:
        w (float): Single scattering albedo.
        x (float): Angle (could be solar zenith or view zenith).
        K (float): Porosity factor.

    Returns:
        float: H function value.
    """
    r = (1 - np.sqrt(1 - w)) / (1 + np.sqrt(1 - w))
    a = (1 - 2*r*(x/K))/2
    b = np.log((1 + (x/K)) / (x/K))
    denominator = 1 - (w * (x/K) *(r + (a*b)))
    h = 1/denominator
    return h





def G1(t_s, t_0, sensor_azimuth, solar_azimuth):
    """
    Calculates the g value for phase function calculation.

    Args:
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.

    Returns:
        float: g value.
    """
    phi = sensor_azimuth - solar_azimuth


    g = np.arccos(
        ( np.cos(np.deg2rad(t_s)) * np.cos(np.deg2rad(t_0)) ) + ( np.sin(np.deg2rad(t_s)) * np.sin(np.deg2rad(t_0)) * np.cos(
            np.deg2rad(phi))))

    return g


def G2(t_s, t_0, sensor_azimuth, solar_azimuth):
    """
     Calculates the g' value for phase function calculation.

     Args:
         t_s (float): Solar zenith angle in degrees.
         t_0 (float): Viewing zenith angle in degrees.
         sensor_azimuth (float): Sensor azimuth angle in degrees.
         solar_azimuth (float): Solar azimuth angle in degrees.

     Returns:
         float: g' value.
     """

    phi = sensor_azimuth - solar_azimuth

    g_p = np.arccos(
        (np.cos(np.deg2rad(t_s)) * np.cos(np.deg2rad(t_0)) ) - (np.sin(np.deg2rad(t_s)) * np.sin(np.deg2rad(t_0)) * np.cos(
            np.deg2rad(phi))))

    return g_p




def phase_function(b1,b2,b3,b4,t_s, t_0, sensor_azimuth, solar_azimuth):
    """
    Calculates the phase function.

    Args:
        b1 (float): B1 coefficient.
        b2 (float): B2 coefficient.
        b3 (float): B3 coefficient.
        b4 (float): B4 coefficient.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.

    Returns:
        float: Phase function value.
    """
    

    g1 = G1(t_s, t_0, sensor_azimuth, solar_azimuth)
    p1 = np.cos(g1)
    p2 = (1/2) * (3 * (np.cos(g1)) ** 2 - 1)
    p3 = (1/2) * (5 * (np.cos(g1)) ** 3 - (3 * np.cos(g1)))
    p4 = (1/8) * (     (35 * ((np.cos(g1)) )** 4)    -     (30 * (np.cos(g1))**2)   +   3 )
    phase_func = 1 + b1*p1 + b2*p2 + b3*p3 + b4*p4
    
    return phase_func



def B_g(B0, t_s, t_0, sensor_azimuth, solar_azimuth, fillFactor, C):

    """
    Calculates the SHOE function.

    Args:
        B0 (float): Intensity size of the hot spot effect.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        fillFactor (float): Porosity factor.
        C (float): Parameter for SHOE function.

    Returns:
        float: SHOE value.
    """
    

    g1 = G1(t_s, t_0, sensor_azimuth, solar_azimuth)
    K = calc_porosity_factor(fillFactor)

    h = C * K * fillFactor

    B = B0/(1 + ((np.tan(g1/2))/h))
    return B




def k(b1,c1,b2,c2,t_s, t_0, B0, sensor_azimuth, solar_azimuth,  fillFactor, C):
    """
    Calculates the K value for reflectance calculation.

    Args:
        b1 (float): B coefficient.
        c1 (float): C coefficient.
        b2 (float): B' coefficient.
        c2 (float): C' coefficient.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        B0 (float): Intensity size of the hot spot effect.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        fillFactor (float): Fill factor of the porous medium.
        C (float): Parameter for SHOE function.

    Returns:
        float: K value for reflectance calculation.
    """
    b_g = B_g(B0, t_s, t_0, sensor_azimuth, solar_azimuth, fillFactor, C)
    p = phase_function(b1,c1,b2,c2,t_s, t_0, sensor_azimuth, solar_azimuth)
    K = (p * (1 + b_g))
    return K

def calc_refl(B0, t_s, t_0, sensor_azimuth, solar_azimuth, w0, b1,c1,b2,c2, fillFactor, C):
    """
    Calculates the reflectance using the Modified (SWAP)-Hapke model.

    Args:
        B0 (float): Intensity size of the hot spot effect.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        w0 (float): Single scattering albedo.
        b1 (float): B coefficient.
        c1 (float): C coefficient.
        b2 (float): B' coefficient.
        c2 (float): C' coefficient.
        fillFactor (float): Porosity factor.
        C (float): Parameter for SHOE function.

    Returns:
        float: Dry soil reflectance value using the Modified (SWAP)-Hapke model.
    """



    x_s = np.cos(t_s * (np.pi/180))
    x_0 = np.cos(t_0 * (np.pi/180))

    K = calc_porosity_factor(fillFactor)

    m = k(b1,c1,b2,c2,t_s, t_0, B0, sensor_azimuth, solar_azimuth, fillFactor, C)
    H_s = hapke(w0, x_s, K)
    H_0 = hapke(w0, x_0, K)
    r_hapke = K * (w0/4) * (1/(x_s + x_0)) * (m + ((H_s * H_0) - 1))
    return r_hapke


def calc_refl_2(alpha,
                L,
                r_hapke,
                epsilon):

    """
    Calculates the final reflectance using the Modified (SWAP)-Hapke model.

    Args:
        alpha (float): Absorption coefficient.
        L (float): Equivalent water thickness.
        r_hapke (float): Dry soil reflectance value using the Modified (SWAP)-Hapke model.
        epsilon (float): Proportion of wet component.

    Returns:
        float: Wet soil reflectance value using the Modified (SWAP)-Hapke model.
    """


    Tw = np.exp(-alpha * L)

    r_wet = r_hapke * Tw
    r_final = epsilon * r_wet + (1 - epsilon) * r_hapke


    return r_final



