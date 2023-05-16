
import numpy as np



def hapke_simplified(w, x):

    """
    Calculates the H function using IMSA 4% error model.

    Args:
        w (float): Single scattering albedo.
        x (float): Angle, could be solar zenith or view zenith.

    Returns:
        float: H function value.
    """
    nom = 1 + 2*x
    den = 1 + (2 * x *np.sqrt(1 - w))
    h = nom/den
    return h

    


def G1(t_s, t_0, sensor_azimuth, solar_azimuth):
    """
    Calculates G1, the angle between the solar and viewing directions.

    Args:
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.

    Returns:
        float: G1 value.
    """
    phi = sensor_azimuth - solar_azimuth

    g = np.arccos(
        ( np.cos(np.deg2rad(t_s)) * np.cos(np.deg2rad(t_0)) ) + ( np.sin(np.deg2rad(t_s)) * np.sin(np.deg2rad(t_0)) * np.cos(
            np.deg2rad(phi))))

    return g


def G2(t_s, t_0, sensor_azimuth, solar_azimuth):
    """
    Calculates G2, the angle between the solar and viewing directions with reversed signs.

    Args:
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.

    Returns:
        float: G2 value.
    """

    phi = sensor_azimuth - solar_azimuth

    g_p = np.arccos(
        (np.cos(np.deg2rad(t_s)) * np.cos(np.deg2rad(t_0)) ) - (np.sin(np.deg2rad(t_s)) * np.sin(np.deg2rad(t_0)) * np.cos(
            np.deg2rad(phi))))

    return g_p



def phase_function(b1,c1,b2,c2,t_s, t_0, sensor_azimuth, solar_azimuth):
    """
    Calculates the phase function.

    Args:
        b1 (float): B coefficient.
        c1 (float): C coefficient.
        b2 (float): B' coefficient.
        c2 (float): C' coefficient.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.

    Returns:
        float: Phase function value.
    """
    

    g1 = G1(t_s, t_0, sensor_azimuth, solar_azimuth)
    g2 = G2(t_s, t_0, sensor_azimuth, solar_azimuth)
    
    phase_func = 1 + (b1 * np.cos(g1)) + ((c1/2) * ((3*(np.cos(g1))**2) -1) ) 
    + (b2 * np.cos(g2)) + ((c2/2) * ((3*(np.cos(g2))**2) -1) )
    
    return phase_func



def B_g(B0, t_s, t_0, sensor_azimuth, solar_azimuth, h):
    
    """
    Calculates the SHOE function.

    Args:
        B0 (float): Intensity size of the hot spot effect.
        t_s (float): Solar zenith angle in degrees.
        t_0 (float): Viewing zenith angle in degrees.
        sensor_azimuth (float): Sensor azimuth angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        h (float): Hot spot parameter.

    Returns:
        float: SHOE value.
    """
    

    g1 = G1(t_s, t_0, sensor_azimuth, solar_azimuth)


    B = B0/(1 + ((np.tan(g1/2))/h))
    return B




def k(b1,c1,b2,c2,t_s, t_0, B0, sensor_azimuth, solar_azimuth,  h):
    """
    Calculates the K function.

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
        h (float): Hot spot parameter.

    Returns:
        float: K function value.
    """
    b_g = B_g(B0, t_s, t_0, sensor_azimuth, solar_azimuth, h)
    p = phase_function(b1,c1,b2,c2,t_s, t_0, sensor_azimuth, solar_azimuth)
    K = (p * (1 + b_g))
    return K

def calc_refl(B0, t_s, t_0, sensor_azimuth, solar_azimuth, w0, b1,c1,b2,c2, h):
    """
        Calculates the reflectance using the Original (SWAP)-Hapke model.

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
            h (float): Hot spot parameter.

        Returns:
            float: Dry soil reflectance value using the Original (SWAP)-Hapke model.
        """


    x_s = np.cos(t_s * (np.pi/180))
    x_0 = np.cos(t_0 * (np.pi/180))


    m = k(b1,c1,b2,c2,t_s, t_0, B0, sensor_azimuth, solar_azimuth, h)
    H_s = hapke_simplified(w0, x_s)
    H_0 = hapke_simplified(w0, x_0)
    r_hapke =  (w0/4) * (1/(x_s + x_0)) * (m + ((H_s * H_0) - 1))
    return r_hapke


def calc_refl_2(alpha,
                L,
                r_hapke):
    """
    Calculates the wet reflectance using the Original (SWAP)-Hapke model.

    Args:
        alpha (float): Absorption coefficient.
        L (float):  Equivalent water thickness.
        r_hapke (float): Dry soil reflectance value using the Original (SWAP)-Hapke model.

    Returns:
        float: Wet soil reflectance value using the Original (SWAP)-Hapke model.
    """


    Tw = np.exp(-alpha * L)

    r_wet = r_hapke * Tw


    return r_wet


