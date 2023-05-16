from hapke_model import *
import lmfit
import numpy as np

def perform_inversion_dry(
                      refl_dry,
                      solar_zenith,
                      sensor_zenith,
                      sensor_azimuth,
                      solar_azimuth,
                      wavelength):
    """
        Perform dry soil inversion using the Modified (SWAP)-Hapke model.

        Args:
            refl_dry (numpy array): Dry soil reflectance values.
            solar_zenith (float): Solar zenith angle.
            sensor_zenith (float): Sensor zenith angle.
            sensor_azimuth (float): Sensor azimuth angle.
            solar_azimuth (float): Solar azimuth angle.
            wavelength (numpy array): Wavelengths corresponding to the reflectance values.

        Returns:
            w, B, b1, c1, b2, c2, fill, C (numpy arrays): Parameters for the Modified (SWAP)-Hapke model.
        """

    Params = lmfit.Parameters()
    Params.add('w', min = 0.01, max = 1)
    Params.add('B', min = 0.01, max = 1)
    Params.add('b1', min = -2, max = 2)
    Params.add('c1', min = -2, max = 2)
    Params.add('b2', min = -2, max = 2)
    Params.add('c2', min = -2, max = 2)
    Params.add('fill', min = 0.01, max = 0.752)
    Params.add('C', min = 0.01, max = 1)


    
    w = np.zeros(len(wavelength))
    B = np.zeros(len(wavelength))
    b1 = np.zeros(len(wavelength))
    c1 = np.zeros(len(wavelength))
    b2 = np.zeros(len(wavelength))
    c2 = np.zeros(len(wavelength))
    fill = np.zeros(len(wavelength))
    C = np.zeros(len(wavelength))



    for k in range(0,len(wavelength)):

        def residual(Params):


            refl_est_dry = calc_refl(Params['B'], solar_zenith, sensor_zenith, sensor_azimuth, solar_azimuth, Params['w'], Params['b1'], Params['c1'], Params['b2'] , Params['c2'], Params['fill'], Params['C'])
            res = abs(refl_dry[k] - refl_est_dry) ** 2

            return res


        mini = lmfit.Minimizer(residual, Params,nan_policy='omit')
        mi = mini.minimize(method ='differential_evolution')

        p = (mi.params)

        w[k] = p['w']._val
        B[k] = p['B']._val
        b1[k] = p['b1']._val
        c1[k] = p['c1']._val
        b2[k] = p['b2']._val
        c2[k] = p['c2']._val
        fill[k] = p['fill']._val
        C[k] = p['C']._val

    return w, B, b1, c1, b2, c2, fill, C


def perform_inversion_wet(refl_meas_wet,
                      solar_zenith,
                      sensor_zenith,
                      sensor_azimuth,
                      solar_azimuth,
                      wavelength,
                      alpha,
                      w,
                      B,
                      b1,
                      c1,
                      b2,
                      c2,
                      fill,
                      C):
    """
        Perform wet soil inversion using the Modified (SWAP)-Hapke model.

        Args:
            refl_meas_wet (numpy array): Measured wet soil reflectance values.
            solar_zenith (float): Solar zenith angle.
            sensor_zenith (float): Sensor zenith angle.
            sensor_azimuth (float): Sensor azimuth angle.
            solar_azimuth (float): Solar azimuth angle.
            wavelength (numpy array): Wavelengths corresponding to the reflectance values.
            alpha, w, B, b1, c1, b2, c2, fill, C (numpy arrays): Parameters for the Modified (SWAP)-Hapke model.

        Returns:
            L, epsilon (numpy arrays): Parameters for the Modified (SWAP)-Hapke model.
        """



    Params = lmfit.Parameters()
    Params.add('L', min=0.01, max=2)
    Params.add('epsilon', min=0.01, max=1)

    L = np.zeros(len(wavelength))
    epsilon = np.zeros(len(wavelength))

    for k in range(0, len(wavelength)):
        def residual(Params):

            refl_est_dry = calc_refl(B[k], solar_zenith[k], sensor_zenith[k], sensor_azimuth[k], solar_azimuth[k],
                                     w[k],b1[k],c1[k],b2[k],c2[k],fill[k], C[k])
            refl_est = calc_refl_2(alpha[k], Params['L'], refl_est_dry,  Params['epsilon'])
            res = abs(refl_meas_wet[k] - refl_est) ** 2

            return res

        mini = lmfit.Minimizer(residual, Params,nan_policy='omit')
        mi = mini.minimize(method='nelder')
        p = (mi.params)

        L[k] = p['L']._val
        epsilon[k] = p['epsilon']._val

    return L, epsilon







