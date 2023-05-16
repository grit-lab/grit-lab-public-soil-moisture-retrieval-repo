from hapke_model import calc_refl
from hapke_model import calc_refl_2
import lmfit
import numpy as np



def perform_inversion_dry(
                      refl_dry,
                      solar_zenith,
                      sensor_zenith,
                      sensor_azimuth,
                      solar_azimuth,
                      wavelength):
    """Performs inversion on dry reflectance data.

    Args:
        refl_dry (numpy.ndarray): Array of dry soil reflectance values.
        solar_zenith (float): Solar zenith angle.
        sensor_zenith (float): Sensor zenith angle.
        sensor_azimuth (float): Sensor azimuth angle.
        solar_azimuth (float): Solar azimuth angle.
        wavelength (numpy.ndarray): Array of wavelengths.

    Returns:
        tuple: Returns tuples of parameter arrays for w, B, b1, b2, b3, b4, fill, and C.
    """

    # Define parameters
    Params = lmfit.Parameters()
    Params.add('w', min=0.01, max=1)
    Params.add('B', min=0.01, max=1)
    Params.add('b1', min=-2, max=2)
    Params.add('b2', min=-2, max=2)
    Params.add('b3', min=-2, max=2)
    Params.add('b4', min=-2, max=2)
    Params.add('fill', min=0.01, max=0.752)
    Params.add('C', min=0.1, max=1)

    # Initialize arrays for parameters
    w = np.zeros(len(wavelength))
    B = np.zeros(len(wavelength))
    b1 = np.zeros(len(wavelength))
    b2 = np.zeros(len(wavelength))
    b3 = np.zeros(len(wavelength))
    b4 = np.zeros(len(wavelength))
    fill = np.zeros(len(wavelength))
    C = np.zeros(len(wavelength))

    for k in range(0,len(wavelength)):

        def residual(Params):

            """Calculate residual between observed and estimated dry reflectance.

            Args:
                Params (lmfit.Parameters): Parameters for the Hapke model.

            Returns:
                float: The residual.
            """

            refl_est_dry = calc_refl(Params['B'], solar_zenith, sensor_zenith, sensor_azimuth, solar_azimuth, Params['w'],  Params['b1'], Params['b2'], Params['b3'] , Params['b4'], Params['fill'], Params['C'])
            res = abs(refl_dry[k] - refl_est_dry) ** 2

            return res


        mini = lmfit.Minimizer(residual, Params,nan_policy='omit')
        mi = mini.minimize(method ='differential_evolution')

        p = (mi.params)
        w[k] = p['w']._val
        B[k] = p['B']._val
        b1[k] = p['b1']._val
        b2[k] = p['b2']._val
        b3[k] = p['b3']._val
        b4[k] = p['b4']._val
        fill[k] = p['fill']._val
        C[k] = p['C']._val

    return w, B, b1, b2, b3, b4, fill, C


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
                      b2,
                      b3,
                      b4,
                      fill,
                      C):
    """Performs inversion on wet reflectance data.

    Args:
        refl_meas_wet (numpy.ndarray): Array of measured wet reflectance values.
        solar_zenith (float): Solar zenith angle.
        sensor_zenith (float): Sensor zenith angle.
        sensor_azimuth (float): Sensor azimuth angle.
        solar_azimuth (float): Solar azimuth angle.
        wavelength (numpy.ndarray): Array of wavelengths.
        alpha, w, B, b1, c1, b2, c2, fill, C (numpy arrays): Parameters for the Modified (SWAP)-Hapke model.

    Returns:
        tuple: Returns tuples of parameter arrays for L and epsilon.
    """


    Params = lmfit.Parameters()
    Params.add('L', min=0.01, max=2)
    Params.add('epsilon', min=0.01, max=1)


    L = np.zeros(len(wavelength))
    epsilon = np.zeros(len(wavelength))

    for k in range(0, len(wavelength)):
        def residual(Params):
            """Calculate residual between observed and estimated wet reflectance.

            Args:
                Params (lmfit.Parameters): Parameters for the Hapke model.

            Returns:
                float: The residual.
            """

            refl_est_dry = calc_refl(B[k], solar_zenith, sensor_zenith, sensor_azimuth, solar_azimuth,
                                     w[k], b1[k], b2[k], b3[k], b4[k], fill[k], C[k])

            refl_est_wet = calc_refl_2(alpha[k],Params['L'], refl_est_dry,Params['epsilon'])

            res = abs(refl_meas_wet[k] - refl_est_wet) ** 2

            return res

        mini = lmfit.Minimizer(residual, Params, nan_policy='omit')
        mi = mini.minimize(method='nelder')
        p = (mi.params)

        L[k] = p['L']._val
        epsilon[k] = p['epsilon']._val


    return  L, epsilon




