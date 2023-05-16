from marmit_model import *
import lmfit
import numpy as np

def perform_inversion(refl_meas_wet,
                      refl_dry,
                      solar_zenith,
                      wavelength,
                      alpha,
                      n):
    """
    Perform inversion of the MARMIT model for given input parameters.

    Args:
        refl_meas_wet (array): Array of measured wet soil reflectance values.
        refl_dry (array): Array of dry soil reflectance values.
        solar_zenith (array): Array of solar zenith angles in degrees.
        wavelength (array): Array of wavelength values.
        alpha (array): Array of absorption coefficients.
        n (array): Array of refractive indices.

    Returns:
        tuple: Arrays of equivalent water thickness (L) and wet surface fraction (epsilon) values.
    """

    Params = lmfit.Parameters()
    Params.add('L', min=0.01, max=2)
    Params.add('epsilon', min=0.01, max=1)

    L = np.zeros(len(wavelength))
    epsilon = np.zeros(len(wavelength))

    for k in range(0,len(wavelength)):

        def residual(Params):

            refl_est = calc_refl(alpha[k],Params['L'],n[k],solar_zenith[k],refl_dry[k],Params['epsilon'])

            res = abs(refl_meas_wet[k] - refl_est) ** 2

            return res

        mini = lmfit.Minimizer(residual, Params,nan_policy='omit')
        mi = mini.minimize(method='nelder')
        p = (mi.params)

        L[k] = p['L']._val
        epsilon[k] = p['epsilon']._val



    return L,epsilon



