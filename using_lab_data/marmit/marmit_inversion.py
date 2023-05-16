from marmit_model import calc_refl
import lmfit
import numpy as np

def perform_inversion(refl_meas_wet,
                      refl_dry,
                      solar_zenith,
                      wavelength,
                      alpha,
                      n):
    """
    Perform inversion on the given inputs to estimate the equivalent water thickness (L) and surface coverage fraction of water (epsilon).

    This function performs a minimization process to minimize the residual between measured reflectance and estimated reflectance.
    The estimated reflectance is calculated using the calc_refl function from the MARMIT model.

    Args:
        refl_meas_wet (numpy array): Array of measured wet soil reflectance values.
        refl_dry (numpy array): Array of measured dry soil reflectance values.
        solar_zenith (numpy array): Array of solar zenith angles.
        wavelength (numpy array): Array of wavelengths.
        alpha (numpy array): Array of absorption coefficients.
        n (numpy array): Array of refractive indices.

    Returns:
        tuple: A tuple containing:
            - L (numpy array): Array of estimated equivalent water thickness values.
            - epsilon (numpy array): Array of estimated surface coverage fraction of water values.
    """

    Params = lmfit.Parameters()
    Params.add('L', min=0.01, max=2)
    Params.add('epsilon', min=0.01, max=1)

    L = np.zeros(len(wavelength))
    epsilon = np.zeros(len(wavelength))

    for k in range(0,len(wavelength)):

        def residual(Params):

            refl_est = calc_refl(alpha[k],Params['L'],n[k],solar_zenith[0],refl_dry[k],Params['epsilon'])
            res = abs(refl_meas_wet[k] - refl_est) ** 2

            return res

        mini = lmfit.Minimizer(residual, Params)
        mi = mini.minimize(method='nelder')
        p = (mi.params)

        L[k] = p['L']._val
        epsilon[k] = p['epsilon']._val

    return L,epsilon
