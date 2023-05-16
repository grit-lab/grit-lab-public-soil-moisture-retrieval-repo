from anc_functions import *
from lmfit import Model
import os
import numpy as np

def get_logistic_function_params(log_output,mwt_output,smc_input):
    """
    Extract logistic function parameters from model outputs and soil moisture content (SMC) data.

    This function fits a logistic function to the SMC data for each run and wavelength.
    The fitted parameters (K, psi, alpha) and the coefficient of determination (R^2) are saved as CSV files.
    Directories are created if they do not exist.

    Args:
        log_output (str): Path to the directory where the output will be saved.
        mwt_output (str): Path to the directory where the mean water thickness data is stored.
        smc_input (str): Path to the SMC data CSV file.

    Returns:
        int: 0, indicating the function has successfully completed its operation.
    """
    
    results_folder = log_output[0:34]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    path, dirs, files = next(os.walk(mwt_output))

    csv_L = [s for s in files if s.endswith('water_level.csv')]
    csv_epsilon = [s for s in files if s.endswith('efficiency.csv')]
    
    num1 = np.zeros(len(csv_L))
    num2 = np.zeros(len(csv_L))
    for i in range(0,len(csv_L)):
        num1[i] = int(csv_L[i].rsplit('run')[-1].rsplit('_')[0])
        num2[i] = int(csv_epsilon[i].rsplit('run')[-1].rsplit('_')[0])
    num_argsort1 = np.argsort(num1)
    num_argsort2 = np.argsort(num2)
    
    smc = get_smc_data(smc_input)
    
    dat = get_L_eps_data(path + csv_L[0])
    L = np.zeros((dat.shape[0],dat.shape[1],len(csv_L)))
    epsilon = np.zeros((dat.shape[0],dat.shape[1],len(csv_L)))
    for i in range(0,len(csv_L)):
        L[:,:,i] = get_L_eps_data(path + csv_L[num_argsort1[i]])
        epsilon[:,:,i] = get_L_eps_data(path + csv_epsilon[num_argsort2[i]])
    mean_water_thickness = L * epsilon
    
    wavelength = np.linspace(350,2500,2151)
    
    K = np.zeros((L.shape[0],L.shape[1]))
    psi = np.zeros((L.shape[0],L.shape[1]))
    alpha = np.zeros((L.shape[0],L.shape[1]))
    r2_val = np.zeros((L.shape[0],L.shape[1]))
    
    for j in range(0,L.shape[0]):
        for i in range(0,len(wavelength)):
            gmodel = Model(logistic_function)
            result = gmodel.fit(smc, phi=mean_water_thickness[j,i,:], K=1,psi=1,alpha=1)
            K[j,i] = result.best_values['K']
            psi[j,i] = result.best_values['psi']
            alpha[j,i] = result.best_values['alpha']
            r2_val[j,i] = coefficient_of_determination(smc,result.best_fit)
                
    np.savetxt(log_output + '_K.csv',K,delimiter=',')
    np.savetxt(log_output + '_psi.csv',psi,delimiter=',')
    np.savetxt(log_output + '_alpha.csv',alpha,delimiter=',')
    np.savetxt(log_output + '_r2.csv',r2_val,delimiter=',')
    
    return 0

