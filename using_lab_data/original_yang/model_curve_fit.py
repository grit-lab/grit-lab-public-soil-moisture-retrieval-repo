from anc_functions import get_smc_data
from anc_functions import get_L_eps_data
from anc_functions import logistic_function
from anc_functions import coefficient_of_determination
from lmfit import Model
import os
import numpy as np

def get_logistic_function_params(log_output, mwt_output, smc_input):
    """
    Calculate logistic function parameters, write results to .csv files, and return 0.

    This function takes in paths to the output files, reads data from various .csv files, performs calculations
    and logistic model fitting, and writes the results to .csv files.

    Args:
        log_output (str): The path to the output directory for the logistic parameters.
        mwt_output (str): The path to the output directory containing the .csv files.
        smc_input (str): The path to the input file containing soil moisture content data.

    Returns:
        int: 0, indicating that the function has finished executing.
    """

    results_folder = log_output[0:34]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    path, dirs, files = next(os.walk(mwt_output))

    csv_w = [s for s in files if s.endswith('single_scattering_albedo.csv')]
    csv_h = [s for s in files if s.endswith('hot_spot_effect.csv')]
    csv_B = [s for s in files if s.endswith('intensity_size_of_h.csv')]
    csv_b1 = [s for s in files if s.endswith('coefficient_b1.csv')]
    csv_b2 = [s for s in files if s.endswith('coefficient_b2.csv')]
    csv_c1 = [s for s in files if s.endswith('coefficient_c1.csv')]
    csv_c2 = [s for s in files if s.endswith('coefficient_c2.csv')]
    csv_L = [s for s in files if s.endswith('water_level.csv')]


    num1 = np.zeros(len(csv_L))
    num2 = np.zeros(len(csv_L))
    num3 = np.zeros(len(csv_L))
    num4 = np.zeros(len(csv_L))
    num5 = np.zeros(len(csv_L))
    num6 = np.zeros(len(csv_L))
    num7 = np.zeros(len(csv_L))
    num8 = np.zeros(len(csv_L))


    for i in range(0, len(csv_L)):
        num1[i] = int(csv_w[i].rsplit('run')[-1].rsplit('_')[0])
        num2[i] = int(csv_h[i].rsplit('run')[-1].rsplit('_')[0])
        num3[i] = int(csv_B[i].rsplit('run')[-1].rsplit('_')[0])
        num4[i] = int(csv_b1[i].rsplit('run')[-1].rsplit('_')[0])
        num5[i] = int(csv_b2[i].rsplit('run')[-1].rsplit('_')[0])
        num6[i] = int(csv_c1[i].rsplit('run')[-1].rsplit('_')[0])
        num7[i] = int(csv_c2[i].rsplit('run')[-1].rsplit('_')[0])
        num8[i] = int(csv_L[i].rsplit('run')[-1].rsplit('_')[0])


    num_argsort1 = np.argsort(num1)
    num_argsort2 = np.argsort(num2)
    num_argsort3 = np.argsort(num3)
    num_argsort4 = np.argsort(num4)
    num_argsort5 = np.argsort(num5)
    num_argsort6 = np.argsort(num6)
    num_argsort7 = np.argsort(num7)
    num_argsort8 = np.argsort(num8)


    smc = get_smc_data(smc_input)

    dat = get_L_eps_data(path + csv_L[0])
    w = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    h = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    B = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b1 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b2 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    c1 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    c2 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    L = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))


    for i in range(0, len(csv_L)):
        w[:, :, i] = get_L_eps_data(path + csv_w[num_argsort1[i]])
        h[:, :, i] = get_L_eps_data(path + csv_h[num_argsort2[i]])
        B[:, :, i] = get_L_eps_data(path + csv_B[num_argsort3[i]])
        b1[:, :, i] = get_L_eps_data(path + csv_b1[num_argsort4[i]])
        b2[:, :, i] = get_L_eps_data(path + csv_b2[num_argsort5[i]])
        c1[:, :, i] = get_L_eps_data(path + csv_c1[num_argsort6[i]])
        c2[:, :, i] = get_L_eps_data(path + csv_c2[num_argsort7[i]])
        L[:, :, i] = get_L_eps_data(path + csv_L[num_argsort8[i]])


    mean_water_thickness = L

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

