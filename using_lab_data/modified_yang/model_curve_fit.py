from anc_functions import get_smc_data
from anc_functions import get_L_eps_data
from anc_functions import logistic_function
from anc_functions import coefficient_of_determination
from lmfit import Model
import os
import numpy as np

def get_logistic_function_params(log_output, mwt_output, smc_input):
    """
    This function calculates the parameters of the logistic function and saves them in csv files.

    Args:
        log_output (str): The output string where the results of the logistic function will be saved.
        mwt_output (str): The path to the output directory of the Mie scattering simulations.
        smc_input (array_like): Array of soil moisture content values.

    Returns:
        int: A value of 0 is returned after successful execution of the function.

    Notes:
        This function assumes that the input directories contain specific csv files with parameters of
        Modified (SWAP)-Hapke model. The function reads these csv files, performs calculations, fits the logistic function
        to the soil moisture content data, and saves the parameters of the logistic function (K, psi, alpha)
        and the coefficient of determination (r2) in separate csv files in the output directory.
    """
    results_folder = log_output[0:34]
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    path, dirs, files = next(os.walk(mwt_output))

    csv_w = [s for s in files if s.endswith('single_scattering_albedo.csv')]
    csv_B = [s for s in files if s.endswith('intensity_size_of_h.csv')]
    csv_b1 = [s for s in files if s.endswith('coefficient_b1.csv')]
    csv_b2 = [s for s in files if s.endswith('coefficient_b2.csv')]
    csv_b3 = [s for s in files if s.endswith('coefficient_b3.csv')]
    csv_b4 = [s for s in files if s.endswith('coefficient_b4.csv')]
    csv_L = [s for s in files if s.endswith('water_level.csv')]
    csv_epsilon = [s for s in files if s.endswith('efficiency.csv')]
    csv_fill = [s for s in files if s.endswith('fill.csv')]
    csv_C = [s for s in files if s.endswith('C.csv')]


    num1 = np.zeros(len(csv_L))
    num3 = np.zeros(len(csv_L))
    num4 = np.zeros(len(csv_L))
    num5 = np.zeros(len(csv_L))
    num6 = np.zeros(len(csv_L))
    num7 = np.zeros(len(csv_L))
    num8 = np.zeros(len(csv_L))
    num9 = np.zeros(len(csv_L))
    num10 = np.zeros(len(csv_L))
    num11 = np.zeros(len(csv_L))

    for i in range(0, len(csv_L)):
        num1[i] = int(csv_w[i].rsplit('run')[-1].rsplit('_')[0])
        num3[i] = int(csv_B[i].rsplit('run')[-1].rsplit('_')[0])
        num4[i] = int(csv_b1[i].rsplit('run')[-1].rsplit('_')[0])
        num5[i] = int(csv_b2[i].rsplit('run')[-1].rsplit('_')[0])
        num6[i] = int(csv_b3[i].rsplit('run')[-1].rsplit('_')[0])
        num7[i] = int(csv_b4[i].rsplit('run')[-1].rsplit('_')[0])
        num8[i] = int(csv_L[i].rsplit('run')[-1].rsplit('_')[0])
        num9[i] = int(csv_epsilon[i].rsplit('run')[-1].rsplit('_')[0])
        num10[i] = int(csv_fill[i].rsplit('run')[-1].rsplit('_')[0])
        num11[i] = int(csv_C[i].rsplit('run')[-1].rsplit('_')[0])


    num_argsort1 = np.argsort(num1)
    num_argsort3 = np.argsort(num3)
    num_argsort4 = np.argsort(num4)
    num_argsort5 = np.argsort(num5)
    num_argsort6 = np.argsort(num6)
    num_argsort7 = np.argsort(num7)
    num_argsort8 = np.argsort(num8)
    num_argsort9 = np.argsort(num9)
    num_argsort10 = np.argsort(num10)
    num_argsort11 = np.argsort(num11)

    smc = get_smc_data(smc_input)


    dat = get_L_eps_data(path + csv_L[0])
    w = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    B = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b1 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b2 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b3 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b4 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    L = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    epsilon = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    fill = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    C = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))


    for i in range(0, len(csv_L)):
        w[:, :, i] = get_L_eps_data(path + csv_w[num_argsort1[i]])
        B[:, :, i] = get_L_eps_data(path + csv_B[num_argsort3[i]])
        b1[:, :, i] = get_L_eps_data(path + csv_b1[num_argsort4[i]])
        b2[:, :, i] = get_L_eps_data(path + csv_b2[num_argsort5[i]])
        b3[:, :, i] = get_L_eps_data(path + csv_b3[num_argsort6[i]])
        b4[:, :, i] = get_L_eps_data(path + csv_b4[num_argsort7[i]])
        L[:, :, i] = get_L_eps_data(path + csv_L[num_argsort8[i]])
        epsilon[:, :, i] = get_L_eps_data(path + csv_epsilon[num_argsort9[i]])
        fill[:, :, i] = get_L_eps_data(path + csv_fill[num_argsort10[i]])
        C[:, :, i] = get_L_eps_data(path + csv_C[num_argsort11[i]])


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

