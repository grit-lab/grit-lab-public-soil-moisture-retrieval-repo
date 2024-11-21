from read_var import *
from marmit_inversion import *
from anc_functions import *
from scipy.stats import norm
import scipy
import csv
import os
import numpy as np
import multiprocess
import matplotlib.pyplot as plt
from sklearn.utils import resample
from matplotlib import colors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from prettytable import PrettyTable
from lmfit import Model
import statistics
import time


""" Initialization of the data"""
start_time1 = time.time()



# Create a dictionary for data structure
csv_file = '../input/reflectance_and_geometry/uas_sample_list.csv'
smc_input = '../input/smc_data/uas_sample.csv'
abs_nw = '../input/abs_coeff_refran_index/nw_abs_grit.csv'

mwt_output = './outputs/anc_files/mwt_data/'
img_output = './outputs/plots'


data_struct = {
        'csv_file': csv_file,
        'smc_input': smc_input,
        'mwt_output': mwt_output,
        'img_output': img_output,
        'abs_nw': abs_nw
        }

# Read in csv file list
rd = open(data_struct['csv_file'], 'r')
csv_reader = csv.reader(rd)
data_list = list(csv_reader)
csvfile_dry = data_list[0][0]
csvfiles_wet = data_list[1:]





def train_dat(wvl,smc1,smc2,mean_water_thickness1,mean_water_thickness2):
    """Fit a logistic function to the data.

    Args:
        wvl (numpy.ndarray): Array of wavelength values.
        smc1 (numpy.ndarray): SMC values for the training data.
        smc2 (numpy.ndarray): SMC values for the testing data.
        mean_water_thickness1 (numpy.ndarray): Mean water thickness values for the training data.
        mean_water_thickness2 (numpy.ndarray): Mean water thickness values for the testing data.

    Returns:
        tuple: A tuple containing the best fit parameters and related values.
    """

    K = np.zeros(len(wvl))
    psi = np.zeros(len(wvl))
    alpha = np.zeros(len(wvl))
    r2_val = np.zeros(len(wvl))

    for i in range(0,len(wvl)):

        gmodel = Model(logistic_function)
        params = gmodel.make_params()

        params['K'].set(22, min=20, max=25)
        params['psi'].set(1,min=1,max=10)
        params['alpha'].set(1,min=1,max=10)

        result = gmodel.fit(smc1, phi=mean_water_thickness1[:,i], params=params)
        K[i] = result.best_values['K']
        psi[i] = result.best_values['psi']
        alpha[i] = result.best_values['alpha']
        r2_val[i] = coefficient_of_determination(smc1,result.best_fit)
        

    r2_ind = int(np.argwhere(r2_val == np.max(r2_val)).flatten())
    K_bv = float(K[r2_ind])
    psi_bv = float(psi[r2_ind])
    alpha_bv = float(alpha[r2_ind])
    wvl_bv = float(wvl[r2_ind])
    print("Best fit was observed in wavelength  ", wvl_bv)

    smc_est = logistic_function(mean_water_thickness2[:,r2_ind], K_bv, psi_bv, alpha_bv)
    r2_val = coefficient_of_determination(smc2, smc_est.flatten())
    nrmse = sqrt(mean_squared_error(smc2, smc_est.flatten())) / np.mean(smc2)

    return r2_ind,r2_val,nrmse,K,psi,alpha,wvl_bv

def find_nearest(a, a0):

    """Find the closest element in an array to a given scalar value.

    Args:
        a (numpy.ndarray): Input array.
        a0 (float): Scalar value.

    Returns:
        float: The closest element in the array to the scalar value.
    """
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def get_logistic_function_params(mwt_output,smc_input):

    """Obtain the logistic function parameters.

    Args:
        mwt_output (str): Path to the mean water thickness output folder.
        smc_input (str): Path to the SMC input data file.

    Returns:
        tuple: A tuple containing the logistic function parameters and related values.
    """
    
    path, dirs, files = next(os.walk(mwt_output))
    file_count = len(files)
    smc = get_smc_data(smc_input)
    
    wvl,_,_ = get_marmit_data(path + files[0])
    
    
    num = np.zeros(len(files))
    for i in range(0,len(files)):
        num[i] = int(files[i].rsplit('_run')[-1].rsplit('.')[0])
    num_argsort = np.argsort(num)
    
    L = np.zeros((len(smc), len(wvl)))
    epsilon = np.zeros((len(smc), len(wvl)))
    
    for i in range(0, file_count):
        wavelength, L[i, :], epsilon[i, :] = get_marmit_data(path + files[num_argsort[i]])
    mean_water_thickness = L * epsilon
    
    wvl_region1 = np.argwhere((wvl > 1000) & (wvl < 1350)).flatten()
    wvl_region2 = np.argwhere((wvl > 1435) & (wvl < 1781)).flatten()
    wvl_region3 = np.argwhere((wvl > 1982) & (wvl < 2450)).flatten()
    
    wvl = wvl[np.concatenate([wvl_region1,wvl_region2,wvl_region3]).flatten()]
    mean_water_thickness = mean_water_thickness[:,np.concatenate([wvl_region1,wvl_region2,wvl_region3]).flatten()]
    
    dat_ind = np.arange(0,mean_water_thickness.shape[0])
    
    # configure bootstrap
    n_iterations = 1000
    n_size = int(len(dat_ind) * 0.80)
    
    r2_final = list()
    nrmse_final = list()
    wvl_final = list()
    K_final = list()
    psi_final = list()
    alpha_final = list()
    r2_ind_final = list()
    test_dat = []
    
    for i in range(n_iterations):
        
        print('iteration no. = ', str(i))
        
        # prepare train and test sets
        train_ind = resample(dat_ind, n_samples=n_size)
        test_ind = np.array([x for x in dat_ind if x.tolist() not in train_ind.tolist()])
    
        mean_water_thickness_train = mean_water_thickness[train_ind,:]
        mean_water_thickness_test = mean_water_thickness[test_ind,:]
        smc_train = smc[train_ind]
        smc_test = smc[test_ind]
        r2_ind,r2_f,nrmse,K_f, psi_f,alpha_f,wvl_f = train_dat(wvl, smc_train, smc_test, mean_water_thickness_train, mean_water_thickness_test)
        r2_final.append(r2_f)
        nrmse_final.append(nrmse)
        wvl_final.append(wvl_f)
        K_final.append(K_f)
        psi_final.append(psi_f)
        alpha_final.append(alpha_f)
        r2_ind_final.append(r2_ind)
        test_dat.append(test_ind)
    wvl_m = statistics.mode(wvl_final)

    
    return wvl,mean_water_thickness,smc,r2_final,nrmse_final,wvl_final,K_final,psi_final,alpha_final,test_dat,wvl_m




def saveplots(wvl, mean_water_thickness, smc, r2_final, nrmse_final, wvl_final, K_final, psi_final, alpha_final,
              test_dat, img_output, wvl_m):

    """Save the output plots of the model.

        Args:
            wvl (numpy.ndarray): Array of wavelength values.
            mean_water_thickness (numpy.ndarray): Mean water thickness values.
            smc (numpy.ndarray): SMC values.
            r2_final (list): List of final R-squared values.
            nrmse_final (list): List of final NRMSE values.
            wvl_final (list): List of final wavelength values.
            K_final (list): List of final K values.
            psi_final (list): List of final psi values.
            alpha_final (list): List of final alpha values.
            test_dat (list): List of test data indices.
            img_output (str): Path to the output folder for plots.
            wvl_m (float): Mode of the best fit wavelengths.
        """

    # Create output directory if it doesn't exist

    if not os.path.exists(img_output):
        os.makedirs(img_output)

    r2_final = np.asarray(r2_final).flatten()
    nrmse_final = np.asarray(nrmse_final).flatten()
    indr2 = np.argwhere(r2_final > 0).flatten()
    mu = np.mean(r2_final[indr2])
    median_r2 = np.median(r2_final[indr2])
    max_r2 = np.max(r2_final[indr2])
    sigma = np.std(r2_final[indr2])

    wvl_final = np.asarray(wvl_final).flatten()

    ind_mean = int(np.argwhere((wvl_final == find_nearest(wvl_final, wvl_m)) & (
                r2_final == find_nearest(r2_final[np.argwhere((wvl_final == find_nearest(wvl_final, wvl_m))).flatten()],
                                         mu)))[0].flatten())
    ind_wvl = int(np.argwhere(wvl == wvl_final[ind_mean]).flatten())
    test_dat_ind = test_dat[ind_mean]
    mwt_final = mean_water_thickness[test_dat_ind, ind_wvl]
    Kv_final = K_final[ind_mean][ind_wvl]
    psiv_final = psi_final[ind_mean][ind_wvl]
    alphav_final = alpha_final[ind_mean][ind_wvl]




    # --------------------------------------- plot hist function ---------------------------------------#
    fig = plt.figure(tight_layout=True)

    murmse = np.mean(nrmse_final[indr2])
    sigmarmse = np.std(nrmse_final[indr2])
    medianrmse = np.median(nrmse_final[indr2])
    minrmse = np.min(nrmse_final[indr2])
    N, bins, patches = plt.hist(nrmse_final[indr2], bins=100, edgecolor='k', alpha=0.75, density=1)
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.plasma(norm(thisfrac))
        thispatch.set_facecolor(color)
    xmin, xmax = plt.xlim(min(nrmse_final[indr2]), 0.6)
    x = np.linspace(xmin, xmax, 100)
    yx = scipy.stats.norm.pdf(x, murmse, sigmarmse)
    plt.plot(x, yx, 'k--', linewidth=2)

    plt.text(0.4, 15,
             'Mean NRMSE = ' + str(np.round(murmse, 3)) + '\n' 'Median NRMSE = ' + str(np.round(medianrmse, 3)),
             fontsize=12)
    plt.xlabel('Normalized root mean square error (NRMSE)', fontsize=15)
    plt.ylabel('Probability Density (%)', fontsize=15)
    plt.xlim(min(nrmse_final[indr2]), 0.6)
    plt.ylim(0, 17)
    plt.tight_layout()
    fig.savefig(img_output + '/bootstrap_hist.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # --------------------------------------- plot hist function ---------------------------------------#





    # --------------------------------------- plot logistic function ---------------------------------------#
    phi_lin = np.linspace(min(mwt_final), max(mwt_final))
    y = logistic_function(phi_lin,
                          Kv_final,
                          psiv_final,
                          alphav_final)
    fig = plt.figure(tight_layout=True)
    plt.plot(mwt_final, smc[test_dat_ind], 'o', markersize=8, markerfacecolor='r', markeredgecolor='k')
    plt.plot(phi_lin, y, 'k-', linewidth=2)
    plt.xlabel('Mean Water Thickness (cm)', fontsize=15)
    plt.ylabel('Soil Moisture Content (%)', fontsize=15)
    plt.tight_layout()

    fig.savefig(img_output + '/bootstarp_smc_log_func.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # --------------------------------------- plot logistic function ---------------------------------------#






    # --------------------------------------- plot measured vs estimated smc ---------------------------------------#
    smc_lin = np.linspace(np.min(smc) - 2, np.max(smc) + 2)
    smc_est_plot = logistic_function(mwt_final,
                                     Kv_final,
                                     psiv_final,
                                     alphav_final)

    r2_plot = r2_score(smc[test_dat_ind], smc_est_plot)
    fig = plt.figure(tight_layout=True)
    plt.plot(smc_lin, smc_lin, 'k-', linewidth=1.5)
    plt.plot(smc[test_dat_ind], smc_est_plot, 'o', markersize=8, markerfacecolor='r', markeredgecolor='k')
    plt.text(0.5, 24, '$R^{2}$ = ' + str(np.round(r2_plot, 3)), fontsize=12)
    plt.xlabel('Measured SMC (%)', fontsize=15)
    plt.ylabel('Estimated SMC (%)', fontsize=15)
    plt.xlim([-1, 26])
    plt.ylim([-1, 26])
    plt.tight_layout()

    fig.savefig(img_output + '/bootstrap_smc_true_v_pred.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # --------------------------------------- plot measured vs estimated smc ---------------------------------------#






    # --------------------------------------- plot nrmse vs wvl ---------------------------------------#
    smc_est_rmse = logistic_function(mean_water_thickness[test_dat_ind, :], K_final[ind_mean], psi_final[ind_mean],
                                     alpha_final[ind_mean])
    rms = np.zeros(len(wvl))
    for i in range(len(wvl)):
        rms[i] = sqrt(mean_squared_error(smc[test_dat_ind], smc_est_rmse[:, i])) / np.mean(smc[test_dat_ind])
    fig = plt.figure(tight_layout=True)
    wvl_region1 = np.argwhere((wvl > 1000) & (wvl < 1350)).flatten()
    wvl_region2 = np.argwhere((wvl > 1435) & (wvl < 1781)).flatten()
    wvl_region3 = np.argwhere((wvl > 1982) & (wvl < 2450)).flatten()
    plt.plot(wvl[wvl_region1], rms[wvl_region1], 'k-', linewidth=2.5)
    plt.plot(wvl[wvl_region2], rms[wvl_region2], 'k-', linewidth=2.5)
    plt.plot(wvl[wvl_region3], rms[wvl_region3], 'k-', linewidth=2.5)
    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Normalized root mean square error (NRMSE)', fontsize=12)
    plt.xlim([950, 2500])
    # plt.ylim([np.min(rms[wvl_region1]) - 0.2,np.max(rms[wvl_region3]) +0.25])
    plt.tight_layout()

    fig.savefig(img_output + '/bootstrap_nrmse.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    # --------------------------------------- plot nrmse vs wvl ---------------------------------------#




    t = PrettyTable(['---+---', ' ---+---'])

    t.add_row(['Mean r2-val', str(np.round(mu, 3))])
    t.add_row(['Mean nrmse', str(np.round(murmse, 3))])
    t.add_row(['St dev. r2-val', str(np.round(sigma, 3))])
    t.add_row(['St dev. nrmse', str(np.round(sigmarmse, 3))])
    t.add_row(['Median r2-val', str(np.round(median_r2, 3))])
    t.add_row(['Median nrmse', str(np.round(medianrmse, 3))])
    t.add_row(['Max r2-val', str(np.round(max_r2, 3))])
    t.add_row(['Min nrmse', str(np.round(minrmse, 3))])
    t.add_row(['--', '--'])
    t.add_row(['Wavelength (nm)', str(np.round(wvl_m, 3))])
    t.add_row(['--', '--'])
    t.add_row(['K-param (mean r2-val)', str(np.round(Kv_final, 3))])
    t.add_row(['psi-param (mean r2-val)', str(np.round(psiv_final, 3))])
    t.add_row(['alpha-param (mean r2-val)', str(np.round(alphav_final, 3))])
    print(' ')
    print(' ')
    print('##################################################################')
    print('|--- Marmit retrieval output stats----|')
    print(t)


    f = open('./outputs' + '/model_stat.csv', 'w')
    f.write('MARMIT retrieval output stats' )
    f.write('\r\n')
    f.write('\r\n')
    f.write('Mean r2-val = ' +str( np.round(mu, 3)))
    f.write('\r\n')
    f.write('Mean nrmse = '+ str(np.round(murmse, 3)))
    f.write('\r\n')
    f.write('\r\n')
    f.write('St dev. r2-val = '+ str(np.round(sigma, 3)))
    f.write('\r\n')
    f.write('St dev. nrmse = '+ str(np.round(sigmarmse, 3)))
    f.write('\r\n')
    f.write('\r\n')
    f.write('Median r2-val = '+ str(np.round(median_r2, 3)))
    f.write('\r\n')
    f.write('Median nrmse = '+ str(np.round(medianrmse, 3)))
    f.write('\r\n')
    f.write('\r\n')
    f.write('Max r2-val = '+ str(np.round(max_r2, 3)))
    f.write('\r\n')
    f.write('Min nrmse = '+ str(np.round(minrmse, 3)))
    f.write('\r\n')
    f.write('\r\n')
    f.write('Wavelength (nm) = '+ str(np.round(wvl_m, 3)))
    f.write('\r\n')
    f.write('\r\n')
    f.write('K-param (mean r2-val) = '+ str(np.round(Kv_final, 3)))
    f.write('\r\n')
    f.write('psi-param (mean r2-val) = '+ str(np.round(psiv_final, 3)))
    f.write('\r\n')
    f.write('alpha-param (mean r2-val) = '+ str(np.round(alphav_final, 3)))

    return 0


wavelength,reflectance_dry = get_reflectance_files(csvfile_dry)
abs_coeff,nw_coeff = get_abs_nw(abs_nw)


    
def output_mw(csvfiles_wet):

    wavelength,reflectance_wet, solar_zenith_wet,solar_azimuth_wet,sensor_zenith_wet,sensor_azimuth_wet = get_reflectance_files_wet(csvfiles_wet)

    L,epsilon = perform_inversion(reflectance_wet,reflectance_dry,solar_zenith_wet,wavelength,abs_coeff,nw_coeff)

    results_folder = data_struct['mwt_output']
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_csv = './' + results_folder + '/mwt_' + csvfiles_wet.rsplit('/')[-1]
    csvfiles_wet[0][0].rsplit('/')[-1]
    flname = open(results_csv, 'wt')
    try:
        writer = csv.writer(flname)
        writer.writerow(('Wavelength', 'Water Level Thickness', 'Efficiency Term'))
        for i in range(len(wavelength)):
            writer.writerow((wavelength[i],L[i],epsilon[i]))
    finally:
        flname.close()
    return 0

pool = multiprocess.Pool()
# if number of processes is not specified, it uses the number of core
pool.map(output_mw,  (csvfiles_wet[i][0] for i in range(len(csvfiles_wet))))

wvl,mean_water_thickness,smc,r2_final,nrmse_final,wvl_final,K_final,psi_final,alpha_final,test_dat,wvl_m = get_logistic_function_params(data_struct['mwt_output'],data_struct['smc_input'])
saveplots(wvl,mean_water_thickness,smc,r2_final,nrmse_final,wvl_final,K_final,psi_final,alpha_final,test_dat,data_struct['img_output'],wvl_m)
print('Finishing MARMIT retrieval, total time taken = ', (time.time() - start_time1) / 3600, 'hours')
