from anc_functions import get_smc_data
from anc_functions import get_L_eps_data
from anc_functions import logistic_function
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import numpy as np
import matplotlib.pyplot as plt



#--------------------------------------- get curve fit data ---------------------------------------#
def get_curve_fit_data(inv_folder,
                       smc_folder,
                       curve_K,
                       curve_psi,
                       curve_alpha,
                       curve_r2):
    """
    Extracts curve fitting data from provided file paths.

    This function reads data files from the input folder, sorts and aligns the data by
    run numbers extracted from file names. It then computes mean water thickness and
    evaluates a logistic function with the mean water thickness and curve fitting parameters
    obtained from the input files. nrmse error between measured and estimated soil moisture
    content (smc) is then computed and sorted.

    Args:
        inv_folder (str): Path to the input data folder.
        smc_folder (str): Path to the folder containing soil moisture content data.
        curve_K (str): Path to the file containing the 'K' values for the logistic function.
        curve_psi (str): Path to the file containing the 'psi' values for the logistic function.
        curve_alpha (str): Path to the file containing the 'alpha' values for the logistic function.
        curve_r2 (str): Path to the file containing the 'r2' values for the logistic function.

    Returns:
        tuple: A tuple containing arrays of mean water thickness, measured soil moisture content (smc),
               estimated smc, nrmse error, 'r2' values, 'K' values, 'psi' values, 'alpha' values,
               and sorted nrmse error indices.
    """
    path, dirs, files = next(os.walk(inv_folder))

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

    smc_meas = get_smc_data(smc_folder)

    dat = get_L_eps_data(path + csv_L[0])
    w = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    B = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b1 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b2 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b3 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    b4 = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    L = np.zeros((dat.shape[0], dat.shape[1], len(csv_L)))
    epsilon = np.zeros((dat.shape[0],dat.shape[1],len(csv_L)))
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
        epsilon[:,:,i] = get_L_eps_data(path + csv_epsilon[num_argsort9[i]])
        fill[:, :, i] = get_L_eps_data(path + csv_fill[num_argsort10[i]])
        C[:, :, i] = get_L_eps_data(path + csv_C[num_argsort11[i]])


    mean_water_thickness = L * epsilon



    K = get_L_eps_data(curve_K).astype(float)
    K = np.stack([K for _ in range(0, len(smc_meas))], axis=2)
    psi = get_L_eps_data(curve_psi).astype(float)
    psi = np.stack([psi for _ in range(0, len(smc_meas))], axis=2)
    alpha = get_L_eps_data(curve_alpha).astype(float)
    alpha = np.stack([alpha for _ in range(0, len(smc_meas))], axis=2)
    r2_val = get_L_eps_data(curve_r2).astype(float)

    smc_est = logistic_function(mean_water_thickness,K,psi,alpha)

    rms = np.zeros(r2_val.shape)
    for i in range(0,r2_val.shape[0]):
        for j in range(0,r2_val.shape[1]):
            rms[i,j] = sqrt(mean_squared_error(smc_meas, smc_est[i,j,:])) / np.mean(smc_meas)

    rms_sort_ind = np.dstack(np.unravel_index(np.argsort(rms.ravel()),(r2_val.shape)))[0]

    return mean_water_thickness,smc_meas,smc_est,rms,r2_val,K,psi,alpha,rms_sort_ind
#--------------------------------------- get curve fit data ---------------------------------------#




#--------------------------------------- plot rms data ---------------------------------------#



def plot_rms(sensor_zenith_dry,sensor_azimuth_dry,wavelength,rms,img_output):
    """
    This function plots the normalized root mean square error (NRMSE) for the given data.

    Args:
        sensor_zenith_dry (np.array): Array of sensor zenith angles in dry conditions.
        sensor_azimuth_dry (np.array): Array of sensor azimuth angles in dry conditions.
        wavelength (np.array): Array of wavelengths.
        rms (np.array): Array of normalized root mean square error (NRMSE) values.
        img_output (str): The directory where the output image will be saved.

    Returns:
        int: Returns 0 upon successful completion.
    """
    if not os.path.exists(img_output[0:22]):
        os.makedirs(img_output[0:22])

    ind60zen = np.argwhere(sensor_zenith_dry == 60).flatten()
    rms60 = rms[ind60zen,:]
    rms60_mean = np.mean(rms60,axis=0)
    ind40zen = np.argwhere(sensor_zenith_dry == 40).flatten()
    rms40 = rms[ind40zen,:]
    rms40_mean = np.mean(rms40,axis=0)
    ind20zen = np.argwhere(sensor_zenith_dry == 20).flatten()
    rms20 = rms[ind20zen,:]
    rms20_mean = np.mean(rms20,axis=0)
    ind0zen = np.argwhere(sensor_zenith_dry == 0).flatten()
    rms0 = rms[ind0zen,:]
    rms0_mean = np.mean(rms0,axis=0)
    rms_avg_zen = np.stack([rms0_mean,rms20_mean,rms40_mean,rms60_mean],axis=0)

    ind0azm = np.argwhere(sensor_azimuth_dry == 0).flatten()
    rms0azm_mean = np.mean(rms[ind0azm,:],axis=0)
    ind36azm = np.argwhere(sensor_azimuth_dry == 36).flatten()
    rms36azm_mean = np.mean(rms[ind36azm,:],axis=0)
    ind72azm = np.argwhere(sensor_azimuth_dry == 72).flatten()
    rms72azm_mean = np.mean(rms[ind72azm,:],axis=0)
    ind108azm = np.argwhere(sensor_azimuth_dry == 108).flatten()
    rms108azm_mean = np.mean(rms[ind108azm,:],axis=0)
    ind144azm = np.argwhere(sensor_azimuth_dry == 144).flatten()
    rms144azm_mean = np.mean(rms[ind144azm,:],axis=0)
    ind180azm = np.argwhere(sensor_azimuth_dry == 180).flatten()
    rms180azm_mean = np.mean(rms[ind180azm,:],axis=0)
    ind216azm = np.argwhere(sensor_azimuth_dry == 216).flatten()
    rms216azm_mean = np.mean(rms[ind216azm,:],axis=0)
    ind252azm = np.argwhere(sensor_azimuth_dry == 252).flatten()
    rms252azm_mean = np.mean(rms[ind252azm,:],axis=0)
    ind288azm = np.argwhere(sensor_azimuth_dry == 288).flatten()
    rms288azm_mean = np.mean(rms[ind288azm,:],axis=0)
    ind324azm = np.argwhere(sensor_azimuth_dry == 324).flatten()
    rms324azm_mean = np.mean(rms[ind324azm,:],axis=0)
    rms_avg_azm = np.stack([rms0azm_mean,rms36azm_mean,rms72azm_mean,rms108azm_mean,rms144azm_mean,
                            rms180azm_mean,rms216azm_mean,rms252azm_mean,rms288azm_mean,rms324azm_mean],axis=0)
    print("az nrmse min = ",np.min(rms_avg_azm))
    print("az nrmse max = ",np.max(rms_avg_azm))

    print("zen nrmse min = ",np.min(rms_avg_zen))
    print("zen nrmse max = ",np.max(rms_avg_zen))

    fig = plt.figure()
    plt.imshow(rms_avg_zen,interpolation='bicubic', aspect='auto',cmap='rainbow',
               vmin=0,vmax=1)
    plt.xticks(range(rms_avg_zen.shape[1])[0::500],wavelength[range(rms_avg_zen.shape[1])[0::500]].astype(int))
    plt.yticks(range(rms_avg_zen.shape[0]),[0,20,40,60])
    plt.xlabel('Wavelength (nm)',fontsize=15)
    plt.ylabel('Sensor Zenith (deg)', fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label('Normalized Root Mean Square Error (NRMSE)', rotation=90, fontsize=10)
    fig.savefig(img_output + '_hapke_nrmse_sza_final.png',dpi=150, bbox_inches='tight',pad_inches=0)

    fig = plt.figure()
    plt.imshow(rms_avg_azm,interpolation='bicubic', aspect='auto',cmap='rainbow',
               vmin=0,vmax=1)
    plt.xticks(range(rms_avg_zen.shape[1])[0::500],wavelength[range(rms_avg_zen.shape[1])[0::500]].astype(int))
    plt.yticks(range(rms_avg_azm.shape[0]),[0,36,72,108,144,180,216,252,288,324])
    plt.xlabel('Wavelength (nm)', fontsize=15)
    plt.ylabel('Sensor Azimuth (deg)', fontsize=15)
    cbar = plt.colorbar()
    cbar.set_label('Normalized Root Mean Square Error (NRMSE)', rotation=90, fontsize=10)
    fig.savefig(img_output + '_hapke_nrmse_azm_final.png',dpi=150, bbox_inches='tight',pad_inches=0)
    #plt.close(fig)

    return 0

#--------------------------------------- plot rms data ---------------------------------------#





#--------------------------------------- plot measured vs estimated smc ---------------------------------------#
def plot_est_meas_smc(ind_val,rms_sort_ind,smc_est,smc_meas,wavelength,sensor_zenith_dry,sensor_azimuth_dry,img_output):
    """
    This function generates a plot of estimated versus measured soil moisture content (SMC).

    Args:
        ind_val (int): Index value for selecting best nrmse indices.
        rms_sort_ind (np.array): Array of sorted nrmse indices.
        smc_est (np.array): Array of estimated soil moisture content (SMC).
        smc_meas (np.array): Array of measured soil moisture content (SMC).
        wavelength (np.array): Array of wavelengths.
        sensor_zenith_dry (np.array): Array of sensor zenith angles in dry conditions.
        sensor_azimuth_dry (np.array): Array of sensor azimuth angles in dry conditions.
        img_output (str): The directory where the output image will be saved.

    Returns:
        int: Returns 0 upon successful completion.
    """
    best_ind = rms_sort_ind[ind_val]
    smc_est_plot = smc_est[best_ind[0],best_ind[1],:]
    smc_std = np.std(smc_est[:,best_ind[1],:],axis=0)
    smc_lin = np.linspace(0, np.max(smc_meas) + 2)

    fig = plt.figure()
    plt.errorbar(smc_meas,smc_est_plot,smc_std,linestyle='',color='k')
    plt.plot(smc_meas,smc_est_plot,'o',color='blue',markersize=10)
    plt.plot(smc_lin,smc_lin,'k-',linewidth=2)
    plt.xlim(np.min(smc_lin),np.max(smc_lin))
    plt.ylim(np.min(smc_lin),np.max(smc_lin))
    plt.xlabel('Measured SMC', fontsize=15)
    plt.ylabel('Estimated SMC', fontsize=15)
    fig.savefig(img_output + '_smc_true_v_pred.png',dpi=150, bbox_inches='tight',pad_inches=0)
    #plt.close(fig)

    return 0
#--------------------------------------- plot measured vs estimated smc ---------------------------------------#





#--------------------------------------- plot logistic function ---------------------------------------#
def plot_logistic_function(ind_val,rms_sort_ind,smc_est,smc_meas,mean_water_thickness,K,psi,alpha,img_output):
    """
    Plots the logistic function for soil moisture content against mean water thickness.

    Args:
        ind_val (int): Index value in the sorted nrmse array.
        rms_sort_ind (numpy array): Indices of the sorted nrmse array.
        smc_est (numpy array): Array of estimated soil moisture content (SMC).
        smc_meas (numpy array): Array of measured soil moisture content.
        mean_water_thickness (numpy array): Array of mean water thickness.
        K (numpy array): Array of values for the logistic function parameter K.
        psi (numpy array): Array of values for the logistic function parameter psi.
        alpha (numpy array): Array of values for the logistic function parameter alpha.
        img_output (str): Path to save the plotted image.

    Returns:
        int: 0 if the function executed successfully.
    """
    best_ind = rms_sort_ind[ind_val]
    smc_est_plot = smc_est[best_ind[0], best_ind[1], :]
    mean_water_thickness_plot = mean_water_thickness[best_ind[0], best_ind[1], :]

    phi_lin = np.linspace(min(mean_water_thickness_plot),max(mean_water_thickness_plot))
    y = logistic_function(phi_lin,
                          K[best_ind[0], best_ind[1],0],
                          psi[best_ind[0], best_ind[1],0],
                          alpha[best_ind[0], best_ind[1],0])



    fig = plt.figure()
    plt.plot(mean_water_thickness_plot, smc_meas,'o',color='blue',markersize=10)
    plt.plot(phi_lin,y,'k-',linewidth=2)
    plt.xlabel('Mean Water Thickness (cm)', fontsize=15)
    plt.ylabel('Soil Moisture Content (%)', fontsize=15)
    fig.savefig(img_output + '_smc_log_func.png',dpi=150, bbox_inches='tight',pad_inches=0)
    #plt.close(fig)
    
    return 0
#--------------------------------------- plot logistic function ---------------------------------------#




#--------------------------------------- save model output ---------------------------------------#
def save_output_data(ind_val,rms_sort_ind,smc_est,smc_meas,wavelength,sensor_zenith_dry,sensor_azimuth_dry,K,psi,alpha,r2_val,smc_output):
    """
    Saves output data from model retrieval, including estimated and measured soil moisture content (SMC),
    the logistic function parameters, and the best fit wavelength and angle.

    Args:
        ind_val (int): Index value in the sorted nrmse array.
        rms_sort_ind (numpy array): Indices of the sorted nrmse array.
        smc_est (numpy array): Array of estimated soil moisture content (SMC).
        smc_meas (numpy array): Array of measured soil moisture content.
        wavelength (numpy array): Array of wavelength values.
        sensor_zenith_dry (numpy array): Array of dry sensor zenith values.
        sensor_azimuth_dry (numpy array): Array of dry sensor azimuth values.
        K (numpy array): Array of values for the logistic function parameter K.
        psi (numpy array): Array of values for the logistic function parameter psi.
        alpha (numpy array): Array of values for the logistic function parameter alpha.
        r2_val (numpy array): Array of r-squared values for model fit.
        smc_output (str): Path to save the output data.

    Returns:
        int: 0 if the function executed successfully.
    """

    if not os.path.exists(smc_output[:20]):
        os.makedirs(smc_output[:20])

    best_ind = rms_sort_ind[ind_val]
    smc_est_plot = smc_est[best_ind[0],best_ind[1],:]
    smc_std = np.std(smc_est[:,best_ind[1],:],axis=0)

    smc_dat = np.transpose(np.asarray([smc_meas,smc_est_plot,smc_std]))
    np.savetxt(smc_output + '_smc_output.csv',smc_dat,delimiter=',')

    f = open(smc_output + '_smc_log.txt', 'w')
    f.write('---Information about the Lab Based Modified SWAP Hapke Retrieval---\r\n')
    f.write('Sample Name: ' + smc_output.rsplit('/')[-1] + '\r\n')
    f.write('\r\n')
    f.write('Best fit was observed in the following wavelength and angle' + '\r\n')
    f.write('\r\n')
    f.write('Wavelength = ' + str(wavelength[best_ind[1]]) + '\r\n')
    f.write('Sensor Zenith = ' + str(sensor_zenith_dry[best_ind[0]]) + '\r\n')
    f.write('Sensor Azimuth = ' + str(sensor_azimuth_dry[best_ind[0]]) + '\r\n')
    f.write('r**2 = ' + str(r2_val[best_ind[0],best_ind[1]]) + '\r\n')
    f.write('\r\n')
    f.write('---Logistic function parameters for the best fit---\n')
    f.write('\r\n')
    f.write('K = ' + str(K[best_ind[0], best_ind[1],0]) + '\r\n')
    f.write('psi = ' + str(psi[best_ind[0], best_ind[1],0]) + '\r\n')
    f.write('alpha = ' + str(alpha[best_ind[0], best_ind[1],0]) + '\r\n')
    f.close()

    return 0
#--------------------------------------- save model output ---------------------------------------#



