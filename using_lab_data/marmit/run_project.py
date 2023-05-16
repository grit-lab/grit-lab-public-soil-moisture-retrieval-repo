from read_var import parse_csv_files
from read_var import get_abs_nw
from marmit_inversion import perform_inversion
from marmit_curve_fit import get_logistic_function_params
from marmit_save_output import plot_rms
from marmit_save_output import get_curve_fit_data
from marmit_save_output import plot_est_meas_smc
from marmit_save_output import plot_logistic_function
from marmit_save_output import save_output_data
from plot_smc_refl import parse_csv_files_smc_plot
from plot_smc_refl import spectral_lib_plot
import numpy as np
import os
import csv
import time
import multiprocessing


# %% Initialization of the data

# Initialization of the data:
#       selecting which of the four lab samples to process
#       pointing towards folder based on the lab sample
#       creating output folder based on the lab sample

start_time1 = time.time()

# Prompt user to choose the dataset to be processed
dataset = input('Choose the lab dataset to process (alg/nev/hogp/hogb): ')
print('lab dataset being processed: ', dataset)


# Set the data paths depending on the selected dataset
if dataset == 'alg':
    csv_file = '../input/reflectance_data/algodones_sample/algodones_sample_list.csv'
    smc_input = '../input/smc_data/algodones_sample.csv'
    plt_smc_input = '../input/smc_data_refl/algodones_sample1.csv'

    mwt_output = './outputs/anc_files/mwt_data/mwt_alg/'
    log_output = './outputs/anc_files/save_curve_fit/alg'
    img_output = './outputs/image_output/alg'
    smc_output = './outputs/smc_output/alg'

elif dataset == 'nev':
    csv_file = '../input/reflectance_data/nevada_sample/nevada_sample_list.csv'
    smc_input = '../input/smc_data/nevada_sample.csv'
    plt_smc_input = '../input/smc_data_refl/nevada_sample1.csv'

    mwt_output = './outputs/anc_files/mwt_data/mwt_nev/'
    log_output = './outputs/anc_files/save_curve_fit/nev'
    img_output = './outputs/image_output/nev'
    smc_output = './outputs/smc_output/nev'

elif dataset == 'hogp':
    csv_file = '../input/reflectance_data/hog_panne_sample/hog_panne_sample_list.csv'
    smc_input = '../input/smc_data/hogp_sample.csv'
    plt_smc_input = '../input/smc_data_refl/hogp_sample1.csv'

    mwt_output = './outputs/anc_files/mwt_data/mwt_hogp/'
    log_output = './outputs/anc_files/save_curve_fit/hogp'
    img_output = './outputs/image_output/hogp'
    smc_output = './outputs/smc_output/hogp'

elif dataset == 'hogb':
    csv_file = '../input/reflectance_data/hog_beach_sample/hog_beach_sample_list.csv'
    smc_input = '../input/smc_data/hogb_sample.csv'
    plt_smc_input = '../input/smc_data_refl/hogb_sample1.csv'

    mwt_output = './outputs/anc_files/mwt_data/mwt_hogb/'
    log_output = './outputs/anc_files/save_curve_fit/hogb'
    img_output = './outputs/image_output/hogb'
    smc_output = './outputs/smc_output/hogb'



# Parsing information for the GRIT data
# column information
azimuth_column = 2
zenith_column = 3
start_wavelength_column = 8

# Solar Angles for the data
solar_zenith_val = 40
solar_azimuth_val = 0

# absorbtion coefficient and relative refractive index data
abs_nw = '../input/abs_coeff_refran_index/nw_abs_grit.csv'

# Construct a data structure for easy access to data values
data_struct = {
    'csv_file': csv_file,
    'smc_input': smc_input,
    'plt_smc_input': plt_smc_input,
    'mwt_output': mwt_output,
    'log_output': log_output,
    'img_output': img_output,
    'smc_output': smc_output,
    'azimuth_column': azimuth_column,
    'zenith_column': zenith_column,
    'start_wavelength_column': start_wavelength_column,
    'solar_zenith_val': solar_zenith_val,
    'solar_azimuth_val': solar_azimuth_val,
    'abs_nw': abs_nw
}




# Read in csv file list
rd = open(data_struct['csv_file'], 'r')
csv_reader = csv.reader(rd)
data_list = list(csv_reader)
csvfile_dry = data_list[0][0]
csvfiles_wet = data_list[1:]

# Read in BRDF data for dry sample
sensor_azimuth_dry, \
sensor_zenith_dry, \
solar_zenith_dry, \
solar_azimuth_dry, \
reflectance_dry, \
wavelength = parse_csv_files(csvfile_dry,
                             data_struct['solar_zenith_val'],
                             data_struct['solar_azimuth_val'],
                             data_struct['azimuth_column'],
                             data_struct['zenith_column'],
                             data_struct['start_wavelength_column'])



# %% Perform inversion to retrieve mean water thickness
print('Starting step 1: marmit inversion to retrieve mean water thickness')


# Function to perform marmit inversion to retrieve mean water thickness
def output_mw(csvfiles_wet):
    sensor_azimuth_wet, \
    sensor_zenith_wet, \
    solar_zenith_wet, \
    solar_azimuth_wet, \
    reflectance_wet, \
    wavelength = parse_csv_files(csvfiles_wet,
                                 data_struct['solar_zenith_val'],
                                 data_struct['solar_azimuth_val'],
                                 data_struct['azimuth_column'],
                                 data_struct['zenith_column'],
                                 data_struct['start_wavelength_column'])

    abs_coeff, nw_coeff = get_abs_nw(data_struct['abs_nw'])

    L = np.zeros(reflectance_wet.shape)
    epsilon = np.zeros(reflectance_wet.shape)

    for i in range(0, reflectance_wet.shape[0]):
        L[i, :], epsilon[i, :] = perform_inversion(reflectance_wet[i, :],
                                                   reflectance_dry[i, :],
                                                   solar_zenith_dry,
                                                   wavelength,
                                                   abs_coeff,
                                                   nw_coeff)

    results_folder = data_struct['mwt_output']
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    L_output = './' + results_folder + '/' + csvfiles_wet.rsplit('/')[-1].rsplit('_singlemode')[0] + '_water_level.csv'
    eps_output = './' + results_folder + '/' + csvfiles_wet.rsplit('/')[-1].rsplit('_singlemode')[0] + '_efficiency.csv'
    np.savetxt(L_output, L, delimiter=',')
    np.savetxt(eps_output, epsilon, delimiter=',')

    return 0


# Use multiprocessing to speed up the processing
pool = multiprocessing.Pool()

# if number of processes is not specified, it uses the number of core
pool.map(output_mw, (csvfiles_wet[i][0] for i in range(len(csvfiles_wet))))
print('Finishing step 1: marmit inversion to retrieve mean water thickness, time taken = ',
      (time.time() - start_time1) / 60, 'mins')


# %% Perform curve fitting to logistic function
print('Starting step 2: marmit curve fitting to logistic function')
start_time2 = time.time()


# Perform curve fitting to logistic function and print time taken
get_logistic_function_params(data_struct['log_output'],
                             data_struct['mwt_output'],
                             data_struct['smc_input'])
print('Finishing step 2: marmit curve fitting to logistic function, time taken = ', (time.time() - start_time2) / 60,
      'mins')



# # %% Saving marmit output
print('Starting to save all marmit outputs')

# Get curve fit data and save output data
mean_water_thickness, \
smc_meas, \
smc_est, \
rms, \
r2_val, \
K, \
psi, \
alpha, \
rms_sort_ind = \
    get_curve_fit_data(data_struct['mwt_output'],
                       data_struct['smc_input'],
                       data_struct['log_output'] + '_K.csv',
                       data_struct['log_output'] + '_psi.csv',
                       data_struct['log_output'] + '_alpha.csv',
                       data_struct['log_output'] + '_r2.csv')

# Save output data, plot nrmse, estimated and measured SMC, and logistic function
save_output_data(0, rms_sort_ind, smc_est, smc_meas, wavelength, sensor_zenith_dry, sensor_azimuth_dry, K, psi, alpha,
                 r2_val, data_struct['smc_output'])
plot_rms(sensor_zenith_dry, sensor_azimuth_dry, wavelength, rms, data_struct['img_output'])
plot_est_meas_smc(0, rms_sort_ind, smc_est, smc_meas, wavelength, sensor_zenith_dry, sensor_azimuth_dry,
                  data_struct['img_output'])
plot_logistic_function(0, rms_sort_ind, smc_est, smc_meas, mean_water_thickness, K, psi, alpha,
                       data_struct['img_output'])



##### Plot spectra at the best-fit sensor geometry for the MARMIT model.
# smc_p, reflectance_p, wavelength_p = parse_csv_files_smc_plot(data_struct['plt_smc_input'])
# spectral_lib_plot(smc_p, wavelength_p, reflectance_p, 'jet', 400, 2350, data_struct['img_output'])

# Print the total time taken for the marmit retrieval
print('Finishing marmit retrieval, total time taken = ', (time.time() - start_time1) / 60, 'mins')
