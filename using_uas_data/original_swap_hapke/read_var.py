import csv
import numpy as np

def get_abs_nw(csv_file):
    """
    Extract absorption coefficients and relative refractive indices from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing two numpy arrays, one for absorption coefficients and one for relative refractive indices.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)

    wavelength = data[1:,0]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    abs_coeff = data[1:,1]
    abs_coeff = abs_coeff.astype(np.float)
    abs_coeff = abs_coeff.flatten()

    nw_coeff = data[1:,2]
    nw_coeff = nw_coeff.astype(np.float)
    nw_coeff = nw_coeff.flatten()

    return abs_coeff,nw_coeff


def get_reflectance_files(csv_file):
    """
    Extract wavelengths and reflectance values from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing two numpy arrays, one for wavelengths and one for reflectance values.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd, delimiter=';')
    data = list(csv_reader)
    data = np.asarray(data)

    wavelength = data[1:,0]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    reflectance = data[1:,1]
    reflectance = reflectance.astype(np.float)
    reflectance = reflectance.flatten()



    return wavelength,reflectance

def get_reflectance_files_wet(csv_file):
    """
    Extract wavelengths, reflectance values, and various solar and sensor angles from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing six numpy arrays for wavelengths, reflectance values, solar zenith angles,
               solar azimuth angles, sensor zenith angles, and sensor azimuth angles.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd, delimiter=',')
    data = list(csv_reader)
    data = np.asarray(data)

    wavelength = data[1:,0]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    reflectance = data[1:,1]
    reflectance = reflectance.astype(np.float)
    reflectance = reflectance.flatten()

    solar_zenith_dry = data[1:, 2]
    solar_zenith_dry = solar_zenith_dry.astype(np.float64)
    solar_zenith_dry = solar_zenith_dry.flatten()

    solar_azimuth_dry = data[1:, 3]
    solar_azimuth_dry = solar_azimuth_dry.astype(np.float64)
    solar_azimuth_dry = solar_azimuth_dry.flatten()

    sensor_zenith_dry = data[1:, 4]
    sensor_zenith_dry = sensor_zenith_dry.astype(np.float64)
    sensor_zenith_dry = sensor_zenith_dry.flatten()

    sensor_azimuth_dry = data[1:, 5]
    sensor_azimuth_dry = sensor_azimuth_dry.astype(np.float64)
    sensor_azimuth_dry = sensor_azimuth_dry.flatten()

    return wavelength,reflectance,solar_zenith_dry,solar_azimuth_dry,sensor_zenith_dry,sensor_azimuth_dry

