import csv
import numpy as np

def parse_csv_files(csv_file,
                    solar_zenith_val,
                    solar_azimuth_val,
                    azimuth_column,
                    zenith_column,
                    start_wavelength_column):
    """Parses a CSV file and extracts relevant data for calculations.

    Args:
        csv_file (str): Path to the CSV file to be parsed.
        solar_zenith_val (float): Value for solar zenith.
        solar_azimuth_val (float): Value for solar azimuth.
        azimuth_column (int): The column number where sensor azimuth data is located.
        zenith_column (int): The column number where sensor zenith data is located.
        start_wavelength_column (int): The column number where wavelength data starts.

    Returns:
        tuple: Contains sensor azimuth data, sensor zenith data, solar zenith data, solar azimuth data, reflectance data, and wavelength data.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)

    sensor_azimuth = data[1:,azimuth_column]
    sensor_azimuth = sensor_azimuth.astype(np.float)
    sensor_azimuth = sensor_azimuth.flatten()

    sensor_zenith = data[1:,zenith_column]
    sensor_zenith = sensor_zenith.astype(np.float)
    sensor_zenith = sensor_zenith.flatten()

    solar_zenith = np.ones(sensor_zenith.shape) * solar_zenith_val
    solar_azimuth = np.ones(sensor_zenith.shape) * solar_azimuth_val

    reflectance = data[1:,start_wavelength_column:]
    reflectance = reflectance.astype(np.float)

    wavelength = data[0,start_wavelength_column:]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    return sensor_azimuth,\
           sensor_zenith,\
           solar_zenith,\
           solar_azimuth,\
           reflectance,\
           wavelength

def get_abs_nw(csv_file):
    """Parses a CSV file and extracts absorption and non-water coefficients.

    Args:
        csv_file (str): Path to the CSV file to be parsed.

    Returns:
        tuple: Contains absorption coefficients and non-water coefficients.
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

