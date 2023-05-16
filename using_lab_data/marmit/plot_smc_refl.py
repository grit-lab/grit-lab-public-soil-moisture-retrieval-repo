import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

def parse_csv_files_smc_plot(csv_file):
    """Parses a CSV file for plotting Soil Moisture Content (SMC).

    Args:
        csv_file (str): Path to the CSV file to be parsed.

    Returns:
        tuple: Contains soil moisture content data, reflectance data, and wavelength data.
    """

    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)

    smc = data[1:,1]
    smc = smc.astype(np.float)
    smc = smc.flatten()

    reflectance = data[1:,2:]
    reflectance = reflectance.astype(np.float)

    wavelength = data[0,2:]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    reflectance = reflectance[:,100:1950]
    wavelength = wavelength[100:1950]

    return smc,\
           reflectance,\
           wavelength

def spectral_lib_plot(smc,
                      wavelength,
                      reflectance,
                      color_scheme,
                      xmin,
                      xmax,
                      img_output):
    """Generates a spectral library plot.

    Args:
        smc (numpy array): Soil moisture content data.
        wavelength (numpy array): Wavelength data.
        reflectance (numpy array): Reflectance data.
        color_scheme (str): The color scheme for the plot.
        xmin (int): The minimum x-axis value for the plot.
        xmax (int): The maximum x-axis value for the plot.
        img_output (str): Path to save the output image.

    Returns:
        int: 0 if the function executed successfully.
    """
    fig = plt.figure()

    norm = matplotlib.colors.Normalize(vmin=np.min(smc), vmax=np.max(smc))

    s_m = matplotlib.cm.ScalarMappable(cmap=color_scheme, norm=norm)
    s_m.set_array([])

    for i in range(len(smc)):
        plt.plot(wavelength,reflectance[i,:], '-', linewidth=2, Markersize=5, color=s_m.to_rgba(smc[i]))
    axes = plt.gca()
    cbar = plt.colorbar(s_m)
    cbar.set_label('Soil Moisture Content (%)', rotation=90, fontsize=15)
    plt.xlabel('Wavelength', fontsize=15)
    plt.ylabel('Reflectance', fontsize=15)
    plt.xlim(xmin,xmax)
    fig.savefig(img_output + '_smc_speclib.png',dpi=150, bbox_inches='tight',pad_inches=0)

    return 0