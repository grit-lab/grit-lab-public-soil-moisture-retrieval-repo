import numpy as np
import csv


def get_L_eps_data(csv_file):
    """Read reflectance data from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing reflectance data.

        Returns:
            numpy array: Reflectance data.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    return data


def get_smc_data(csv_file):
    """Read SMC data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing SMC data.

    Returns:
        numpy array: SMC data.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    smc = data[1:,1]
    smc = smc.astype(np.float)
    smc = smc.flatten()
    return smc


def squared_error(ys_orig, ys_line):
    """Calculate squared error.

    Args:
        ys_orig (numpy array): Original data points.
        ys_line (numpy array): Fitted data points.

    Returns:
        float: Squared error.
    """
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    """Calculate the coefficient of determination (R-squared).

    Args:
        ys_orig (numpy array): Original data points.
        ys_line (numpy array): Fitted data points.

    Returns:
        float: Coefficient of determination.
    """
    y_mean_line = [np.mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)



def logistic_function(phi,K,psi,alpha):
    """Expression of the logistic function (numerically stable sigmoid function).

    Args:
        phi (numpy array): Independent variable.
        K (float): Carrying capacity.
        psi (float): Growth rate.
        alpha (float): Scaling parameter.

    Returns:
        numpy array: Logistic function values.
    """

    x = psi * phi
    x = np.float64(x)
    K = np.float64(K)
    alpha = np.float64(alpha)
    log_func = np.where(x >= 0, (K / (1 + (alpha * (np.exp(-x))))),((K * (np.exp(x))) / (alpha + (np.exp(x)))))
    log_func = np.float64(log_func)
    return log_func


def rmse(predictions, targets):
    """Calculate the root mean squared error (RMSE).

    Args:
        predictions (numpy array): Predicted data points.
        targets (numpy array): Actual data points.

    Returns:
        float: Root mean squared error.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_swap_hapke_data(data_file):
    """Read Original (SWAP)-Hapke data from a CSV file.

     Args:
         data_file (str): Path to the CSV file containing Original (SWAP)-Hapke data.

     Returns:
         tuple: Wavelength and water level data as numpy arrays.
     """
    rd = open(data_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)

    wavelength = data[1:, 0]
    wavelength = wavelength.astype(np.float)
    wavelength = wavelength.flatten()

    water_level = data[1:, 7]
    water_level = water_level.astype(np.float)
    water_level = water_level.flatten()

    return wavelength, water_level

