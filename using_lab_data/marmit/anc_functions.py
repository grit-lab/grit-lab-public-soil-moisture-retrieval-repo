import numpy as np
import csv


def get_L_eps_data(csv_file):
    """
        Reads CSV file and returns the data as a numpy array.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            numpy.ndarray: Data from the CSV file as a numpy array.
        """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    return data

def get_smc_data(csv_file):
    """
    Reads CSV file, extracts the soil moisture content (smc) data, and returns it as a numpy array.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        numpy.ndarray: Soil moisture content data from the CSV file as a numpy array.
    """
    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    smc = data[2:,1]
    smc = smc.astype(np.float)
    smc = smc.flatten()
    return smc

def squared_error(ys_orig, ys_line):
    """
    Computes the sum of squared differences between two sets of values.

    Args:
        ys_orig (array): Original values.
        ys_line (array): Fitted line values.

    Returns:
        float: Sum of squared differences.
    """
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))


def coefficient_of_determination(ys_orig, ys_line):
    """
    Computes the coefficient of determination (R-squared) of the fitted line.

    Args:
        ys_orig (array): Original values.
        ys_line (array): Fitted line values.

    Returns:
        float: Coefficient of determination (R-squared).
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
    x = np.float128(x)
    K = np.float128(K)
    alpha = np.float128(alpha)
    log_func = np.where(x >= 0, (K / (1 + (alpha * (np.exp(-x))))),((K * (np.exp(x))) / (alpha + (np.exp(x)))))
    log_func = np.float64(log_func)
    return log_func


def rmse(predictions, targets):
    """
    Computes the root-mean-square error (RMSE) between predictions and targets.

    Args:
        predictions (array): Predicted values.
        targets (array): Target or actual values.

    Returns:
        float: The computed RMSE.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

