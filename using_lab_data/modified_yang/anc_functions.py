import numpy as np
import csv


def get_L_eps_data(csv_file):
    """Read data from csv file.

    Args:
        csv_file (str): path to the csv file.

    Returns:
        numpy.ndarray: the data read from the csv file.
    """

    rd = open(csv_file, 'r')
    csv_reader = csv.reader(rd)
    data = list(csv_reader)
    data = np.asarray(data)
    return data



def get_smc_data(csv_file):
    """Read Soil Moisture Content (SMC) data from csv file.

    Args:
        csv_file (str): path to the csv file.

    Returns:
        numpy.ndarray: the SMC data read from the csv file.
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
    """Calculate the squared error.

     Args:
         ys_orig (numpy.ndarray): original y values.
         ys_line (numpy.ndarray): line y values.

     Returns:
         float: the squared error.
     """
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))




def coefficient_of_determination(ys_orig, ys_line):
    """Calculate the coefficient of determination (R^2).

     Args:
         ys_orig (numpy.ndarray): original y values.
         ys_line (numpy.ndarray): line y values.

     Returns:
         float: the coefficient of determination.
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
    """Calculate the Root Mean Square Error (RMSE).

    Args:
        predictions (numpy.ndarray): predicted values.
        targets (numpy.ndarray): actual values.

    Returns:
        float: the RMSE.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())

