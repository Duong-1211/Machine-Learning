import numpy as np


def rmse(y_true, y_pred):
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return np.sqrt(np.mean(errors ** 2))


def log_rmse_from_log_predictions(y_true, y_pred_log):
    return rmse(np.log1p(y_true), y_pred_log)
