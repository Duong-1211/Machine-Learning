import numpy as np


def rmse(y_true, y_pred):
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return np.sqrt(np.mean(errors ** 2))

def r2_score_custom(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    rss = np.sum((y_true - y_pred) ** 2)
    
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    
    if tss == 0:
        return 0.0
        
    return 1 - (rss / tss)

def mae(y_true, y_pred):
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.mean(errors)

def log_rmse_from_log_predictions(y_true, y_pred_log):
    return rmse(np.log1p(y_true), y_pred_log)

def log_r2_from_log_predictions(y_true, y_pred_log):
    return r2_score_custom(np.log1p(y_true), y_pred_log)

def log_mae_from_log_predictions(y_true, y_pred_log):
    return mae(np.log1p(y_true), y_pred_log)

def print_evaluation_report(y_true, y_pred_log, set_name="EVAL"):

    log_rmse_val = log_rmse_from_log_predictions(y_true, y_pred_log)
    log_r2_val = log_r2_from_log_predictions(y_true, y_pred_log)
    log_mae_val = log_mae_from_log_predictions(y_true, y_pred_log)
    
    y_pred_actual = np.expm1(y_pred_log)
    actual_rmse = rmse(y_true, y_pred_actual)
    actual_r2 = r2_score_custom(y_true, y_pred_actual)
    actual_mae = mae(y_true, y_pred_actual)
    
    print(f"\n================ RESULTS: {set_name.upper()} ================")
    print(f"--- LOG SPACE (Kaggle Metric Optimization) ---")
    print(f"  * Log-RMSE : {log_rmse_val:.5f}")
    print(f"  * Log-MAE  : {log_mae_val:.5f}")
    print(f"  * Log-R2   : {log_r2_val:.4f}")
    print(f"--- Real Space ( USD) ---")
    print(f"  * RMSE     : ${actual_rmse:,.2f}")
    print(f"  * MAE      : ${actual_mae:,.2f}")
    print(f"  * R2 Score : {actual_r2:.4f}")
    print("=========================================================\n")