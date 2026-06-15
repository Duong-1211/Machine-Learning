import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import DEFAULT_RANDOM_STATE,MODEL_NAMES, EVAL_FEATURES_PATH, TRAIN_FEATURES_PATH, TEST_FEATURES_PATH
from src.training import train_model, remove_outliers, split_features_target
from src.evaluation import log_rmse_from_log_predictions, r2_score_custom, mae
from src.prediction import predict as general_predict


def run_custom_cross_validation(model_name, X_train_full, y_train_full, cv_folds=5):

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    fold_rmse_scores = []
    
    print(f"   Executing {cv_folds}-Fold CV validation process...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), start=1):

        X_tr_fold, X_va_fold = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr_fold, y_va_fold = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        try:

            trained_fold_object, _, _ = train_model(model_name, cv_folds=0)
            
            y_pred_val_actual = general_predict(model_name, trained_fold_object, X_va_fold)
            y_pred_val_log = np.log1p(y_pred_val_actual)
            
            fold_score = log_rmse_from_log_predictions(y_va_fold, y_pred_val_log)
            fold_rmse_scores.append(fold_score)
            print(f"      -> Fold {fold}/{cv_folds} Log-RMSE: {fold_score:.5f}")
            
        except Exception as e:
            print(f"      -> Fold {fold}/{cv_folds} failed due to internal error: {e}")
            continue
            
    if len(fold_rmse_scores) == 0:
        return np.nan, np.nan
        
    return np.mean(fold_rmse_scores), np.std(fold_rmse_scores)


def evaluate_models_scientifically(cv_folds=5):
    train_df = pd.read_csv(TRAIN_FEATURES_PATH, index_col="Id")
    eval_df = pd.read_csv(EVAL_FEATURES_PATH, index_col="Id")
    test_df = pd.read_csv(TEST_FEATURES_PATH, index_col="Id")

    train_df = remove_outliers(train_df)
    X_train, y_train, X_eval, y_eval, X_test = split_features_target(train_df, eval_df, test_df)
    
    comparison_metrics = []
    
    for model_name in MODEL_NAMES:
        print(f"-> Evaluating model architecture: {model_name.upper()}")
        try:
            start_time = time.time()
            trained_object, _, _ = train_model(model_name, cv_folds=0)
            training_duration = time.time() - start_time

            y_pred_train_actual = general_predict(model_name, trained_object, X_train)
            y_pred_eval_actual = general_predict(model_name, trained_object, X_eval)
            
            y_pred_train_log = np.log1p(y_pred_train_actual)
            y_pred_eval_log = np.log1p(y_pred_eval_actual)
            
            cv_mean, cv_std = run_custom_cross_validation(model_name, X_train, y_train, cv_folds=cv_folds)
            
            comparison_metrics.append({
                "Model Algorithm": model_name.upper(),
                "Train Log-RMSE": round(log_rmse_from_log_predictions(y_train, y_pred_train_log), 5),
                "Eval Log-RMSE": round(log_rmse_from_log_predictions(y_eval, y_pred_eval_log), 5),
                "CV Log-RMSE (Mean)": round(cv_mean, 5) if not np.isnan(cv_mean) else "N/A",
                "CV Log-RMSE (Std)": round(cv_std, 5) if not np.isnan(cv_std) else "N/A",
                "Eval MAE ($)": round(mae(y_eval, y_pred_eval_actual), 2),
                "Eval R² Score": round(r2_score_custom(y_eval, y_pred_eval_actual), 4),
                "Train Time (sec)": round(training_duration, 4)
            })
            
        except Exception as e:
            print(f"Skipping execution for {model_name}: {e}")
            continue

    df_comparison = pd.DataFrame(comparison_metrics)
    df_comparison = df_comparison.sort_values(by="Eval Log-RMSE").reset_index(drop=True)
    return df_comparison


if __name__ == "__main__":
    import os
    report = evaluate_models_scientifically(cv_folds=5)
    
    print("\nMODEL PERFORMANCE BENCHMARK REPORT")
    print("=" * 115)
    print(report.to_string(index=False))
    print("=" * 115)
    
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "model_benchmark.csv")
    report.to_csv(csv_path, index=False)
    print(f"CSV data file saved at: {csv_path}")
    
    txt_path = os.path.join(output_dir, "model_benchmark_report.txt")
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("===================================================================================\n")
        f.write(f"MODEL PERFORMANCE BENCHMARK REPORT (Generated at: {current_time})\n")
        f.write("===================================================================================\n\n")
        f.write(report.to_string(index=False))
        f.write("\n\n* Note: Lower Eval Log-RMSE values indicate superior predictive accuracy.")
        
    print(f"TXT human-readable report saved at: {txt_path}")