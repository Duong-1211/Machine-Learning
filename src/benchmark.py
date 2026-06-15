import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.config import MODEL_NAMES, EVAL_FEATURES_PATH, TRAIN_FEATURES_PATH, TEST_FEATURES_PATH
from src.training import train_model, remove_outliers, split_features_target
from src.evaluation import log_rmse_from_log_predictions, r2_score_custom, mae

from src.prediction import predict as general_predict


def evaluate_models_scientifically(cv_folds=5):
    train_df = pd.read_csv(TRAIN_FEATURES_PATH, index_col="Id")
    eval_df = pd.read_csv(EVAL_FEATURES_PATH, index_col="Id")
    test_df = pd.read_csv(TEST_FEATURES_PATH, index_col="Id")

    train_df = remove_outliers(train_df)
    X_train, y_train, X_eval, y_eval, X_test = split_features_target(train_df, eval_df, test_df)
    y_train_log = np.log1p(y_train)
    
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
            
            if model_name == "deep_learning":
                cv_mean, cv_std = np.nan, np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(
                        trained_object, X_train, y_train_log, cv=cv_folds, scoring="neg_mean_squared_error"
                    )
                cv_rmse = np.sqrt(-cv_scores)
                cv_mean, cv_std = cv_rmse.mean(), cv_rmse.std()
            
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