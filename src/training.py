import os
import warnings

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_DIR,
    TEST_FEATURES_PATH,
    EVAL_FEATURES_PATH,
    TRAIN_FEATURES_PATH,
    DEFAULT_RANDOM_STATE,
)

from src.evaluation import print_evaluation_report

def ensure_feature_data_exists():
    if os.path.exists(TRAIN_FEATURES_PATH) and os.path.exists(TEST_FEATURES_PATH):
        return

    from src.datapreprocessing import run_preprocessing

    run_preprocessing(generate_feature_files=True)


def load_feature_data():
    ensure_feature_data_exists()
    train_df = pd.read_csv(TRAIN_FEATURES_PATH, index_col="Id")
    eval_df = pd.read_csv(EVAL_FEATURES_PATH, index_col="Id")
    test_df = pd.read_csv(TEST_FEATURES_PATH, index_col="Id")
    return train_df, eval_df, test_df


def remove_outliers(df):
    df = df.copy()

    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)

    if "LotArea" in df.columns:
        df = df.drop(df[df["LotArea"] > 100000].index)

    return df


def split_features_target(train_df, eval_df, test_df):
    X_train = train_df.drop(columns=["SalePrice"])
    y_train = train_df["SalePrice"]

    X_eval = eval_df.drop(columns=["SalePrice"])
    y_eval = eval_df["SalePrice"]

    X_test = test_df.copy()
    return X_train, y_train, X_eval, y_eval, X_test


def get_model_path(model_name):
    extension = "pt" if model_name == "deep_learning" else "pkl"
    return os.path.join(MODEL_DIR, f"{model_name}_model.{extension}")


def build_model(model_name, X_train):
    if model_name in {"linear", "ridge", "lasso"}:
        from src.models.regression import build_pipeline

        return build_pipeline(X_train, model_name)

    if model_name == "random_forest":
        from src.models.random_forest import build_pipeline

        return build_pipeline(X_train)

    if model_name == "gradient_boosting":
        from src.models.gradient_boosting import build_pipeline

        return build_pipeline(X_train)

    if model_name == "xgboost":
        from src.models.xgboost import build_pipeline

        return build_pipeline(X_train)

    raise ValueError(f"Unsupported sklearn model: {model_name}")


def save_model(model_name, model):
    model_path = get_model_path(model_name)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if model_name == "deep_learning":
        from src.models.deep_learning import save_model as save_deep_learning_model

        save_deep_learning_model(model, model_path)
        return model_path

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def train_model(model_name, epochs=50, cv_folds=0):
    train_df, eval_df, test_df = load_feature_data()
    train_df = remove_outliers(train_df)
    X_train, y_train, X_eval, y_eval, X_test = split_features_target(train_df, eval_df, test_df)

    if model_name == "deep_learning":
        from src.models.deep_learning import train_model as train_deep_learning_model

        model = train_deep_learning_model(X_train, y_train, epochs=epochs)
        save_model(model_name, model)
        return model, X_test, test_df.index

    model = build_model(model_name, X_train)
    y_log = np.log1p(y_train)

    if cv_folds > 1:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            model,
            X_train,
            y_log,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
        )
        rmse_scores = np.sqrt(-scores)
        print(f"--- {model_name.upper()} Cross-Validation ---")
        print(f"CV RMSE log mean: {rmse_scores.mean():.4f}")
        print(f"CV RMSE log std : {rmse_scores.std():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_log)

    y_pred_eval_log = model.predict(X_eval)
    print_evaluation_report(y_eval, y_pred_eval_log, set_name="Eval")

    save_model(model_name, model)
    return model, X_test, test_df.index
