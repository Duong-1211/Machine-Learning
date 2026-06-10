import os
import argparse
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score


TRAIN_PATH = os.path.join("data", "preprocessed", "train_features.csv")
TEST_PATH = os.path.join("data", "preprocessed", "test_features.csv")

MODEL_DIR = "models"


def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col="Id")
    test = pd.read_csv(TEST_PATH, index_col="Id")
    return train, test


def remove_outliers(df):
    df = df.copy()

    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)

    if "LotArea" in df.columns:
        df = df.drop(df[df["LotArea"] > 100000].index)

    return df


def preprocess(train, test):
    n_train = len(train)

    combined = pd.concat([train, test], axis=0)

    y = combined["SalePrice"]
    X = combined.drop(columns=["SalePrice"], errors="ignore")

    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]

    return X_train, X_test, y_train


def choose_model(model_name):
    if model_name == "linear":
        return LinearRegression()

    if model_name == "ridge":
        return Ridge(alpha=10)

    if model_name == "lasso":
        return Lasso(alpha=0.001, max_iter=10000)

    raise ValueError("model_name must be one of: linear, ridge, lasso")


def build_pipeline(X_train, model_name):
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model = choose_model(model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline


def train_model(X_train, y_train, model_name):
    y_log = np.log1p(y_train)

    pipeline = build_pipeline(X_train, model_name)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_log,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(-scores)

    print(f"Model: {model_name}")
    print(f"CV RMSE log mean: {rmse_scores.mean():.4f}")
    print(f"CV RMSE log std : {rmse_scores.std():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_log)

    return pipeline


def get_model_path(model_name):
    return os.path.join(MODEL_DIR, f"{model_name}_regression_model.pkl")


def save_model(model, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = get_model_path(model_name)
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


def load_model(model_name):
    model_path = get_model_path(model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    return joblib.load(model_path)


def predict(model, X_test):
    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    return predictions


def create_submission(predictions, ids, model_name):
    output_dir = os.path.join("data", "submissions")
    os.makedirs(output_dir, exist_ok=True)

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions
    })

    submission_path = os.path.join(
        output_dir,
        f"{model_name}_regression_submission.csv"
    )

    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "ridge", "lasso"],
        help="Choose regression model: linear, ridge, or lasso"
    )

    args = parser.parse_args()
    model_name = args.model

    train, test = load_data()

    train = remove_outliers(train)

    X_train, X_test, y_train = preprocess(train, test)

    model = train_model(X_train, y_train, model_name)

    save_model(model, model_name)

    preds = predict(model, X_test)

    create_submission(preds, test.index, model_name)


if __name__ == "__main__":
    main()