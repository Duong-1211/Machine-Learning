import os
import joblib
import warnings
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score


TRAIN_PATH = os.path.join("data", "raw", "train.csv")
TEST_PATH = os.path.join("data", "raw", "test.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pkl")


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


def add_features(df):
    df = df.copy()

    if "TotalSF" not in df.columns:
        df["TotalSF"] = (
            df["TotalBsmtSF"].fillna(0)
            + df["1stFlrSF"].fillna(0)
            + df["2ndFlrSF"].fillna(0)
        )

    if "TotalBathrooms" not in df.columns:
        df["TotalBathrooms"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )

    if "TotalPorchSF" not in df.columns:
        df["TotalPorchSF"] = (
            df["OpenPorchSF"].fillna(0)
            + df["EnclosedPorch"].fillna(0)
            + df["3SsnPorch"].fillna(0)
            + df["ScreenPorch"].fillna(0)
        )

    if "HouseAge" not in df.columns:
        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    if "YearsSinceRemodel" not in df.columns:
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    if "IsRemodeled" not in df.columns:
        df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    if "HasGarage" not in df.columns:
        df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)

    if "HasBasement" not in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)

    if "HasFireplace" not in df.columns:
        df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)

    if "HasPool" not in df.columns:
        df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)

    if "OverallQual_TotalSF" not in df.columns:
        df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    return df


def preprocess(train, test):
    n_train = len(train)

    combined = pd.concat([train, test], axis=0)
    combined = add_features(combined)

    y = combined["SalePrice"] if "SalePrice" in combined.columns else None
    X = combined.drop(columns=["SalePrice"], errors="ignore")

    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train] if y is not None else None

    return X_train, X_test, y_train


def build_pipeline(X_train):
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

    model = Ridge(alpha=10)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline


def train_model(X_train, y_train):
    y_log = np.log1p(y_train)

    pipeline = build_pipeline(X_train)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_log,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(-scores)

    print(f"CV RMSE log mean: {rmse_scores.mean():.4f}")
    print(f"CV RMSE log std : {rmse_scores.std():.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_log)

    return pipeline


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict(model, X_test):
    predictions_log = model.predict(X_test)
    predictions = np.expm1(predictions_log)
    return predictions


def create_submission(predictions, ids):
    output_dir = os.path.join("data", "submissions")
    os.makedirs(output_dir, exist_ok=True)

    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions
    })

    submission_path = os.path.join(output_dir, "linear_regression_submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    train, test = load_data()

    train = remove_outliers(train)

    X_train, X_test, y_train = preprocess(train, test)

    model = train_model(X_train, y_train)

    save_model(model)

    preds = predict(model, X_test)

    create_submission(preds, test.index)