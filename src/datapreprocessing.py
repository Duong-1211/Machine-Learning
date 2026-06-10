import os
from typing import Dict, Tuple

import pandas as pd


TRAIN_RAW_PATH = os.path.join("data", "raw", "train.csv")
TEST_RAW_PATH = os.path.join("data", "raw", "test.csv")
PREPROCESSED_DIR = os.path.join("data", "preprocessed")
TRAIN_PREPROCESSED_PATH = os.path.join(PREPROCESSED_DIR, "train_preprocessed.csv")
TEST_PREPROCESSED_PATH = os.path.join(PREPROCESSED_DIR, "test_preprocessed.csv")

# In Ames data these NA values usually mean "feature not present" rather than bad data.
NONE_CATEGORY_COLUMNS = {
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
}


def load_raw_data(
    train_path: str = TRAIN_RAW_PATH,
    test_path: str = TEST_RAW_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, index_col="Id")
    test_df = pd.read_csv(test_path, index_col="Id")
    return train_df, test_df


def _standardize_object_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].apply(lambda val: val.strip() if isinstance(val, str) else val)
    return df


def _build_fill_values(train_X: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    numeric_cols = train_X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = train_X.select_dtypes(include=["object"]).columns.tolist()

    numeric_fill = {
        col: train_X[col].median() for col in numeric_cols
    }

    categorical_fill: Dict[str, object] = {}
    for col in categorical_cols:
        if col in NONE_CATEGORY_COLUMNS:
            categorical_fill[col] = "None"
            continue

        mode_series = train_X[col].dropna().mode()
        categorical_fill[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

    return {
        "numeric": numeric_fill,
        "categorical": categorical_fill,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def _apply_fill_values(df: pd.DataFrame, fill_values: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    df = df.copy()

    for col in fill_values["categorical_cols"]:
        if col in df.columns:
            df[col] = df[col].fillna(fill_values["categorical"].get(col, "Unknown"))

    for col in fill_values["numeric_cols"]:
        if col in df.columns and col in fill_values["numeric"]:
            df[col] = df[col].fillna(fill_values["numeric"][col])

    return df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _standardize_object_values(train_df)
    test_df = _standardize_object_values(test_df)

    y_train = train_df["SalePrice"]
    train_X = train_df.drop(columns=["SalePrice"])

    fill_values = _build_fill_values(train_X)

    train_X_clean = _apply_fill_values(train_X, fill_values)
    test_X_clean = _apply_fill_values(test_df, fill_values)

    train_clean = train_X_clean.copy()
    train_clean["SalePrice"] = y_train

    return train_clean, test_X_clean


def save_preprocessed_data(
    train_clean: pd.DataFrame,
    test_clean: pd.DataFrame,
    train_output_path: str = TRAIN_PREPROCESSED_PATH,
    test_output_path: str = TEST_PREPROCESSED_PATH,
) -> None:
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    train_clean.to_csv(train_output_path)
    test_clean.to_csv(test_output_path)
    print(f"Preprocessed train data saved to {train_output_path}")
    print(f"Preprocessed test data saved to {test_output_path}")


def run_preprocessing(generate_feature_files: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_raw_data()

    train_missing_before = int(train_df.isnull().sum().sum())
    test_missing_before = int(test_df.isnull().sum().sum())

    train_clean, test_clean = preprocess_data(train_df, test_df)
    save_preprocessed_data(train_clean, test_clean)

    train_missing_after = int(train_clean.isnull().sum().sum())
    test_missing_after = int(test_clean.isnull().sum().sum())

    print(f"Train missing values: {train_missing_before} -> {train_missing_after}")
    print(f"Test missing values : {test_missing_before} -> {test_missing_after}")

    if generate_feature_files:
        try:
            from src.features import save_feature_data
        except ModuleNotFoundError:
            from features import save_feature_data

        save_feature_data(
            train_path=TRAIN_PREPROCESSED_PATH,
            test_path=TEST_PREPROCESSED_PATH,
            train_output_path=os.path.join(PREPROCESSED_DIR, "train_features.csv"),
            test_output_path=os.path.join(PREPROCESSED_DIR, "test_features.csv"),
        )

    return train_clean, test_clean


if __name__ == "__main__":
    run_preprocessing(generate_feature_files=True)