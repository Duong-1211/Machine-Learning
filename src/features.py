import os
import pandas as pd
import numpy as np


def add_features(df):
    df = df.copy()

    #Total square feet
    if "TotalSF" not in df.columns:
        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
            df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Total bathrooms
    if "TotalBathrooms" not in df.columns:
        needed_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
        if needed_cols.issubset(df.columns):
            df["TotalBathrooms"] = (
                df["FullBath"]
                + 0.5 * df["HalfBath"]
                + df["BsmtFullBath"]
                + 0.5 * df["BsmtHalfBath"]
            )

    #Total porch area
    if "TotalPorchSF" not in df.columns:
        needed_cols = {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}
        if needed_cols.issubset(df.columns):
            df["TotalPorchSF"] = (
                df["OpenPorchSF"]
                + df["EnclosedPorch"]
                + df["3SsnPorch"]
                + df["ScreenPorch"]
            )

    #House age
    if "HouseAge" not in df.columns:
        if {"YrSold", "YearBuilt"}.issubset(df.columns):
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    #Years since remodel
    if "YearsSinceRemodel" not in df.columns:
        if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
            df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    #Whether house was remodeled
    if "IsRemodeled" not in df.columns:
        if {"YearBuilt", "YearRemodAdd"}.issubset(df.columns):
            df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)

    #Has second floor
    if "Has2ndFloor" not in df.columns:
        if "2ndFlrSF" in df.columns:
            df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    #Has basement
    if "HasBasement" not in df.columns:
        if "TotalBsmtSF" in df.columns:
            df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

    #Has garage
    if "HasGarage" not in df.columns:
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    #Has fireplace
    if "HasFireplace" not in df.columns:
        if "Fireplaces" in df.columns:
            df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    #Has pool
    if "HasPool" not in df.columns:
        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)

    #Overall quality x total square feet
    if "OverallQual_TotalSF" not in df.columns:
        if {"OverallQual", "TotalSF"}.issubset(df.columns):
            df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

    #Overall quality x garage area
    if "OverallQual_GarageArea" not in df.columns:
        if {"OverallQual", "GarageArea"}.issubset(df.columns):
            df["OverallQual_GarageArea"] = df["OverallQual"] * df["GarageArea"]

    return df


def process_feature_file(input_path, output_path):
    df = pd.read_csv(input_path, index_col="Id")

    old_columns = set(df.columns)

    df = add_features(df)

    new_columns = set(df.columns) - old_columns

    if len(new_columns) == 0:
        print(f"No new features added. Data already has feature columns: {input_path}")
    else:
        print(f"Added {len(new_columns)} new features:")
        for col in sorted(new_columns):
            print(f"- {col}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"Feature data saved to: {output_path}")


def save_feature_data(
    train_path="data/preprocessed/train_preprocessed.csv",
    test_path="data/preprocessed/test_preprocessed.csv",
    train_output_path="data/preprocessed/train_features.csv",
    test_output_path="data/preprocessed/test_features.csv"
):
    process_feature_file(train_path, train_output_path)
    process_feature_file(test_path, test_output_path)


if __name__ == "__main__":
    save_feature_data()