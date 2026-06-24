import os
import sys
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import add_features

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = (
    "xgboost_model.pkl",
    "random_forest_model.pkl",
    "gradient_boosting_model.pkl",
    "ridge_model.pkl",
    "lasso_model.pkl",
    "linear_model.pkl",
)
FEATURE_TRAIN_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "train_features.csv")


def resolve_model_path():
    model_dir = os.path.join(BASE_DIR, "models")
    for filename in MODEL_CANDIDATES:
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            return candidate

    searched = ", ".join(MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"No trained model found in {model_dir}. Expected one of: {searched}"
    )


@st.cache_resource
def load_model():
    return joblib.load(resolve_model_path())


@st.cache_data
def load_defaults():
    train_df = pd.read_csv(FEATURE_TRAIN_PATH, index_col="Id")
    num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = train_df.select_dtypes(include=["object"]).columns
    defaults = {}
    for c in num_cols:
        defaults[c] = float(train_df[c].median())
    for c in cat_cols:
        defaults[c] = train_df[c].mode().iloc[0]
    defaults.pop("SalePrice", None)
    return defaults, list(train_df.columns)


def build_input_df(form_data, defaults, all_columns):
    row = deepcopy(defaults)
    row.update(form_data)
    df = pd.DataFrame([row])
    df = df[[c for c in all_columns if c != "SalePrice"]]
    df = add_features(df)
    return df


CODE_MAP = {
    "MSZoning": {
        "Residential Low Density": "RL",
        "Residential Medium Density": "RM",
        "Residential High Density": "RH",
        "Floating Village": "FV",
        "Commercial": "C (all)",
    },
    "BldgType": {
        "Single-Family": "1Fam",
        "Two-Family Conversion": "2fmCon",
        "Duplex": "Duplex",
        "Townhouse": "Twnhs",
        "Townhouse End": "TwnhsE",
    },
    "HouseStyle": {
        "One Story": "1Story",
        "One & Half Story Finished": "1.5Fin",
        "One & Half Story Unfinished": "1.5Unf",
        "Two Story": "2Story",
        "Two & Half Story Finished": "2.5Fin",
        "Two & Half Story Unfinished": "2.5Unf",
        "Split Foyer": "SFoyer",
        "Split Level": "SLvl",
    },
    "ExterQual": {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa"},
    "KitchenQual": {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa"},
    "BsmtQual": {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa"},
    "GarageType": {
        "Attached": "Attchd",
        "Detached": "Detchd",
        "Built-In": "BuiltIn",
        "Basement": "Basment",
        "Carport": "CarPort",
        "Two Types": "2Types",
    },
    "GarageFinish": {
        "Finished": "Fin",
        "Rough Finished": "RFn",
        "Unfinished": "Unf",
    },
    "CentralAir": {"Yes": "Y", "No": "N"},
    "PavedDrive": {"Yes": "Y", "Partial": "P", "No": "N"},
}

NEIGHBORHOODS = [
    "Bloomington", "Bluestem", "Briardale", "Brookside", "Clear Creek",
    "College Creek", "Crawford", "Edwards", "Gilbert", "Iowa DOT & Railroad",
    "Meadow Village", "Mitchell", "North Ames", "Northpark Village",
    "Northwest Ames", "Northridge", "Northridge Heights", "Old Town",
    "Southwest Iowa State University", "Sawyer", "Sawyer West",
    "Somerset", "Stone Brook", "Timberland", "Veenker",
]

NEIGHBORHOOD_CODE = {
    "Bloomington": "Blmngtn", "Bluestem": "Blueste", "Briardale": "BrDale",
    "Brookside": "BrkSide", "Clear Creek": "ClearCr", "College Creek": "CollgCr",
    "Crawford": "Crawfor", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "Iowa DOT & Railroad": "IDOTRR", "Meadow Village": "MeadowV",
    "Mitchell": "Mitchel", "North Ames": "NAmes", "Northpark Village": "NPkVill",
    "Northwest Ames": "NWAmes", "Northridge": "NoRidge",
    "Northridge Heights": "NridgHt", "Old Town": "OldTown",
    "Southwest Iowa State University": "SWISU", "Sawyer": "Sawyer",
    "Sawyer West": "SawyerW", "Somerset": "Somerst", "Stone Brook": "StoneBr",
    "Timberland": "Timber", "Veenker": "Veenker",
}

LABEL_MAP = {
    "MSZoning": "Zoning Classification",
    "Neighborhood": "Neighborhood",
    "LotArea": "Lot Area (sq ft)",
    "BldgType": "Building Type",
    "HouseStyle": "House Style",
    "OverallQual": "Overall Quality (1-10)",
    "OverallCond": "Overall Condition (1-10)",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Year Remodeled",
    "ExterQual": "Exterior Quality",
    "KitchenQual": "Kitchen Quality",
    "GrLivArea": "Above Grade Living Area (sq ft)",
    "TotalBsmtSF": "Basement Area (sq ft)",
    "1stFlrSF": "1st Floor Area (sq ft)",
    "2ndFlrSF": "2nd Floor Area (sq ft)",
    "BedroomAbvGr": "Bedrooms Above Grade",
    "TotRmsAbvGrd": "Total Rooms Above Grade",
    "FullBath": "Full Bathrooms",
    "HalfBath": "Half Bathrooms",
    "BsmtFullBath": "Basement Full Bathrooms",
    "BsmtHalfBath": "Basement Half Bathrooms",
    "GarageType": "Garage Type",
    "GarageFinish": "Garage Finish",
    "GarageCars": "Garage Car Capacity",
    "GarageArea": "Garage Area (sq ft)",
    "Fireplaces": "Fireplaces",
    "BsmtQual": "Basement Quality",
    "CentralAir": "Central Air Conditioning",
    "PavedDrive": "Paved Driveway",
    "WoodDeckSF": "Wood Deck Area (sq ft)",
    "OpenPorchSF": "Open Porch Area (sq ft)",
}


def friendly_label(col):
    return LABEL_MAP.get(col, col)


QUALITY_OPTIONS = ["Excellent", "Good", "Average", "Fair"]


def main():
    model = load_model()
    defaults, all_columns = load_defaults()

    st.title("House Price Predictor")
    st.markdown(
        "Adjust the features of a home in the sidebar panel, then click "
        "**Predict Price** to get an instant estimate using a trained XGBoost model."
    )

    with st.sidebar:
        st.header("House Features")

        with st.expander("Location & Zoning", expanded=True):
            ms_zoning = st.selectbox(
                "Zoning Classification",
                list(CODE_MAP["MSZoning"].keys()),
                help="Type of zoning for the property",
                index=0,
            )
            neighborhood = st.selectbox(
                "Neighborhood", NEIGHBORHOODS,
                index=NEIGHBORHOODS.index("North Ames"),
                help="Physical location within Ames city limits",
            )
            lot_area = st.number_input(
                "Lot Area (sq ft)", min_value=1000, max_value=300000,
                value=int(defaults["LotArea"]), step=500,
                help="Total size of the property lot in square feet",
            )
            bldg_type = st.selectbox(
                "Building Type", list(CODE_MAP["BldgType"].keys()),
                help="Type of dwelling",
                index=0,
            )
            house_style = st.selectbox(
                "House Style", list(CODE_MAP["HouseStyle"].keys()),
                help="Architectural style of the home",
                index=0,
            )

        with st.expander("Quality & Condition", expanded=True):
            overall_qual = st.slider(
                "Overall Quality (1-10)", 1, 10, int(defaults["OverallQual"]), 1,
                help="Rates the overall material and finish. 10 = Excellent, 1 = Very Poor",
            )
            overall_cond = st.slider(
                "Overall Condition (1-10)", 1, 10, int(defaults["OverallCond"]), 1,
                help="Rates the overall condition. 10 = Excellent, 1 = Very Poor",
            )
            year_built = st.number_input(
                "Year Built", min_value=1850, max_value=2025,
                value=int(defaults["YearBuilt"]), step=1,
                help="Original construction date",
            )
            year_remod = st.number_input(
                "Year Remodeled", min_value=1850, max_value=2025,
                value=int(defaults["YearRemodAdd"]), step=1,
                help="Year of last remodel (same as Year Built if never remodeled)",
            )
            exter_qual = st.selectbox(
                "Exterior Quality", QUALITY_OPTIONS,
                help="Quality of the exterior material and finish",
                index=QUALITY_OPTIONS.index("Average"),
            )
            kitchen_qual = st.selectbox(
                "Kitchen Quality", QUALITY_OPTIONS,
                help="Quality of the kitchen",
                index=QUALITY_OPTIONS.index("Average"),
            )

        with st.expander("Square Footage & Rooms", expanded=True):
            gr_liv_area = st.number_input(
                "Above Grade Living Area (sq ft)", min_value=300, max_value=7000,
                value=int(defaults["GrLivArea"]), step=50,
                help="Finished square footage above ground (excludes basement)",
            )
            total_bsmt_sf = st.number_input(
                "Basement Area (sq ft)", min_value=0, max_value=7000,
                value=int(defaults["TotalBsmtSF"]), step=50,
                help="Total square footage of basement area",
            )
            first_flr_sf = st.number_input(
                "1st Floor Area (sq ft)", min_value=300, max_value=5000,
                value=int(defaults["1stFlrSF"]), step=50,
                help="Square footage of the first floor",
            )
            second_flr_sf = st.number_input(
                "2nd Floor Area (sq ft)", min_value=0, max_value=3000,
                value=int(defaults["2ndFlrSF"]), step=50,
                help="Square footage of the second floor (0 if no second floor)",
            )
            bed_abv_gr = st.number_input(
                "Bedrooms Above Grade", min_value=0, max_value=10,
                value=int(defaults["BedroomAbvGr"]), step=1,
                help="Number of bedrooms above ground level (excludes basement)",
            )
            tot_rms = st.number_input(
                "Total Rooms Above Grade", min_value=1, max_value=20,
                value=int(defaults["TotRmsAbvGrd"]), step=1,
                help="Total number of rooms above ground (excluding bathrooms)",
            )

        with st.expander("Bathrooms", expanded=False):
            full_bath = st.number_input(
                "Full Bathrooms", min_value=0, max_value=5,
                value=int(defaults["FullBath"]), step=1,
                help="Full bathrooms above grade",
            )
            half_bath = st.number_input(
                "Half Bathrooms", min_value=0, max_value=5,
                value=int(defaults["HalfBath"]), step=1,
                help="Half bathrooms above grade",
            )
            bsmt_full = st.number_input(
                "Basement Full Bathrooms", min_value=0, max_value=5,
                value=int(defaults["BsmtFullBath"]), step=1,
                help="Full bathrooms in the basement",
            )
            bsmt_half = st.number_input(
                "Basement Half Bathrooms", min_value=0, max_value=5,
                value=int(defaults["BsmtHalfBath"]), step=1,
                help="Half bathrooms in the basement",
            )

        with st.expander("Garage", expanded=False):
            garage_type = st.selectbox(
                "Garage Type", list(CODE_MAP["GarageType"].keys()),
                help="Type of garage",
                index=0,
            )
            garage_finish = st.selectbox(
                "Garage Finish", list(CODE_MAP["GarageFinish"].keys()),
                help="Interior finish of the garage",
                index=list(CODE_MAP["GarageFinish"].keys()).index("Unfinished"),
            )
            garage_cars = st.number_input(
                "Car Capacity", min_value=0, max_value=6,
                value=int(defaults["GarageCars"]), step=1,
                help="Number of cars the garage can hold",
            )
            garage_area = st.number_input(
                "Garage Area (sq ft)", min_value=0, max_value=2000,
                value=int(defaults["GarageArea"]), step=50,
                help="Size of the garage in square feet",
            )

        with st.expander("Extra Features", expanded=False):
            fireplaces = st.number_input(
                "Fireplaces", min_value=0, max_value=5,
                value=int(defaults["Fireplaces"]), step=1,
                help="Number of fireplaces in the home",
            )
            bsmt_qual = st.selectbox(
                "Basement Quality", QUALITY_OPTIONS,
                help="Quality of the basement",
                index=QUALITY_OPTIONS.index("Average"),
            )
            central_air = st.selectbox(
                "Central Air Conditioning",
                list(CODE_MAP["CentralAir"].keys()),
                help="Does the home have central air conditioning?",
                index=0,
            )
            paved_drive = st.selectbox(
                "Paved Driveway",
                list(CODE_MAP["PavedDrive"].keys()),
                help="Does the home have a paved driveway?",
                index=0,
            )
            wood_deck = st.number_input(
                "Wood Deck Area (sq ft)", min_value=0, max_value=1500,
                value=int(defaults["WoodDeckSF"]), step=10,
                help="Size of the wood deck in square feet",
            )
            open_porch = st.number_input(
                "Open Porch Area (sq ft)", min_value=0, max_value=1000,
                value=int(defaults["OpenPorchSF"]), step=10,
                help="Size of the open porch in square feet",
            )

        predict_btn = st.button("Predict Price", type="primary", use_container_width=True)

    if predict_btn:
        form_data = {
            "MSZoning": CODE_MAP["MSZoning"][ms_zoning],
            "Neighborhood": NEIGHBORHOOD_CODE[neighborhood],
            "LotArea": lot_area,
            "BldgType": CODE_MAP["BldgType"][bldg_type],
            "HouseStyle": CODE_MAP["HouseStyle"][house_style],
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "ExterQual": CODE_MAP["ExterQual"][exter_qual],
            "KitchenQual": CODE_MAP["KitchenQual"][kitchen_qual],
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "BedroomAbvGr": bed_abv_gr,
            "TotRmsAbvGrd": tot_rms,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BsmtFullBath": bsmt_full,
            "BsmtHalfBath": bsmt_half,
            "GarageType": CODE_MAP["GarageType"][garage_type],
            "GarageFinish": CODE_MAP["GarageFinish"][garage_finish],
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "Fireplaces": fireplaces,
            "BsmtQual": CODE_MAP["BsmtQual"][bsmt_qual],
            "CentralAir": CODE_MAP["CentralAir"][central_air],
            "PavedDrive": CODE_MAP["PavedDrive"][paved_drive],
            "WoodDeckSF": wood_deck,
            "OpenPorchSF": open_porch,
        }

        input_df = build_input_df(form_data, defaults, all_columns)

        pred_log = model.predict(input_df)
        pred_price = float(np.expm1(pred_log)[0])

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Predicted Sale Price",
                value=f"${pred_price:,.0f}",
            )

        st.divider()
        st.subheader("Your Input Summary")
        summary_data = {}
        for raw_key, val in form_data.items():
            label = friendly_label(raw_key)
            summary_data[label] = val
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["Feature", "Value"])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("What Matters Most - Top Features")
        imp = model.named_steps["regressor"].feature_importances_
        try:
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feat_names = [f"f{i}" for i in range(len(imp))]

        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": imp})
        imp_df = imp_df.sort_values("Importance", ascending=False).head(10)
        imp_df["Feature"] = imp_df["Feature"].str.replace("num__", "").str.replace("cat__", "")
        st.caption("These are the top 10 factors the model relies on most to make its prediction.")
        st.bar_chart(imp_df.set_index("Feature"))


if __name__ == "__main__":
    main()
