import os
import joblib
import warnings
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


TRAIN_PATH = os.path.join('data', 'raw', 'train.csv')
TEST_PATH = os.path.join('data', 'raw', 'test.csv')
MODEL_DIR = os.path.join('models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')


ORDINAL_MAPPINGS = {
    'LotShape':     {'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0},
    'LandSlope':    {'Gtl': 2, 'Mod': 1, 'Sev': 0},
    'ExterQual':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'ExterCond':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'BsmtQual':     {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'BsmtCond':     {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0},
    'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0},
    'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0},
    'HeatingQC':    {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'KitchenQual':  {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'FireplaceQu':  {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'GarageQual':   {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'GarageCond':   {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0},
    'PoolQC':       {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NA': 0},
    'Functional':   {'Typ': 6, 'Min1': 5, 'Min2': 4, 'Mod': 3, 'Maj1': 2, 'Maj2': 1, 'Sev': 0},
    'PavedDrive':   {'Y': 2, 'P': 1, 'N': 0},
    'Fence':        {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0},
}


def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col='Id')
    test = pd.read_csv(TEST_PATH, index_col='Id')
    return train, test


def remove_outliers(df):
    df = df.copy()
    df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)
    df = df.drop(df[(df['LotArea'] > 100000)].index)
    df = df.reset_index(drop=True)
    return df


def engineer_features(df):
    df = df.copy()
    df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath']
                       + df['BsmtFullBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch']
                          + df['3SsnPorch'] + df['ScreenPorch'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    df['IsNew'] = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['HasGarage'] = (~df['GarageType'].isna()).astype(int)
    df['HasBasement'] = (~df['BsmtQual'].isna()).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasFence'] = (~df['Fence'].isna()).astype(int)
    df['OverallQual_Sq'] = df['OverallQual'] ** 2
    df['OverallCond_Sq'] = df['OverallCond'] ** 2
    df['LotArea_log'] = np.log1p(df['LotArea'])
    df['LotFrontage_fill'] = df['LotFrontage'].fillna(0)
    return df


def apply_ordinal(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].fillna('NA').map(mapping).fillna(-1).astype(int)
    return df


NOMINAL_COLS = [
    'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig',
    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
    'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType',
    'GarageFinish', 'MiscFeature', 'SaleType', 'SaleCondition',
]


def preprocess(train, test):
    n_train = len(train)
    combined = pd.concat([train, test], axis=0)

    combined = engineer_features(combined)
    combined = apply_ordinal(combined)

    X = combined.drop(columns=['SalePrice'])
    y = combined['SalePrice'] if 'SalePrice' in combined.columns else None

    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train] if y is not None else None

    return X_train, X_test, y_train


def build_pipeline():
    numeric_cols = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold', 'TotalSF', 'TotalBath', 'TotalPorchSF',
        'HouseAge', 'YearsSinceRemodel', 'IsNew', 'HasGarage', 'HasBasement',
        'HasFireplace', 'HasPool', 'HasFence', 'OverallQual_Sq',
        'OverallCond_Sq', 'LotArea_log', 'LotFrontage_fill',
    ]

    ordinal_cols = list(ORDINAL_MAPPINGS.keys())
    cat_cols_in_pipeline = [c for c in NOMINAL_COLS if c not in ordinal_cols]

    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]), numeric_cols),
    ]

    if cat_cols_in_pipeline:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat_cols_in_pipeline))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    xgb = XGBRegressor(
        n_estimators=1500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', xgb),
    ])

    return pipeline


def train_model(X_train, y_train):
    y_log = np.log1p(y_train)

    pipeline = build_pipeline()

    param_dist = {
        'regressor__max_depth': [4, 5, 6, 7],
        'regressor__learning_rate': [0.01, 0.03, 0.05],
        'regressor__subsample': [0.7, 0.8, 0.9],
        'regressor__colsample_bytree': [0.7, 0.8, 0.9],
        'regressor__reg_alpha': [0, 0.1, 1],
        'regressor__reg_lambda': [0.1, 1, 5],
    }

    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=30, cv=5,
        scoring='neg_mean_squared_error',
        random_state=42, n_jobs=-1, verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(X_train, y_log)

    print(f'Best CV RMSE (log): {-search.best_score_ ** 0.5:.4f}')
    print(f'Best params: {search.best_params_}')
    return search.best_estimator_


def save_model(pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found at {MODEL_PATH}')
    return joblib.load(MODEL_PATH)


def predict(pipeline, X_test):
    return np.expm1(pipeline.predict(X_test))


def create_submission(predictions, ids):
    output_dir = os.path.join('data', 'submissions')
    os.makedirs(output_dir, exist_ok=True)
    submission = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
    submission_path = os.path.join(output_dir, 'xgboost_submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f'Submission saved to {submission_path}')
    return submission_path


if __name__ == '__main__':
    train, test = load_data()
    train = remove_outliers(train)
    X_train, X_test, y_train = preprocess(train, test)
    pipeline = train_model(X_train, y_train)
    save_model(pipeline)
    preds = predict(pipeline, X_test)
    create_submission(preds, test.index)
