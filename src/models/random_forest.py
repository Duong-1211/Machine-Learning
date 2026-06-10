import os
import subprocess
import sys
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")
TRAIN_PATH = os.path.join("data", "preprocessed", "train_features.csv")
TEST_PATH = os.path.join("data", "preprocessed", "test_features.csv")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ensure_feature_data_exists() -> None:
	if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
		return

	preprocessing_script = os.path.join(PROJECT_ROOT, "src", "datapreprocessing.py")
	subprocess.run([sys.executable, preprocessing_script], cwd=PROJECT_ROOT, check=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
	_ensure_feature_data_exists()
	train_df = pd.read_csv(TRAIN_PATH, index_col="Id")
	test_df = pd.read_csv(TEST_PATH, index_col="Id")
	return train_df, test_df


def split_features_target(train_df: pd.DataFrame, test_df: pd.DataFrame):
	X_train = train_df.drop(columns=["SalePrice"])
	y_train = train_df["SalePrice"]
	X_test = test_df.copy()
	return X_train, y_train, X_test


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
	numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
	categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
		]
	)

	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore")),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_cols),
			("cat", categorical_pipeline, categorical_cols),
		]
	)

	model = RandomForestRegressor(
		random_state=42,
		n_jobs=-1,
	)

	pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("regressor", model),
		]
	)
	return pipeline


def tune_and_train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
	y_train_log = np.log1p(y_train)

	pipeline = build_pipeline(X_train)

	param_dist = {
		"regressor__n_estimators": [500, 800, 1000, 1200],
		"regressor__max_depth": [None, 12, 18, 24, 32],
		"regressor__min_samples_split": [2, 4, 8, 12],
		"regressor__min_samples_leaf": [1, 2, 4],
		"regressor__max_features": ["sqrt", "log2", 0.6, 0.8],
		"regressor__bootstrap": [True],
	}

	search = RandomizedSearchCV(
		estimator=pipeline,
		param_distributions=param_dist,
		n_iter=25,
		cv=5,
		scoring="neg_mean_squared_error",
		random_state=42,
		n_jobs=-1,
		verbose=1,
	)

	search.fit(X_train, y_train_log)

	best_pipeline = search.best_estimator_

	cv_scores = cross_val_score(
		best_pipeline,
		X_train,
		y_train_log,
		cv=5,
		scoring="neg_mean_squared_error",
		n_jobs=-1,
	)
	cv_rmse = np.sqrt(-cv_scores)

	print(f"Best params: {search.best_params_}")
	print(f"CV RMSE (log) mean: {cv_rmse.mean():.4f}")
	print(f"CV RMSE (log) std : {cv_rmse.std():.4f}")

	return best_pipeline


def save_model(model: Pipeline) -> None:
	os.makedirs(MODEL_DIR, exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	print(f"Model saved to: {MODEL_PATH}")


def load_model() -> Pipeline:
	if not os.path.exists(MODEL_PATH):
		raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
	return joblib.load(MODEL_PATH)


def predict(model: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
	pred_log = model.predict(X_test)
	predictions = np.expm1(pred_log)
	return predictions


def create_submission(predictions: np.ndarray, test_index: pd.Index) -> str:
	output_dir = os.path.join("data", "submissions")
	os.makedirs(output_dir, exist_ok=True)
	submission_path = os.path.join(output_dir, "Random_Forest_Submission.csv")

	submission = pd.DataFrame(
		{
			"Id": test_index,
			"SalePrice": predictions,
		}
	)
	submission.to_csv(submission_path, index=False)
	print(f"Submission saved to: {submission_path}")
	return submission_path


def main() -> None:
	train_df, test_df = load_data()
	X_train, y_train, X_test = split_features_target(train_df, test_df)
	model = tune_and_train_model(X_train, y_train)
	save_model(model)
	predictions = predict(model, X_test)
	create_submission(predictions, test_df.index)


if __name__ == "__main__":
	main()
