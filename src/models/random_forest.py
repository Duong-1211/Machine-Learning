from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DEFAULT_RANDOM_STATE


def build_pipeline(X_train) -> Pipeline:
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
		n_estimators=500,
		random_state=DEFAULT_RANDOM_STATE,
		n_jobs=-1,
	)

	pipeline = Pipeline(
		steps=[
			("preprocessor", preprocessor),
			("regressor", model),
		]
	)
	return pipeline
