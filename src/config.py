import os


RAW_DATA_DIR = os.path.join("data", "raw")
PREPROCESSED_DIR = os.path.join("data", "preprocessed")
SUBMISSION_DIR = os.path.join("data", "submissions")
MODEL_DIR = "models"

TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_RAW_PATH = os.path.join(RAW_DATA_DIR, "test.csv")
TRAIN_FEATURES_PATH = os.path.join(PREPROCESSED_DIR, "train_features.csv")
TEST_FEATURES_PATH = os.path.join(PREPROCESSED_DIR, "test_features.csv")
EVAL_FEATURES_PATH = os.path.join(PREPROCESSED_DIR, "eval_features.csv")

TRAIN_PREPROCESSED_PATH = os.path.join(PREPROCESSED_DIR, "train_preprocessed.csv")
EVAL_PREPROCESSED_PATH = os.path.join(PREPROCESSED_DIR, "eval_preprocessed.csv")  # Thêm mới
TEST_PREPROCESSED_PATH = os.path.join(PREPROCESSED_DIR, "test_preprocessed.csv")

MODEL_NAMES = (
    "linear",
    "ridge",
    "lasso",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "deep_learning",
)

DEFAULT_TRAIN_EVAL_SPLIT = 0.2
DEFAULT_MODEL = "ridge"
DEFAULT_RANDOM_STATE = 42
DEFAULT_DEEP_LEARNING_EPOCHS = 50
