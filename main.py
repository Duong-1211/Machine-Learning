import argparse

from src.config import (
    DEFAULT_DEEP_LEARNING_EPOCHS,
    DEFAULT_MODEL,
    MODEL_NAMES,
)
from src.prediction import predict
from src.submission import create_submission
from src.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a house-price model and create a submission.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODEL_NAMES,
        help="Model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_DEEP_LEARNING_EPOCHS,
        help="Epochs used only for the deep_learning model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Optional sklearn cross-validation folds. Use 0 to skip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model, X_test, test_index = train_model(
        args.model,
        epochs=args.epochs,
        cv_folds=args.cv_folds,
    )
    predictions = predict(args.model, model, X_test)
    create_submission(predictions, test_index, args.model)


if __name__ == "__main__":
    main()

