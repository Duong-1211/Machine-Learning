import os

import pandas as pd

from src.config import SUBMISSION_DIR


def create_submission(predictions, ids, model_name):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    submission_path = os.path.join(SUBMISSION_DIR, f"{model_name}_submission.csv")
    submission = pd.DataFrame({
        "Id": ids,
        "SalePrice": predictions,
    })
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to {submission_path}")
    return submission_path
