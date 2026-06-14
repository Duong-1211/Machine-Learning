import numpy as np


def predict(model_name, model, X_test):
    if model_name == "deep_learning":
        from src.models.deep_learning import predict as predict_deep_learning

        return predict_deep_learning(model, X_test)

    pred_log = model.predict(X_test)
    return np.expm1(pred_log)
