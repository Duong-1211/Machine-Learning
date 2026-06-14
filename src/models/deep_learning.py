import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DEFAULT_RANDOM_STATE


@dataclass
class DeepLearningArtifact:
    model: object
    preprocessor: ColumnTransformer


def _build_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
    ]

    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))

    return ColumnTransformer(transformers)


def _build_model(input_dim):
    import torch
    from torch import nn

    torch.manual_seed(DEFAULT_RANDOM_STATE)

    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


def train_model(X_train, y_train, epochs=50, learning_rate=0.001, batch_size=64):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    preprocessor = _build_preprocessor(X_train)
    X_array = preprocessor.fit_transform(X_train).astype(np.float32)
    y_array = np.log1p(y_train.to_numpy()).astype(np.float32).reshape(-1, 1)

    dataset = TensorDataset(
        torch.tensor(X_array, dtype=torch.float32),
        torch.tensor(y_array, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _build_model(X_array.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_X)

        if epoch == 0 or epoch == epochs - 1:
            mean_loss = epoch_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{epochs} - MSE log loss: {mean_loss:.4f}")

    return DeepLearningArtifact(model=model, preprocessor=preprocessor)


def predict(artifact, X_test):
    import torch

    X_array = artifact.preprocessor.transform(X_test).astype(np.float32)
    artifact.model.eval()
    with torch.no_grad():
        pred_log = artifact.model(torch.tensor(X_array, dtype=torch.float32)).numpy().ravel()
    return np.expm1(pred_log)


def save_model(artifact, model_path):
    import torch

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(artifact.model.state_dict(), model_path)
    joblib.dump(artifact.preprocessor, f"{model_path}.preprocessor.pkl")
    print(f"Model saved to {model_path}")
    print(f"Preprocessor saved to {model_path}.preprocessor.pkl")
