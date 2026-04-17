"""Model training entry points."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


DEFAULT_MODEL_PATH = Path("models") / "model.pkl"


def train_classifier(
    dataset_path: str | Path,
    model_output_path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[RandomForestClassifier, float]:
    """Train a Random Forest classifier, print metrics, and save the model."""

    dataset = pd.read_csv(dataset_path)
    if "label" not in dataset.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    features = dataset.drop(columns=["label"])
    labels = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if labels.nunique() > 1 else None,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=model.classes_)

    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(matrix, index=model.classes_, columns=model.classes_))

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    return model, float(accuracy)
