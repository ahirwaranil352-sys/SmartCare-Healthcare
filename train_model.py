"""Train, evaluate and save the student-support risk classifier."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, FEATURES, MODEL_DIR, MODEL_PATH, REPORTS_DIR, TARGET


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",
        )),
    ])


def train_and_evaluate() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    report = classification_report(y_test, predictions, output_dict=True)
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "classification_report": report}, f, indent=2)

    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation="vertical")
    plt.title("Student Support-Risk Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    print("\nModel evaluation")
    print("----------------")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print(f"\nSaved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
