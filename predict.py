"""Prediction helpers used by the Streamlit application."""

import joblib
import pandas as pd

from config import FEATURES, MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `python src/generate_data.py` and "
            "`python src/train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_student(values: dict) -> tuple[str, dict]:
    model = load_model()
    row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
    label = str(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    classes = [str(x) for x in model.classes_]
    return label, {
        name: float(probability)
        for name, probability in zip(classes, probabilities)
    }
