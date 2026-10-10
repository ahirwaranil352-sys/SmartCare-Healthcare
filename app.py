"""Streamlit dashboard for the Student Success AI demo."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PATH, MODEL_PATH
from predict import predict_student


st.set_page_config(page_title="Student Success AI", page_icon="🎓", layout="wide")

st.title("🎓 Student Success AI")
st.subheader("AI-powered academic support-risk demonstration")

st.info(
    "Educational demo using synthetic data. Predictions are not validated "
    "for real academic decisions."
)

if not MODEL_PATH.exists():
    st.warning(
        "Model not trained yet. Run:\n\n"
        "`python src/generate_data.py`\n\n"
        "`python src/train_model.py`"
    )
    st.stop()

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    c1, c2, c3 = st.columns(3)
    c1.metric("Demo students", len(df))
    c2.metric("Features", 6)
    c3.metric("Model", "Random Forest")
    st.markdown("### Demo dataset")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.markdown("### Predict support-risk category")

left, right = st.columns(2)

with left:
    attendance = st.slider("Attendance (%)", 35.0, 100.0, 78.0)
    study_hours = st.slider("Study hours / week", 1.0, 30.0, 12.0)
    assignment = st.slider("Assignment completion (%)", 20.0, 100.0, 76.0)

with right:
    previous_score = st.slider("Previous score (%)", 20.0, 100.0, 70.0)
    participation = st.slider("Participation (%)", 10.0, 100.0, 72.0)
    late_submissions = st.slider("Late submissions", 0, 10, 1)

if st.button("Predict Support Risk", type="primary", use_container_width=True):
    values = {
        "attendance_pct": attendance,
        "study_hours_per_week": study_hours,
        "assignment_completion_pct": assignment,
        "previous_score_pct": previous_score,
        "participation_pct": participation,
        "late_submissions": late_submissions,
    }

    label, probabilities = predict_student(values)

    if label == "High":
        st.error(f"Predicted support risk: **{label}**")
    elif label == "Medium":
        st.warning(f"Predicted support risk: **{label}**")
    else:
        st.success(f"Predicted support risk: **{label}**")

    probability_df = pd.Series(
        probabilities, name="Probability"
    ).sort_values(ascending=False).to_frame()
    probability_df["Probability"] = probability_df["Probability"].map(
        lambda x: f"{x:.1%}"
    )
    st.markdown("#### Model probabilities")
    st.dataframe(probability_df, use_container_width=True, hide_index=True)

st.caption(
    "AICTE × IBM SkillsBuild learning project. Human review and appropriate "
    "institutional safeguards are required for real use."
)
