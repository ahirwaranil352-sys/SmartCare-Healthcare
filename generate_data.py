"""Generate reproducible synthetic student records for the learning project."""

import numpy as np
import pandas as pd

from config import DATA_DIR, DATA_PATH


def generate_dataset(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(78, 12, n_samples), 35, 100)
    study_hours = np.clip(rng.normal(12, 5, n_samples), 1, 30)
    assignment = np.clip(rng.normal(76, 15, n_samples), 20, 100)
    previous_score = np.clip(rng.normal(70, 15, n_samples), 20, 100)
    participation = np.clip(rng.normal(72, 18, n_samples), 10, 100)
    late_submissions = np.clip(
        rng.poisson(np.maximum((100 - assignment) / 25, 0.4)), 0, 10
    )

    # Transparent demo rule: this creates labels for learning only.
    risk_score = (
        (100 - attendance) * 0.30
        + (100 - assignment) * 0.22
        + (100 - previous_score) * 0.20
        + (100 - participation) * 0.10
        + np.maximum(8 - study_hours, 0) * 1.4
        + late_submissions * 2.2
        + rng.normal(0, 5, n_samples)
    )

    low_cut, high_cut = np.quantile(risk_score, [0.55, 0.82])
    risk = np.select(
        [risk_score <= low_cut, risk_score <= high_cut],
        ["Low", "Medium"],
        default="High",
    )

    return pd.DataFrame({
        "attendance_pct": np.round(attendance, 1),
        "study_hours_per_week": np.round(study_hours, 1),
        "assignment_completion_pct": np.round(assignment, 1),
        "previous_score_pct": np.round(previous_score, 1),
        "participation_pct": np.round(participation, 1),
        "late_submissions": late_submissions.astype(int),
        "support_risk": risk,
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df)} demo records.")
    print(f"Saved to: {DATA_PATH}")


if __name__ == "__main__":
    main()
