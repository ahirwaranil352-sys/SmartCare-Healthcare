from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_PATH = DATA_DIR / "student_success.csv"
MODEL_PATH = MODEL_DIR / "student_success_model.joblib"

FEATURES = [
    "attendance_pct",
    "study_hours_per_week",
    "assignment_completion_pct",
    "previous_score_pct",
    "participation_pct",
    "late_submissions",
]
TARGET = "support_risk"
RISK_LABELS = ["Low", "Medium", "High"]
