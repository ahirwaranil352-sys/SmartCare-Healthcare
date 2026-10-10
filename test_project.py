from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import FEATURES, TARGET
from generate_data import generate_dataset


def test_generated_dataset_has_expected_schema():
    df = generate_dataset(n_samples=100)
    assert len(df) == 100
    assert all(feature in df.columns for feature in FEATURES)
    assert TARGET in df.columns


def test_generated_values_are_in_expected_ranges():
    df = generate_dataset(n_samples=100)
    assert df["attendance_pct"].between(35, 100).all()
    assert df["assignment_completion_pct"].between(20, 100).all()
    assert df["previous_score_pct"].between(20, 100).all()
    assert df["participation_pct"].between(10, 100).all()
    assert df["late_submissions"].between(0, 10).all()
    assert set(df[TARGET].unique()) <= {"Low", "Medium", "High"}


def test_dataset_is_reproducible():
    first = generate_dataset(n_samples=20, seed=7)
    second = generate_dataset(n_samples=20, seed=7)
    pd.testing.assert_frame_equal(first, second)
