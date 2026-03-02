"""
Tests for split integrity: temporal ordering, no overlap violations.

Ensures train/val/test have per-user temporal ordering and no future leakage.
"""

import os
import pytest
import pandas as pd

from src.eval.temporal import TemporalEvaluator


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def _splits_available():
    for name in ("train.csv", "val.csv", "test.csv"):
        if not os.path.exists(os.path.join(DATA_DIR, name)):
            return False
    return True


@pytest.mark.skipif(not _splits_available(), reason="data/processed splits not found")
def test_per_user_temporal_ordering():
    """Per-user ordering: for each user, train < val < test by time."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

    evaluator = TemporalEvaluator(interactions)
    results = evaluator.comprehensive_validation(train, val, test)

    assert results["per_user_ordering"].iloc[0], (
        "Per-user temporal ordering must hold (no future leakage within user)."
    )


@pytest.mark.skipif(not _splits_available(), reason="data/processed splits not found")
def test_no_future_leakage():
    """No future leakage: temporal ordering checks pass."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

    evaluator = TemporalEvaluator(interactions)
    results = evaluator.comprehensive_validation(train, val, test)

    assert results["no_future_leakage"].iloc[0], (
        "no_future_leakage must be True."
    )


def test_temporal_ordering_on_synthetic():
    """Temporal ordering holds on synthetic in-order splits."""
    interactions = pd.DataFrame({
        "user_id": ["u1"] * 6 + ["u2"] * 6,
        "item_id": ["a", "b", "c", "d", "e", "f"] * 2,
        "timestamp": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"] * 2
        ),
    })
    train = interactions.iloc[[0, 1, 6, 7]]
    val = interactions.iloc[[2, 8]]
    test = interactions.iloc[[3, 4, 9, 10]]

    evaluator = TemporalEvaluator(interactions)
    results = evaluator.comprehensive_validation(train, val, test)

    assert results["per_user_ordering"].iloc[0]
    assert results["no_future_leakage"].iloc[0]
