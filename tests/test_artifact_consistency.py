"""
Tests that canonical artifact CSVs exist and have expected structure and bounds.

If results/final/ or results/ contain comprehensive_metrics.csv, check that
precision_mean, recall_mean, ndcg_mean are in [0, 1] and required columns exist.
"""

import os
import pytest
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _find_metrics_path():
    for sub in ("results/final", "results"):
        path = os.path.join(BASE_DIR, sub, "comprehensive_metrics.csv")
        if os.path.exists(path):
            return path
    return None


@pytest.mark.skipif(_find_metrics_path() is None, reason="comprehensive_metrics.csv not found")
def test_comprehensive_metrics_columns():
    """Canonical metrics CSV has expected columns."""
    path = _find_metrics_path()
    df = pd.read_csv(path)
    required = {"model", "precision_mean", "recall_mean", "ndcg_mean"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"


@pytest.mark.skipif(_find_metrics_path() is None, reason="comprehensive_metrics.csv not found")
def test_comprehensive_metrics_bounds():
    """precision_mean, recall_mean, ndcg_mean are in [0, 1]."""
    path = _find_metrics_path()
    df = pd.read_csv(path)
    for col in ("precision_mean", "recall_mean", "ndcg_mean"):
        if col not in df.columns:
            continue
        assert (df[col] >= 0.0).all(), f"{col} has values < 0"
        assert (df[col] <= 1.0).all(), f"{col} has values > 1"


@pytest.mark.skipif(_find_metrics_path() is None, reason="comprehensive_metrics.csv not found")
def test_comprehensive_metrics_models():
    """Expected models appear in comprehensive_metrics."""
    path = _find_metrics_path()
    df = pd.read_csv(path)
    models = set(df["model"].astype(str).str.lower())
    expected = {"popularity", "tfidf", "itemknn", "hybrid"}
    assert expected.issubset(models), f"Missing models: {expected - models}"
