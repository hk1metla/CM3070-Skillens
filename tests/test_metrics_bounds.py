"""
Tests that precision, recall, and NDCG stay in [0, 1].
"""

import pytest
from src.eval.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k_bounds():
    """Precision@K is always in [0, 1]."""
    assert 0.0 <= precision_at_k([], [], 10) <= 1.0
    assert 0.0 <= precision_at_k(["a", "b"], ["a"], 2) <= 1.0
    assert 0.0 <= precision_at_k(["a", "b"], ["a", "b"], 2) <= 1.0
    assert precision_at_k(["a"], ["a"], 1) == 1.0
    assert precision_at_k(["a"], ["b"], 1) == 0.0


def test_recall_at_k_bounds():
    """Recall@K is always in [0, 1]."""
    assert 0.0 <= recall_at_k([], [], 10) <= 1.0
    assert recall_at_k([], ["a"], 10) == 0.0
    assert 0.0 <= recall_at_k(["a", "b"], ["a"], 2) <= 1.0
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert 0.0 <= recall_at_k(["a"], ["a", "b"], 1) <= 1.0


def test_ndcg_at_k_bounds():
    """NDCG@K is always in [0, 1]."""
    assert 0.0 <= ndcg_at_k([], [], 10) <= 1.0
    assert ndcg_at_k([], ["a"], 10) == 0.0
    assert 0.0 <= ndcg_at_k(["a"], ["a"], 1) <= 1.0
    assert ndcg_at_k(["a"], ["a"], 1) == 1.0
    assert ndcg_at_k(["a"], ["b"], 1) == 0.0
    rec = ["a", "b", "c"]
    rel = ["a", "b"]
    assert 0.0 <= ndcg_at_k(rec, rel, 3) <= 1.0
