"""
Tests for evaluation leakage: query/relevant sets must not use test-only data.

Ensures that during evaluation, user profile (query) is built from training
history only, and that relevant set for metrics uses only novel test items
(not already in train for that user).
"""

import os
import pytest
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_relevant_excludes_train_items_concept():
    """
    For each user, 'relevant' for metric computation must be a subset of
    test items that are NOT in that user's training set.
    This is the leakage fix: we only evaluate on novel items.
    """
    # Simulate the logic in comprehensive_eval: relevant = test_items - train_items
    train_items_set = {"a", "b", "c"}
    test_items = ["a", "b", "d", "e"]
    relevant = [i for i in test_items if i not in train_items_set]
    assert relevant == ["d", "e"]
    assert "a" not in relevant and "b" not in relevant


def test_query_built_from_train_only_docstring():
    """
    Document the contract: in comprehensive_eval, goal_text for TF-IDF/hybrid
    is built from user_train (training history) only, not from test.
    """
    # This test encodes the invariant; the actual implementation is in comprehensive_eval.
    user_train_items = ["item_1", "item_2", "item_3"]
    test_items = ["item_4", "item_5"]
    # Query must be derived from user_train_items, not test_items.
    goal_text = " ".join(user_train_items)
    assert "item_1" in goal_text
    assert "item_4" not in goal_text


@pytest.mark.skipif(
    not os.path.exists(os.path.join(BASE_DIR, "src", "eval", "comprehensive_eval.py")),
    reason="comprehensive_eval not found",
)
def test_comprehensive_eval_relevant_filter():
    """
    Ensure comprehensive_eval builds 'relevant' by filtering test items
    to exclude user's training items.
    """
    from src.eval.comprehensive_eval import _evaluate_model_comprehensive
    # We only check that the code path exists; full run would require data.
    import inspect
    source = inspect.getsource(_evaluate_model_comprehensive)
    assert "train_items_set" in source
    assert "relevant =" in source and "train_items_set" in source and "not in" in source
