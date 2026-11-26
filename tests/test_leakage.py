import pandas as pd
from src.eval.comprehensive_eval import _evaluate_model_comprehensive


def test_only_novel_test_targets_are_scored_and_query_uses_training_text():
    items = pd.DataFrame({"item_id": ["a", "b"], "title": ["earlier title", "held-out title"]})
    train = pd.DataFrame({"user_id": ["eligible", "repeat"], "item_id": ["a", "a"]})
    test = pd.DataFrame({"user_id": ["eligible", "eligible", "repeat"], "item_id": ["a", "b", "a"]})
    queries = []

    class Recorder:
        def recommend(self, goal, k):
            queries.append(goal)
            return pd.DataFrame({"item_id": ["b"]})

    result = _evaluate_model_comprehensive("tfidf", Recorder(), items, test, train, 1,
                                           {}, {}, set(), return_per_user=True)
    assert queries == ["earlier title"]
    assert result["_per_user_metrics"]["recall"] == {"eligible": 1.0}
