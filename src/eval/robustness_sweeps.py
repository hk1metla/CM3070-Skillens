# Test-set sensitivity analysis. These sweeps do not select the reported fusion weights.
"""
Robustness sweeps for Skillens evaluation.

Runs K sweep, fusion weight sweep, and ItemKNN neighbourhood sweep.

Usage:
    python -m src.eval.robustness_sweeps --out results/final
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k
from src.models.hybrid import HybridRecommender
from src.models.itemknn import ItemKNNRecommender
from src.models.popularity import PopularityRecommender
from src.models.tfidf import TfidfRecommender

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def _load_data():
    """Load and filter data to OULAD only."""
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"), dtype={"item_id": str})
    items = items[items["item_id"].str.startswith("oulad_", na=False)]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), dtype={"user_id": str, "item_id": str})
    train = train[train["item_id"].str.startswith("oulad_", na=False)]
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), dtype={"user_id": str, "item_id": str})
    test = test[test["item_id"].str.startswith("oulad_", na=False)]
    return items, train, test


def _evaluate_model(model_name, model, items, test_df, train_df, k,
                    _train_grouped=None, _test_grouped=None):
    """Evaluate a single model and return mean NDCG, Precision, Recall."""
    ndcg_scores = []
    prec_scores = []
    recall_scores = []
    item_lookup = items.set_index("item_id")

    # Use pre-grouped data if available for performance
    test_grouped = _test_grouped if _test_grouped is not None else dict(list(test_df.groupby("user_id")))
    train_grouped = _train_grouped if _train_grouped is not None else dict(list(train_df.groupby("user_id")))

    for user_id, user_test in test_grouped.items():
        user_train = train_grouped.get(user_id, pd.DataFrame())
        train_items_set = set(user_train["item_id"].tolist()) if len(user_train) > 0 else set()
        relevant = [item for item in user_test["item_id"].tolist() if item not in train_items_set]

        if not relevant:
            continue

        if model_name == "popularity":
            recs_df = model.recommend(k=k)
        elif model_name == "tfidf":
            if len(user_train) == 0:
                continue
            train_items = user_train["item_id"].tolist()
            goal_text = " ".join(
                item_lookup.loc[iid, "title"]
                for iid in train_items[:3]
                if iid in item_lookup.index
            )
            if not goal_text.strip():
                continue
            recs_df = model.recommend(goal_text, k=k)
        elif model_name == "itemknn":
            if len(user_train) == 0:
                continue
            recs_df = model.recommend(user_id, user_train, k=k)
        elif model_name == "hybrid":
            if len(user_train) == 0:
                continue
            train_items = user_train["item_id"].tolist()
            goal_text = " ".join(
                item_lookup.loc[iid, "title"]
                for iid in train_items[:3]
                if iid in item_lookup.index
            )
            if not goal_text.strip():
                continue
            recs_df = model.recommend(
                goal=goal_text, user_id=user_id,
                user_interactions=user_train, k=k,
            )
        else:
            continue

        recs = recs_df["item_id"].tolist()
        if not recs:
            continue

        ndcg_scores.append(ndcg_at_k(recs, relevant, k))
        prec_scores.append(precision_at_k(recs, relevant, k))
        recall_scores.append(recall_at_k(recs, relevant, k))

    return {
        "ndcg": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
        "precision": round(float(np.mean(prec_scores)), 4) if prec_scores else 0.0,
        "recall": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
        "n_users": len(ndcg_scores),
    }


def run_k_sweep(items, train, test, out_dir, train_grouped=None, test_grouped=None):
    """Sweep K in {5, 10, 20} for all models."""
    print("Running K sweep...")
    results = []

    # Fit models once
    popularity = PopularityRecommender()
    popularity.fit(train)
    tfidf = TfidfRecommender()
    tfidf.fit(items)
    itemknn = ItemKNNRecommender(k=50)
    itemknn.fit(train)
    hybrid = HybridRecommender(w_content=0.6, w_cf=0.4)
    hybrid.fit(items, train)

    models = {
        "popularity": popularity,
        "tfidf": tfidf,
        "itemknn": itemknn,
        "hybrid": hybrid,
    }

    for k in [5, 10, 20]:
        for name, model in models.items():
            metrics = _evaluate_model(name, model, items, test, train, k,
                                      _train_grouped=train_grouped, _test_grouped=test_grouped)
            results.append({"k": k, "model": name, **metrics})
            print(f"  K={k}, {name}: NDCG={metrics['ndcg']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "k_sweep.csv"), index=False)
    print(f"K sweep saved to {out_dir}/k_sweep.csv\n", flush=True)
    return df


def run_weight_sweep(items, train, test, out_dir, k=10, train_grouped=None, test_grouped=None):
    """Sweep hybrid fusion weight from 0.0 to 1.0."""
    print("Running weight sweep...")
    results = []

    for w_c in [round(i / 10, 1) for i in range(11)]:
        w_k = round(1 - w_c, 1)
        hybrid = HybridRecommender(w_content=w_c, w_cf=w_k)
        hybrid.fit(items, train)
        metrics = _evaluate_model("hybrid", hybrid, items, test, train, k,
                                  _train_grouped=train_grouped, _test_grouped=test_grouped)
        results.append({"w_content": w_c, "w_cf": w_k, **metrics})
        print(f"  w_content={w_c}, w_cf={w_k}: NDCG={metrics['ndcg']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "weight_sweep.csv"), index=False)
    print(f"Weight sweep saved to {out_dir}/weight_sweep.csv\n", flush=True)
    return df


def run_knn_k_sweep(items, train, test, out_dir, k=10, train_grouped=None, test_grouped=None):
    """Sweep ItemKNN neighbourhood size in {25, 50, 100}."""
    print("Running ItemKNN k sweep...")
    results = []

    for knn_k in [25, 50, 100]:
        knn = ItemKNNRecommender(k=knn_k)
        knn.fit(train)
        metrics = _evaluate_model("itemknn", knn, items, test, train, k,
                                  _train_grouped=train_grouped, _test_grouped=test_grouped)
        results.append({"knn_k": knn_k, **metrics})
        print(f"  knn_k={knn_k}: NDCG={metrics['ndcg']:.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "knn_k_sweep.csv"), index=False)
    print(f"KNN k sweep saved to {out_dir}/knn_k_sweep.csv\n", flush=True)
    return df


def run_all_sweeps(out_dir: str = None):
    """Run all robustness sweeps."""
    save_dir = out_dir or os.path.join(BASE_DIR, "results", "final")
    os.makedirs(save_dir, exist_ok=True)

    items, train, test = _load_data()
    print(f"Loaded {len(items)} items, {len(train)} train, {len(test)} test interactions")

    # Pre-group data once (avoids repeated O(N) scans per evaluation)
    print("Pre-grouping data by user_id...", flush=True)
    train_grouped = dict(list(train.groupby("user_id")))
    test_grouped = dict(list(test.groupby("user_id")))
    print(f"Grouped {len(train_grouped)} train users, {len(test_grouped)} test users\n", flush=True)

    run_k_sweep(items, train, test, save_dir, train_grouped, test_grouped)
    run_weight_sweep(items, train, test, save_dir, train_grouped=train_grouped, test_grouped=test_grouped)
    run_knn_k_sweep(items, train, test, save_dir, train_grouped=train_grouped, test_grouped=test_grouped)

    print("All robustness sweeps complete.")


def main():
    parser = argparse.ArgumentParser(description="Run robustness sweeps")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "results", "final"))
    args = parser.parse_args()
    run_all_sweeps(out_dir=args.out)


if __name__ == "__main__":
    main()
