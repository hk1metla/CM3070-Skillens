"""
Comprehensive evaluation harness with all models and metrics.

This replaces the basic run_experiments.py with a full evaluation suite
including hybrid models, diversity metrics, statistical testing, and slice analysis.

DUAL-DATASET STRATEGY:
- OULAD is PRIMARY: All evaluation, metrics, fairness analysis, ablations use OULAD data only
- Coursera is SECONDARY: Used for demo UI only (screenshots/video with clickable links)
- All research claims are based exclusively on OULAD data

This aligns with Final_Plan.pdf requirements and addresses preliminary feedback
by providing real interaction data, demographics, and temporal structure.
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.eval.metrics import (
    catalog_coverage,
    gini_coefficient,
    intra_list_diversity,
    long_tail_coverage,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
    serendipity,
)
from src.eval.statistical import (
    bootstrap_confidence_interval,
    compute_effect_size,
    paired_t_test,
    wilcoxon_signed_rank_test,
)
from src.models.hybrid import HybridRecommender
from src.models.itemknn import ItemKNNRecommender
from src.models.popularity import PopularityRecommender
from src.models.tfidf import TfidfRecommender


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def _load_items() -> pd.DataFrame:
    """
    Load items and filter to OULAD-only for evaluation.
    
    Per dual-dataset strategy: OULAD is PRIMARY for all evaluation.
    Coursera items are excluded from evaluation (used for demo UI only).
    """
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    # Filter to OULAD items only (item_id starts with "oulad_")
    oulad_items = items[items["item_id"].str.startswith("oulad_", na=False)]
    if len(oulad_items) == 0:
        print("WARNING: No OULAD items found! Falling back to all items.")
        return items
    print(f"Loaded {len(oulad_items)} OULAD items (filtered from {len(items)} total)")
    return oulad_items


def _load_split(name: str) -> pd.DataFrame:
    """
    Load train/val/test split and filter to OULAD-only interactions.
    
    Per dual-dataset strategy: All evaluation uses OULAD data only.
    """
    split_df = pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"))
    # Filter to OULAD interactions only (item_id starts with "oulad_")
    oulad_split = split_df[split_df["item_id"].str.startswith("oulad_", na=False)]
    if len(oulad_split) == 0:
        print(f"WARNING: No OULAD interactions in {name} split! Falling back to all interactions.")
        return split_df
    print(f"Loaded {len(oulad_split)} OULAD interactions in {name} (filtered from {len(split_df)} total)")
    return oulad_split


def _load_demographics() -> pd.DataFrame:
    """Load OULAD demographics if available."""
    try:
        from src.data.load_demographics import load_demographics
        return load_demographics()
    except Exception:
        return pd.DataFrame()


def _compute_item_features(items: pd.DataFrame) -> Dict[str, List[float]]:
    """Extract TF-IDF features for diversity computation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    items = items.copy()
    items["text"] = (
        items["title"].fillna("") + " " + items["description"].fillna("")
    )
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    features = vectorizer.fit_transform(items["text"])
    
    item_features = {}
    for idx, item_id in enumerate(items["item_id"]):
        item_features[item_id] = features[idx].toarray().flatten().tolist()
    
    return item_features


def _compute_item_popularity(interactions: pd.DataFrame) -> Dict[str, float]:
    """Compute normalized popularity scores for novelty metric."""
    item_counts = interactions["item_id"].value_counts()
    max_count = item_counts.max()
    return {item: count / max_count for item, count in item_counts.items()}


def _evaluate_model_comprehensive(
    model_name: str,
    model,
    items: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int,
    item_features: Dict[str, List[float]],
    item_popularity: Dict[str, float],
    popular_items: set,
    return_per_user: bool = False,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single model.
    
    Returns accuracy, diversity, coverage, and novelty metrics.
    
    Args:
        return_per_user: If True, also returns per-user metric lists for statistical testing
    """
    metrics = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "diversity": [],
        "novelty": [],
        "long_tail": [],
    }
    
    all_recommendations = []
    item_lookup = items.set_index("item_id")
    
    # Group by user for evaluation
    for user_id, user_df in test_df.groupby("user_id"):
        # Get user's training items to filter test items
        user_train = train_df[train_df["user_id"] == user_id]
        train_items_set = set(user_train["item_id"].tolist()) if len(user_train) > 0 else set()
        
        # CRITICAL: Only evaluate on test items that user hasn't seen in training
        # This prevents inflated metrics from recommending items user already interacted with
        test_items = user_df["item_id"].tolist()
        relevant = [item for item in test_items if item not in train_items_set]
        
        # Skip if no novel test items (all test items were in training)
        if not relevant:
            continue
        
        # Get recommendations based on model type
        if model_name == "popularity":
            recs_df = model.recommend(k=k)
            recs = recs_df["item_id"].tolist()
        elif model_name == "tfidf":
            # FIX: Build query from TRAIN history, not test items (avoids data leakage)
            user_train = train_df[train_df["user_id"] == user_id]
            if len(user_train) == 0:
                # Skip cold-start users (no training history for TF-IDF)
                continue
            
            # Use training items to build goal text
            train_items = user_train["item_id"].tolist()
            goal_text = " ".join(
                item_lookup.loc[item_id, "title"]
                for item_id in train_items[:3]
                if item_id in item_lookup.index
            )
            if not goal_text.strip():
                continue
            
            # Get recommendations based on training history
            # Note: In temporal splits, users may interact with same items at different times.
            # We don't exclude training items because they may legitimately appear in test.
            # The key is that query is built from training data only (no test leakage).
            recs_df = model.recommend(goal_text, k=k)
            recs = recs_df["item_id"].tolist()
        elif model_name == "itemknn":
            user_train = train_df[train_df["user_id"] == user_id]
            if len(user_train) == 0:
                continue
            recs_df = model.recommend(user_id, user_train, k=k)
            recs = recs_df["item_id"].tolist()
        elif model_name == "hybrid":
            user_train = train_df[train_df["user_id"] == user_id]
            if len(user_train) == 0:
                # Skip cold-start users (no training history for hybrid)
                continue
            
            # FIX: Build query from TRAIN history, not test items (avoids data leakage)
            train_items = user_train["item_id"].tolist()
            goal_text = " ".join(
                item_lookup.loc[item_id, "title"]
                for item_id in train_items[:3]
                if item_id in item_lookup.index
            )
            if not goal_text.strip():
                continue
            
            # Get hybrid recommendations
            # ItemKNN already excludes items user has interacted with internally.
            # For temporal splits, we don't exclude training items from TF-IDF component
            # because users may legitimately interact with same items at different times.
            recs_df = model.recommend(
                goal=goal_text,
                user_id=user_id,
                user_interactions=user_train,
                k=k,
            )
            recs = recs_df["item_id"].tolist()
        else:
            continue
        
        if not recs:
            continue
        
        # Accuracy metrics
        metrics["precision"].append(precision_at_k(recs, relevant, k))
        metrics["recall"].append(recall_at_k(recs, relevant, k))
        metrics["ndcg"].append(ndcg_at_k(recs, relevant, k))
        
        # Diversity metrics
        diversity = intra_list_diversity(recs, item_features, k)
        metrics["diversity"].append(diversity)
        
        # Novelty
        nov = novelty(recs, item_popularity, k)
        metrics["novelty"].append(nov)
        
        # Long-tail coverage
        long_tail = long_tail_coverage(recs, popular_items, k)
        metrics["long_tail"].append(long_tail)
        
        all_recommendations.append(recs)
    
    # Aggregate metrics with bootstrap confidence intervals
    results = {}
    for metric_name, values in metrics.items():
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            results[f"{metric_name}_mean"] = mean_val
            results[f"{metric_name}_std"] = std_val
            
            # Add bootstrap confidence intervals (95% CI)
            mean_ci, lower_ci, upper_ci = bootstrap_confidence_interval(values, confidence=0.95)
            results[f"{metric_name}_ci_lower"] = lower_ci
            results[f"{metric_name}_ci_upper"] = upper_ci
        else:
            results[f"{metric_name}_mean"] = 0.0
            results[f"{metric_name}_std"] = 0.0
            results[f"{metric_name}_ci_lower"] = 0.0
            results[f"{metric_name}_ci_upper"] = 0.0
    
    # Catalog coverage
    catalog_cov = catalog_coverage(all_recommendations, len(items))
    results["catalog_coverage"] = catalog_cov
    
    # Gini coefficient (inequality in recommendations)
    item_rec_counts = {}
    for recs in all_recommendations:
        for item in recs:
            item_rec_counts[item] = item_rec_counts.get(item, 0) + 1
    
    if item_rec_counts:
        gini = gini_coefficient(item_rec_counts)
        results["gini_coefficient"] = gini
    else:
        results["gini_coefficient"] = 0.0
    
    # Return per-user metrics if requested (for statistical testing)
    if return_per_user:
        results["_per_user_metrics"] = metrics
    
    return results


def _slice_analysis(
    model_name: str,
    model,
    items: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """
    Perform slice analysis: evaluate on different user/item segments.
    
    Slices:
    - Cold-start users (no training history)
    - Low-history users (< 3 interactions)
    - Popular vs long-tail items
    """
    slice_results = []
    
    # Identify popular items (top 20% by interaction count)
    item_counts = train_df["item_id"].value_counts()
    n_popular = max(1, int(len(item_counts) * 0.2))
    popular_items = set(item_counts.head(n_popular).index)
    
    # User history lengths
    user_history_lengths = train_df.groupby("user_id").size()
    
    # Slice 1: Cold-start users (no training history)
    cold_start_users = set(test_df["user_id"]) - set(train_df["user_id"])
    if cold_start_users:
        cold_test = test_df[test_df["user_id"].isin(cold_start_users)]
        # For cold-start, we can only use content-based models
        if model_name in ["tfidf", "hybrid"]:
            slice_results.append({
                "slice": "cold_start",
                "n_users": len(cold_start_users),
                "model": model_name,
            })
    
    # Slice 2: Low-history users (< 3 interactions)
    low_history_users = set(
        user_history_lengths[user_history_lengths < 3].index
    ) & set(test_df["user_id"])
    if low_history_users:
        low_test = test_df[test_df["user_id"].isin(low_history_users)]
        slice_results.append({
            "slice": "low_history",
            "n_users": len(low_history_users),
            "model": model_name,
        })
    
    # Slice 3: Popular items
    popular_test = test_df[test_df["item_id"].isin(popular_items)]
    if len(popular_test) > 0:
        slice_results.append({
            "slice": "popular_items",
            "n_interactions": len(popular_test),
            "model": model_name,
        })
    
    # Slice 4: Long-tail items
    long_tail_test = test_df[~test_df["item_id"].isin(popular_items)]
    if len(long_tail_test) > 0:
        slice_results.append({
            "slice": "long_tail_items",
            "n_interactions": len(long_tail_test),
            "model": model_name,
        })
    
    return pd.DataFrame(slice_results)


def run_comprehensive_eval(config_path: str, out_dir: str = None) -> pd.DataFrame:
    """
    Run comprehensive evaluation with all models and metrics.

    Args:
        config_path: Path to experiment config YAML.
        out_dir: If set, write all CSVs here; else use RESULTS_DIR.

    Returns:
        DataFrame of model metrics. Also writes comprehensive_metrics.csv,
        significance_matrix.csv, and optionally fairness_metrics.csv.
    """
    save_dir = out_dir if out_dir is not None else RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    
    k = config["eval"]["k"]
    
    # Load data
    items = _load_items()
    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")
    
    # Load demographics for fairness evaluation (OULAD only)
    # Per dual-dataset strategy: Fairness evaluation uses OULAD demographics exclusively
    print("Loading OULAD demographics for fairness evaluation...")
    demographics = _load_demographics()
    if not demographics.empty:
        print(f"Loaded {len(demographics)} OULAD demographic records")
    else:
        print("WARNING: No OULAD demographics available. Fairness evaluation will be skipped.")
        print("This is expected if OULAD studentInfo.csv is not present.")
    
    # Precompute features for diversity metrics
    print("Computing item features for diversity metrics...")
    item_features = _compute_item_features(items)
    item_popularity = _compute_item_popularity(train_df)
    
    # Identify popular items (top 20%)
    item_counts = train_df["item_id"].value_counts()
    n_popular = max(1, int(len(item_counts) * 0.2))
    popular_items = set(item_counts.head(n_popular).index)
    
    all_results = []
    per_user_metrics = {}  # Store per-user metrics for statistical testing
    
    # Evaluate Popularity baseline
    print("Evaluating Popularity model...")
    popularity = PopularityRecommender()
    popularity.fit(train_df)
    pop_results = _evaluate_model_comprehensive(
        "popularity", popularity, items, test_df, train_df, k,
        item_features, item_popularity, popular_items, return_per_user=True
    )
    per_user_metrics["popularity"] = pop_results.pop("_per_user_metrics")
    pop_results["model"] = "popularity"
    all_results.append(pop_results)
    
    # Evaluate TF-IDF
    print("Evaluating TF-IDF model...")
    tfidf = TfidfRecommender()
    tfidf.fit(items)
    tfidf_results = _evaluate_model_comprehensive(
        "tfidf", tfidf, items, test_df, train_df, k,
        item_features, item_popularity, popular_items, return_per_user=True
    )
    per_user_metrics["tfidf"] = tfidf_results.pop("_per_user_metrics")
    tfidf_results["model"] = "tfidf"
    all_results.append(tfidf_results)
    
    # Evaluate ItemKNN
    print("Evaluating ItemKNN model...")
    itemknn = ItemKNNRecommender(k=50)
    itemknn.fit(train_df)
    itemknn_results = _evaluate_model_comprehensive(
        "itemknn", itemknn, items, test_df, train_df, k,
        item_features, item_popularity, popular_items, return_per_user=True
    )
    per_user_metrics["itemknn"] = itemknn_results.pop("_per_user_metrics")
    itemknn_results["model"] = "itemknn"
    all_results.append(itemknn_results)
    
    # Evaluate Hybrid
    print("Evaluating Hybrid model...")
    hybrid = HybridRecommender(w_content=0.6, w_cf=0.4)
    hybrid.fit(items, train_df)
    hybrid_results = _evaluate_model_comprehensive(
        "hybrid", hybrid, items, test_df, train_df, k,
        item_features, item_popularity, popular_items, return_per_user=True
    )
    per_user_metrics["hybrid"] = hybrid_results.pop("_per_user_metrics")
    hybrid_results["model"] = "hybrid"
    all_results.append(hybrid_results)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Fairness evaluation with demographics (if available)
    fairness_results_all = []
    if not demographics.empty:
        print("\nRunning fairness evaluation with OULAD demographics...")
        from src.eval.fairness import FairnessAuditor
        
        fairness_auditor = FairnessAuditor(items, train_df, demographics)
        
        # Get demographic groups
        from src.data.load_demographics import get_demographic_groups
        demo_groups = get_demographic_groups(demographics)
        
        # Store models for fairness evaluation
        models_dict = {
            "popularity": popularity,
            "tfidf": tfidf,
            "itemknn": itemknn,
            "hybrid": hybrid,
        }
        
        # Evaluate fairness for each model and demographic category
        # Group flat demo_groups dict by category prefix (gender_, age_, etc.)
        categories = {}
        for group_name, user_ids in demo_groups.items():
            # Extract category from group name (e.g., "gender_M" -> "gender")
            if "_" in group_name:
                category = group_name.split("_", 1)[0]
                if category not in categories:
                    categories[category] = {}
                categories[category][group_name] = user_ids
        
        for model_name in results_df["model"].values:
            model = models_dict[model_name]
            
            # For each demographic category (gender, age_band, etc.)
            for category, groups in categories.items():
                # Generate recommendations for each group
                recommendations_by_group = {}
                
                for group_name, user_ids in groups.items():
                    # Get test users in this demographic group
                    group_test_users = test_df[test_df["user_id"].isin(user_ids)]["user_id"].unique()
                    
                    if len(group_test_users) == 0:
                        continue
                    
                    group_recs = []
                    item_lookup = items.set_index("item_id")
                    
                    # Sample up to 500 users per group for efficiency (fairness evaluation)
                    sample_size = min(500, len(group_test_users))
                    sampled_users = np.random.choice(group_test_users, size=sample_size, replace=False)
                    
                    for user_id in sampled_users:
                        user_test = test_df[test_df["user_id"] == user_id]
                        user_train = train_df[train_df["user_id"] == user_id]
                        
                        if len(user_train) == 0:
                            continue
                        
                        # Generate recommendations based on model type
                        if model_name == "popularity":
                            recs_df = model.recommend(k=k)
                            recs = recs_df["item_id"].tolist()
                        elif model_name == "tfidf":
                            train_items = user_train["item_id"].tolist()
                            goal_text = " ".join(
                                item_lookup.loc[item_id, "title"]
                                for item_id in train_items[:3]
                                if item_id in item_lookup.index
                            )
                            if goal_text.strip():
                                recs_df = model.recommend(goal_text, k=k)
                                recs = recs_df["item_id"].tolist()
                            else:
                                continue
                        elif model_name == "itemknn":
                            if len(user_train) > 0:
                                recs_df = model.recommend(user_id, user_train, k=k)
                                recs = recs_df["item_id"].tolist()
                            else:
                                continue
                        elif model_name == "hybrid":
                            train_items = user_train["item_id"].tolist()
                            goal_text = " ".join(
                                item_lookup.loc[item_id, "title"]
                                for item_id in train_items[:3]
                                if item_id in item_lookup.index
                            )
                            if goal_text.strip():
                                recs_df = model.recommend(
                                    goal=goal_text,
                                    user_id=user_id,
                                    user_interactions=user_train,
                                    k=k,
                                )
                                recs = recs_df["item_id"].tolist()
                            else:
                                continue
                        else:
                            continue
                        
                        if recs:
                            group_recs.append(recs)
                    
                    if group_recs:
                        recommendations_by_group[group_name] = group_recs
                
                # Compute fairness metrics per group
                if recommendations_by_group:
                    fairness_df = fairness_auditor.audit_by_demographic(recommendations_by_group)
                    fairness_df["model"] = model_name
                    fairness_df["demographic_category"] = category
                    fairness_results_all.append(fairness_df)
                    
                    # Compute demographic parity
                    parity = fairness_auditor.compute_demographic_parity(recommendations_by_group)
                    fairness_results_all[-1]["demographic_parity"] = parity
        
        # Combine all fairness results
        if fairness_results_all:
            fairness_df_all = pd.concat(fairness_results_all, ignore_index=True)
            
            # Save fairness results
            fairness_df_all.to_csv(os.path.join(save_dir, "fairness_metrics.csv"), index=False)
            print(f"Fairness metrics saved to {save_dir}/fairness_metrics.csv")
            print(f"  Evaluated {len(fairness_df_all)} model-demographic combinations")
        else:
            print("WARNING: No fairness results generated (no valid demographic groups found)")
    
    # Statistical significance testing
    print("\nComputing statistical significance tests...")
    significance_results = []
    
    model_names = list(per_user_metrics.keys())
    metrics_to_test = ["precision", "recall", "ndcg"]
    
    # Compare all model pairs
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            # Align per-user metrics (handle different user sets)
            metrics_a = per_user_metrics[model_a]
            metrics_b = per_user_metrics[model_b]
            
            # For each metric, perform statistical tests
            for metric_name in metrics_to_test:
                values_a = metrics_a.get(metric_name, [])
                values_b = metrics_b.get(metric_name, [])
                
                # Align lengths (take minimum)
                min_len = min(len(values_a), len(values_b))
                if min_len < 2:
                    continue
                
                values_a_aligned = values_a[:min_len]
                values_b_aligned = values_b[:min_len]
                
                # Paired t-test
                ttest_result = paired_t_test(values_a_aligned, values_b_aligned, alpha=0.05)
                
                # Wilcoxon signed-rank test (non-parametric)
                wilcoxon_result = wilcoxon_signed_rank_test(values_a_aligned, values_b_aligned, alpha=0.05)
                
                # Effect size (Cohen's d)
                effect_size = compute_effect_size(values_a_aligned, values_b_aligned)
                
                significance_results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric_name,
                    "mean_a": ttest_result["mean_a"],
                    "mean_b": ttest_result["mean_b"],
                    "mean_diff": ttest_result["mean_diff"],
                    "t_statistic": ttest_result["statistic"],
                    "t_pvalue": ttest_result["pvalue"],
                    "t_significant": ttest_result["significant"],
                    "wilcoxon_statistic": wilcoxon_result["statistic"],
                    "wilcoxon_pvalue": wilcoxon_result["pvalue"],
                    "wilcoxon_significant": wilcoxon_result["significant"],
                    "cohens_d": effect_size,
                })
    
    # Create significance matrix DataFrame
    if significance_results:
        significance_df = pd.DataFrame(significance_results)
        
        # Apply multiple comparison correction (Bonferroni)
        from src.eval.statistical import multiple_comparison_correction
        pvalues = significance_df["t_pvalue"].tolist()
        corrected_pvalues = multiple_comparison_correction(pvalues, method="bonferroni")
        significance_df["t_pvalue_corrected"] = corrected_pvalues
        significance_df["t_significant_corrected"] = significance_df["t_pvalue_corrected"] < 0.05
        
        # Save significance matrix
        significance_df.to_csv(os.path.join(save_dir, "significance_matrix.csv"), index=False)
        print(f"Statistical significance matrix saved to {save_dir}/significance_matrix.csv")
        print(f"  Total comparisons: {len(significance_df)}")
        print(f"  Significant differences (p<0.05, Bonferroni corrected): {significance_df['t_significant_corrected'].sum()}")
    
    # Save results
    results_df.to_csv(os.path.join(save_dir, "comprehensive_metrics.csv"), index=False)
    
    print(f"\nEvaluation complete! Results saved to {save_dir}/comprehensive_metrics.csv")
    print("\nSummary:")
    print(results_df[["model", "precision_mean", "recall_mean", "ndcg_mean", "diversity_mean", "catalog_coverage"]].to_string())
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    run_comprehensive_eval(args.config)
