"""
Split validation script.

Validates temporal integrity and logs validation results.
This ensures evaluation is leakage-free and reproducible.
"""

import os

import pandas as pd

from src.eval.temporal import TemporalEvaluator


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def validate_splits(out_dir: str = None) -> pd.DataFrame:
    """
    Validate data splits for temporal integrity.

    Args:
        out_dir: If set, write split_validation.csv here; else use RESULTS_DIR.

    Returns:
        DataFrame with validation results
    """
    save_dir = out_dir if out_dir is not None else RESULTS_DIR
    # Load splits
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Load full interactions for context
    interactions = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    
    # Run validation
    evaluator = TemporalEvaluator(interactions)
    validation_results = evaluator.comprehensive_validation(train, val, test)
    
    # Detect cold-start and new items
    cold_start_users, n_cold_start = evaluator.detect_cold_start_users(train, test)
    new_items, n_new_items = evaluator.detect_new_items(train, test)
    
    validation_results["n_cold_start_users"] = n_cold_start
    validation_results["n_new_items"] = n_new_items
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    validation_results.to_csv(
        os.path.join(save_dir, "split_validation.csv"), index=False
    )
    
    print("Split validation complete!")
    print("\nValidation Results:")
    print(validation_results.to_string())
    
    # Check for issues
    if not validation_results["global_ordering"].iloc[0]:
        print("\n⚠️  NOTE: Global temporal ordering shows some overlap.")
        print("   This is expected for per-user temporal splits and is OK")
        print("   as long as per-user ordering is maintained.")
    
    if not validation_results["per_user_ordering"].iloc[0]:
        print("\n❌ ERROR: Per-user temporal ordering violated!")
        print("   This indicates data leakage and must be fixed!")
    
    if n_cold_start > 0:
        print(f"\n📊 Found {n_cold_start} cold-start users in test set")
    
    if n_new_items > 0:
        print(f"📊 Found {n_new_items} new items in test set")
    
    return validation_results


if __name__ == "__main__":
    validate_splits()
