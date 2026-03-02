"""
Canonical evaluation pipeline for Skillens.

Single entrypoint that regenerates all report artifacts into a given output
directory (e.g. results/final/). Produces CSVs, plots, and a run manifest
for reproducibility.

Usage:
    python -m src.eval.pipeline --config configs/experiment.yaml --out results/final
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

REQUIRED_DATA_FILES = ["train.csv", "val.csv", "test.csv", "interactions.csv", "items.csv"]


def _data_ready() -> bool:
    """Return True if data/processed has all files needed for evaluation."""
    if not os.path.isdir(DATA_PROCESSED):
        return False
    for f in REQUIRED_DATA_FILES:
        if not os.path.isfile(os.path.join(DATA_PROCESSED, f)):
            return False
    return True


def _run_data_pipeline() -> None:
    """Run ingest -> build_interactions -> make_splits to create data/processed."""
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    print("Running data pipeline: ingest -> build_interactions -> make_splits ...")
    from src.data.ingest import ingest
    ingest()
    from src.data.build_interactions import build_interactions
    build_interactions()
    from src.data.make_splits import make_splits
    make_splits()
    print("Data pipeline complete.")
    print()


def _git_hash():
    """Return current git commit hash or 'unknown' if not a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_run_manifest(out_dir: str, config_path: str, command_line: list) -> None:
    """Write run_manifest.json with git hash, config, timestamps, versions."""
    manifest = {
        "git_commit": _git_hash(),
        "config_path": os.path.abspath(config_path) if config_path else None,
        "command_line": command_line,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
    }
    try:
        import pandas as pd
        import numpy as np
        import yaml
        manifest["packages"] = {
            "pandas": getattr(pd, "__version__", "?"),
            "numpy": getattr(np, "__version__", "?"),
        }
    except Exception:
        pass
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest written to {path}")


def run_pipeline(config_path: str, out_dir: str, prepare_data: bool = False) -> None:
    """
    Run the full canonical evaluation pipeline.

    Writes to out_dir:
        - comprehensive_metrics.csv
        - significance_matrix.csv
        - fairness_metrics.csv (if demographics available)
        - ablation_study.csv
        - split_validation.csv
        - plots/*.png
        - run_manifest.json
    """
    if prepare_data:
        _run_data_pipeline()
    if not _data_ready():
        missing = []
        for f in REQUIRED_DATA_FILES:
            p = os.path.join(DATA_PROCESSED, f)
            if not os.path.isfile(p):
                missing.append(f)
        print("ERROR: Required data files are missing. Cannot run evaluation.")
        print(f"  Directory: {DATA_PROCESSED}")
        print(f"  Missing: {', '.join(missing)}")
        print()
        print("Generate them by either:")
        print("  1. Run with --prepare-data to build from raw data:")
        print("     python -m src.eval.pipeline --config configs/experiment.yaml --out results/final --prepare-data")
        print("  2. Or run the data steps yourself, then re-run the pipeline:")
        print("     python -m src.data.ingest")
        print("     python -m src.data.build_interactions")
        print("     python -m src.data.make_splits")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    cmd = [sys.executable, "-m", "src.eval.pipeline", "--config", config_path, "--out", out_dir]
    if prepare_data:
        cmd.append("--prepare-data")
    print("=" * 60)
    print("Skillens canonical evaluation pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")
    print()

    # 1. Split validation
    from src.data.validate_splits import validate_splits
    validate_splits(out_dir=out_dir)
    print()

    # 2. Comprehensive evaluation (writes comprehensive_metrics, significance, fairness)
    from src.eval.comprehensive_eval import run_comprehensive_eval
    results_df = run_comprehensive_eval(config_path, out_dir=out_dir)
    print()

    # 3. Ablation study (reuses results_df, writes ablation_study.csv)
    from src.eval.ablation import run_ablation_study
    run_ablation_study(config_path, results_df=results_df, out_dir=out_dir)
    print()

    # 4. Plots from canonical CSVs
    from src.eval.generate_plots import generate_all_plots
    generate_all_plots(metrics_dir=out_dir, plots_dir=plots_dir)
    print()

    # 5. Run manifest
    _write_run_manifest(out_dir, config_path, cmd)
    print()
    print("Pipeline complete. All artifacts in:", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Canonical evaluation pipeline: regenerate all report artifacts."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(BASE_DIR, "configs", "experiment.yaml"),
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(BASE_DIR, "results", "final"),
        help="Output directory for CSVs and plots (e.g. results/final)",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Run ingest, build_interactions, make_splits first if data/processed is missing",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.out, prepare_data=args.prepare_data)


if __name__ == "__main__":
    main()
