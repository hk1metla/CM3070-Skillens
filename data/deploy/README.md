# Bundled demo data (Streamlit Cloud & quick demos)

These files are **small subsets** of the full processed OULAD pipeline output so the app can run **without** running `--prepare-data` (e.g. on Streamlit Community Cloud, where raw OULAD is not present).

| File | Purpose |
|------|---------|
| `items.csv` | Course/item catalogue for recommendations |
| `train.csv` | Sample interactions (~20k rows) for hybrid / popularity |
| `comprehensive_metrics.csv` | Control Room **Performance / Metrics** tabs (copy from `results/final` after a local run) |
| `ablation_study.csv` | Control Room **Ablation** tab |

For full experiments and report numbers, use the real pipeline locally:

```bash
python -m src.eval.pipeline --config configs/experiment.yaml --prepare-data
```

That writes the full `data/processed/` outputs (still gitignored).
