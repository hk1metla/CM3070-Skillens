# Streamlit deployment — test checklist

Check | Where | What
---|---|---
[ ] | Deployed URL | App loads (no traceback on first paint).
[ ] | `?page=home` | Home renders.
[ ] | Explore / recommendations | Enter a goal → recommendations appear.
[ ] | Preset goal buttons | Apply and return results.
[ ] | OULAD vs Coursera items | OULAD: internal view; Coursera: link behaves.
[ ] | `?page=control` | Opens or empty state without crash.
[ ] | Model switch (if used) | TF-IDF default; hybrid/semantic if you demo them.
[ ] | Hard refresh | Page recovers.
[ ] | Streamlit Cloud → **Logs** | No repeating errors after normal use.

**Regenerate cold-start figure (report):**

```bash
python -m src.eval.history_truncation --out results/final
python generate_coldstart_plot.py
```

Copy `results/final/plots/28.png` next to your `.tex` if needed.

**Control Room metrics on Cloud:** commit `data/deploy/comprehensive_metrics.csv` and `data/deploy/ablation_study.csv` (refresh from local `results/final/` after evaluation).
