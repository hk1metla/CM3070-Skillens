# Skillens — Data-Driven Personalised Educational Content Recommendation

**CM3070 Computer Science Final Project | University of London**
**BSc Computer Science (Machine Learning and Artificial Intelligence)**

> Student: Shahzaib Naeem | Submission: March 2026
> Project Template: Project Template 1 (CM3005 Data Science) — Idea 1.1

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://cm370-skillens-kuxgzkqtmag6eozssjk5s6.streamlit.app/)
[![Repository](https://img.shields.io/badge/Repo-GitHub-181717?logo=github)](https://github.com/hk1metla/CM3070-Skillens)

---

## About

**Skillens** is a personalised recommendation system for educational content that helps reduce choice overload in online learning by generating ranked Top-K recommendations. The system targets learners on platforms like the Open University, where interaction data can be sparse and resources are described through short text metadata.

The core idea is that neither content-based nor collaborative filtering alone is sufficient for educational settings — Skillens combines both through a **weighted hybrid fusion** architecture, with a cold-start fallback for learners who have no interaction history. A deterministic template-based explanation is generated for each recommendation so learners can understand why an item was suggested.

### Key results (OULAD test set, K = 10)

| Model | NDCG@10 | Precision@10 | Recall@10 |
|---|---|---|---|
| Popularity baseline | 0.384 | 0.074 | 0.744 |
| TF-IDF (content-based) | 0.440 | 0.095 | 0.953 |
| ItemKNN (collaborative) | 0.574 | 0.100 | 1.000 |
| **Hybrid (TF-IDF + ItemKNN)** | **0.703** | **0.100** | **1.000** |

Hybrid outperforms the popularity baseline by **83%** on NDCG@10. All differences are statistically significant (Wilcoxon signed-rank, Bonferroni corrected). Effect size for hybrid vs popularity: Cohen's d = 1.06.

---

## System Architecture

```
src/
├── data/           # OULAD ingestion, interaction construction, temporal splitting
├── models/         # Popularity, TF-IDF, ItemKNN, Hybrid, LambdaMART (LTR)
├── eval/           # Evaluation pipeline, metrics, robustness sweeps, fairness
├── explain/        # Template-based explanation generation
└── app/            # Streamlit web interface
```

The pipeline is fully modular: baselines, hybrid, and the LTR re-ranker can each be evaluated independently. Evaluation and UI are kept separate so offline metrics are never contaminated by UI data.

---

## Live Demo

**Hosted on Streamlit Community Cloud:**
👉 https://cm370-skillens-kuxgzkqtmag6eozssjk5s6.streamlit.app/

> **Note:** The app may take 30–60 seconds to wake from sleep on first load (Streamlit Community Cloud hibernates inactive apps). If you see a loading screen, wait briefly and the app will start.

The demo uses the lightweight `data/deploy/` dataset (pre-processed `items.csv` + sample `train.csv`) so it runs without the full OULAD download. For full-fidelity results matching the report, run the evaluation pipeline locally (see below).

**What the UI does:**
- Enter a free-text learning goal or select from quick-goal presets
- Choose a recommender type: Popularity, TF-IDF, ItemKNN, or Hybrid
- Set Top-N (slider)
- View ranked recommendations with match confidence scores and template-based explanations
- Submit binary relevance feedback (Helpful / Not for me), logged to `data/feedback/`

---

## Setup

**Requirements:** Python 3.9+, pip

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.3.3 | Data processing |
| numpy | 2.3.5 | Numerical computation |
| scikit-learn | 1.8.0 | TF-IDF, cosine similarity, metrics |
| scipy | 1.17.0 | Sparse matrices, Wilcoxon tests |
| lightgbm | 4.6.0 | LambdaMART learning-to-rank |
| streamlit | 1.54.0 | Web interface |
| sentence-transformers | 5.2.2 | Sentence-BERT semantic embeddings |
| statsmodels | — | Bonferroni correction |
| shap | — | SHAP explainability (extension module) |

---

## Data Preparation

Download the [OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) and place the `anonymisedData/` folder in the repo root. Then run:

```bash
python -m src.eval.pipeline --config configs/experiment.yaml --prepare-data
```

This ingestion step:
1. Loads `studentVle.csv` (10,655,280 VLE interaction events), `vle.csv` (resource metadata), and `studentInfo.csv` (demographics)
2. Maps individual VLE activities to the 22 OULAD module-presentation items
3. Constructs log-transformed implicit feedback interactions (Hu et al., 2008)
4. Creates per-user temporal splits: 60% train / 20% validation / 20% test
5. Writes all processed artefacts to `data/processed/`

> The temporal split is a hard evaluation constraint — training interactions always precede test interactions for each user, preventing temporal leakage.

---

## Run Evaluation

Regenerates **all** report tables and figures in one command:

```bash
python -m src.eval.pipeline --config configs/experiment.yaml --out results/final
```

To run data preparation and evaluation together:

```bash
python -m src.eval.pipeline --config configs/experiment.yaml --prepare-data --out results/final
```

The pipeline writes `results/final/run_manifest.json` on every run, recording the git hash, config path, timestamps, Python version, command line, and package versions for reproducibility.

---

## Launch UI

```bash
streamlit run src/app/app.py
```

---

## Run Tests

Four automated correctness checks are included:

```bash
pytest tests/
```

| Test file | What it checks |
|---|---|
| `tests/test_split_integrity.py` | No user's test interactions precede their training interactions |
| `tests/test_leakage.py` | No test item IDs appear in the constructed TF-IDF query text |
| `tests/test_metrics_bounds.py` | All NDCG and Precision values fall within [0, 1] |
| `tests/test_artifact_consistency.py` | Output CSV row counts match expected split totals |

---

## Expected Outputs

All files below are written to `results/final/` by the evaluation pipeline. Table and figure numbers refer to the final report.

| Output file | Report reference |
|---|---|
| `comprehensive_metrics.csv` | Table 10 — Baseline comparison (Precision, Recall, NDCG@10) |
| `ablation_study.csv` | Table 11 — Ablation study (component contributions) |
| `fairness_accuracy.csv` | Table 12 — Fairness analysis by demographic group |
| `history_truncation.csv` | Table 13 — Cold-start performance by user history length |
| `significance_matrix.csv` | Figure 9 — Statistical significance heatmap |
| `weight_sweep.csv` | Table 9 — Robustness sweep over hybrid fusion weights |
| `run_manifest.json` | Reproducibility log (git hash, versions, timestamps) |
| `split_validation.csv` | Split integrity verification |
| `plots/*.png` | All report figures (Figures 8–12) |

---

## Streamlit Community Cloud Deployment

- **Main file path:** `src/app/app.py`
- **Data:** The repo includes `data/deploy/` (`items.csv` + sample `train.csv`) so the app starts without the full OULAD download on the server. For full-fidelity metrics, run `--prepare-data` locally and use those processed outputs (not committed to the repo due to size).
- **Python version:** 3.9+
- **Requirements file:** `requirements.txt`

---

## Dataset

This project uses the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) (Kuzilek et al., 2017).

| Statistic | Value |
|---|---|
| Total VLE interaction events | 10,655,280 |
| Unique users (in interaction set) | 26,074 |
| Unique items (OULAD module-presentations) | 22 |
| Train interactions | 6,382,979 (59.9%) |
| Validation interactions | 2,130,988 (20.0%) |
| Test interactions | 2,141,313 (20.1%) |
| Matrix sparsity | > 95% |

> **User count note:** `demographics.csv` contains 32,593 records, but only 26,074 students appear in the VLE interaction mapping after filtering for at least one recorded activity. The 6,519 remaining rows are registered students with no VLE activity log entries.
>
> **Item count note:** 22 items refers to OULAD module-presentation IDs (e.g. `AAA_2013J`), not individual VLE resources. The UI also loads a Coursera catalogue (`items.csv`) for demonstration purposes — Coursera items carry a `coursera_` prefix and are excluded from all OULAD evaluation metrics.

---

## Evaluation Leakage Fix

A critical bug was identified and corrected during development. The original TF-IDF implementation built user-goal queries from the full interaction history including test items, yielding an inflated NDCG@10 of **0.98**. The fix restricts query construction to training-set items only, dropping TF-IDF to a realistic **0.440**. This fix is verified automatically by `tests/test_leakage.py`. All reported results use the leak-free protocol.

---

## References

- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data*, 4, 170171.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM 2008*, 263–272.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. *Proceedings of WWW*, 285–295.
- Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An overview. *Microsoft Research Technical Report MSR-TR-2010-82*.
- Sabiri, B. et al. (2025). Hybrid quality-based recommender systems: A systematic literature review. *Journal of Imaging*, 11(1), 12.

Full references are in the report (`Final_Report_Skillens.pdf`).

