![Skillens](assets/banner.svg)

# Skillens

**Hybrid content recommendation with an evaluation-focused ML pipeline.**

Skillens combines text-based retrieval, collaborative signals and weighted fusion to recommend learning content. A Streamlit interface connects model outputs with explanations and interactive exploration.

**Python · scikit-learn · pandas · Streamlit · recommender systems**

## System overview

| Layer | Implementation |
| --- | --- |
| Content retrieval | TF-IDF representations and similarity-based ranking |
| Collaborative recommendation | ItemKNN from interaction data |
| Fusion and fallback | Weighted hybrid ranking with content-based cold-start handling |
| Explanations | Template-based descriptions of recommendation signals |
| Evaluation | Ranking metrics, temporal splits, ablation and robustness scripts |
| Interface | Streamlit pages for exploration and a model control room |

The repository also contains experimental semantic, pedagogical and learning-to-rank modules. Their presence does not mean every variant has been validated or deployed.

## Run locally

Install the pinned environment from `requirements.txt`, using a compatible Python environment:

```bash
python -m pip install -r requirements.txt
python -m streamlit run src/app/app.py
```

The demonstration uses its available catalog and deployment artifacts. [Open the existing Streamlit demo](https://cm370-skillens-kuxgzkqtmag6eozssjk5s6.streamlit.app/). The demo interface and offline research pipeline serve different purposes.

## Offline evaluation

The canonical entry point is:

```bash
python -m src.eval.pipeline --config configs/experiment.yaml --out results/reproduction --prepare-data
```

Provide the required OULAD files under `data/raw/oulad/` first. Inspect `src/data/ingest.py`, `build_interactions.py` and `make_splits.py` for the exact inputs and item definitions. OULAD interactions are aggregated to module/presentation items, while the separate external course catalog supports the interface.

**Data mode matters:** the interaction builder includes a synthetic fallback when real interaction files are unavailable. Check input provenance and generated artifacts before interpreting any metric as an OULAD result. A successful demo run is not a real-data benchmark reproduction.

The split helper sorts each user's history by timestamp, with default 60/20/20 proportions. Histories shorter than three records remain in training. Evaluation should state the candidate universe, repeated-item handling, relevance definition and split coverage alongside ranking scores.

## Repository map

| Folder | Contents |
| --- | --- |
| `src/models/` | Baseline, content, collaborative and experimental ranking models |
| `src/data/` | Ingestion, interaction construction, demographic loading and split checks |
| `src/eval/` | Metrics, canonical pipeline, fairness/robustness experiments and reporting |
| `src/explain/` | Explanation templates and attribution experiments |
| `src/app/` | Streamlit UI |
| `tests/` | Metrics, schema, leakage, split and artifact checks |
| `data/deploy/` | Artifacts used by the demonstration |

## Results and verification

The repository contains historical evaluation artifacts. Full training, evaluation and deployment were **not rerun during this portfolio documentation pass**. Older numerical claims are preserved with their context in [the historical README](docs/HISTORICAL_README.md); they are not presented here as newly verified results.

A credible reproduction should retain input hashes, configuration, split integrity checks and a run manifest. Report performance only for the exact verified data mode and evaluation protocol.

## Source and scope

Developed by Shahzaib Naeem from an academic research project and maintained as an applied ML portfolio project. Source history is recorded in [PROVENANCE.md](PROVENANCE.md). Third-party datasets and libraries retain their own terms; a code release does not grant rights to redistribute external catalog content.

Current research directions include stronger candidate-set evaluation, sparse-feedback robustness, transparent fallback behavior and separating recommendation usefulness from offline ranking metrics.
