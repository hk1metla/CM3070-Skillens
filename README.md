# Skillens

**Data-Driven Personalised Educational Content Recommendation**  
CM3070 Computer Science Final Project, University of London  
Shahzaib Naeem  
Project Template 1 (CM3005 Data Science), Idea 1.1

**Repository:** https://github.com/hk1metla/CM3070-Skillens  
**Live Demo:** https://cm370-skillens-kuxgzkqtmag6eozssjk5s6.streamlit.app/

> The live demo may take a short time to wake on first load because Streamlit Community Cloud hibernates inactive apps.

---

## Overview

Skillens is a Streamlit-based educational content recommender. It combines:

- TF-IDF content-based recommendation
- ItemKNN collaborative filtering
- weighted hybrid fusion
- template-based explanations for recommended items

The system is designed for sparse educational interaction data and includes a cold-start fallback when collaborative evidence is unavailable.

The deployed app is a lightweight demonstration interface. All reported quantitative results in the final report are generated locally from the OULAD evaluation pipeline.

---

## Main Results

OULAD test set at K = 10:

| Model | NDCG@10 | Precision@10 | Recall@10 |
|---|---:|---:|---:|
| Popularity | 0.384 | 0.074 | 0.744 |
| TF-IDF | 0.440 | 0.095 | 0.953 |
| ItemKNN | 0.574 | 0.100 | 1.000 |
| **Hybrid** | **0.703** | **0.100** | **1.000** |

Hybrid outperforms the popularity baseline by 83% on NDCG@10. Statistical significance was assessed using the Wilcoxon signed-rank test with Bonferroni correction.

---

## Project Structure

```text
src/
├── app/       # Streamlit interface
├── data/      # ingestion, interaction construction, temporal splits
├── eval/      # canonical evaluation pipeline and plots
├── explain/   # template-based explanation logic
└── models/    # recommender models
