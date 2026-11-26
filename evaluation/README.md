# Reproduce the OULAD ranking comparison

Download the Open University Learning Analytics Dataset ZIP with `studentVle.csv`, `vle.csv`, `courses.csv` and `studentInfo.csv` at its root. The run reads the archive and writes aggregate outputs. It uses a temporary directory for extracted files. [UCI dataset and citation](https://doi.org/10.24432/C5KK69)

From the repository root, use a separate virtual environment for the experiment.

```bash
python -m pip install -r evaluation/requirements.txt
PYTHONHASHSEED=0 python evaluation/reproduce.py --data-zip /path/to/oulad.zip --out audit-local
```

On PowerShell, set `$env:PYTHONHASHSEED="0"` before running the Python command. Compare `audit-local/ranking_results.csv`, `paired_tests.csv`, `paired_rank_counts.csv` and `data_audit.json` with the committed files here. The run manifest records the checked-out commit, input archive hash, package versions, seeds and fusion weights. The verified input ZIP SHA-256 was `f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4`. Different archive containers can package equivalent CSVs.

## What is measured

Each activity maps to one of 22 anonymized module-presentations. The split sorts each learner's events on a relative-day scale, then assigns 60%, 20% and 20% to train, validation and test, subject to short-history handling. Day zero is relative to each presentation. Tied days and different presentation starts limit calendar interpretations.

Of 26,074 learners, 25,797 had test records. Only 43 had a test item absent from their training history. Each had exactly one such item. The remaining test users contribute no ranking score. Both ItemKNN and the hybrid retrieved all 43 targets in the top ten. Their nDCG difference describes where those targets appeared in the list.

The repository policy retains each model's original candidate handling. The common-unseen policy first excludes the same training-history items for every model. The weights are fixed at 0.6 content and 0.4 collaborative. The comparison uses 10,000 bootstrap resamples and paired Wilcoxon tests. Across all 12 paired comparisons, the adjusted hybrid versus ItemKNN p-values are 0.221 and 0.0475 for the respective policies. The result depends on how candidates are chosen. The small, selective cohort cannot support population-wide fairness estimates or a claim about learning outcomes.

The reproduction keeps the historical model APIs and reports their diagnostic checks. It does not validate the anonymized modules' subject labels. [Ranking estimates and intervals](ranking_results.csv) · [Paired tests](paired_tests.csv) · [Data coverage](data_audit.json)
