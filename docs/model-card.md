# Model Card: CGM-Derived Tabular Proxy Classifier

## Summary

This repository trains **binary classifiers** on a **small tabular feature vector** derived from CGM time series and static demographics. The **prediction target is a research proxy** (HbA1c threshold), not a clinical diagnosis and not a validated CGM trajectory forecast.

## Intended use

- **Research and engineering education**: reproducible pipelines, baselines, cross-validation, and prototype HTTP/UI layers.
- **Explicitly out of scope**: treatment decisions, screening, diagnosis, or remote patient monitoring without independent clinical validation.

## Model families

| Family | Role |
|--------|------|
| **Logistic regression** | Linear, calibrated-friendly baseline on the same scaled features. |
| **Random forest** | Nonlinear tree ensemble baseline. |
| **Attention MLP** | Neural baseline with feature-wise gating. |
| **TabTransformer-style encoder** | Transformer encoder over an embedded feature token sequence (tabular adaptation). |
| **Patient GNN (GCN)** | Graph convolution on a **fully connected patient graph built per minibatch** during training; at single-subject inference the graph collapses to a single node with a self-loop. This is a **methodological placeholder**, not a longitudinal patient graph from real-world care. |
| **Ensemble** | Unweighted average of logits from the three neural heads (TT + MLP + GNN). |

## Inputs

Five **z-scored** features (fit via `StandardScaler` on the training corpus loaded from disk):

1. Mean CGM (mg/dL)  
2. Standard deviation of CGM  
3. Root mean square of successive differences (variability proxy)  
4. Age (coerced from text; missing → imputed in demographics table)  
5. HbA1c % (missing → imputed in demographics table)

## Outputs

- **Two logits** → softmax → class 0 or 1 for the proxy label.

## Training / selection protocol

- **Outer loop**: repeated stratified K-fold cross-validation (`RepeatedStratifiedKFold`), aligned between training and evaluation scripts so fold `k` checkpoints match fold `k` test partitions.
- **Inner split (torch models)**: within each training fold, a **stratified hold-out fraction** (`Config.VAL_FRACTION`) selects checkpoints by **validation loss** when the fold is large enough; if the fold is too small, the code **falls back to training loss only** (see `src/training/splits.py`).
- **Baselines**: fit on the **full training fold** (no epochal early stopping); generalization is assessed on the outer CV test fold.

## Metrics reported

Loss (cross-entropy for torch, log loss for sklearn where returned), accuracy, ROC-AUC, sensitivity, and specificity from a **0.5 threshold on predicted probability of class 1**.

## Data and evaluation dependencies

Performance is **not meaningful** without the real DiaTrend cohort in `data/raw/` and the study’s official train/validation policy. The bundled **synthetic fixtures** exist only for unit tests.

## Ethical and safety considerations

See `docs/clinical-limitations.md`. Outputs must not be presented as medical advice.

## Caveats

- **Label leakage risk**: HbA1c appears both as a feature and defines the proxy label in the default pipeline. This is documented as a **methodological limitation** for research transparency, not as a clinically sound setup.
- **Small-sample GNN**: batch graphs built from handfuls of patients can be unstable; BatchNorm and very small graphs may behave poorly—monitor training logs.

## How to reproduce

1. Place approved DiaTrend exports under `data/raw/` (see `docs/data-card.md`).  
2. `PYTHONPATH=src:. python scripts/train.py --model TabTransformer --folds 5 --repeats 1`  
3. `PYTHONPATH=src:. python scripts/evaluate.py --model TabTransformer --folds 5 --repeats 1`  

Repeat for baselines via `scripts/train_baseline.py`.
