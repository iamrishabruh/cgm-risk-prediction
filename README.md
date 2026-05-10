# CGM-Derived Tabular Modeling (Research Prototype)

Honest ML engineering layout for **continuous glucose monitor (CGM) spreadsheet exports**: preprocessing → baselines + neural models → repeated cross-validation → optional FastAPI inference and a Streamlit UI. This is **not** a medical device, **not** deployment-ready for care, and **not** validated for diagnosis or treatment decisions. See `docs/clinical-limitations.md`.

---

## 1. Problem

Given a short CGM window and basic demographics, can we build a **reproducible supervised learning baseline** for a **research-only proxy label** related to glycemic control? The emphasis is on **pipeline structure**, **baselines**, **CV discipline**, and **clear limitations**—not on claiming clinical utility.

## 2. What the model predicts

A **binary proxy class** derived from a fixed HbA1c threshold in preprocessing (class 1 when HbA1c ≤ 7.0% in the default configuration). This is **not** a CGM trajectory forecast and **not** a hypoglycemia predictor.

## 3. Dataset access

Real DiaTrend subject files are distributed via Synapse with controlled access. This repository **does not include** patient data. Place approved exports under `data/raw/` as described in `docs/data-card.md`. **CI and unit tests** use fabricated spreadsheets in `tests/fixtures/` only.

## 4. Feature engineering

Per subject, after cleaning the CGM sheet:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | CGM mean | Mean glucose (mg/dL) |
| 1 | CGM std | Standard deviation |
| 2 | Variability proxy | RMS of successive differences |
| 3 | Age | Parsed from demographics text |
| 4 | HbA1c | Numeric laboratory value |

Features are **standardized with `StandardScaler`** fit on all loaded subjects. Missing glucose rows are **dropped** before summaries; missing demographics numerics are imputed with **column means** (research default—see limitations).

Implementation: `src/data/features.py`, `src/data/processor.py`.

## 5. Models implemented

| Model | Notes |
|-------|------|
| **LogisticRegression** | Linear baseline (`scikit-learn`), same scaled features. |
| **RandomForest** | Nonlinear ensemble baseline. |
| **AttentionMLP** | Neural model with attention gating. |
| **TabTransformer-style** | Transformer encoder over embedded tabular inputs. |
| **PatientGNN (GCN)** | Requires **PyTorch Geometric**; training builds dense patient graphs per batch. Single-subject inference uses a **one-node graph** (self-loop)—a simplification for API demos. |
| **Ensemble** | Average logits of the three neural heads. |

## 6. Evaluation protocol

- **Outer**: `RepeatedStratifiedKFold` ties together training checkpoints and test partitions by fold index (`repFold1`, `repFold2`, …).  
- **Inner (torch only)**: stratified **train/validation split inside each training fold** when `n_train ≥ 2 * VAL_MIN_SAMPLES`; checkpoints minimize **validation loss**. Small folds fall back to **training loss** (see `src/training/splits.py` and `docs/model-card.md`).  
- **Baselines**: fit on the full outer training fold; no epochal validation.  
- **Metrics**: accuracy, ROC-AUC, sensitivity/specificity at 0.5 threshold; torch runs also report cross-entropy on the held-out fold.

## 7. Environment setup

Python 3.10+ recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU PyTorch example (pick the wheel index matching your platform):
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**PyTorch Geometric**: required for `PatientGNN` and ensemble paths. If `pip install torch-geometric` fails, follow the official install matrix for your Torch/CUDA version: [PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

Working imports assume:

```bash
export PYTHONPATH=src:.
```

## 8. How to run training

Torch models (inner validation split for checkpoint selection when possible):

```bash
PYTHONPATH=src:. python scripts/train.py --model TabTransformer --folds 5 --repeats 1
PYTHONPATH=src:. python scripts/train.py --model AttentionMLP --folds 5 --repeats 1
PYTHONPATH=src:. python scripts/train.py --model GNN --folds 5 --repeats 1
```

Sklearn baselines (same outer CV indices; `.joblib` checkpoints):

```bash
PYTHONPATH=src:. python scripts/train_baseline.py --model LogisticRegression --folds 5 --repeats 1
PYTHONPATH=src:. python scripts/train_baseline.py --model RandomForest --folds 5 --repeats 1
```

## 9. How to run evaluation

```bash
PYTHONPATH=src:. python scripts/evaluate.py --model LogisticRegression --folds 5 --repeats 1
PYTHONPATH=src:. python scripts/evaluate.py --model TabTransformer --folds 5 --repeats 1
PYTHONPATH=src:. python scripts/evaluate.py --model Ensemble --folds 5 --repeats 1
```

## 10. How to run the API

```bash
PYTHONPATH=src:. uvicorn api.main:app --reload
```

- `GET /health` — process health.  
- `POST /predict/features` — JSON body with `features` (length 5), `model`, `fold_index`.  
- `POST /predict` — multipart XLSX upload (still requires `data/raw/demographics.xlsx` for subject id matching).

## 11. How to run the Streamlit dashboard

```bash
PYTHONPATH=src:. streamlit run ui/streamlit_app.py
```

The UI shells out to `scripts/train.py` / `scripts/train_baseline.py` so it never embeds a FastAPI app (previous coupling removed).

## 12. Limitations and clinical safety

- **Label/feature coupling**: HbA1c may appear as both input and label definition in the default proxy—reporting metrics without acknowledging this is misleading.  
- **Collapsed time series**: only three CGM scalars are used; no explicit modeling of time-in-range, trends, or sensor noise distributions.  
- **GNN semantics**: training-time graphs are **synthetic fully connected batches**, not verified clinical similarity graphs.  
- **No prospective validation**, drift monitoring, or fairness audits are included.

Read `docs/clinical-limitations.md` before any external communication about this work.

## 13. Next improvements

- Replace the HbA1c proxy with **temporal prediction** (e.g., forecasting glucose) using proper rolling-origin evaluation.  
- Add **group / site held-out validation** and calibration curves.  
- Remove HbA1c from inputs when predicting HbA1c-derived labels, or switch labels to outcomes that do not leak through features.  
- Implement **explicit missing-data models** instead of mean imputation.  
- Add **subgroup reporting** and **uncertainty** (ensembles, temperature scaling).

## Repository layout

```
api/main.py              # FastAPI inference only
ui/streamlit_app.py      # Streamlit dashboard only
config/config.py         # Paths and hyperparameters
src/data/                # Feature extraction + DataProcessor
src/models/              # Torch + sklearn factory helpers
src/training/            # CV training loops (torch + baselines)
src/evaluation/          # Metrics + out-of-fold evaluation
src/inference/           # Shared torch inference helpers
scripts/                 # CLI entrypoints
tests/fixtures/          # Synthetic XLSX for CI
docs/                    # Model, data, and clinical limitation cards
```

## Testing

```bash
PYTHONPATH=src:. python -m pytest
```

## License

MIT — see `LICENSE`.