"""
Streamlit dashboard for training, evaluation, and local inference.

Run from repository root:
  PYTHONPATH=src:. streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config.config import Config
from data.processor import DataProcessor
from evaluation.evaluate import evaluate_all
from inference.predict import ensemble_predict_features, predict_from_features

TORCH_MODELS = sorted(Config.TORCH_MODELS)
BASELINES = sorted(Config.BASELINE_MODELS)
ALL_EVAL = TORCH_MODELS + BASELINES + [Config.ENSEMBLE_NAME]


def run_training_subprocess(model: str, folds: int, repeats: int) -> None:
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    if model in Config.TORCH_MODELS:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--model",
            model,
            "--folds",
            str(folds),
            "--repeats",
            str(repeats),
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_baseline.py"),
            "--model",
            model,
            "--folds",
            str(folds),
            "--repeats",
            str(repeats),
        ]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main():
    st.set_page_config(page_title="CGM tabular ML (research)", layout="wide")
    st.caption("Research tooling only — not a medical device and not for diagnosis.")

    tabs = st.tabs(["Train", "Evaluate", "Inference", "Checkpoints"])

    with tabs[0]:
        st.header("Train models")
        model = st.selectbox("Model", TORCH_MODELS + BASELINES)
        folds = st.number_input("Folds", min_value=2, max_value=10, value=2)
        repeats = st.number_input("Repeats", min_value=1, max_value=5, value=1)
        if st.button("Start training"):
            with st.spinner("Training…"):
                run_training_subprocess(model, int(folds), int(repeats))
            st.success("Training finished (see logs/training.log for torch runs).")

    with tabs[1]:
        st.header("Evaluate (out-of-fold)")
        model = st.selectbox("Model", ALL_EVAL, key="eval_model")
        folds = st.number_input("Folds", 2, 10, key="eval_folds")
        repeats = st.number_input("Repeats", 1, 5, key="eval_repeats")
        if st.button("Run evaluation"):
            with st.spinner("Evaluating…"):
                fold_metrics, overall = evaluate_all(model, int(folds), int(repeats))
            df = pd.DataFrame(
                fold_metrics,
                columns=["Loss", "Accuracy", "AUC", "Sensitivity", "Specificity"],
            )
            st.dataframe(df)
            st.json(
                {
                    "mean_loss": overall[0],
                    "mean_accuracy": overall[1],
                    "mean_auc": overall[2],
                    "mean_sensitivity": overall[3],
                    "mean_specificity": overall[4],
                }
            )

    with tabs[2]:
        st.header("Single-sample inference")
        st.markdown(
            "Uses checkpoints on disk. For HTTP access, run the FastAPI app separately "
            "(`PYTHONPATH=src:. uvicorn api.main:app`)."
        )
        mode = st.selectbox("Model", ["TabTransformer", "AttentionMLP", "GNN", "Ensemble"])
        fold_idx = st.number_input("Fold index", min_value=1, value=1)
        up = st.file_uploader("Subject XLSX", type=["xlsx"])
        if up and st.button("Predict"):
            processor = DataProcessor()
            demo = processor._load_demographics()
            feats, _ = processor.process_file(up, demo, filename=up.name)
            if mode == "Ensemble":
                pred = ensemble_predict_features(int(fold_idx), feats)
            else:
                pred = predict_from_features(mode, int(fold_idx), feats)
            st.metric("Predicted class (proxy label)", pred)
            st.caption("Class definition: derived from HbA1c threshold in preprocessing (research proxy only).")

    with tabs[3]:
        st.header("Checkpoint directory")
        st.write(str(Config.CHECKPOINTS))
        if st.button("Delete all checkpoints"):
            shutil.rmtree(Config.CHECKPOINTS, ignore_errors=True)
            Config.CHECKPOINTS.mkdir(exist_ok=True)
            st.success("Checkpoints removed.")


if __name__ == "__main__":
    main()
