"""
FastAPI inference service for CGM-derived tabular features.

Run from repository root:
  PYTHONPATH=src:. uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from config.config import Config
from data.processor import DataProcessor
from inference.predict import ensemble_predict_features, predict_from_features

logger = logging.getLogger("cgm_api")
logging.basicConfig(level=logging.INFO)

TorchModelName = Literal["TabTransformer", "AttentionMLP", "GNN"]
ModelName = Literal["TabTransformer", "AttentionMLP", "GNN", "Ensemble"]


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str = str(Config.DEVICE)


class PredictFeaturesRequest(BaseModel):
    features: list[float] = Field(..., min_length=Config.NUM_FEATURES, max_length=Config.NUM_FEATURES)
    model: ModelName = "Ensemble"
    fold_index: int = Field(1, ge=1, description="Repeated CV fold checkpoint index (1-based)")


class PredictResponse(BaseModel):
    prediction: int
    model: str
    fold_index: int


app = FastAPI(
    title="CGM tabular inference (research prototype)",
    description="Binary proxy classifier from CGM summary features. Not for medical use.",
    version="0.2.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse()


@app.post("/predict/features", response_model=PredictResponse)
def predict_features(body: PredictFeaturesRequest):
    """Predict from a feature vector (for tests and programmatic clients)."""
    try:
        if body.model == "Ensemble":
            pred = ensemble_predict_features(body.fold_index, body.features)
        else:
            pred = predict_from_features(body.model, body.fold_index, body.features)
        return PredictResponse(prediction=pred, model=body.model, fold_index=body.fold_index)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict_features failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict")
async def predict_file(
    file: UploadFile = File(...),
    model: ModelName = Query("Ensemble"),
    fold_index: int = Query(1, gt=0, description="Repeated fold index (>=1)"),
):
    """
    Upload a DiaTrend-style Subject XLSX (CGM sheet + demographics on disk for subject id match).
    """
    try:
        contents = await file.read()
        buf = BytesIO(contents)
        buf.name = file.filename or "uploaded.xlsx"

        processor = DataProcessor()
        demo_df = processor._load_demographics()
        features, _label = processor.process_file(buf, demo_df, filename=buf.name)

        if model == "Ensemble":
            pred_class = ensemble_predict_features(fold_index, features)
        else:
            pred_class = predict_from_features(model, fold_index, features)

        return {"prediction": pred_class, "model": model, "fold_index": fold_index}
    except FileNotFoundError as fnf:
        logger.error(str(fnf))
        raise HTTPException(status_code=404, detail=str(fnf)) from fnf
    except Exception as e:
        logger.exception("predict_file failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
