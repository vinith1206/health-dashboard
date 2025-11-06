from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

from ._shared import preprocess_features, FEATURE_COLS, map_columns, load_artifacts


app = FastAPI(title="Health Dashboard API", version="1.0.0")

# CORS for flexibility (e.g., if a separate frontend domain is used)
try:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


class Patient(BaseModel):
    age: Optional[float] = Field(None, description="Age in years")
    sex: Optional[int] = Field(None, description="1=male, 0=female")
    trestbps: Optional[float] = Field(None, description="Resting blood pressure (mmHg)")
    chol: Optional[float] = Field(None, description="Serum cholesterol (mg/dl)")
    fbs: Optional[int] = Field(None, description="Fasting blood sugar > 120 mg/dl (1/0)")
    thalach: Optional[float] = Field(None, description="Max heart rate achieved")
    exang: Optional[int] = Field(None, description="Exercise induced angina (1/0)")
    oldpeak: Optional[float] = Field(None, description="ST depression induced by exercise")


class PredictRequest(BaseModel):
    instances: List[Patient]


@app.get("/health")
def health() -> Dict[str, Any]:
    model, scaler, _ = load_artifacts()
    status = model is not None and scaler is not None
    return {"ok": status}


@app.get("/metrics")
def metrics():
    _, __, metrics = load_artifacts()
    return JSONResponse(metrics or {})


@app.post("/predict")
def predict(req: PredictRequest):
    model, scaler, _ = load_artifacts()
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not available. Train and upload artifacts to /model")

    # Convert payload to DataFrame
    rows = [p.dict(exclude_none=True) for p in req.instances]
    if not rows:
        raise HTTPException(status_code=400, detail="No instances provided")

    df_in = pd.DataFrame(rows)
    # Normalize column names
    if map_columns is not None:
        df_in = map_columns(df_in)
    # Ensure all expected features exist
    for c in FEATURE_COLS:
        if c not in df_in.columns:
            df_in[c] = np.nan

    X, _, _ = preprocess_features(df_in, fit_scaler=False, scaler=scaler)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        "predictions": preds.tolist(),
        "probabilities": proba.tolist(),
        "features": FEATURE_COLS,
    }


# Vercel expects a top-level handler for ASGI
from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute


def handler(request: Request) -> Response:  # type: ignore
    # This will be patched by Vercel's Python runtime to ASGI app
    # Placeholder to satisfy imports if needed; actual entry is 'app'
    return Response(status_code=404)
