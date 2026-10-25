"""
prediction.py — /api/predict and /api/battery/* routes
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field, validator

from backend.services.model_service import battery_service

router = APIRouter(prefix="/api", tags=["prediction"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CycleRow(BaseModel):
    """One discharge cycle's 12 features (exact names from config.json)."""
    voltage_mean:       float = Field(..., description="Mean discharge voltage (V)")
    voltage_min:        float = Field(..., description="Minimum voltage (V)")
    voltage_max:        float = Field(..., description="Maximum voltage (V)")
    voltage_std:        float = Field(..., description="Voltage standard deviation")
    current_mean:       float = Field(..., description="Mean discharge current (A)")
    current_std:        float = Field(..., description="Current standard deviation")
    temp_mean:          float = Field(..., description="Mean temperature (°C)")
    temp_max:           float = Field(..., description="Peak temperature (°C)")
    temp_range:         float = Field(..., description="Temperature range (°C)")
    discharge_duration: float = Field(..., description="Discharge duration (seconds)")
    discharge_capacity: float = Field(..., description="Discharge capacity (Ah)")
    cycle_number:       float = Field(..., description="Discharge cycle index")


class PredictRequest(BaseModel):
    """
    Manual prediction — supply exactly `window_size` (10) consecutive cycles.
    The first cycle in the list is the oldest (t-9), the last is the current (t0).
    """
    cycles:           list[CycleRow] = Field(..., min_items=10, max_items=10)
    initial_capacity: Optional[float] = Field(
        None,
        description="Battery's first measured capacity (Ah). "
                    "Used for SOH%. Defaults to the capacity of the first cycle supplied.",
    )

    @validator("cycles")
    def must_be_window_size(cls, v):
        expected = 10
        if len(v) != expected:
            raise ValueError(f"Exactly {expected} cycles are required (got {len(v)}).")
        return v


class PredictResponse(BaseModel):
    predicted_capacity: float
    soh:                float
    rul_estimate:       int
    is_below_eol:       bool


class BatteryCycleRequest(BaseModel):
    battery_id: str
    cycle_idx:  int = Field(..., ge=10, description="Row index within the processed dataset (≥ window_size)")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse, summary="Run DNN inference on a manual window")
def predict(req: PredictRequest):
    """
    Feed exactly 10 consecutive discharge-cycle feature vectors.
    Returns predicted capacity (Ah), SOH (%), estimated RUL (cycles), and EOL flag.
    """
    if not battery_service.is_loaded:
        raise HTTPException(503, "Model not loaded yet — try again in a moment.")

    rows = [r.dict() for r in req.cycles]
    window_df = pd.DataFrame(rows)

    ic = req.initial_capacity or float(window_df["discharge_capacity"].iloc[0])

    try:
        result = battery_service.predict(window_df, initial_capacity=ic)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return PredictResponse(**result)


@router.post("/predict/csv", response_model=PredictResponse, summary="Run DNN inference from uploaded CSV")
async def predict_csv(
    file: UploadFile = File(..., description="CSV with exactly 10 rows and the 12 feature columns"),
    initial_capacity: Optional[float] = Query(None, description="Battery first capacity (Ah)"),
):
    """
    Upload a CSV file with 10 rows (one per historical cycle) and 12 feature columns.
    Column names must match the 12 feature names defined in config.json.
    """
    if not battery_service.is_loaded:
        raise HTTPException(503, "Model not loaded yet.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    try:
        contents = await file.read()
        window_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    expected_cols = battery_service.base_features
    missing = [c for c in expected_cols if c not in window_df.columns]
    if missing:
        raise HTTPException(
            422,
            f"CSV is missing required columns: {missing}. "
            f"Required: {expected_cols}",
        )

    if len(window_df) != battery_service.window_size:
        raise HTTPException(
            422,
            f"CSV must have exactly {battery_service.window_size} rows (one per cycle). "
            f"Got {len(window_df)}.",
        )

    ic = initial_capacity or float(window_df["discharge_capacity"].iloc[0])

    try:
        result = battery_service.predict(window_df, initial_capacity=ic)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return PredictResponse(**result)


@router.get("/batteries", summary="List available battery IDs")
def list_batteries():
    """Return the battery IDs available in the processed NASA dataset."""
    if not battery_service.is_loaded:
        raise HTTPException(503, "Model not loaded.")
    return {"batteries": battery_service.list_batteries()}


@router.get("/batteries/{battery_id}/history", summary="Full degradation history for one battery")
def battery_history(battery_id: str):
    """
    Return the complete cycle-by-cycle capacity / SOH / RUL trajectory
    for a specific battery, sourced directly from the processed NASA dataset.
    """
    if not battery_service.is_loaded:
        raise HTTPException(503, "Model not loaded.")

    available = battery_service.list_batteries()
    if battery_id not in available:
        raise HTTPException(404, f"Battery '{battery_id}' not found. Available: {available}")

    return {
        "battery_id": battery_id,
        "history":    battery_service.battery_history(battery_id),
    }


@router.post("/batteries/predict-at-cycle", summary="Predict using a known battery at a specific cycle index")
def predict_at_cycle(req: BatteryCycleRequest):
    """
    Use a battery from the NASA dataset to run a single-step prediction.
    Builds the W=10 window from the dataset at the requested cycle index.
    Returns predicted + actual values for comparison.
    """
    if not battery_service.is_loaded:
        raise HTTPException(503, "Model not loaded.")

    try:
        result = battery_service.predict_battery_at_cycle(req.battery_id, req.cycle_idx)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return result


@router.get("/results/memory-comparison", summary="Memory-window benchmark results")
def memory_comparison():
    """Return the Phase-2 Experiment 1 memory-window LOBO comparison CSV as JSON."""
    import os
    from backend.services.model_service import _PROJECT_ROOT
    csv_path = os.path.join(_PROJECT_ROOT, "results", "final", "memory_comparison.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, "memory_comparison.csv not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/results/ablation", summary="Ablation study results")
def ablation_results():
    import os
    from backend.services.model_service import _PROJECT_ROOT
    csv_path = os.path.join(_PROJECT_ROOT, "results", "final", "ablation_study.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, "ablation_study.csv not found.")
    return pd.read_csv(csv_path).to_dict(orient="records")
