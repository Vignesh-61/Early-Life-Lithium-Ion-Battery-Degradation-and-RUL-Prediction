"""
main.py — FastAPI application entry point
==========================================
Starts the battery-RUL prognostics API.
The existing research model is loaded once at startup via lifespan().

Run:
    uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.model_service import battery_service
from backend.routes.prediction import router as prediction_router


# ── Lifespan: load model once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained DNN and scaler before serving any request."""
    battery_service.load()
    print("[startup] Battery DNN loaded — ready to serve predictions.")
    yield
    print("[shutdown] Battery DNN service stopped.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Battery RUL Prognostics API",
    description = (
        "FastAPI integration layer around the existing "
        "'Adaptive Memory-Window Deep Neural Network' research implementation. "
        "The DNN, preprocessing, and feature extraction are unchanged."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

# Allow the Vite dev server (localhost:5173) and any production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(prediction_router)


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["system"], summary="API health check")
def health():
    return {"status": "ok", "model_loaded": battery_service.is_loaded}


@app.get("/api/model-info", tags=["system"], summary="Trained model metadata")
def model_info():
    """
    Returns the metadata stored in models/final/config.json — no invented values.
    """
    if not battery_service.is_loaded:
        return {"error": "Model not yet loaded."}
    return battery_service.model_info()
