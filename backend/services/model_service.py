"""
model_service.py — Single authoritative inference service
==========================================================
Wraps the existing research pipeline:
  - src/prediction.py  → load_inference_pipeline, predict_next_cycle
  - src/preprocessing.py → compute_soh, compute_rul, DEFAULT_EOL_THRESHOLD_AH
  - src/feature_engineering.py → FEATURE_NAMES
  - models/final/config.json  → all configuration

NO preprocessing is duplicated here.
This class is the sole integration point between the API and the research code.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd

# ── Locate project root (battery-dnn-project/) and inject src/ into path ──
_BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))           # backend/services/
_BACKEND_ROOT = os.path.abspath(os.path.join(_BACKEND_DIR, ".."))    # backend/
_PROJECT_ROOT = os.path.abspath(os.path.join(_BACKEND_ROOT, ".."))   # battery-dnn-project/
_SRC_DIR      = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── Import existing research modules (do NOT re-implement) ──────────────────
from prediction import load_inference_pipeline, predict_next_cycle
from preprocessing import compute_soh, compute_rul, DEFAULT_EOL_THRESHOLD_AH
from feature_engineering import FEATURE_NAMES


# Paths to serialised artefacts
_MODELS_DIR  = os.path.join(_PROJECT_ROOT, "models", "final")
_MODEL_PATH  = os.path.join(_MODELS_DIR, "final_model.pt")
_SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.joblib")
_CONFIG_PATH = os.path.join(_MODELS_DIR, "config.json")
_DATA_PATH   = os.path.join(_PROJECT_ROOT, "data", "processed", "cycle_features.csv")


class BatteryModelService:
    """
    Singleton service that loads the trained DNN once and exposes a clean
    predict() interface consumed by the FastAPI routes.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    # ── Initialisation ────────────────────────────────────────────────────────

    def load(self):
        """Load model, scaler, config, and dataset exactly once."""
        if self._initialised:
            return

        # Config
        with open(_CONFIG_PATH, "r") as f:
            self.config = json.load(f)

        self.window_size   = self.config["selected_memory_window"]        # 10
        self.num_features  = self.config["num_base_features_per_cycle"]   # 12
        self.base_features = self.config["base_feature_names"]            # list[str]
        self.eol_threshold = self.config["eol_threshold_Ah"]              # 1.4
        self.rated_capacity = self.config["rated_capacity_Ah"]            # 2.0

        # Model + scaler (calls existing research function)
        self.model, self.scaler = load_inference_pipeline(
            model_path   = _MODEL_PATH,
            scaler_path  = _SCALER_PATH,
            window_size  = self.window_size,
            num_features = self.num_features,
        )

        # Dataset (used for battery history queries)
        self._df_raw = pd.read_csv(_DATA_PATH)
        self._df     = compute_soh(self._df_raw.copy())
        self._df     = compute_rul(self._df, eol_threshold_ah=self.eol_threshold)

        self._initialised = True

    @property
    def is_loaded(self) -> bool:
        return self._initialised

    # ── Public API ────────────────────────────────────────────────────────────

    def model_info(self) -> dict:
        """Return metadata from config.json (no invented values)."""
        return {
            "model_name":       self.config["model_name"],
            "memory_window":    self.window_size,
            "input_dimension":  self.config["input_dimension"],
            "features":         self.base_features,
            "target":           self.config["target_variable"],
            "architecture":     self.config["model_architecture"],
            "dataset":          self.config["dataset"],
            "eol_threshold_ah": self.eol_threshold,
            "rated_capacity_ah": self.rated_capacity,
            "validation_metrics": self.config["validation_metrics_lobo"],
        }

    def list_batteries(self) -> list[str]:
        """Return available battery IDs from the processed dataset."""
        return sorted(self._df["battery_id"].unique().tolist())

    def battery_history(self, battery_id: str) -> list[dict]:
        """
        Return the full SOH / capacity / RUL trajectory for a given battery.
        All values come from the existing dataset — nothing is fabricated.
        """
        bat = self._df[self._df["battery_id"] == battery_id].sort_values("cycle_number")
        if bat.empty:
            return []

        initial_cap = float(bat.iloc[0]["discharge_capacity"])
        records = []
        for _, row in bat.iterrows():
            cap = float(row["discharge_capacity"])
            records.append({
                "cycle":     int(row["cycle_number"]),
                "capacity":  round(cap, 4),
                "soh":       round(float(row["soh_percent"]), 2),
                "rul":       int(row["rul_cycles"]),
            })
        return records

    def predict(
        self,
        window_df: pd.DataFrame,
        initial_capacity: float | None = None,
    ) -> dict:
        """
        Run inference using the existing research predict_next_cycle() function.

        Parameters
        ----------
        window_df : pd.DataFrame
            Exactly `window_size` (10) consecutive rows with the 12 base feature columns.
        initial_capacity : float, optional
            The battery's first-ever measured capacity (Ah) — used for SOH computation.

        Returns
        -------
        dict with keys: predicted_capacity, soh, rul_estimate, is_below_eol
        """
        if len(window_df) != self.window_size:
            raise ValueError(
                f"Expected exactly {self.window_size} cycles in the window, "
                f"got {len(window_df)}."
            )

        missing = [c for c in self.base_features if c not in window_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        # Delegate entirely to existing research function
        result = predict_next_cycle(
            model            = self.model,
            scaler           = self.scaler,
            past_window_df   = window_df,
            initial_capacity = initial_capacity,
            eol_threshold    = self.eol_threshold,
        )

        predicted_cap = result["predicted_capacity_ah"]
        soh_pct       = result["predicted_soh_percent"]
        is_below_eol  = result["is_below_eol"]

        # Estimate RUL from degradation rate implied by the window
        ic = initial_capacity or float(window_df["discharge_capacity"].iloc[0])
        last_cap = float(window_df["discharge_capacity"].iloc[-1])
        last_cycle = int(window_df["cycle_number"].iloc[-1]) if "cycle_number" in window_df.columns else len(window_df)

        if last_cycle > 0 and ic > 0:
            deg_rate = max(1e-6, (ic - last_cap) / max(1, last_cycle))
            rul_est  = max(0, int((predicted_cap - self.eol_threshold) / deg_rate)) if predicted_cap > self.eol_threshold else 0
        else:
            rul_est = 0

        return {
            "predicted_capacity": predicted_cap,
            "soh":                soh_pct,
            "rul_estimate":       rul_est,
            "is_below_eol":       is_below_eol,
        }

    def predict_battery_at_cycle(self, battery_id: str, cycle_idx: int) -> dict:
        """
        Predict the next-cycle capacity for a known battery at a specific cycle index
        using the existing W=10 window from the processed dataset.
        """
        bat = self._df[self._df["battery_id"] == battery_id].sort_values("cycle_number").reset_index(drop=True)
        if bat.empty:
            raise ValueError(f"Battery '{battery_id}' not found.")

        n = len(bat)
        if cycle_idx < self.window_size or cycle_idx >= n:
            raise ValueError(
                f"cycle_idx must be between {self.window_size} and {n - 1}, got {cycle_idx}."
            )

        window_df        = bat.iloc[cycle_idx - self.window_size: cycle_idx][self.base_features].reset_index(drop=True)
        initial_capacity = float(bat.iloc[0]["discharge_capacity"])

        result   = self.predict(window_df, initial_capacity=initial_capacity)
        actual   = bat.iloc[cycle_idx]

        return {
            **result,
            "battery_id":      battery_id,
            "cycle":           int(actual["cycle_number"]),
            "actual_capacity": round(float(actual["discharge_capacity"]), 4),
            "actual_soh":      round(float(actual["soh_percent"]), 2),
            "actual_rul":      int(actual["rul_cycles"]),
        }


# Module-level singleton
battery_service = BatteryModelService()
