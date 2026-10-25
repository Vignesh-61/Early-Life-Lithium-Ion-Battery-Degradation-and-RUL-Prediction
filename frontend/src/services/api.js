/**
 * api.js — single source of truth for all HTTP calls to the FastAPI backend.
 * All fetch calls go through axios; base URL is read from env (or defaults to localhost:8000).
 */

import axios from "axios";

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// ── Health & info ─────────────────────────────────────────────────────────────

export const fetchHealth     = () => api.get("/api/health").then(r => r.data);
export const fetchModelInfo  = () => api.get("/api/model-info").then(r => r.data);

// ── Battery data ──────────────────────────────────────────────────────────────

export const fetchBatteries      = () => api.get("/api/batteries").then(r => r.data.batteries);
export const fetchBatteryHistory = (batteryId) =>
  api.get(`/api/batteries/${batteryId}/history`).then(r => r.data);

// ── Prediction ────────────────────────────────────────────────────────────────

/** Predict using a known NASA battery at a specific cycle index. */
export const predictAtCycle = (batteryId, cycleIdx) =>
  api.post("/api/batteries/predict-at-cycle", { battery_id: batteryId, cycle_idx: cycleIdx })
     .then(r => r.data);

/** Manual prediction from 10 cycle-feature rows. */
export const predictManual = (cycles, initialCapacity = null) =>
  api.post("/api/predict", { cycles, initial_capacity: initialCapacity })
     .then(r => r.data);

/** CSV upload prediction. */
export const predictCSV = (file, initialCapacity = null) => {
  const form = new FormData();
  form.append("file", file);
  const params = initialCapacity ? `?initial_capacity=${initialCapacity}` : "";
  return api.post(`/api/predict/csv${params}`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  }).then(r => r.data);
};

// ── Research results ──────────────────────────────────────────────────────────

export const fetchMemoryComparison = () =>
  api.get("/api/results/memory-comparison").then(r => r.data);

export const fetchAblation = () =>
  api.get("/api/results/ablation").then(r => r.data);

export default api;
