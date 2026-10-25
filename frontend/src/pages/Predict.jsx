// Predict.jsx — Battery prediction page (NASA dataset selector + CSV upload)
import { useState, useEffect, useRef } from "react";
import {
  fetchBatteries,
  fetchBatteryHistory,
  predictAtCycle,
  predictCSV,
} from "../services/api";
import PredictionResult from "../components/PredictionResult";
import DegradationChart from "../components/DegradationChart";

// Tab IDs
const TAB_DATASET = "dataset";
const TAB_CSV     = "csv";

export default function Predict() {
  const [tab,        setTab]        = useState(TAB_DATASET);
  const [batteries,  setBatteries]  = useState([]);
  const [battery,    setBattery]    = useState("B0005");
  const [history,    setHistory]    = useState([]);
  const [maxCycle,   setMaxCycle]   = useState(167);
  const [cycleIdx,   setCycleIdx]   = useState(60);
  const [running,    setRunning]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);

  // CSV tab state
  const [csvFile,    setCsvFile]    = useState(null);
  const [dragging,   setDragging]   = useState(false);
  const [initCap,    setInitCap]    = useState("");
  const fileRef = useRef();

  useEffect(() => { fetchBatteries().then(setBatteries).catch(() => {}); }, []);

  useEffect(() => {
    if (!battery) return;
    fetchBatteryHistory(battery)
      .then(r => {
        const h = r.history || [];
        setHistory(h);
        const max = h.length - 1;
        setMaxCycle(max);
        setCycleIdx(prev => Math.min(prev, max));
      })
      .catch(() => {});
    setResult(null);
    setError(null);
  }, [battery]);

  // Run prediction (dataset tab)
  const runDatasetPredict = async () => {
    setRunning(true); setResult(null); setError(null);
    try {
      const r = await predictAtCycle(battery, cycleIdx);
      setResult(r);
    } catch (e) {
      setError(e.response?.data?.detail || "Prediction failed.");
    } finally { setRunning(false); }
  };

  // Run prediction (CSV tab)
  const runCsvPredict = async () => {
    if (!csvFile) { setError("Please upload a CSV file first."); return; }
    setRunning(true); setResult(null); setError(null);
    try {
      const ic = initCap ? parseFloat(initCap) : null;
      const r  = await predictCSV(csvFile, ic);
      setResult(r);
    } catch (e) {
      const msg = e.response?.data?.detail;
      setError(typeof msg === "string" ? msg : JSON.stringify(msg) || "Prediction failed.");
    } finally { setRunning(false); }
  };

  // Drag-and-drop
  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.name.endsWith(".csv")) { setCsvFile(f); setError(null); }
    else setError("Only .csv files are accepted.");
  };

  const currentHistPoint = history[cycleIdx] || null;

  return (
    <div>
      <div className="page-header">
        <h2>Battery Health Prediction</h2>
        <p>
          Run the trained Adaptive Memory-Window DNN (W=10) on a NASA battery
          or upload your own CSV with 10 discharge-cycle rows.
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, marginBottom: 20, borderBottom: "1px solid var(--border)" }}>
        {[
          { id: TAB_DATASET, label: "⬡  NASA Dataset Battery" },
          { id: TAB_CSV,     label: "⬆  Upload CSV" },
        ].map(t => (
          <button
            key={t.id}
            onClick={() => { setTab(t.id); setResult(null); setError(null); }}
            style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "10px 20px", fontSize: "0.85rem", fontWeight: 600,
              color: tab === t.id ? "var(--accent)" : "var(--text-secondary)",
              borderBottom: `2px solid ${tab === t.id ? "var(--accent)" : "transparent"}`,
              marginBottom: -1,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="two-col" style={{ alignItems: "start" }}>

        {/* Left panel */}
        <div>
          {tab === TAB_DATASET ? (
            <div className="card">
              <div className="section-header">Select Battery & Cycle</div>
              <p className="section-desc">
                The DNN builds a W=10 memory window ending at the selected cycle
                and predicts capacity at cycle index + 1.
              </p>

              <div style={{ marginBottom: 16 }}>
                <label className="form-label">Battery Cell</label>
                <select
                  className="form-select"
                  value={battery}
                  onChange={e => setBattery(e.target.value)}
                >
                  {batteries.map(b => <option key={b}>{b}</option>)}
                </select>
              </div>

              <div style={{ marginBottom: 6 }}>
                <label className="form-label">
                  Current Cycle Index: <span style={{ color: "var(--accent)" }}>{cycleIdx}</span>
                  {currentHistPoint && (
                    <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                      {" "}(Cycle #{currentHistPoint.cycle})
                    </span>
                  )}
                </label>
                <input
                  type="range"
                  className="slider"
                  min={10}
                  max={maxCycle > 10 ? maxCycle : 10}
                  value={cycleIdx}
                  onChange={e => { setCycleIdx(Number(e.target.value)); setResult(null); }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                  <span>10 (min)</span><span>{maxCycle} (max)</span>
                </div>
              </div>

              {currentHistPoint && (
                <div style={{ marginTop: 14, padding: "10px 14px", background: "var(--bg-base)", borderRadius: 7 }}>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginBottom: 6 }}>
                    Ground-truth at cycle index {cycleIdx}
                  </div>
                  <div style={{ display: "flex", gap: 24 }}>
                    <div><span className="card-title">Capacity</span><br/><span className="mono">{currentHistPoint.capacity.toFixed(4)} Ah</span></div>
                    <div><span className="card-title">SOH</span><br/><span className="mono">{currentHistPoint.soh.toFixed(2)} %</span></div>
                    <div><span className="card-title">RUL</span><br/><span className="mono">{currentHistPoint.rul} cycles</span></div>
                  </div>
                </div>
              )}

              <div style={{ marginTop: 20 }}>
                <button
                  className="btn btn-primary"
                  style={{ width: "100%" }}
                  onClick={runDatasetPredict}
                  disabled={running}
                >
                  {running ? <><span className="spinner" /> Running DNN…</> : "⚡ Run Prediction"}
                </button>
              </div>

              <div className="alert alert-info" style={{ marginTop: 14, marginBottom: 0, fontSize: "0.75rem" }}>
                Window: cycles {cycleIdx - 9}–{cycleIdx} → predict cycle {cycleIdx + 1}.
                Calls <code>POST /api/batteries/predict-at-cycle</code>.
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="section-header">Upload Battery Cycle CSV</div>
              <p className="section-desc">
                CSV must have exactly <strong>10 rows</strong> (consecutive discharge cycles)
                and the 12 feature columns listed below.
              </p>

              {/* Upload zone */}
              <div
                className={`upload-zone ${dragging ? "drag-over" : ""}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  style={{ display: "none" }}
                  onChange={e => { setCsvFile(e.target.files[0]); setError(null); }}
                />
                <div className="upload-icon">📄</div>
                {csvFile
                  ? <p style={{ color: "var(--green)" }}>{csvFile.name}</p>
                  : <p>Drag & drop a CSV or click to browse</p>
                }
              </div>

              {/* Optional initial capacity */}
              <div style={{ marginTop: 14 }}>
                <label className="form-label">Initial Capacity (Ah) — optional</label>
                <input
                  className="form-input"
                  type="number"
                  step="0.001"
                  placeholder="e.g. 1.856 — used for SOH computation"
                  value={initCap}
                  onChange={e => setInitCap(e.target.value)}
                />
              </div>

              <button
                className="btn btn-primary"
                style={{ width: "100%", marginTop: 16 }}
                onClick={runCsvPredict}
                disabled={running || !csvFile}
              >
                {running ? <><span className="spinner" /> Running DNN…</> : "⚡ Run Prediction"}
              </button>

              {/* Required columns */}
              <div className="divider" />
              <div className="card-title">Required CSV Columns (12)</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
                {[
                  "voltage_mean","voltage_min","voltage_max","voltage_std",
                  "current_mean","current_std","temp_mean","temp_max","temp_range",
                  "discharge_duration","discharge_capacity","cycle_number",
                ].map(c => (
                  <span key={c} className="badge badge-blue mono" style={{ fontSize: "0.68rem" }}>{c}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right panel — result */}
        <div>
          {result
            ? <PredictionResult result={result} actual={result} />
            : (
              <div className="card" style={{ textAlign: "center", padding: 40, color: "var(--text-muted)" }}>
                <div style={{ fontSize: 2.5 + "rem", marginBottom: 12 }}>⚡</div>
                <p>Prediction will appear here after you click <strong>Run Prediction</strong>.</p>
                <p style={{ marginTop: 8, fontSize: "0.78rem" }}>
                  Values come directly from the trained DNN via the FastAPI backend.
                </p>
              </div>
            )
          }

          {/* Degradation chart below result */}
          {history.length > 0 && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="card-title" style={{ marginBottom: 12 }}>
                {battery} — Capacity Degradation Curve
              </div>
              <DegradationChart
                data={history}
                mode="capacity"
                eolLine={1.40}
                highlight={result ? { cycle: result.cycle ?? cycleIdx, value: result.predicted_capacity } : null}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
