// Dashboard.jsx — main overview page
import { useState, useEffect, useCallback } from "react";
import { fetchBatteries, fetchBatteryHistory, fetchModelInfo } from "../services/api";
import MetricCard from "../components/MetricCard";
import DegradationChart from "../components/DegradationChart";
import ModelInfoCard from "../components/ModelInfoCard";

export default function Dashboard() {
  const [batteries, setBatteries]   = useState([]);
  const [selected,  setSelected]    = useState("B0005");
  const [history,   setHistory]     = useState([]);
  const [modelInfo, setModelInfo]   = useState(null);
  const [chartMode, setChartMode]   = useState("capacity");
  const [loadingH,  setLoadingH]    = useState(false);
  const [error,     setError]       = useState(null);

  // Load batteries + model info once
  useEffect(() => {
    fetchBatteries().then(setBatteries).catch(() => {});
    fetchModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  // Load history whenever selected battery changes
  const loadHistory = useCallback((batteryId) => {
    setLoadingH(true);
    setError(null);
    fetchBatteryHistory(batteryId)
      .then(r => setHistory(r.history || []))
      .catch(e => setError(e.response?.data?.detail || "Failed to load battery history."))
      .finally(() => setLoadingH(false));
  }, []);

  useEffect(() => { if (selected) loadHistory(selected); }, [selected, loadHistory]);

  // Latest data point
  const latest = history.at(-1) || null;
  const first  = history.at(0)  || null;

  const sohColor = latest?.soh == null ? "var(--text-primary)"
                 : latest.soh >= 85    ? "var(--green)"
                 : latest.soh >= 70    ? "var(--yellow)"
                 : "var(--red)";

  const healthBadge = latest?.soh == null ? { color: "blue",   text: "Unknown" }
                    : latest.soh >= 85    ? { color: "green",  text: "Healthy" }
                    : latest.soh >= 70    ? { color: "yellow", text: "Degraded" }
                    : { color: "red", text: "Critical" };

  const eolLine = chartMode === "capacity" ? 1.40 : 70;

  return (
    <div>
      <div className="page-header">
        <h2>Battery Intelligence Dashboard</h2>
        <p>Real-time degradation tracking from the NASA Li-ion Battery Aging Dataset (B0005–B0018)</p>
      </div>

      {/* Battery selector */}
      <div style={{ display: "flex", gap: 10, marginBottom: 20, alignItems: "center" }}>
        <label className="form-label" style={{ margin: 0, whiteSpace: "nowrap" }}>Battery cell:</label>
        <select
          className="form-select"
          style={{ maxWidth: 180 }}
          value={selected}
          onChange={e => setSelected(e.target.value)}
        >
          {batteries.map(b => <option key={b} value={b}>{b}</option>)}
        </select>
        {loadingH && <span className="spinner" />}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {/* KPI row */}
      <div className="metric-grid">
        <MetricCard
          label="Latest Capacity"
          value={latest ? `${latest.capacity.toFixed(4)} Ah` : null}
          sub={`Cycle ${latest?.cycle ?? "—"}`}
          loading={loadingH}
        />
        <MetricCard
          label="State of Health"
          value={latest ? `${latest.soh.toFixed(2)} %` : null}
          sub={`Initial: ${first ? first.capacity.toFixed(4) : "—"} Ah`}
          badge={latest ? healthBadge : null}
          loading={loadingH}
        />
        <MetricCard
          label="Remaining Useful Life"
          value={latest ? `${latest.rul} cycles` : null}
          sub="Until EOL (1.40 Ah)"
          loading={loadingH}
        />
        <MetricCard
          label="Total Cycles"
          value={history.length || null}
          sub={`Memory window: W=${modelInfo?.memory_window ?? 10}`}
          loading={loadingH}
        />
      </div>

      {/* Degradation chart */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div className="section-header" style={{ marginBottom: 0 }}>
            Degradation Trajectory — {selected}
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {["capacity", "soh"].map(m => (
              <button
                key={m}
                className={`btn ${chartMode === m ? "btn-primary" : "btn-ghost"}`}
                style={{ padding: "5px 12px", fontSize: "0.78rem" }}
                onClick={() => setChartMode(m)}
              >
                {m === "capacity" ? "Capacity (Ah)" : "SOH (%)"}
              </button>
            ))}
          </div>
        </div>
        <DegradationChart
          data={history}
          mode={chartMode}
          eolLine={eolLine}
        />
        <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 8 }}>
          Yellow dashed line = EOL threshold ({eolLine}{chartMode === "capacity" ? " Ah" : "%"}).
          Data sourced directly from NASA dataset via <code>/api/batteries/{selected}/history</code>.
        </div>
      </div>

      {/* Bottom row: model info + SOH bar */}
      <div className="two-col">
        <ModelInfoCard info={modelInfo} />

        <div className="card">
          <div className="card-title">SOH Health Gauge</div>
          <div style={{ marginTop: 12 }}>
            {[
              { label: "Optimal (≥ 85%)", min: 85, color: "var(--green)"  },
              { label: "Degraded (70–84%)", min: 70, color: "var(--yellow)" },
              { label: "Critical (< 70%)", min: 0,  color: "var(--red)"   },
            ].map(({ label, min, color }) => {
              const pct = latest?.soh != null ? Math.min(100, Math.max(0, (latest.soh / 100) * 100)) : 0;
              const active = latest?.soh != null &&
                (min === 85 ? latest.soh >= 85 :
                 min === 70 ? latest.soh >= 70 && latest.soh < 85 :
                              latest.soh < 70);
              return (
                <div key={label} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", marginBottom: 4 }}>
                    <span style={{ color: active ? color : "var(--text-muted)" }}>{label}</span>
                    {active && <span style={{ color }}>{latest?.soh?.toFixed(1)}%</span>}
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{
                        width: active && latest ? `${latest.soh}%` : "0%",
                        background: color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="divider" />
          <div className="card-title">EOL Definition</div>
          <p style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            End-of-Life is defined as the first cycle where discharge capacity ≤ 1.40 Ah (70% of 2.0 Ah nominal).
            RUL = max(0, EOL cycle − current cycle).
          </p>
        </div>
      </div>
    </div>
  );
}
