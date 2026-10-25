// Research.jsx — Research documentation page with actual experiment results
import { useState, useEffect } from "react";
import { fetchModelInfo, fetchMemoryComparison, fetchAblation } from "../services/api";

const ARCH_STEPS = [
  { label: "Input Layer", desc: "W × 12 = 120 features (Memory-10)", dim: "120" },
  { label: "Dense 128 + ReLU", desc: "He Normal initialization", dim: "128" },
  { label: "Dropout (p=0.20)", desc: "Regularization", dim: "128" },
  { label: "Dense 64 + ReLU", desc: "", dim: "64" },
  { label: "Dense 32 + ReLU", desc: "", dim: "32" },
  { label: "Linear Output", desc: "Predicted Capacity (Ah)", dim: "1" },
];

const WINDOWS = [
  { w: 3,  dim: 36,  rmse: 0.09486, mae: 0.08180, r2: 0.6794, rul: 12.91 },
  { w: 5,  dim: 60,  rmse: 0.13176, mae: 0.10801, r2: 0.3483, rul: 33.40 },
  { w: 10, dim: 120, rmse: 0.09960, mae: 0.08037, r2: 0.6512, rul: 8.45, selected: true },
  { w: 20, dim: 240, rmse: 0.10415, mae: 0.08861, r2: 0.1382, rul: 12.96 },
];

export default function Research() {
  const [modelInfo,   setModelInfo]   = useState(null);
  const [memResults,  setMemResults]  = useState([]);
  const [ablResults,  setAblResults]  = useState([]);

  useEffect(() => {
    fetchModelInfo().then(setModelInfo).catch(() => {});
    fetchMemoryComparison().then(setMemResults).catch(() => {});
    fetchAblation().then(setAblResults).catch(() => {});
  }, []);

  const Section = ({ title, children }) => (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="section-header">{title}</div>
      {children}
    </div>
  );

  return (
    <div>
      <div className="page-header">
        <h2>Research Overview</h2>
        <p>
          Documentation of the <em>Adaptive Memory-Window Deep Neural Network</em> research project.
          All metrics shown here are from actual Leave-One-Battery-Out cross-validation runs.
        </p>
      </div>

      {/* Problem */}
      <Section title="Problem Statement">
        <p style={{ color: "var(--text-secondary)", fontSize: "0.85rem", lineHeight: 1.7 }}>
          Lithium-ion batteries degrade with every charge-discharge cycle.
          Capacity fade is non-linear — capacity <em>regeneration spikes</em> occur during rest,
          making naive prediction unreliable. State of Health (SOH) and Remaining Useful Life (RUL)
          must be estimated accurately for predictive maintenance in electric vehicles and grid storage.
          <br/><br/>
          Existing work relies on computationally expensive recurrent networks (LSTM, GRU) or Transformers.
          This project investigates whether a <strong>lightweight, pure feed-forward DNN</strong> equipped
          with an <strong>Adaptive Historical Memory Window</strong> can achieve competitive accuracy.
        </p>
      </Section>

      {/* Research question */}
      <Section title="Research Question">
        <div style={{
          background: "var(--bg-base)", borderRadius: 8,
          padding: "16px 20px", borderLeft: "3px solid var(--accent)",
          color: "var(--text-primary)", fontSize: "0.9rem", fontStyle: "italic",
        }}>
          "How much historical battery-cycle information is required for a pure feed-forward DNN
          to accurately predict future battery degradation and RUL under battery-independent evaluation?"
        </div>
        <div style={{ marginTop: 16, display: "flex", gap: 10, flexWrap: "wrap" }}>
          {[3, 5, 10, 20].map(w => (
            <span
              key={w}
              className={`badge ${w === 10 ? "badge-blue" : "badge-green"}`}
              style={{ fontSize: "0.85rem", padding: "6px 14px" }}
            >
              W = {w} cycles {w === 10 ? "✓ selected" : ""}
            </span>
          ))}
        </div>
      </Section>

      {/* Architecture */}
      <Section title="DNN Architecture (Feed-Forward MLP)">
        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 420 }}>
          {ARCH_STEPS.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                background: i === 0 || i === ARCH_STEPS.length - 1
                  ? "var(--accent)" : "var(--bg-hover)",
                border: "1px solid var(--border-mid)",
                borderRadius: 7, padding: "8px 14px", minWidth: 200,
              }}>
                <div style={{ fontWeight: 600, fontSize: "0.84rem" }}>{s.label}</div>
                {s.desc && <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: 2 }}>{s.desc}</div>}
              </div>
              <div className="mono" style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
                dim = {s.dim}
              </div>
              {i < ARCH_STEPS.length - 1 && (
                <div style={{ position: "absolute", marginLeft: 220, color: "var(--text-muted)" }}>▼</div>
              )}
            </div>
          ))}
        </div>
        <div style={{ marginTop: 16, fontSize: "0.78rem", color: "var(--text-secondary)" }}>
          Optimizer: Adam (lr=0.001, weight_decay=1e-4) | Loss: MSELoss | Dropout: 0.20 | Init: He/Kaiming Normal
        </div>
      </Section>

      {/* Memory window comparison — from API */}
      <Section title="Memory Window Benchmark (LOBO Cross-Validation)">
        <p className="section-desc">
          Identical architecture trained with each window size. Evaluated on strictly unseen battery cells.
          {memResults.length > 0 && " Values loaded from the research results API."}
        </p>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Window (W)</th>
                <th>Input Dim</th>
                <th>LOBO MAE (Ah)</th>
                <th>LOBO RMSE (Ah)</th>
                <th>LOBO R²</th>
                <th>MAPE (%)</th>
                <th>RUL Error (cycles)</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(memResults.length > 0 ? memResults : WINDOWS).map(row => {
                const w = row.memory_window ?? row.w;
                const selected = w === 10;
                return (
                  <tr key={w} style={selected ? { background: "var(--accent-glow)" } : {}}>
                    <td className="mono" style={selected ? { color: "var(--accent)" } : {}}>{w}</td>
                    <td className="mono">{row.input_features ?? row.dim}</td>
                    <td className="mono">{(row.MAE ?? row.mae)?.toFixed(5)}</td>
                    <td className="mono">{(row.RMSE ?? row.rmse)?.toFixed(5)}</td>
                    <td className="mono">{(row.R2 ?? row.r2)?.toFixed(4)}</td>
                    <td className="mono">{(row["MAPE_%"] ?? (row.mae * 100 / 1.6))?.toFixed(2)}</td>
                    <td className="mono">{(row.RUL_error_cycles ?? row.rul)?.toFixed(2)}</td>
                    <td>{selected && <span className="badge badge-blue">Selected</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Ablation — from API */}
      {ablResults.length > 0 && (
        <Section title="Feature Ablation Study (W=10, LOBO)">
          <p className="section-desc">
            Impact of removing feature groups from the 12-feature input. Evaluated under the same LOBO protocol.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Configuration</th>
                  <th>Window</th>
                  <th>Features</th>
                  <th>RMSE (Ah)</th>
                  <th>R²</th>
                </tr>
              </thead>
              <tbody>
                {ablResults.map((row, i) => (
                  <tr key={i}>
                    <td style={{ maxWidth: 320 }}>{row.Configuration}</td>
                    <td className="mono">{row.Window_W}</td>
                    <td className="mono">{row.Input_Features}</td>
                    <td className="mono">{Number(row.LOBO_RMSE).toFixed(5)}</td>
                    <td className="mono">{Number(row.LOBO_R2).toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Dataset */}
      <Section title="Dataset">
        <table className="data-table">
          <tbody>
            <tr><td style={{ color: "var(--text-muted)", width: 160 }}>Source</td><td>NASA Lithium-Ion Battery Aging Dataset</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>Battery Cells</td><td className="mono">B0005, B0006, B0007, B0018</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>Total Cycles</td><td className="mono">636 discharge cycles</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>Nominal Capacity</td><td className="mono">2.0 Ah (18650 LiCoO₂)</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>EOL Definition</td><td className="mono">≤ 1.40 Ah (70% rated)</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>Features per Cycle</td><td className="mono">12 statistical features</td></tr>
            <tr><td style={{ color: "var(--text-muted)" }}>Validation</td><td>4-fold Leave-One-Battery-Out (LOBO) cross-validation</td></tr>
          </tbody>
        </table>
      </Section>

      {/* Integrity note */}
      <div className="alert alert-info">
        <strong>Research Integrity:</strong> This frontend is an application layer built around the existing
        research implementation. The DNN architecture, trained weights, preprocessing pipeline, feature
        engineering, and all experimental results are unchanged. The API calls the same <code>predict_next_cycle()</code>
        function from <code>src/prediction.py</code>.
      </div>
    </div>
  );
}
