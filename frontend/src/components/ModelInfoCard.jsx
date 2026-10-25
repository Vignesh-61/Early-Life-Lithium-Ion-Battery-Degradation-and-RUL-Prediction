// ModelInfoCard.jsx — shows config.json metadata from /api/model-info
export default function ModelInfoCard({ info }) {
  if (!info) return null;

  const m = info.model_architecture || {};
  const v = info.validation_metrics || {};

  return (
    <div className="card">
      <div className="card-title">Model Information</div>
      <table className="data-table" style={{ marginTop: 4 }}>
        <tbody>
          <tr><td style={{ color: "var(--text-muted)", width: 140 }}>Architecture</td>
              <td className="mono">{m.type}</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Hidden Layers</td>
              <td className="mono">{(m.hidden_layers || []).join(" → ")}</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Activation</td>
              <td className="mono">{m.activation}</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Optimizer</td>
              <td className="mono">{m.optimizer}</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Memory Window</td>
              <td className="mono">{info.memory_window} cycles</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Input Dim</td>
              <td className="mono">{info.input_dimension}</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>EOL Threshold</td>
              <td className="mono">{info.eol_threshold_ah} Ah (70%)</td></tr>
          <tr><td style={{ color: "var(--text-muted)" }}>Dataset</td>
              <td style={{ fontSize: "0.78rem" }}>{info.dataset}</td></tr>
        </tbody>
      </table>

      {Object.keys(v).length > 0 && (
        <>
          <div className="divider" />
          <div className="card-title">LOBO Validation Metrics</div>
          <table className="data-table" style={{ marginTop: 4 }}>
            <tbody>
              {v.mean_MAE_Ah   != null && <tr><td style={{ color: "var(--text-muted)" }}>MAE</td>     <td className="mono">{v.mean_MAE_Ah} Ah</td></tr>}
              {v.mean_RMSE_Ah  != null && <tr><td style={{ color: "var(--text-muted)" }}>RMSE</td>    <td className="mono">{v.mean_RMSE_Ah} Ah</td></tr>}
              {v.mean_R2       != null && <tr><td style={{ color: "var(--text-muted)" }}>R²</td>      <td className="mono">{v.mean_R2}</td></tr>}
              {v.mean_MAPE_percent != null && <tr><td style={{ color: "var(--text-muted)" }}>MAPE</td><td className="mono">{v.mean_MAPE_percent}%</td></tr>}
              {v.mean_RUL_MAE_cycles != null && <tr><td style={{ color: "var(--text-muted)" }}>RUL MAE</td><td className="mono">{v.mean_RUL_MAE_cycles} cycles</td></tr>}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
