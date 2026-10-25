// PredictionResult.jsx — show DNN output after inference
export default function PredictionResult({ result, actual = null }) {
  if (!result) return null;

  const soh = result.soh ?? null;
  const cap = result.predicted_capacity ?? result.predicted_capacity_ah;
  const rul = result.rul_estimate ?? null;

  const sohColor = soh === null ? "var(--text-primary)"
                 : soh >= 85   ? "var(--green)"
                 : soh >= 70   ? "var(--yellow)"
                 : "var(--red)";

  const healthLabel = soh === null ? "—"
                    : soh >= 85   ? "Good"
                    : soh >= 70   ? "Degraded"
                    : "Critical";

  const healthBadge = soh === null ? "blue"
                    : soh >= 85   ? "green"
                    : soh >= 70   ? "yellow"
                    : "red";

  return (
    <div className="card" style={{ borderColor: "var(--accent)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <h3 style={{ fontSize: "0.9rem", fontWeight: 700 }}>⚡ DNN Prediction Result</h3>
        <span className={`badge badge-${healthBadge}`}>{healthLabel}</span>
      </div>

      <div className="metric-grid" style={{ gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
        {/* Capacity */}
        <div>
          <div className="card-title">Predicted Capacity</div>
          <div className="card-value mono" style={{ fontSize: "1.5rem" }}>
            {cap != null ? cap.toFixed(4) : "—"}
          </div>
          <div className="card-sub">Ah</div>
          {actual?.actual_capacity != null && (
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>
              Actual: {actual.actual_capacity.toFixed(4)} Ah
            </div>
          )}
        </div>

        {/* SOH */}
        <div>
          <div className="card-title">State of Health</div>
          <div className="card-value mono" style={{ fontSize: "1.5rem", color: sohColor }}>
            {soh != null ? soh.toFixed(2) : "—"}
          </div>
          <div className="card-sub">%</div>
          {actual?.actual_soh != null && (
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>
              Actual: {actual.actual_soh.toFixed(2)} %
            </div>
          )}
        </div>

        {/* RUL */}
        <div>
          <div className="card-title">Est. Remaining Useful Life</div>
          <div className="card-value mono" style={{ fontSize: "1.5rem", color: "var(--yellow)" }}>
            {rul != null ? rul : "—"}
          </div>
          <div className="card-sub">cycles</div>
          {actual?.actual_rul != null && (
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 4 }}>
              Actual: {actual.actual_rul} cycles
            </div>
          )}
        </div>
      </div>

      {/* EOL flag */}
      {result.is_below_eol && (
        <div className="alert alert-error" style={{ marginTop: 16, marginBottom: 0 }}>
          ⚠ Predicted capacity is <strong>below the EOL threshold (1.40 Ah)</strong>.
          This battery has reached end-of-life per the research definition.
        </div>
      )}

      {/* Disclaimer */}
      <div style={{ marginTop: 14, padding: "8px 12px", background: "var(--bg-base)", borderRadius: 6, fontSize: "0.72rem", color: "var(--text-muted)" }}>
        Research prototype. Values from the trained Adaptive Memory-Window DNN (W=10). Not for safety-critical use.
      </div>
    </div>
  );
}
