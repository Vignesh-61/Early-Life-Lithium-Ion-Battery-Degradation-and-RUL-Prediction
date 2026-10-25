// MetricCard.jsx — Single KPI tile
export default function MetricCard({ label, value, sub, accent = false, badge, loading = false }) {
  return (
    <div className="card" style={accent ? { borderColor: "var(--accent)" } : {}}>
      <div className="card-title">{label}</div>
      {loading
        ? <div style={{ marginTop: 8 }}><span className="spinner" /></div>
        : <div className="card-value">{value ?? "—"}</div>
      }
      {sub && <div className="card-sub">{sub}</div>}
      {badge && (
        <div style={{ marginTop: 8 }}>
          <span className={`badge badge-${badge.color}`}>{badge.text}</span>
        </div>
      )}
    </div>
  );
}
