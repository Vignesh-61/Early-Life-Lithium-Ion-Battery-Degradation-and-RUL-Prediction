// DegradationChart.jsx — Recharts line chart of capacity/SOH over cycles
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--bg-panel)", border: "1px solid var(--border-mid)",
      borderRadius: 7, padding: "10px 14px", fontSize: "0.78rem",
    }}>
      <p style={{ color: "var(--text-muted)", marginBottom: 6 }}>Cycle {label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <b>{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</b>
        </p>
      ))}
    </div>
  );
};

/**
 * Props:
 *   data      — array of { cycle, capacity, soh, rul }  (from battery history API)
 *   mode      — "capacity" | "soh"
 *   eolLine   — numeric threshold to draw (e.g. 1.4 for capacity, 70 for SOH)
 *   highlight — optional { cycle, value } to mark a prediction point
 */
export default function DegradationChart({ data = [], mode = "capacity", eolLine, highlight }) {
  if (!data.length) {
    return (
      <div style={{ height: 260, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)" }}>
        No data — select a battery
      </div>
    );
  }

  const isCapacity = mode === "capacity";
  const dataKey    = isCapacity ? "capacity" : "soh";
  const label      = isCapacity ? "Capacity (Ah)" : "SOH (%)";
  const color      = "var(--accent)";
  const yMin       = isCapacity ? 1.0 : 50;

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 6, right: 20, bottom: 6, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="cycle"
          stroke="var(--text-muted)"
          tick={{ fontSize: 11 }}
          label={{ value: "Cycle", position: "insideBottomRight", offset: -4, fontSize: 11, fill: "var(--text-muted)" }}
        />
        <YAxis
          stroke="var(--text-muted)"
          tick={{ fontSize: 11 }}
          domain={[yMin, "auto"]}
          label={{ value: label, angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-muted)" }}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: "0.78rem", color: "var(--text-secondary)" }} />

        {eolLine && (
          <ReferenceLine
            y={eolLine}
            stroke="var(--yellow)"
            strokeDasharray="5 3"
            label={{ value: "EOL", position: "right", fontSize: 10, fill: "var(--yellow)" }}
          />
        )}

        <Line
          type="monotone"
          dataKey={dataKey}
          name={label}
          stroke={color}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: color }}
        />

        {highlight && (
          <ReferenceLine
            x={highlight.cycle}
            stroke="var(--red)"
            strokeDasharray="4 2"
            label={{ value: "▶ Prediction", position: "top", fontSize: 10, fill: "var(--red)" }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
