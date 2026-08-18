import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function RiskChart({ timeline, threshold }) {
  const data = timeline.map((point) => ({
    hour: point.hour,
    risk: point.risk_score,
  }));

  return (
    <div className="chart-wrapper">
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="hour"
            stroke="var(--text-muted)"
            label={{ value: "ICU hour", position: "insideBottom", offset: -5, fill: "var(--text-muted)" }}
          />
          <YAxis domain={[0, 1]} stroke="var(--text-muted)" tickFormatter={(v) => `${Math.round(v * 100)}%`} />
          <Tooltip
            formatter={(value) => [`${(value * 100).toFixed(1)}%`, "Risk"]}
            labelFormatter={(hour) => `Hour ${hour}`}
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)" }}
          />
          <ReferenceLine y={threshold} stroke="var(--danger)" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="risk" stroke="var(--accent)" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
