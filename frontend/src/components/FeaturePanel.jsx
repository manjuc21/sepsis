export default function FeaturePanel({ features }) {
  if (!features || features.length === 0) {
    return (
      <div className="panel feature-panel">
        <h3>Why this score?</h3>
        <p className="empty-state">No explanation available for this point.</p>
      </div>
    );
  }

  return (
    <div className="panel feature-panel">
      <h3>Why this score?</h3>
      <p className="feature-panel-subtitle">Top contributing factors, most influential first.</p>
      <ol>
        {features.map((f, i) => (
          <li key={i}>{f}</li>
        ))}
      </ol>
    </div>
  );
}
