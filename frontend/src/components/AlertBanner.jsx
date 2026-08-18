export default function AlertBanner({ alert, riskScore, threshold }) {
  if (!alert) {
    return (
      <div className="alert-banner ok">
        <strong>No active alert.</strong> Current risk {(riskScore * 100).toFixed(0)}%
        is below the {(threshold * 100).toFixed(0)}% threshold.
      </div>
    );
  }

  return (
    <div className="alert-banner danger">
      <strong>Sepsis risk alert.</strong> Current risk{" "}
      {(riskScore * 100).toFixed(0)}% has crossed the {(threshold * 100).toFixed(0)}%
      threshold — review contributing features below.
    </div>
  );
}
