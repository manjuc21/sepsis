export default function PatientList({ patients, selectedId, onSelect }) {
  if (patients.length === 0) {
    return (
      <div className="panel patient-list">
        <h2>Patients</h2>
        <p className="empty-state">
          No demo patients loaded yet. Run{" "}
          <code>scripts/prepare_demo_patients.py</code> after training a model.
        </p>
      </div>
    );
  }

  const sorted = [...patients].sort((a, b) => b.current_risk_score - a.current_risk_score);

  return (
    <div className="panel patient-list">
      <h2>Patients</h2>
      <ul>
        {sorted.map((p) => (
          <li
            key={p.patient_id}
            className={`patient-row ${p.patient_id === selectedId ? "selected" : ""} ${p.alert ? "alert" : ""}`}
            onClick={() => onSelect(p.patient_id)}
          >
            <span className="patient-id">{p.patient_id}</span>
            <span className={`risk-pill ${p.alert ? "risk-high" : "risk-low"}`}>
              {(p.current_risk_score * 100).toFixed(0)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
