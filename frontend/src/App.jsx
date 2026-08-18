import { useEffect, useState } from "react";
import "./App.css";
import AlertBanner from "./components/AlertBanner";
import FeaturePanel from "./components/FeaturePanel";
import PatientList from "./components/PatientList";
import RiskChart from "./components/RiskChart";
import { fetchPatients, fetchPatientTimeline } from "./api";

const THRESHOLD = 0.5;

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [cursor, setCursor] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPatients()
      .then((data) => {
        setPatients(data);
        if (data.length > 0) setSelectedId(data[0].patient_id);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    fetchPatientTimeline(selectedId)
      .then((data) => {
        setTimeline(data);
        setCursor(data.timeline.length - 1);
      })
      .catch((e) => setError(e.message));
  }, [selectedId]);

  const currentPoint = timeline ? timeline.timeline[cursor] : null;

  return (
    <div className="app">
      <header className="app-header">
        <h1>Sepsis Early Prediction — Demo Dashboard</h1>
        <p className="subtitle">Simulated ICU patients, retrospective risk trends</p>
      </header>

      {error && <div className="alert-banner danger">Could not reach API: {error}</div>}

      <div className="layout">
        <PatientList patients={patients} selectedId={selectedId} onSelect={setSelectedId} />

        <main className="detail">
          {!timeline ? (
            <div className="panel">
              <p className="empty-state">Select a patient to view their risk trend.</p>
            </div>
          ) : (
            <>
              <div className="panel detail-header">
                <h2>{timeline.patient_id}</h2>
                <input
                  type="range"
                  min={0}
                  max={timeline.timeline.length - 1}
                  value={cursor}
                  onChange={(e) => setCursor(Number(e.target.value))}
                />
                <span className="cursor-label">
                  Hour {currentPoint.hour} of {timeline.timeline.length}
                </span>
              </div>

              <AlertBanner
                alert={currentPoint.risk_score >= THRESHOLD}
                riskScore={currentPoint.risk_score}
                threshold={THRESHOLD}
              />

              <div className="panel">
                <h3>Risk trend</h3>
                <RiskChart timeline={timeline.timeline} threshold={THRESHOLD} />
              </div>

              <FeaturePanel features={currentPoint.top_features} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
