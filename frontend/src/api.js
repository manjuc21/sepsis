const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function get(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json();
}

export function fetchPatients() {
  return get("/patients");
}

export function fetchPatientTimeline(patientId) {
  return get(`/patients/${encodeURIComponent(patientId)}/timeline`);
}

export function fetchHealth() {
  return get("/health");
}
