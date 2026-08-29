// Base URL of the FastAPI backend (deployed on Railway).
// Set VITE_API_BASE_URL in Vercel project settings / .env for local dev.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: "application/json" },
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore body parse errors */
    }
    throw new Error(`${res.status} — ${detail}`);
  }

  return res.json();
}

export const api = {
  getDashboard: () => getJSON("/api/dashboard"),
  getAnalytics: () => getJSON("/api/analytics/"),
  getPredictions: () => getJSON("/api/predictions/"),
};

export { API_BASE_URL };
