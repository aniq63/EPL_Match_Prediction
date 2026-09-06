// Base URL of the FastAPI backend (deployed on Railway).
// Set VITE_API_BASE_URL in Vercel project settings / .env for local dev.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const requestCache = new Map();

// Loud, one-time console warning if this build is running somewhere other than
// localhost but is still pointed at a localhost backend — the #1 cause of
// "nothing loads" once deployed to Vercel without setting the env var there.
if (
  typeof window !== "undefined" &&
  window.location.hostname !== "localhost" &&
  API_BASE_URL.includes("localhost")
) {
  console.warn(
    "[PL Nexus] VITE_API_BASE_URL is not set for this deployment — " +
      "the app is falling back to http://localhost:8000, which will never be " +
      "reachable from a deployed site. Set VITE_API_BASE_URL in your Vercel " +
      "project's Environment Variables to your Railway backend URL and redeploy."
  );
}

async function getJSON(path) {
  const url = `${API_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  let res;
  try {
    res = await fetch(url, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") {
      throw new Error(`Timed out waiting for ${url}. Is the backend awake/running?`);
    }
    // Almost always: backend is down, URL is wrong, or CORS blocked the request.
    throw new Error(
      `Couldn't reach ${url}. Check VITE_API_BASE_URL is correct and the backend is running. (${err.message})`
    );
  }
  clearTimeout(timeout);

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      /* ignore body parse errors */
    }
    throw new Error(`${url} → ${res.status} ${detail}`);
  }

  return res.json();
}

function getCachedJSON(path) {
  if (!requestCache.has(path)) {
    const request = getJSON(path).catch((error) => {
      requestCache.delete(path);
      throw error;
    });
    requestCache.set(path, request);
  }
  return requestCache.get(path);
}

export const api = {
  getDashboard: () => getCachedJSON("/api/dashboard"),
  getAnalytics: () => getCachedJSON("/api/analytics/"),
  getPredictions: () => getCachedJSON("/api/predictions/"),
};

export { API_BASE_URL };
