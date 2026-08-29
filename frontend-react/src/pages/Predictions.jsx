import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
} from "chart.js";
import PageHero from "../components/PageHero.jsx";
import { Loader, ErrorBlock, EmptyBlock } from "../components/States.jsx";
import { api } from "../api.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

const OUTCOME_MAP = {
  Win: { label: "Home Win", cls: "outcome-win" },
  Draw: { label: "Draw", cls: "outcome-draw" },
  Lose: { label: "Away Win", cls: "outcome-lose" },
  ERROR: { label: "Unavailable", cls: "outcome-error" },
  SKIPPED: { label: "Skipped", cls: "outcome-error" },
};

function formatDate(value) {
  if (!value) return "TBD";
  const d = new Date(value);
  if (isNaN(d)) return value;
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
}

function PredictionCard({ p }) {
  const outcome = OUTCOME_MAP[p.predicted_result] || OUTCOME_MAP.ERROR;
  const confidence = typeof p.confidence === "number" ? p.confidence : null;

  return (
    <div className="card card-white prediction-card">
      <span className={`prediction-outcome ${outcome.cls}`}>{outcome.label}</span>

      <div className="match-card" style={{ padding: 0 }}>
        <div className="match-side home">
          <span className="match-team-name">{p.home_team}</span>
        </div>
        <div className="match-mid">
          <span className="vs-pill">VS</span>
          <span className="match-date">{formatDate(p.match_date)}</span>
        </div>
        <div className="match-side away">
          <span className="match-team-name">{p.away_team}</span>
        </div>
      </div>

      {confidence !== null && (
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span className="stat-label">Model confidence</span>
            <span className="stat-label" style={{ color: "var(--purple)" }}>{confidence.toFixed(1)}%</span>
          </div>
          <div className="confidence-bar-track">
            <div className="confidence-bar-fill" style={{ width: `${Math.min(confidence, 100)}%` }} />
          </div>
        </div>
      )}

      {(p.prob_win != null || p.prob_draw != null || p.prob_lose != null) && (
        <div className="pip-row" style={{ gap: 8 }}>
          {p.prob_win != null && <span className="vs-pill">Home {p.prob_win.toFixed?.(0) ?? p.prob_win}%</span>}
          {p.prob_draw != null && <span className="vs-pill">Draw {p.prob_draw.toFixed?.(0) ?? p.prob_draw}%</span>}
          {p.prob_lose != null && <span className="vs-pill">Away {p.prob_lose.toFixed?.(0) ?? p.prob_lose}%</span>}
        </div>
      )}
    </div>
  );
}

function FeatureImportanceChart({ rows }) {
  if (!rows || rows.length === 0) return <EmptyBlock message="Feature importance hasn't been logged yet." />;

  const sorted = [...rows].sort((a, b) => (b.importance ?? 0) - (a.importance ?? 0)).slice(0, 8);
  const labels = sorted.map((r) => r.feature ?? r.name ?? "—");
  const values = sorted.map((r) => r.importance ?? 0);

  const chartData = {
    labels,
    datasets: [
      {
        label: "Importance",
        data: values,
        backgroundColor: "#00c96a",
        borderRadius: 6,
        barThickness: 18,
      },
    ],
  };

  const options = {
    indexAxis: "y",
    responsive: true,
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: {
      x: { grid: { color: "#e6def0" }, ticks: { color: "#6b5c74" } },
      y: { grid: { display: false }, ticks: { color: "#14061a", font: { weight: 600 } } },
    },
  };

  return (
    <div className="card card-white" style={{ height: 340 }}>
      <Bar data={chartData} options={options} />
    </div>
  );
}

export default function Predictions() {
  const [predictions, setPredictions] = useState([]);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [status, setStatus] = useState("loading");
  const [errMsg, setErrMsg] = useState("");

  function load() {
    setStatus("loading");
    api
      .getPredictions()
      .then((res) => {
        setPredictions(res.predictions || []);
        setFeatureImportance(res.feature_importance || []);
        setStatus("ready");
      })
      .catch((err) => {
        setErrMsg(err.message);
        setStatus("error");
      });
  }

  useEffect(load, []);

  return (
    <>
      <PageHero
        eyebrow="Machine Learning"
        title="Gameweek Predictions"
        sub="Match outcomes predicted by the trained classifier, served straight from the MLflow model registry."
      />
      <div className="page-body container">
        {status === "loading" && <Loader label="Fetching model predictions…" />}
        {status === "error" && <ErrorBlock message={errMsg} onRetry={load} />}

        {status === "ready" && (
          <>
            <section style={{ marginBottom: 44 }}>
              <div className="section-head">
                <span className="section-title">This Gameweek</span>
                <span className="section-count">{predictions.length} match(es)</span>
                <div className="section-line" />
              </div>
              {predictions.length === 0 ? (
                <EmptyBlock message="No predictions have been generated for the upcoming gameweek yet." />
              ) : (
                <div className="grid grid-2">
                  {predictions.map((p, i) => (
                    <PredictionCard key={i} p={p} />
                  ))}
                </div>
              )}
            </section>

            <section>
              <div className="section-head">
                <span className="section-title">What the Model Weighs Most</span>
                <div className="section-line" />
              </div>
              <FeatureImportanceChart rows={featureImportance} />
            </section>
          </>
        )}
      </div>
    </>
  );
}
