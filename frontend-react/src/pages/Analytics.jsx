import { useEffect, useMemo, useState } from "react";
import PageHero from "../components/PageHero.jsx";
import { Loader, ErrorBlock, EmptyBlock } from "../components/States.jsx";
import { api } from "../api.js";

// Each tab maps to a backend table key + a preferred metric field name
// (the schema returns raw SQL rows, so we guess sensibly and fall back
// to the first numeric column if the preferred one isn't present).
const TABS = [
  { key: "top_players_goals", label: "Top Scorers", group: "Players", metricHints: ["goals"] },
  { key: "top_players_assists", label: "Top Assists", group: "Players", metricHints: ["assists"] },
  { key: "top_players_shots", label: "Most Shots", group: "Players", metricHints: ["shots"] },
  { key: "top_players_key_passes", label: "Key Passes", group: "Players", metricHints: ["key_passes", "keypasses"] },
  { key: "top_players_yellow_cards", label: "Yellow Cards", group: "Players", metricHints: ["yellow_cards", "yellowcards"] },
  { key: "top_players_red_cards", label: "Red Cards", group: "Players", metricHints: ["red_cards", "redcards"] },
  { key: "top_teams_goals", label: "Team Goals", group: "Teams", metricHints: ["goals"] },
  { key: "top_teams_shots", label: "Team Shots", group: "Teams", metricHints: ["shots"] },
  { key: "top_teams_yellow_cards", label: "Team Yellows", group: "Teams", metricHints: ["yellow_cards", "yellowcards"] },
  { key: "top_teams_red_cards", label: "Team Reds", group: "Teams", metricHints: ["red_cards", "redcards"] },
  { key: "top_teams_created_chances", label: "Chances Created", group: "Teams", metricHints: ["created_chances", "chances_created", "chances"] },
];

function findKey(record, hints) {
  const keys = Object.keys(record);
  for (const hint of hints) {
    const match = keys.find((k) => k.toLowerCase().replace(/_/g, "") === hint.replace(/_/g, ""));
    if (match) return match;
  }
  return null;
}

function inferColumns(record, metricHints) {
  const keys = Object.keys(record);
  const nameKey =
    keys.find((k) => k.toLowerCase() === "player") ||
    keys.find((k) => k.toLowerCase() === "team" && !keys.includes("player")) ||
    keys.find((k) => typeof record[k] === "string");
  const teamKey =
    nameKey?.toLowerCase() !== "team" ? keys.find((k) => k.toLowerCase() === "team") : null;
  const metricKey =
    findKey(record, metricHints) ||
    keys.find((k) => typeof record[k] === "number" && k !== "id");

  return { nameKey, teamKey, metricKey };
}

function Leaderboard({ rows, metricHints }) {
  if (!rows || rows.length === 0) return <EmptyBlock message="No records for this category." />;

  const { nameKey, teamKey, metricKey } = inferColumns(rows[0], metricHints);

  return (
    <div className="card card-white">
      <div className="leaderboard-list">
        {rows.slice(0, 10).map((row, i) => (
          <div className="leaderboard-row" key={i}>
            <span className="leaderboard-rank">{i + 1}</span>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="leaderboard-name">{nameKey ? row[nameKey] : "—"}</div>
              {teamKey && <div className="leaderboard-team">{row[teamKey]}</div>}
            </div>
            <span className="leaderboard-value">{metricKey ? row[metricKey] : "—"}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Analytics() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");
  const [errMsg, setErrMsg] = useState("");
  const [activeTab, setActiveTab] = useState(TABS[0].key);

  function load() {
    setStatus("loading");
    api
      .getAnalytics()
      .then((res) => {
        setData(res.data);
        setStatus("ready");
      })
      .catch((err) => {
        setErrMsg(err.message);
        setStatus("error");
      });
  }

  useEffect(load, []);

  const active = useMemo(() => TABS.find((t) => t.key === activeTab), [activeTab]);

  return (
    <>
      <PageHero
        eyebrow="Deep Dive"
        title="Player & Team Analytics"
        sub="Season-long leaderboards for goals, assists, discipline and chance creation."
      />
      <div className="page-body container">
        {status === "loading" && <Loader label="Crunching the numbers…" />}
        {status === "error" && <ErrorBlock message={errMsg} onRetry={load} />}

        {status === "ready" && (
          <>
            <div className="tab-row">
              {TABS.map((tab) => (
                <button
                  key={tab.key}
                  className={`tab-btn${activeTab === tab.key ? " active" : ""}`}
                  onClick={() => setActiveTab(tab.key)}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="section-head">
              <span className="section-title">{active.label}</span>
              <span className="section-count">{active.group}</span>
              <div className="section-line" />
            </div>

            <Leaderboard rows={data?.[active.key]} metricHints={active.metricHints} />
          </>
        )}
      </div>
    </>
  );
}
