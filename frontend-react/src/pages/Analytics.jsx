import { useEffect, useState } from "react";
import PageHero from "../components/PageHero.jsx";
import { Loader, ErrorBlock, EmptyBlock } from "../components/States.jsx";
import TeamBadge from "../components/TeamBadge.jsx";
import { api } from "../api.js";

const PLAYER_METRICS = [
  { key: "top_players_goals", label: "Top Scorers", metricHints: ["goals"] },
  { key: "top_players_assists", label: "Top Assists", metricHints: ["assists"] },
  { key: "top_players_shots", label: "Most Shots", metricHints: ["shots"] },
  { key: "top_players_key_passes", label: "Key Passes", metricHints: ["key_passes", "keypasses"] },
  { key: "top_players_yellow_cards", label: "Yellow Cards", metricHints: ["yellow_cards", "yellowcards"] },
  { key: "top_players_red_cards", label: "Red Cards", metricHints: ["red_cards", "redcards"] },
];

const TEAM_METRICS = [
  { key: "top_teams_goals", label: "Team Goals", metricHints: ["goals"] },
  { key: "top_teams_shots", label: "Team Shots", metricHints: ["shots"] },
  { key: "top_teams_created_chances", label: "Chances Created", metricHints: ["created_chances", "chances_created", "chances"] },
  { key: "top_teams_yellow_cards", label: "Team Yellow Cards", metricHints: ["yellow_cards", "yellowcards"] },
  { key: "top_teams_red_cards", label: "Team Red Cards", metricHints: ["red_cards", "redcards"] },
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

function LeaderboardCard({ title, rows, metricHints }) {
  if (!rows || rows.length === 0) return (
    <div className="card card-white" style={{ height: "100%" }}>
      <div className="section-head" style={{ marginBottom: 12 }}>
        <span className="section-title" style={{ fontSize: "1.05rem" }}>{title}</span>
      </div>
      <EmptyBlock message="No records available." />
    </div>
  );

  const { nameKey, teamKey, metricKey } = inferColumns(rows[0], metricHints);
  const badgeSource = teamKey || nameKey;

  return (
    <div className="card card-white" style={{ display: "flex", flexDirection: "column" }}>
      <div className="section-head" style={{ marginBottom: 12 }}>
        <span className="section-title" style={{ fontSize: "1.05rem" }}>{title}</span>
        <span className="section-count">{rows.length} records</span>
      </div>
      <div className="leaderboard-list">
        {rows.slice(0, 10).map((row, i) => (
          <div className="leaderboard-row" key={i}>
            <span className="leaderboard-rank">{i + 1}</span>
            <TeamBadge name={badgeSource ? row[badgeSource] : ""} size="sm" />
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
  const [filterGroup, setFilterGroup] = useState("all"); // 'all' | 'players' | 'teams'

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

  return (
    <>
      <PageHero
        eyebrow="Deep Dive"
        title="Player & Team Analytics"
        sub="All season-long leaderboards for goals, assists, discipline, and chance creation on a single page."
      />
      <div className="page-body container">
        {status === "loading" && <Loader label="Crunching the numbers…" />}
        {status === "error" && <ErrorBlock message={errMsg} onRetry={load} />}

        {status === "ready" && (
          <>
            {/* Quick Filter Pill Buttons */}
            <div className="tab-row" style={{ justifyContent: "center", marginBottom: 32 }}>
              <button
                className={`tab-btn${filterGroup === "all" ? " active" : ""}`}
                onClick={() => setFilterGroup("all")}
              >
                All Stats ({PLAYER_METRICS.length + TEAM_METRICS.length})
              </button>
              <button
                className={`tab-btn${filterGroup === "players" ? " active" : ""}`}
                onClick={() => setFilterGroup("players")}
              >
                Player Stats ({PLAYER_METRICS.length})
              </button>
              <button
                className={`tab-btn${filterGroup === "teams" ? " active" : ""}`}
                onClick={() => setFilterGroup("teams")}
              >
                Team Stats ({TEAM_METRICS.length})
              </button>
            </div>

            {/* PLAYER STATS SECTION */}
            {(filterGroup === "all" || filterGroup === "players") && (
              <section style={{ marginBottom: 48 }}>
                <div className="section-head">
                  <span className="section-title">Player Statistics</span>
                  <span className="section-count">Individual Leaders</span>
                  <div className="section-line" />
                </div>
                <div className="grid grid-3">
                  {PLAYER_METRICS.map((metric) => (
                    <LeaderboardCard
                      key={metric.key}
                      title={metric.label}
                      rows={data?.[metric.key]}
                      metricHints={metric.metricHints}
                    />
                  ))}
                </div>
              </section>
            )}

            {/* TEAM STATS SECTION */}
            {(filterGroup === "all" || filterGroup === "teams") && (
              <section style={{ marginBottom: 32 }}>
                <div className="section-head">
                  <span className="section-title">Team Statistics</span>
                  <span className="section-count">Club Leaders</span>
                  <div className="section-line" />
                </div>
                <div className="grid grid-3">
                  {TEAM_METRICS.map((metric) => (
                    <LeaderboardCard
                      key={metric.key}
                      title={metric.label}
                      rows={data?.[metric.key]}
                      metricHints={metric.metricHints}
                    />
                  ))}
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </>
  );
}

