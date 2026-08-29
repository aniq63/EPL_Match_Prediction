import { useEffect, useState } from "react";
import PageHero from "../components/PageHero.jsx";
import { Loader, ErrorBlock, EmptyBlock } from "../components/States.jsx";
import { PipRow } from "../components/Pips.jsx";
import TeamBadge from "../components/TeamBadge.jsx";
import { api } from "../api.js";

function formatDate(iso) {
  if (!iso) return "TBD";
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" });
}

function formToPips(formStr) {
  if (!formStr) return [];
  return formStr.split(",").map((r) => r.trim().toUpperCase());
}

function StandingsTable({ rows }) {
  if (!rows || rows.length === 0) return <EmptyBlock message="Standings aren't available right now." />;

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Club</th>
            <th>P</th>
            <th>W</th>
            <th>D</th>
            <th>L</th>
            <th>GD</th>
            <th>Pts</th>
            <th>Form</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.team?.id || row.position}>
              <td><span className="rank-num">{row.position}</span></td>
              <td>
                <div className="team-cell">
                  <TeamBadge name={row.team?.name} crestUrl={row.team?.crest} size="sm" />
                  {row.team?.name}
                </div>
              </td>
              <td>{row.playedGames}</td>
              <td>{row.won}</td>
              <td>{row.draw}</td>
              <td>{row.lost}</td>
              <td>{row.goalDifference}</td>
              <td style={{ fontWeight: 700 }}>{row.points}</td>
              <td><PipRow results={formToPips(row.form)} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MatchList({ matches, showScore }) {
  if (!matches || matches.length === 0) return <EmptyBlock message="No matches to show." />;

  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr" }}>
      {matches.map((m) => (
        <div className="card card-white match-card" key={m.id}>
          <div className="match-side home">
            <TeamBadge name={m.homeTeam?.name} crestUrl={m.homeTeam?.crest} size="md" />
            <span className="match-team-name">{m.homeTeam?.name}</span>
          </div>
          <div className="match-mid">
            {showScore ? (
              <span className="match-score">
                {m.score?.fullTime?.home ?? "–"} : {m.score?.fullTime?.away ?? "–"}
              </span>
            ) : (
              <span className="vs-pill">VS</span>
            )}
            <span className="match-date">{formatDate(m.utcDate)}</span>
          </div>
          <div className="match-side away">
            <TeamBadge name={m.awayTeam?.name} crestUrl={m.awayTeam?.crest} size="md" />
            <span className="match-team-name">{m.awayTeam?.name}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function NewsGrid({ articles }) {
  if (!articles || articles.length === 0) return <EmptyBlock message="No news articles found." />;

  return (
    <div className="grid grid-3">
      {articles.map((a, i) => (
        <a className="card news-card" href={a.url} target="_blank" rel="noreferrer" key={i}>
          <div className="news-source">{a.source?.name || "News"}</div>
          <div className="news-title">{a.title}</div>
          <div className="news-date">{formatDate(a.publishedAt)}</div>
        </a>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | ready | error
  const [errMsg, setErrMsg] = useState("");

  function load() {
    setStatus("loading");
    api
      .getDashboard()
      .then((res) => {
        setData(res);
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
        eyebrow="Gameweek Overview"
        title="Premier League Dashboard"
        sub="Live standings, upcoming fixtures, recent results and the latest headlines — all in one place."
      />
      <div className="page-body container">
        {status === "loading" && <Loader label="Pulling the latest matchday data…" />}
        {status === "error" && <ErrorBlock message={errMsg} onRetry={load} />}

        {status === "ready" && (
          <>
            <section style={{ marginBottom: 44 }}>
              <div className="section-head">
                <span className="section-title">Standings</span>
                <div className="section-line" />
              </div>
              <StandingsTable rows={data.standings} />
            </section>

            <div className="grid grid-2" style={{ marginBottom: 44 }}>
              <section>
                <div className="section-head">
                  <span className="section-title">Upcoming Fixtures</span>
                  {data.metadata?.next_mw && (
                    <span className="section-count">GW {data.metadata.next_mw}</span>
                  )}
                  <div className="section-line" />
                </div>
                <MatchList matches={data.fixtures} showScore={false} />
              </section>

              <section>
                <div className="section-head">
                  <span className="section-title">Recent Results</span>
                  {data.metadata?.prev_mw && (
                    <span className="section-count">GW {data.metadata.prev_mw}</span>
                  )}
                  <div className="section-line" />
                </div>
                <MatchList matches={data.results} showScore={true} />
              </section>
            </div>

            <section>
              <div className="section-head">
                <span className="section-title">Latest News</span>
                <div className="section-line" />
              </div>
              <NewsGrid articles={data.news} />
            </section>
          </>
        )}
      </div>
    </>
  );
}
