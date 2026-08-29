import PageHero from "../components/PageHero.jsx";
import TeamBadge from "../components/TeamBadge.jsx";
import { CHAMPIONS, getTitleLeaders } from "../data/champions.js";

export default function History() {
  const leaders = getTitleLeaders();
  const recent = CHAMPIONS.slice(0, 12);

  return (
    <>
      <PageHero
        eyebrow="Since 1992"
        title="Premier League Champions"
        sub="33 seasons — every title, every dynasty, every miracle run to the line."
      />
      <div className="page-body container">
        <section style={{ marginBottom: 44 }}>
          <div className="section-head">
            <span className="section-title">All-Time Title Leaders</span>
            <div className="section-line" />
          </div>
          <div className="grid grid-3">
            {leaders.map(([team, count]) => (
              <div className="card stat-card" key={team}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                  <TeamBadge name={team} size="md" />
                  <span className="stat-value">{count}</span>
                </div>
                <span className="stat-label">{team}</span>
              </div>
            ))}
          </div>
        </section>

        <section style={{ marginBottom: 44 }}>
          <div className="section-head">
            <span className="section-title">Recent Champions</span>
            <span className="section-count">Last 12 seasons</span>
            <div className="section-line" />
          </div>
          <div className="grid grid-3">
            {recent.map((c) => (
              <div className="card card-white" key={c.season}>
                <div className="stat-label" style={{ marginBottom: 6 }}>{c.season}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                  <TeamBadge name={c.team} size="md" />
                  <div className="champion-club">{c.team}</div>
                </div>
                <div className="champion-manager">{c.manager} · {c.pts} pts</div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="section-head">
            <span className="section-title">Full Timeline</span>
            <span className="section-count">1992/93 – 2024/25</span>
            <div className="section-line" />
          </div>
          <div className="table-wrap">
            {CHAMPIONS.map((c, i) => (
              <div
                className="champion-row"
                key={c.season}
                style={{ background: i % 2 === 0 ? "#fff" : "var(--surface)" }}
              >
                <span className="champion-season">{c.season}</span>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <TeamBadge name={c.team} size="sm" />
                  <div>
                    <div className="champion-club">{c.team}</div>
                    <div className="champion-manager">{c.manager}</div>
                  </div>
                </div>
                <span className="champion-points">{c.pts} pts</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
