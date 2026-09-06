import PageHero from "../components/PageHero.jsx";
import TeamBadge from "../components/TeamBadge.jsx";
import { CHAMPIONS, getTitleLeaders } from "../data/champions.js";

const HISTORY_VIDEOS = [
  { title: "Premier League Greatest Moments", channel: "Premier League Official", label: "Greatest Moments", image: "https://i.ytimg.com/vi/F-3hBVzMqj4/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=F-3hBVzMqj4" },
  { title: "Top 10 Premier League Title Races", channel: "Sky Sports Football", label: "Title Races", image: "https://i.ytimg.com/vi/2Fg5PCZV4qE/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=2Fg5PCZV4qE" },
  { title: "Leicester City 2015/16: The Greatest Story", channel: "LCFC Official", label: "Miracle Season", image: "https://i.ytimg.com/vi/mJqOjLKJqW4/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=mJqOjLKJqW4" },
  { title: "Sir Alex Ferguson: 13 Premier League Titles", channel: "Man United Official", label: "SAF Legacy", image: "https://i.ytimg.com/vi/j4nJbS25SsM/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=j4nJbS25SsM" },
  { title: "Pep Guardiola: City's Dominant Era", channel: "Man City Official", label: "City Dynasty", image: "https://i.ytimg.com/vi/1d8FWsHQmHg/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=1d8FWsHQmHg" },
  { title: "The Invincibles 2003/04", channel: "Arsenal Official", label: "The Invincibles", image: "https://i.ytimg.com/vi/OPVqNbPr1IE/maxresdefault.jpg", url: "https://www.youtube.com/watch?v=OPVqNbPr1IE" },
];

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
            <span className="section-count">1992/93 - 2024/25</span>
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

        <section>
          <div className="section-head">
            <span className="section-title">Watch PL History</span>
            <span className="section-count">YouTube</span>
            <div className="section-line" />
          </div>
          <div className="grid grid-3 history-video-grid">
            {HISTORY_VIDEOS.map((video) => (
              <a className="card history-video-card" href={video.url} target="_blank" rel="noreferrer" key={video.url}>
                <img src={video.image} alt="" loading="lazy" onError={(event) => { event.currentTarget.style.display = "none"; }} />
                <div className="news-source">{video.label}</div>
                <div className="news-title">{video.title}</div>
                <div className="news-date">{video.channel}</div>
              </a>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
