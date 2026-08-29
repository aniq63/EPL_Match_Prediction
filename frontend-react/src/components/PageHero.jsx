export default function PageHero({ eyebrow, title, sub }) {
  return (
    <div className="page-hero">
      <div className="container">
        {eyebrow && <div className="page-eyebrow">{eyebrow}</div>}
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
    </div>
  );
}
