/**
 * The recurring visual motif across the app — small colored squares
 * mirroring FPL's own fixture-ticker / form-guide pips.
 * W = green, D = grey, L = pink, pending = dashed outline.
 */
export function Pip({ result }) {
  const map = {
    W: { cls: "pip-w", label: "W" },
    D: { cls: "pip-d", label: "D" },
    L: { cls: "pip-l", label: "L" },
  };
  const entry = map[result];
  if (!entry) {
    return <span className="pip pip-pending">–</span>;
  }
  return <span className={`pip ${entry.cls}`}>{entry.label}</span>;
}

export function PipRow({ results = [] }) {
  return (
    <div className="pip-row">
      {results.map((r, i) => (
        <Pip key={i} result={r} />
      ))}
    </div>
  );
}
