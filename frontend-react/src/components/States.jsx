import { API_BASE_URL } from "../api.js";

export function Loader({ label = "Loading data…" }) {
  return (
    <div className="state-block">
      <div className="spinner" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBlock({ message, onRetry }) {
  return (
    <div className="state-block state-block-error">
      <div className="state-title">Couldn't load this</div>
      <span>{message || "The server didn't respond."}</span>
      <code className="state-diagnostic">API target: {API_BASE_URL}</code>
      {onRetry && (
        <button className="retry-btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function EmptyBlock({ message = "Nothing here yet." }) {
  return (
    <div className="state-block">
      <div className="state-title">No data</div>
      <span>{message}</span>
    </div>
  );
}
