import { useState } from "react";
import { getCrest, getFallbackColor, getInitials } from "../data/teamCrests.js";

const SIZES = { sm: 20, md: 28, lg: 40 };

export default function TeamBadge({ name, crestUrl, size = "md" }) {
  const [failed, setFailed] = useState(false);
  const px = SIZES[size] || SIZES.md;
  const resolvedUrl = crestUrl || getCrest(name);

  if (!resolvedUrl || failed) {
    return (
      <span
        className="team-badge-fallback"
        style={{
          width: px,
          height: px,
          fontSize: px * 0.36,
          background: getFallbackColor(name),
        }}
        title={name}
      >
        {getInitials(name)}
      </span>
    );
  }

  return (
    <img
      className="team-badge-img"
      style={{ width: px, height: px }}
      src={resolvedUrl}
      alt={name}
      title={name}
      onError={() => setFailed(true)}
      loading="lazy"
      referrerPolicy="no-referrer"
    />
  );
}
