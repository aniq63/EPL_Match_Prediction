import { useState } from "react";
import { NavLink } from "react-router-dom";
import { api } from "../api.js";

const LINKS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/analytics", label: "Analytics" },
  { to: "/predictions", label: "ML Predictions" },
  { to: "/history", label: "PL History" },
];

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="navbar">
      <div className="container navbar-inner">
        <div className="brand">
          <span className="brand-mark">PL</span>
          Nexus
        </div>
        
        <button
          className="mobile-menu-btn"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
        >
          <span className={`hamburger-icon ${mobileOpen ? "open" : ""}`} />
        </button>

        <nav className={`nav-links ${mobileOpen ? "mobile-open" : ""}`}>
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              onClick={() => setMobileOpen(false)}
              onMouseEnter={() => {
                if (link.to === "/analytics") api.getAnalytics().catch(() => {});
                if (link.to === "/predictions") api.getPredictions().catch(() => {});
              }}
              onFocus={() => {
                if (link.to === "/analytics") api.getAnalytics().catch(() => {});
                if (link.to === "/predictions") api.getPredictions().catch(() => {});
              }}
              className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

