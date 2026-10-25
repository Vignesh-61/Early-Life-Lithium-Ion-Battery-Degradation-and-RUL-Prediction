// Navbar.jsx — Sidebar navigation
import { NavLink } from "react-router-dom";

const NAV = [
  { to: "/",          icon: "⬡",  label: "Dashboard"   },
  { to: "/predict",   icon: "⚡",  label: "Predict"     },
  { to: "/research",  icon: "🔬", label: "Research"    },
];

export default function Navbar({ modelLoaded }) {
  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🔋</div>
        <h1>Battery<br/>Intelligence</h1>
        <p>Adaptive Memory DNN</p>
      </div>

      {NAV.map(({ to, icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
        >
          <span className="icon">{icon}</span>
          {label}
        </NavLink>
      ))}

      <div className="sidebar-footer">
        <span className={`status-dot ${modelLoaded ? "online" : "offline"}`} />
        <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
          {modelLoaded ? "Model online" : "Model offline"}
        </span>
      </div>
    </nav>
  );
}
