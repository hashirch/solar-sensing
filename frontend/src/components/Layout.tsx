import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  Sun,
  Upload,
  LogOut,
  Menu,
  X,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <div className="app-layout">
      {/* ─── Top navigation bar ─── */}
      <header className="navbar">
        <div className="navbar-inner">
          <div className="navbar-brand">
            <Sun size={22} />
            <span>Solar Sensing</span>
          </div>

          <button
            className="navbar-mobile-toggle"
            onClick={() => setMobileOpen((o) => !o)}
            aria-label="Toggle navigation"
          >
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>

          <nav className={`navbar-links ${mobileOpen ? "open" : ""}`}>
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `nav-link ${isActive ? "active" : ""}`
              }
              onClick={() => setMobileOpen(false)}
            >
              <Upload size={16} />
              Workspace
            </NavLink>
          </nav>

          <div className="navbar-actions">
            <span className="status-indicator">
              <span className="status-dot green" />
              Inference engine online
            </span>
            <div className="user-menu">
              <NavLink to="/profile" className="avatar-btn" title="Your profile" aria-label="Your profile">
                {(user?.displayName || user?.email || "U")[0].toUpperCase()}
              </NavLink>
              <span className="user-name">{user?.displayName}</span>
              <button
                className="btn-icon"
                onClick={handleLogout}
                title="Sign out"
              >
                <LogOut size={16} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ─── Page content ─── */}
      <main className="page-content page-fade" key={location.pathname}>
        <Outlet />
      </main>

      {/* ─── Footer ─── */}
      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-brand">
            <Sun size={14} />
            <span>Solar Sensing</span>
          </div>
          <span>Solar panel segmentation research</span>
          <span>Local inference · Privacy first</span>
        </div>
      </footer>
    </div>
  );
}
