import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, KeyRound, LogOut, Trash2, UserRound } from "lucide-react";
import { useAuth } from "../context/AuthContext";

interface AnalysisRow {
  id: number;
  filename: string;
  model_source: string;
  image_width: number;
  image_height: number;
  panel_count: number;
  total_area_m2: number;
  capacity_kw: number;
  avg_confidence: number;
  created_at: number;
}

const fmtDate = (ts: number | null) =>
  ts ? new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

export default function ProfilePage() {
  const { user, updateName, requestPasswordChange, confirmPasswordChange, authFetch, logout } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState(user?.displayName || "");
  const [nameMsg, setNameMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [pwMsg, setPwMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [pwStep, setPwStep] = useState<"form" | "code">("form");
  const [pwCode, setPwCode] = useState("");
  const [pwBusy, setPwBusy] = useState(false);

  const [rows, setRows] = useState<AnalysisRow[] | null>(null);

  useEffect(() => {
    authFetch("/api/analyses").then((r) => setRows(r.ok ? r.data.analyses : []));
  }, [authFetch]);

  if (!user) return null;

  async function saveName(e: React.FormEvent) {
    e.preventDefault();
    const r = await updateName(name);
    setNameMsg({ ok: r.ok, text: r.ok ? "Profile updated." : r.error || "Could not update." });
  }

  async function sendPasswordCode(e?: React.FormEvent) {
    e?.preventDefault();
    if (next.length < 8) return setPwMsg({ ok: false, text: "New password must be at least 8 characters." });
    setPwBusy(true);
    const r = await requestPasswordChange(current, next);
    setPwBusy(false);
    if (r.ok) {
      setPwStep("code");
      setPwCode("");
      setPwMsg({ ok: true, text: `We emailed a 6-digit code to ${user!.email}.` });
    } else {
      setPwMsg({ ok: false, text: r.error || "Could not start the password change." });
    }
  }

  async function confirmPassword(e: React.FormEvent) {
    e.preventDefault();
    if (pwCode.length !== 6) return setPwMsg({ ok: false, text: "Enter the 6-digit code." });
    setPwBusy(true);
    const r = await confirmPasswordChange(pwCode);
    setPwBusy(false);
    if (r.ok) {
      setCurrent("");
      setNext("");
      setPwCode("");
      setPwStep("form");
      setPwMsg({ ok: true, text: "Password changed. Other devices were signed out." });
    } else {
      setPwMsg({ ok: false, text: r.error || "Verification failed." });
    }
  }

  function cancelPasswordChange() {
    setPwStep("form");
    setPwCode("");
    setPwMsg(null);
  }

  async function removeRow(id: number) {
    const r = await authFetch(`/api/analyses/${id}`, { method: "DELETE" });
    if (r.ok) setRows((cur) => (cur || []).filter((x) => x.id !== id));
  }

  const totals = (rows || []).reduce(
    (a, r) => ({ panels: a.panels + r.panel_count, area: a.area + r.total_area_m2 }),
    { panels: 0, area: 0 }
  );

  return (
    <div className="profile-page">
      <div className="page-header profile-header">
        <div className="profile-identity">
          <div className="profile-avatar">{user.displayName[0]?.toUpperCase()}</div>
          <div>
            <span className="kicker">Your profile</span>
            <h1>{user.displayName}</h1>
            <p>{user.email}</p>
          </div>
        </div>
        <button className="btn-secondary" onClick={() => { logout(); navigate("/login"); }}>
          <LogOut size={16} /> Sign out
        </button>
      </div>

      <div className="profile-stats">
        <div className="cum-stat"><span className="stat-label">Analyses saved</span><strong className="stat-val">{rows?.length ?? "—"}</strong></div>
        <div className="cum-stat"><span className="stat-label">Panels detected</span><strong className="stat-val">{rows ? totals.panels : "—"}</strong></div>
        <div className="cum-stat"><span className="stat-label">Area mapped</span><strong className="stat-val">{rows ? `${Math.round(totals.area * 10) / 10} m²` : "—"}</strong></div>
        <div className="cum-stat"><span className="stat-label">Member since</span><strong className="stat-val sm">{fmtDate(user.createdAt).split(",")[0]}</strong></div>
      </div>

      <div className="profile-grid">
        <section className="profile-card">
          <h2><UserRound size={17} /> Account details</h2>
          <form onSubmit={saveName} className="auth-form">
            <label htmlFor="pname">Full name</label>
            <div className="input-group">
              <input id="pname" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
            </div>
            <label htmlFor="pemail">Email address</label>
            <div className="input-group readonly">
              <input id="pemail" value={user.email} readOnly />
            </div>
            <dl className="profile-meta">
              <div><dt>Role</dt><dd>{user.role}</dd></div>
              <div><dt>Last sign-in</dt><dd>{fmtDate(user.lastLoginAt)}</dd></div>
            </dl>
            {nameMsg && <p className={nameMsg.ok ? "field-success" : "field-error"} role="status">{nameMsg.text}</p>}
            <button className="btn-primary" type="submit" disabled={!name.trim() || name.trim() === user.displayName}>
              <Check size={16} /> Save changes
            </button>
          </form>
        </section>

        <section className="profile-card">
          <h2><KeyRound size={17} /> Password</h2>
          {pwStep === "form" ? (
            <form onSubmit={sendPasswordCode} className="auth-form">
              <label htmlFor="cur">Current password</label>
              <div className="input-group">
                <input id="cur" type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
              </div>
              <label htmlFor="new">New password</label>
              <div className="input-group">
                <input id="new" type="password" autoComplete="new-password" placeholder="At least 8 characters" value={next} onChange={(e) => setNext(e.target.value)} />
              </div>
              <p className="field-hint">We will email a verification code to confirm the change. Passwords are stored only as salted hashes.</p>
              {pwMsg && <p className={pwMsg.ok ? "field-success" : "field-error"} role="status">{pwMsg.text}</p>}
              <button className="btn-primary" type="submit" disabled={!current || !next || pwBusy}>
                {pwBusy ? "Sending code…" : "Send verification code"}
              </button>
            </form>
          ) : (
            <form onSubmit={confirmPassword} className="auth-form">
              <label htmlFor="pwcode">Verification code</label>
              <div className="input-group">
                <input id="pwcode" inputMode="numeric" autoComplete="one-time-code" maxLength={6} placeholder="000000"
                  value={pwCode} onChange={(e) => setPwCode(e.target.value.replace(/\D/g, "").slice(0, 6))} />
              </div>
              {pwMsg && <p className={pwMsg.ok ? "field-success" : "field-error"} role="status">{pwMsg.text}</p>}
              <div className="pw-actions">
                <button className="btn-primary" type="submit" disabled={pwCode.length !== 6 || pwBusy}>
                  {pwBusy ? "Verifying…" : "Confirm password change"}
                </button>
                <button type="button" className="btn-link" onClick={() => sendPasswordCode()}>Resend code</button>
                <button type="button" className="btn-link" onClick={cancelPasswordChange}>Cancel</button>
              </div>
            </form>
          )}
        </section>
      </div>

      <section className="profile-card history-card">
        <h2>Analysis history</h2>
        {rows === null ? (
          <p className="empty-note">Loading…</p>
        ) : rows.length === 0 ? (
          <p className="empty-note">No analyses yet. Results from the workspace are saved here automatically.</p>
        ) : (
          <div className="history-table-wrap">
            <table className="history-table">
              <thead>
                <tr><th>Image</th><th>Date</th><th>Size</th><th>Panels</th><th>Area</th><th>Capacity</th><th>Confidence</th><th /></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <td className="hist-name">{r.filename}</td>
                    <td>{fmtDate(r.created_at)}</td>
                    <td>{r.image_width}×{r.image_height}</td>
                    <td>{r.panel_count}</td>
                    <td>{r.total_area_m2} m²</td>
                    <td>{r.capacity_kw} kWp</td>
                    <td>{r.avg_confidence}%</td>
                    <td>
                      <button className="btn-icon-sm" onClick={() => removeRow(r.id)} aria-label={`Delete ${r.filename}`}>
                        <Trash2 size={15} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
