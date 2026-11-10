import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Mail, ArrowRight, Shield, Sun, User, KeyRound, Eye, EyeOff } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"signin" | "signup">("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function switchMode(next: "signin" | "signup") {
    setMode(next);
    setError("");
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const cleanEmail = email.trim().toLowerCase();
    if (mode === "signup" && !name.trim()) return setError("Please enter your name.");
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(cleanEmail))
      return setError("Please enter a valid email address.");
    if (mode === "signup" && password.length < 8)
      return setError("Password must be at least 8 characters.");
    if (!password) return setError("Please enter your password.");
    setError("");
    setLoading(true);
    const result =
      mode === "signup"
        ? await register(name, cleanEmail, password)
        : await login(cleanEmail, password);
    setLoading(false);
    if (!result.ok) return setError(result.error || "Something went wrong.");
    // Signing in goes straight to the workspace; sign-up still verifies the email.
    if (mode === "signup") navigate("/verify");
  }

  return (
    <div className="auth-page page-fade">
      {/* Left hero panel */}
      <div className="auth-hero">
        <div className="auth-hero-content">
          <div className="auth-hero-logo">
            <Sun size={24} />
            <span>Solar Sensing</span>
          </div>
          <h1>
            Intelligent Solar Panel
            <br />
            <em>Segmentation</em>
          </h1>
          <p>
            Analyze satellite imagery with state-of-the-art CNN/U-Net and
            Mask&nbsp;R-CNN models. Privacy-first, local inference.
          </p>
        </div>
        <div className="auth-hero-grid" aria-hidden="true" />
        <div className="auth-circle auth-circle-1" aria-hidden="true" />
        <div className="auth-circle auth-circle-2" aria-hidden="true" />
      </div>

      {/* Right form panel */}
      <div className="auth-form-panel">
        <div className="auth-form-wrapper">
          <div className="auth-form-header">
            <span className="badge">Secure access</span>
            <h2>{mode === "signin" ? "Sign in to your workspace" : "Create your account"}</h2>
            <p>
              {mode === "signin"
                ? "Enter your email and password to open your workspace."
                : "Create a profile. We will email you a code to verify your address."}
            </p>
          </div>

          <div className="auth-tabs" role="tablist">
            <button type="button" role="tab" aria-selected={mode === "signin"}
              className={mode === "signin" ? "active" : ""} onClick={() => switchMode("signin")}>
              Sign in
            </button>
            <button type="button" role="tab" aria-selected={mode === "signup"}
              className={mode === "signup" ? "active" : ""} onClick={() => switchMode("signup")}>
              Create account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="auth-form" noValidate>
            {mode === "signup" && (
              <>
                <label htmlFor="name">Full name</label>
                <div className="input-group">
                  <User size={18} className="input-icon" />
                  <input id="name" type="text" autoComplete="name" placeholder="Your name"
                    maxLength={60} value={name} onChange={(e) => setName(e.target.value)} />
                </div>
              </>
            )}

            <label htmlFor="email">Email address</label>
            <div className="input-group">
              <Mail size={18} className="input-icon" />
              <input id="email" type="email" autoComplete="email" placeholder="you@example.com"
                value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>

            <label htmlFor="password">Password</label>
            <div className="input-group">
              <KeyRound size={18} className="input-icon" />
              <input id="password" type={showPw ? "text" : "password"}
                autoComplete={mode === "signin" ? "current-password" : "new-password"}
                placeholder={mode === "signup" ? "At least 8 characters" : "Your password"}
                maxLength={128} value={password} onChange={(e) => setPassword(e.target.value)} />
              <button type="button" className="btn-icon" onClick={() => setShowPw((v) => !v)}
                aria-label={showPw ? "Hide password" : "Show password"}>
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>

            {error && (
              <p className="field-error" role="alert">
                {error}
              </p>
            )}

            <button type="submit" className="btn-primary btn-full" disabled={loading}>
              {loading ? <span className="spinner light" /> : null}
              {loading ? "Please wait…" : mode === "signin" ? "Continue" : "Create account"}
              {!loading && <ArrowRight size={16} />}
            </button>
          </form>

          <p className="auth-privacy-note">
            <Shield size={14} />
            Passwords are stored as salted hashes. Imagery is processed locally.
          </p>
        </div>
      </div>
    </div>
  );
}
