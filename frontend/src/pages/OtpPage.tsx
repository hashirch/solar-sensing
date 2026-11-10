import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  LockKeyhole,
  ArrowRight,
  ArrowLeft,
  Shield,
  Sun,
  RefreshCw,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function OtpPage() {
  const { email, resendCode, verifyOtp } = useAuth();
  const [otp, setOtp] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!email) navigate("/login");
  }, [email, navigate]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (otp.length !== 6) {
      setError("Please enter the 6-digit verification code.");
      return;
    }
    setError("");
    setLoading(true);
    const result = await verifyOtp(otp);
    setLoading(false);
    if (result.ok) navigate("/");
    else setError(result.error || "Verification failed.");
  }

  async function handleResend() {
    setError("");
    setOtp("");
    const result = await resendCode();
    if (result.ok) {
      setNotice("A new code has been sent.");
      setTimeout(() => setNotice(""), 4500);
    } else {
      setError(result.error || "Could not send a new code.");
    }
  }

  return (
    <div className="auth-page page-fade">
      <div className="auth-hero">
        <div className="auth-hero-content">
          <div className="auth-hero-logo">
            <Sun size={24} />
            <span>Solar Sensing</span>
          </div>
          <h1>
            Verify your
            <br />
            <em>identity</em>
          </h1>
          <p>
            We sent a 6-digit code to your email. Enter it to open your
            segmentation workspace.
          </p>
        </div>
        <div className="auth-hero-grid" aria-hidden="true" />
        <div className="auth-circle auth-circle-1" aria-hidden="true" />
        <div className="auth-circle auth-circle-2" aria-hidden="true" />
      </div>

      <div className="auth-form-panel">
        <div className="auth-form-wrapper">
          <div className="auth-form-header">
            <span className="badge">Two-step verification</span>
            <h2>Enter verification code</h2>
            <p>
              Code sent to <strong>{email}</strong>. It expires in 5 minutes.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            <label htmlFor="otp">6-digit verification code</label>
            <div className="input-group">
              <LockKeyhole size={18} className="input-icon" />
              <input
                id="otp"
                type="text"
                inputMode="numeric"
                pattern="[0-9]{6}"
                autoComplete="one-time-code"
                maxLength={6}
                placeholder="000000"
                value={otp}
                onChange={(e) =>
                  setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))
                }
              />
            </div>

            {error && (
              <p className="field-error" role="alert">
                {error}
              </p>
            )}
            {notice && (
              <p className="field-success" role="status">
                {notice}
              </p>
            )}

            <button type="submit" className="btn-primary btn-full" disabled={loading}>
              {loading ? <span className="spinner light" /> : null}
              {loading ? "Verifying…" : "Verify & open workspace"}
              {!loading && <ArrowRight size={16} />}
            </button>
          </form>

          <div className="auth-actions">
            <button className="btn-link" onClick={() => navigate("/login")}>
              <ArrowLeft size={14} /> Back
            </button>
            <button className="btn-link" onClick={handleResend}>
              <RefreshCw size={14} /> Resend code
            </button>
          </div>

          <p className="auth-privacy-note">
            <Shield size={14} />
            Codes are single-use and limited to 5 attempts.
          </p>
        </div>
      </div>
    </div>
  );
}
