"""Email OTP generation and delivery for the optional Solar Sensing access gate."""

from __future__ import annotations

import html
import logging
import os
import secrets
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OTPChallenge:
    """An OTP and the time at which it stops being valid."""

    code: str
    expires_at: datetime


class OTPService:
    """Generate short-lived email OTPs and send verification messages."""

    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Solar Sensing")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.expiry_minutes = int(os.getenv("OTP_EXPIRY_MINUTES", "5"))

    @staticmethod
    def generate_otp() -> str:
        """Return a cryptographically secure six-digit numeric code."""
        return f"{secrets.randbelow(1_000_000):06d}"

    def create_challenge(self) -> OTPChallenge:
        return OTPChallenge(
            code=self.generate_otp(),
            expires_at=datetime.now(timezone.utc)
            + timedelta(minutes=self.expiry_minutes),
        )

    @staticmethod
    def is_otp_expired(expires_at: datetime) -> bool:
        """Check expiry using timezone-aware UTC timestamps."""
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) >= expires_at

    def send_email(
        self, to_email: str, subject: str, html_content: str, text_content: str = ""
    ) -> bool:
        """Send an HTML email, returning False only for transport/auth failures."""
        if not self.smtp_username or not self.smtp_password or not self.from_email:
            logger.error("SMTP credentials are not configured; cannot send OTP email")
            return False

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.app_name} <{self.from_email}>"
        message["To"] = to_email
        if text_content:
            message.attach(MIMEText(text_content, "plain"))
        message.attach(MIMEText(html_content, "html"))

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=20) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.from_email, [to_email], message.as_string())
        except (OSError, smtplib.SMTPException):
            logger.exception("Unable to send OTP email to %s", to_email)
            return False
        return True

    _COPY = {
        "verify": (
            "Verify your email",
            "Use the code below to finish setting up your Solar Sensing workspace.",
            "Verify your {app} workspace",
        ),
        "password": (
            "Confirm your password change",
            "Use the code below to confirm the change to your Solar Sensing password.",
            "Confirm your password change - {app}",
        ),
        "resend": (
            "Your new verification code",
            "Here is your new code. Any earlier code no longer works.",
            "New verification code - {app}",
        ),
    }

    def render_code_email(self, recipient_name: str, code: str, purpose: str = "verify"):
        """Return (subject, html, plain_text) styled like the Solar Sensing web app."""
        heading, intro, subject = self._COPY.get(purpose, self._COPY["verify"])
        name = html.escape(recipient_name or "there")
        app = html.escape(self.app_name)
        digits = html.escape(code)
        minutes = self.expiry_minutes
        font = "Inter,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
        mono = "'JetBrains Mono','SFMono-Regular',Consolas,'Courier New',monospace"
        body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(heading)}</title></head>
<body style="margin:0;padding:0;background:#F6F7F9;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#F6F7F9;">
<tr><td align="center" style="padding:32px 16px;">
  <table role="presentation" width="560" cellpadding="0" cellspacing="0" style="width:100%;max-width:560px;">
    <tr><td style="padding:0 4px 16px;font-family:{font};">
      <table role="presentation" cellpadding="0" cellspacing="0"><tr>
        <td align="center" valign="middle" width="32" height="32" style="width:32px;height:32px;background:#0F5132;border-radius:8px;color:#ffffff;font-size:18px;line-height:32px;font-family:{font};">&#9728;</td>
        <td style="padding-left:10px;font-size:17px;font-weight:700;color:#0F172A;letter-spacing:-0.02em;font-family:{font};">{app}</td>
      </tr></table>
    </td></tr>
    <tr><td style="background:#ffffff;border:1px solid #E3E7ED;border-radius:12px;padding:36px 32px;font-family:{font};">
      <span style="display:inline-block;padding:4px 12px;background:#EEF6F1;color:#1F7A4D;border-radius:999px;font-size:11px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;font-family:{font};">Two-step verification</span>
      <h1 style="margin:18px 0 10px;font-size:26px;line-height:1.2;font-weight:700;letter-spacing:-0.03em;color:#0F172A;font-family:{font};">{html.escape(heading)}</h1>
      <p style="margin:0 0 4px;font-size:15px;line-height:1.6;color:#475569;font-family:{font};">Hi {name},</p>
      <p style="margin:0 0 24px;font-size:15px;line-height:1.6;color:#475569;font-family:{font};">{html.escape(intro)}</p>
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
        <td align="center" style="background:#EEF6F1;border:1px solid #D4EDDA;border-radius:12px;padding:22px 12px;">
          <div style="font-family:{mono};font-size:38px;line-height:1;font-weight:700;letter-spacing:0.28em;color:#0F5132;padding-left:0.28em;">{digits}</div>
        </td>
      </tr></table>
      <p style="margin:20px 0 0;font-size:14px;line-height:1.6;color:#475569;font-family:{font};">This code expires in <strong style="color:#0F172A;">{minutes} minutes</strong> and can only be used once.</p>
      <hr style="border:none;border-top:1px solid #E3E7ED;margin:28px 0 18px;">
      <p style="margin:0;font-size:13px;line-height:1.6;color:#94A3B8;font-family:{font};">If you didn't request this, you can safely ignore this email. Nobody can access your account without this code.</p>
    </td></tr>
    <tr><td align="center" style="padding:20px 8px 0;font-size:12px;line-height:1.6;color:#94A3B8;font-family:{font};">{app} &middot; Satellite solar panel segmentation</td></tr>
  </table>
</td></tr></table>
</body></html>"""
        text = (
            f"{heading}\n\nHi {recipient_name or 'there'},\n{intro}\n\n"
            f"    {code}\n\nThis code expires in {minutes} minutes and can only be used once.\n"
            "If you didn't request this, you can safely ignore this email.\n"
        )
        return subject.format(app=self.app_name), body, text

    def send_verification_email(
        self, recipient_name: str, email: str, code: str, purpose: str = "verify"
    ) -> bool:
        subject, body, text = self.render_code_email(recipient_name, code, purpose)
        return self.send_email(email, subject, body, text)

    def send_resend_email(self, recipient_name: str, email: str, code: str) -> bool:
        return self.send_verification_email(recipient_name, email, code, "resend")

    @staticmethod
    def normalize_code(code: str) -> str:
        return code.strip().lower()

    def verify(self, challenge: OTPChallenge, submitted_code: str) -> bool:
        """Verify a code without exposing the expected OTP in logs or errors."""
        normalized = self.normalize_code(submitted_code)
        return (
            not self.is_otp_expired(challenge.expires_at)
            and len(normalized) == 6
            and normalized.isdecimal()
            and secrets.compare_digest(challenge.code, normalized)
        )


otp_service = OTPService()
