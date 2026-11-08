"""Email OTP generation and delivery for the optional SolarSight access gate."""

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
        self.app_name = os.getenv("APP_NAME", "SolarSight")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_username)
        self.expiry_minutes = int(os.getenv("OTP_EXPIRY_MINUTES", "5"))

    @staticmethod
    def generate_otp() -> str:
        """Return a six-character hexadecimal code, matching the source flow."""
        return secrets.token_hex(3)

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

    def send_email(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send an HTML email, returning False only for transport/auth failures."""
        if not self.smtp_username or not self.smtp_password or not self.from_email:
            logger.error("SMTP credentials are not configured; cannot send OTP email")
            return False

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.app_name} <{self.from_email}>"
        message["To"] = to_email
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

    def send_verification_email(self, recipient_name: str, email: str, code: str) -> bool:
        safe_name = html.escape(recipient_name)
        return self.send_email(
            email,
            f"Verify your {self.app_name} workspace",
            f"<h2>Hello {safe_name}</h2><p>Your verification code is:</p>"
            f"<h1>{html.escape(code)}</h1>"
            f"<p>This code expires in {self.expiry_minutes} minutes.</p>",
        )

    def send_resend_email(self, recipient_name: str, email: str, code: str) -> bool:
        safe_name = html.escape(recipient_name)
        return self.send_email(
            email,
            f"New verification code - {self.app_name}",
            f"<h2>Hello {safe_name}</h2><p>Your new verification code is:</p>"
            f"<h1>{html.escape(code)}</h1>"
            f"<p>This code expires in {self.expiry_minutes} minutes.</p>",
        )

    @staticmethod
    def normalize_code(code: str) -> str:
        return code.strip().lower()

    def verify(self, challenge: OTPChallenge, submitted_code: str) -> bool:
        """Verify a code without exposing the expected OTP in logs or errors."""
        normalized = self.normalize_code(submitted_code)
        return (
            not self.is_otp_expired(challenge.expires_at)
            and len(normalized) == 6
            and secrets.compare_digest(challenge.code, normalized)
        )


otp_service = OTPService()
