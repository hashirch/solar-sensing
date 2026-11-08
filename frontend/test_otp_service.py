import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from otp_service import OTPChallenge, OTPService


class OTPServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = OTPService()

    def test_challenge_has_six_character_code_and_expiry(self):
        challenge = self.service.create_challenge()
        self.assertEqual(len(challenge.code), 6)
        self.assertFalse(self.service.is_otp_expired(challenge.expires_at))

    def test_verify_accepts_normalized_code(self):
        challenge = OTPChallenge(
            code="a1b2c3",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        self.assertTrue(self.service.verify(challenge, " A1B2C3 "))

    def test_verify_rejects_expired_or_invalid_code(self):
        expired = OTPChallenge(
            code="a1b2c3",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
        self.assertFalse(self.service.verify(expired, "a1b2c3"))
        self.assertFalse(
            self.service.verify(
                OTPChallenge(
                    code="a1b2c3",
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                ),
                "wrong!",
            )
        )

    @patch("otp_service.smtplib.SMTP")
    def test_send_email_returns_false_without_credentials(self, smtp):
        with patch.dict(
            "os.environ",
            {"SMTP_USERNAME": "", "SMTP_PASSWORD": "", "FROM_EMAIL": ""},
            clear=False,
        ):
            service = OTPService()
            self.assertFalse(service.send_email("user@example.com", "OTP", "<p>code</p>"))
        smtp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
