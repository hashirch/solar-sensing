"""End-to-end API tests: runs the real HTTP handler on a random port (models are not loaded)."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer
from unittest.mock import patch

from tests import _env  # noqa: F401
from backend import db
from backend import server

PASSWORD = "correct horse"


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()
        cls.httpd = HTTPServer(("127.0.0.1", 0), server.InferenceHandler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        # Pretend email is configured and capture the codes instead of sending mail.
        cls.codes = []
        server.otp_service.smtp_username = "test@example.com"
        server.otp_service.smtp_password = "x"

        def fake_send(name, email, code, purpose="verify"):
            cls.codes.append((email, code, purpose))
            return True

        cls.patcher = patch.object(server.otp_service, "send_verification_email", side_effect=fake_send)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        cls.httpd.shutdown()

    def setUp(self):
        for store in (server._otp_challenges, server._pending, server._login_fails):
            store.clear()
        self.codes.clear()
        with db.connect() as conn:
            for table in ("detections", "analyses", "sessions", "users"):
                conn.execute(f"DELETE FROM {table}")

    # ---- helpers ----
    def call(self, method, path, body=None, token=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def last_code(self, email=None):
        return [c for e, c, _ in self.codes if email in (None, e)][-1]

    def signup(self, email="ann@example.com", name="Ann Lee", password=PASSWORD):
        status, body = self.call("POST", "/api/auth/register", {"name": name, "email": email, "password": password})
        self.assertEqual(status, 200, body)
        status, body = self.call("POST", "/api/otp/verify", {"email": email, "code": self.last_code(email)})
        self.assertEqual(status, 200, body)
        return body["token"]

    def signin(self, email="ann@example.com", password=PASSWORD):
        return self.call("POST", "/api/auth/login", {"email": email, "password": password})


class HealthTests(ApiTestCase):
    def test_health_endpoint(self):
        status, body = self.call("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "online")

    def test_unknown_route_404(self):
        status, _ = self.call("GET", "/api/nope")
        self.assertEqual(status, 404)


class SignUpTests(ApiTestCase):
    def reg(self, **kw):
        body = {"name": "Ann", "email": "ann@example.com", "password": PASSWORD}
        body.update(kw)
        return self.call("POST", "/api/auth/register", body)

    def test_rejects_empty_name(self):
        self.assertEqual(self.reg(name="   ")[0], 400)

    def test_rejects_invalid_emails(self):
        for bad in ("", "plain", "a@b", "@x.com"):
            self.assertEqual(self.reg(email=bad)[0], 400, bad)

    def test_rejects_short_and_overlong_passwords(self):
        self.assertEqual(self.reg(password="short")[0], 400)
        self.assertEqual(self.reg(password="x" * 129)[0], 400)

    def test_account_not_created_until_code_verified(self):
        self.assertEqual(self.reg()[0], 200)
        self.assertIsNone(db.get_user_by_email("ann@example.com"))

    def test_full_signup_returns_session_and_profile(self):
        token = self.signup()
        status, body = self.call("GET", "/api/me", token=token)
        self.assertEqual(status, 200)
        self.assertEqual(body["user"]["email"], "ann@example.com")
        self.assertEqual(body["user"]["displayName"], "Ann Lee")

    def test_password_is_stored_hashed(self):
        self.signup()
        stored = db.get_user_by_email("ann@example.com")["password_hash"]
        self.assertNotIn(PASSWORD, stored)
        self.assertTrue(stored.startswith("scrypt$"))

    def test_display_name_whitespace_is_normalised(self):
        self.signup(name="  Ann   Lee  ")
        self.assertEqual(db.get_user_by_email("ann@example.com")["display_name"], "Ann Lee")

    def test_signup_again_within_cooldown_is_throttled(self):
        self.assertEqual(self.reg()[0], 200)
        self.assertEqual(self.reg()[0], 429)


class OneTimeCodeTests(ApiTestCase):
    def start(self):
        self.call("POST", "/api/auth/register", {"name": "Ann", "email": "ann@example.com", "password": PASSWORD})

    def verify(self, code):
        return self.call("POST", "/api/otp/verify", {"email": "ann@example.com", "code": code})

    def test_code_is_six_digits_and_not_stored_in_plaintext(self):
        self.start()
        code = self.last_code()
        self.assertRegex(code, r"^\d{6}$")
        self.assertNotIn(code, json.dumps(server._otp_challenges, default=str))

    def test_wrong_code_rejected_with_attempts_left(self):
        self.start()
        wrong = "000000" if self.last_code() != "000000" else "111111"
        status, body = self.verify(wrong)
        self.assertEqual(status, 401)
        self.assertIn("4 attempt", body["error"])

    def test_non_numeric_code_rejected(self):
        self.start()
        self.assertEqual(self.verify("abcdef")[0], 401)

    def test_code_is_single_use(self):
        self.start()
        code = self.last_code()
        self.assertEqual(self.verify(code)[0], 200)
        self.assertEqual(self.verify(code)[0], 400)

    def test_locked_out_after_five_wrong_attempts_even_with_right_code(self):
        self.start()
        code = self.last_code()
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            self.assertEqual(self.verify(wrong)[0], 401)
        self.assertIn(self.verify(code)[0], (400, 429))

    def test_expired_code_rejected(self):
        self.start()
        server._otp_challenges["ann@example.com"]["expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=1)
        status, body = self.verify(self.last_code())
        self.assertEqual(status, 400)
        self.assertIn("expired", body["error"])

    def test_verify_without_starting_is_rejected(self):
        self.assertEqual(self.verify("123456")[0], 400)

    def test_resend_requires_prior_signup_or_signin(self):
        status, _ = self.call("POST", "/api/otp/request", {"email": "stranger@example.com"})
        self.assertEqual(status, 400)
        self.assertEqual(self.codes, [])

    def test_resend_has_cooldown(self):
        self.start()
        self.assertEqual(self.call("POST", "/api/otp/request", {"email": "ann@example.com"})[0], 429)


class SignInTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.signup()
        self.codes.clear()

    def test_correct_password_signs_in_without_a_code(self):
        status, body = self.signin()
        self.assertEqual(status, 200)
        self.assertTrue(body["token"])
        self.assertEqual(self.codes, [])  # no email code needed

    def test_wrong_password_gives_generic_error(self):
        status, body = self.signin(password="not the password")
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "Incorrect email or password.")

    def test_unknown_email_gives_same_error_as_wrong_password(self):
        a = self.signin(email="ghost@example.com")
        b = self.signin(password="not the password")
        self.assertEqual(a, b)

    def test_email_is_case_insensitive(self):
        self.assertEqual(self.signin(email="ANN@Example.com")[0], 200)

    def test_account_locks_after_five_failures(self):
        for _ in range(5):
            self.assertEqual(self.signin(password="bad password 1")[0], 401)
        self.assertEqual(self.signin()[0], 429)  # even the right password is refused

    def test_successful_login_resets_failure_counter(self):
        for _ in range(4):
            self.signin(password="bad password 1")
        self.assertEqual(self.signin()[0], 200)
        for _ in range(4):
            self.assertEqual(self.signin(password="bad password 1")[0], 401)

    def test_signing_up_again_needs_a_code_and_keeps_old_credentials(self):
        status, _ = self.call("POST", "/api/auth/register",
                              {"name": "Intruder", "email": "ann@example.com", "password": "attacker pass 1"})
        self.assertEqual(status, 200)
        self.assertEqual(len(self.codes), 1)
        status, body = self.call("POST", "/api/otp/verify",
                                 {"email": "ann@example.com", "code": self.last_code()})
        self.assertEqual(status, 200)
        self.assertEqual(body["user"]["displayName"], "Ann Lee")
        self.assertEqual(self.signin()[0], 200)
        self.assertEqual(self.signin(password="attacker pass 1")[0], 401)


class SessionAndProfileTests(ApiTestCase):
    def test_protected_endpoints_require_a_token(self):
        for method, path in (("GET", "/api/me"), ("GET", "/api/analyses"),
                             ("POST", "/api/me"), ("DELETE", "/api/analyses/1"),
                             ("POST", "/api/me/password/request")):
            self.assertEqual(self.call(method, path, {} if method == "POST" else None)[0], 401, path)

    def test_garbage_token_rejected(self):
        self.assertEqual(self.call("GET", "/api/me", token="garbage")[0], 401)

    def test_session_check_endpoint(self):
        token = self.signup()
        self.assertEqual(self.call("POST", "/api/session", {"token": token})[0], 200)
        self.assertEqual(self.call("POST", "/api/session", {"token": "bad"})[0], 401)

    def test_logout_invalidates_token(self):
        token = self.signup()
        self.call("POST", "/api/logout", {}, token=token)
        self.assertEqual(self.call("GET", "/api/me", token=token)[0], 401)

    def test_update_display_name(self):
        token = self.signup()
        status, body = self.call("POST", "/api/me", {"displayName": "  New   Name "}, token=token)
        self.assertEqual(status, 200)
        self.assertEqual(body["user"]["displayName"], "New Name")

    def test_empty_display_name_rejected(self):
        token = self.signup()
        self.assertEqual(self.call("POST", "/api/me", {"displayName": "   "}, token=token)[0], 400)

    def test_display_name_is_capped_at_60_chars(self):
        token = self.signup()
        _, body = self.call("POST", "/api/me", {"displayName": "x" * 200}, token=token)
        self.assertEqual(len(body["user"]["displayName"]), 60)


class PasswordChangeTests(ApiTestCase):
    NEW = "brand new pass 1"

    def request(self, token, current=PASSWORD, new=NEW):
        return self.call("POST", "/api/me/password/request",
                         {"currentPassword": current, "newPassword": new}, token=token)

    def test_wrong_current_password_rejected_and_no_code_sent(self):
        token = self.signup()
        self.codes.clear()
        self.assertEqual(self.request(token, current="wrong")[0], 401)
        self.assertEqual(self.codes, [])

    def test_new_password_validation(self):
        token = self.signup()
        self.assertEqual(self.request(token, new="short")[0], 400)
        self.assertEqual(self.request(token, new=PASSWORD)[0], 400)  # same as current

    def test_confirm_without_request_rejected(self):
        token = self.signup()
        self.assertEqual(self.call("POST", "/api/me/password/confirm", {"code": "123456"}, token=token)[0], 400)

    def test_password_not_changed_until_code_confirmed(self):
        token = self.signup()
        self.assertEqual(self.request(token)[0], 200)
        self.assertEqual(self.signin(password=self.NEW)[0], 401)
        self.assertEqual(self.signin()[0], 200)

    def test_wrong_code_does_not_change_password(self):
        token = self.signup()
        self.request(token)
        wrong = "000000" if self.last_code() != "000000" else "111111"
        self.assertEqual(self.call("POST", "/api/me/password/confirm", {"code": wrong}, token=token)[0], 401)
        self.assertEqual(self.signin()[0], 200)

    def test_full_change_flow(self):
        token = self.signup()
        self.request(token)
        self.assertEqual(self.codes[-1][2], "password")
        status, _ = self.call("POST", "/api/me/password/confirm", {"code": self.last_code()}, token=token)
        self.assertEqual(status, 200)
        self.assertEqual(self.signin(password=self.NEW)[0], 200)
        self.assertEqual(self.signin()[0], 401)

    def test_change_signs_out_other_devices_but_not_this_one(self):
        token = self.signup()
        other = self.signin()[1]["token"]
        self.request(token)
        self.call("POST", "/api/me/password/confirm", {"code": self.last_code()}, token=token)
        self.assertEqual(self.call("GET", "/api/me", token=token)[0], 200)
        self.assertEqual(self.call("GET", "/api/me", token=other)[0], 401)

    def test_code_cannot_be_reused(self):
        token = self.signup()
        self.request(token)
        code = self.last_code()
        self.call("POST", "/api/me/password/confirm", {"code": code}, token=token)
        self.assertEqual(self.call("POST", "/api/me/password/confirm", {"code": code}, token=token)[0], 400)


class AnalysisHistoryTests(ApiTestCase):
    def add(self, email, n=2, name="img.jpg"):
        user = db.get_user_by_email(email)
        return db.save_analysis(user["id"], name, "pytorch_trained_weights",
                                {"width": 432, "height": 432},
                                {"panelCount": n, "totalAreaM2": 1.0, "estimatedCapacityKw": 0.2,
                                 "avgConfidence": 90.0, "coveragePercent": 1.0, "inferenceTimeMs": 5,
                                 "instances": [{"id": i + 1, "bbox": [0, 0, 1, 1], "confidence": .9, "areaM2": 1.0}
                                               for i in range(n)]})

    def test_list_and_fetch_own_analyses(self):
        token = self.signup()
        aid = self.add("ann@example.com", n=3)
        status, body = self.call("GET", "/api/analyses", token=token)
        self.assertEqual([a["id"] for a in body["analyses"]], [aid])
        status, body = self.call("GET", f"/api/analyses/{aid}", token=token)
        self.assertEqual(len(body["analysis"]["detections"]), 3)

    def test_users_cannot_see_or_delete_each_others_analyses(self):
        ann = self.signup("ann@example.com")
        bob = self.signup("bob@example.com", name="Bob")
        aid = self.add("ann@example.com")
        self.assertEqual(self.call("GET", f"/api/analyses/{aid}", token=bob)[0], 404)
        self.assertEqual(self.call("DELETE", f"/api/analyses/{aid}", token=bob)[0], 404)
        self.assertEqual(self.call("GET", "/api/analyses", token=bob)[1]["analyses"], [])
        self.assertEqual(self.call("DELETE", f"/api/analyses/{aid}", token=ann)[0], 200)

    def test_non_numeric_id_is_404(self):
        token = self.signup()
        self.assertEqual(self.call("GET", "/api/analyses/abc", token=token)[0], 404)


if __name__ == "__main__":
    unittest.main()
