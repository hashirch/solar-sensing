"""Unit tests for db.py: password hashing, accounts, sessions and analysis history."""
import time
import unittest

from tests import _env  # noqa: F401  (must come before importing db)
from backend import db


def sample_metrics(n=2):
    return {
        "panelCount": n, "totalAreaM2": 10.5, "estimatedCapacityKw": 2.1,
        "avgConfidence": 90.1, "coveragePercent": 3.2, "inferenceTimeMs": 120,
        "instances": [
            {"id": i + 1, "bbox": [1, 2, 3, 4], "confidence": 0.9, "areaM2": 5.0} for i in range(n)
        ],
    }


class DatabaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init()

    def setUp(self):
        with db.connect() as conn:
            for table in ("detections", "analyses", "sessions", "users"):
                conn.execute(f"DELETE FROM {table}")

    def make_user(self, email="a@example.com", name="Ann", password="correct horse"):
        return db.create_or_claim_account(email, name, db.hash_password(password))


class PasswordHashingTests(unittest.TestCase):
    def test_hash_is_not_plaintext_and_verifies(self):
        h = db.hash_password("s3cret-pass")
        self.assertNotIn("s3cret-pass", h)
        self.assertTrue(h.startswith("scrypt$"))
        self.assertTrue(db.verify_password("s3cret-pass", h))

    def test_wrong_password_rejected(self):
        self.assertFalse(db.verify_password("wrong", db.hash_password("right-pass")))

    def test_same_password_gets_different_salt(self):
        self.assertNotEqual(db.hash_password("same-pass"), db.hash_password("same-pass"))

    def test_malformed_hash_is_rejected_not_crashing(self):
        for bad in ("", "garbage", "scrypt$x$y", "scrypt$16384$8$1$zz$zz"):
            self.assertFalse(db.verify_password("anything", bad))


class AccountTests(DatabaseTestCase):
    def test_create_account_stores_hash_and_lowercases_email(self):
        user = self.make_user("Ann@Example.COM")
        self.assertEqual(user["email"], "ann@example.com")
        self.assertTrue(user["password_hash"].startswith("scrypt$"))
        self.assertEqual(user["role"], "researcher")

    def test_email_lookup_is_case_insensitive(self):
        self.make_user("bob@example.com")
        self.assertIsNotNone(db.get_user_by_email("BOB@EXAMPLE.COM"))

    def test_existing_password_account_is_not_overwritten(self):
        original = self.make_user("c@example.com", "Carol", "original-pass")
        again = db.create_or_claim_account("c@example.com", "Intruder", db.hash_password("attacker-pass"))
        self.assertEqual(again["display_name"], "Carol")
        self.assertTrue(db.verify_password("original-pass", again["password_hash"]))
        self.assertEqual(again["id"], original["id"])

    def test_legacy_account_without_password_can_be_claimed(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO users (email, display_name, created_at) VALUES ('old@example.com','old',1)")
        claimed = db.create_or_claim_account("old@example.com", "Old", db.hash_password("new-pass-1"))
        self.assertTrue(db.verify_password("new-pass-1", claimed["password_hash"]))

    def test_update_display_name(self):
        user = self.make_user()
        self.assertEqual(db.update_display_name(user["id"], "Ann Lee")["display_name"], "Ann Lee")


class SessionTests(DatabaseTestCase):
    def test_valid_token_returns_user_and_token_is_stored_hashed(self):
        user = self.make_user()
        db.create_session(user["id"], "tok-123")
        self.assertEqual(db.get_user_by_token("tok-123")["id"], user["id"])
        with db.connect() as conn:
            stored = conn.execute("SELECT token_hash FROM sessions").fetchone()["token_hash"]
        self.assertNotIn("tok-123", stored)

    def test_unknown_and_empty_tokens_rejected(self):
        self.assertIsNone(db.get_user_by_token("nope"))
        self.assertIsNone(db.get_user_by_token(""))

    def test_expired_session_rejected(self):
        user = self.make_user()
        db.create_session(user["id"], "old-tok")
        with db.connect() as conn:
            conn.execute("UPDATE sessions SET expires_at = ?", (int(time.time()) - 1,))
        self.assertIsNone(db.get_user_by_token("old-tok"))

    def test_delete_session(self):
        user = self.make_user()
        db.create_session(user["id"], "t1")
        db.delete_session("t1")
        self.assertIsNone(db.get_user_by_token("t1"))

    def test_password_change_keeps_current_session_and_drops_others(self):
        user = self.make_user()
        db.create_session(user["id"], "keep")
        db.create_session(user["id"], "drop")
        db.set_password(user["id"], db.hash_password("brand-new-1"), keep_token="keep")
        self.assertIsNotNone(db.get_user_by_token("keep"))
        self.assertIsNone(db.get_user_by_token("drop"))
        self.assertTrue(db.verify_password("brand-new-1", db.get_user(user["id"])["password_hash"]))


class AnalysisTests(DatabaseTestCase):
    def save(self, user, name="x.jpg", n=2):
        return db.save_analysis(user["id"], name, "pytorch_trained_weights",
                                {"width": 512, "height": 512}, sample_metrics(n))

    def test_save_and_fetch_with_detections(self):
        user = self.make_user()
        aid = self.save(user, n=3)
        item = db.get_analysis(user["id"], aid)
        self.assertEqual(item["panel_count"], 3)
        self.assertEqual(len(item["detections"]), 3)

    def test_history_lists_only_own_analyses_newest_first(self):
        a, b = self.make_user("a@x.com"), self.make_user("b@x.com")
        first = self.save(a, "first.jpg")
        second = self.save(a, "second.jpg")
        self.save(b, "other.jpg")
        ids = [r["id"] for r in db.list_analyses(a["id"])]
        self.assertEqual(ids, [second, first])

    def test_cannot_read_or_delete_another_users_analysis(self):
        a, b = self.make_user("a@x.com"), self.make_user("b@x.com")
        aid = self.save(a)
        self.assertIsNone(db.get_analysis(b["id"], aid))
        self.assertFalse(db.delete_analysis(b["id"], aid))
        self.assertIsNotNone(db.get_analysis(a["id"], aid))

    def test_delete_removes_detections_too(self):
        user = self.make_user()
        aid = self.save(user, n=4)
        self.assertTrue(db.delete_analysis(user["id"], aid))
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"], 0)

    def test_deleting_user_cascades(self):
        user = self.make_user()
        self.save(user)
        db.create_session(user["id"], "t")
        with db.connect() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
            self.assertEqual(conn.execute("SELECT COUNT(*) c FROM analyses").fetchone()["c"], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"], 0)


if __name__ == "__main__":
    unittest.main()
