"""SQLite persistence for Solar Sensing: user profiles, sessions and analysis history.

Uses only the standard library. The database file lives in <repo>/data/solar_sensing.db
(override with the SOLAR_DB_PATH environment variable).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

DB_PATH = Path(
    os.getenv("SOLAR_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "solar_sensing.db")
)
SESSION_TTL_S = 8 * 3600

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name  TEXT NOT NULL,
    password_hash TEXT,                     -- scrypt$n$r$p$salt$hash (never the password)
    role          TEXT NOT NULL DEFAULT 'researcher' CHECK (role IN ('researcher','admin')),
    created_at    INTEGER NOT NULL,
    last_login_at INTEGER
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,           -- SHA-256 of the bearer token (never the token)
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS analyses (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename          TEXT NOT NULL,
    model_source      TEXT NOT NULL,
    image_width       INTEGER NOT NULL,
    image_height      INTEGER NOT NULL,
    panel_count       INTEGER NOT NULL,
    total_area_m2     REAL NOT NULL,
    capacity_kw       REAL NOT NULL,
    avg_confidence    REAL NOT NULL,
    coverage_percent  REAL NOT NULL,
    inference_ms      INTEGER NOT NULL,
    created_at        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_user_time ON analyses(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id  INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    instance_no  INTEGER NOT NULL,
    confidence   REAL NOT NULL,
    bbox_x       REAL NOT NULL,
    bbox_y       REAL NOT NULL,
    bbox_w       REAL NOT NULL,
    bbox_h       REAL NOT NULL,
    area_m2      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_analysis ON detections(analysis_id);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        if "password_hash" not in cols:  # migrate databases created before passwords existed
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row else None


# ─── Passwords (scrypt, per-user random salt) ───
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, maxmem=64 * 1024 * 1024
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=int(n), r=int(r), p=int(p), maxmem=64 * 1024 * 1024,
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


# ─── Users / profiles ───
def get_user_by_email(email: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        return _row(conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone())


def get_user(user_id: int) -> Optional[dict[str, Any]]:
    with connect() as conn:
        return _row(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


def create_or_claim_account(email: str, display_name: str, password_hash: str) -> dict[str, Any]:
    """Create the account, or attach a password to a legacy profile that has none.

    Only call this after the email address has been verified with a one-time code.
    """
    email = email.strip().lower()
    now = int(time.time())
    with connect() as conn:
        conn.execute(
            "INSERT INTO users (email, display_name, password_hash, created_at, last_login_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(email) DO UPDATE SET "
            "display_name = excluded.display_name, password_hash = excluded.password_hash, "
            "last_login_at = excluded.last_login_at WHERE users.password_hash IS NULL",
            (email, display_name, password_hash, now, now),
        )
        return _row(conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone())  # type: ignore[return-value]


def touch_login(user_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (int(time.time()), user_id))


def set_password(user_id: int, password_hash: str, keep_token: str = "") -> None:
    """Change the password and sign out every other session."""
    with connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user_id, _hash(keep_token))
        )


def update_display_name(user_id: int, display_name: str) -> Optional[dict[str, Any]]:
    with connect() as conn:
        conn.execute("UPDATE users SET display_name = ? WHERE id = ?", (display_name, user_id))
        return _row(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


# ─── Sessions ───
def create_session(user_id: int, token: str) -> None:
    now = int(time.time())
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (_hash(token), user_id, now, now + SESSION_TTL_S),
        )


def get_user_by_token(token: str) -> Optional[dict[str, Any]]:
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (_hash(token), int(time.time())),
        ).fetchone()
        return _row(row)


def delete_session(token: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))


# ─── Analyses ───
def save_analysis(
    user_id: int,
    filename: str,
    model_source: str,
    dims: dict[str, int],
    metrics: dict[str, Any],
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO analyses (user_id, filename, model_source, image_width, image_height, "
            "panel_count, total_area_m2, capacity_kw, avg_confidence, coverage_percent, "
            "inference_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                user_id, filename[:255], model_source, dims["width"], dims["height"],
                metrics["panelCount"], metrics["totalAreaM2"], metrics["estimatedCapacityKw"],
                metrics["avgConfidence"], metrics["coveragePercent"], metrics["inferenceTimeMs"],
                int(time.time()),
            ),
        )
        analysis_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO detections (analysis_id, instance_no, confidence, bbox_x, bbox_y, "
            "bbox_w, bbox_h, area_m2) VALUES (?,?,?,?,?,?,?,?)",
            [
                (analysis_id, i["id"], i["confidence"], *i["bbox"], i["areaM2"])
                for i in metrics.get("instances", [])
            ],
        )
        return int(analysis_id)  # type: ignore[arg-type]


def list_analyses(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    with connect() as conn:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, limit),
            )
        ]


def get_analysis(user_id: int, analysis_id: int) -> Optional[dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["detections"] = [
            dict(r)
            for r in conn.execute(
                "SELECT instance_no, confidence, bbox_x, bbox_y, bbox_w, bbox_h, area_m2 "
                "FROM detections WHERE analysis_id = ? ORDER BY instance_no",
                (analysis_id,),
            )
        ]
        return result


def delete_analysis(user_id: int, analysis_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
        )
        return cur.rowcount > 0
