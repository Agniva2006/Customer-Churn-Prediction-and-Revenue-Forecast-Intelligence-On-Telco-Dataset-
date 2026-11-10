"""Authentication module — token-based auth with SQLite user storage.

Provides user registration, login, profile management, and session token validation.

Security notes:
- Passwords are hashed with PBKDF2-HMAC-SHA256 (per-user random salt, configurable
  iterations via ``AUTH_PBKDF2_ITERATIONS``). Legacy unsalted sha256 hashes are
  verified transparently and upgraded to PBKDF2 on the next successful login.
- Session tokens carry an ISO expiry timestamp enforced on every lookup
  (``AUTH_TOKEN_TTL_HOURS``, default 72).
"""

import hashlib
import hmac
import logging
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

AUTH_DB_DIR = Path(__file__).resolve().parent.parent / "logs"
AUTH_DB_PATH = AUTH_DB_DIR / "users.db"

# PBKDF2 iterations: override via environment for stronger hardware.
PBKDF2_ITERATIONS = int(os.environ.get("AUTH_PBKDF2_ITERATIONS", "200000"))
TOKEN_TTL_HOURS = int(os.environ.get("AUTH_TOKEN_TTL_HOURS", "72"))


def _get_conn() -> sqlite3.Connection:
    AUTH_DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create users table if it does not exist; migrate legacy schemas."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL DEFAULT '',
            company TEXT DEFAULT '',
            role TEXT DEFAULT 'analyst',
            avatar_color TEXT DEFAULT '#6366f1',
            default_threshold REAL DEFAULT 0.15,
            notifications_enabled INTEGER DEFAULT 1,
            dark_mode INTEGER DEFAULT 1,
            token TEXT,
            token_expires_at TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    # Lightweight migration for pre-existing databases without token expiry column
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "token_expires_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN token_expires_at TEXT")
        logger.info("Migrated users table: added token_expires_at column")
    conn.commit()
    conn.close()


init_auth_db()


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a per-user random salt.

    Returns the string ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``
    so iterations can be raised later without invalidating stored hashes.
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    """Constant-time password verification against an encoded PBKDF2 hash.

    Also transparently verifies legacy unsalted sha256 hashes (``<hex64>``);
    callers upgrade those hashes to PBKDF2 format on successful login.
    """
    try:
        if encoded.startswith("pbkdf2_sha256$"):
            _, iters, salt_hex, hash_hex = encoded.split("$", 3)
            dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
            return hmac.compare_digest(dk.hex(), hash_hex)
        # Legacy format fallback (fixed-salt sha256) — vulnerable; verify then upgrade.
        legacy = hashlib.sha256(f"telco_churn_v3_salt{password}".encode()).hexdigest()
        return hmac.compare_digest(legacy, encoded)
    except (ValueError, TypeError):
        return False


def _new_token() -> tuple:
    """Create a fresh session token and its ISO expiry timestamp."""
    token = secrets.token_hex(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)).isoformat()
    return token, expires


def register_user(email: str, password: str, full_name: str) -> Dict[str, Any]:
    """Register a new user. Returns user dict with token on success."""
    conn = _get_conn()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return {"error": "An account with this email already exists."}

        pw_hash = _hash_password(password)
        token, expires = _new_token()
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            "INSERT INTO users (email, password_hash, full_name, token, token_expires_at, created_at, last_login) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (email.lower().strip(), pw_hash, full_name.strip(), token, expires, now, now)
        )
        conn.commit()

        user = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user and return user dict with a fresh token."""
    conn = _get_conn()
    try:
        clean_email = email.lower().strip()

        user = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,)).fetchone()
        if not user:
            return {"error": "Invalid email or password."}

        if not _verify_password(password, user["password_hash"]):
            return {"error": "Invalid email or password."}

        # Transparently upgrade legacy unsalted hashes to PBKDF2
        if not user["password_hash"].startswith("pbkdf2_sha256$"):
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (_hash_password(password), user["id"]),
            )
            logger.info("Upgraded legacy password hash for user id=%s", user["id"])

        token, expires = _new_token()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE users SET token = ?, token_expires_at = ?, last_login = ? WHERE id = ?",
            (token, expires, now, user["id"]),
        )
        conn.commit()

        user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Look up a user by their session token, enforcing expiry."""
    if not token:
        return None
    conn = _get_conn()
    try:
        user = conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
        if not user:
            return None
        expires_raw = None
        try:
            expires_raw = user["token_expires_at"]
        except (IndexError, KeyError):
            pass
        if expires_raw:
            try:
                expires = datetime.fromisoformat(expires_raw)
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) >= expires:
                    conn.execute(
                        "UPDATE users SET token = NULL, token_expires_at = NULL WHERE id = ?",
                        (user["id"],),
                    )
                    conn.commit()
                    return None
            except ValueError:
                pass
        return _user_to_dict(user)
    finally:
        conn.close()


def update_profile(user_id: int, full_name: str, company: str, role: str) -> Dict[str, Any]:
    """Update user profile fields."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE users SET full_name = ?, company = ?, role = ? WHERE id = ?",
            (full_name.strip(), company.strip(), role.strip(), user_id)
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def update_settings(user_id: int, default_threshold: float, notifications_enabled: bool, dark_mode: bool) -> Dict[str, Any]:
    """Update user settings."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE users SET default_threshold = ?, notifications_enabled = ?, dark_mode = ? WHERE id = ?",
            (default_threshold, int(notifications_enabled), int(dark_mode), user_id)
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def logout_user(token: str) -> bool:
    """Invalidate a user's session token."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE users SET token = NULL, token_expires_at = NULL WHERE token = ?",
            (token,),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def _user_to_dict(row) -> Dict[str, Any]:
    """Convert a sqlite Row to a safe dict (no password hash)."""
    return {
        "id": row["id"],
        "email": row["email"],
        "full_name": row["full_name"],
        "company": row["company"] or "",
        "role": row["role"] or "analyst",
        "avatar_color": row["avatar_color"] or "#6366f1",
        "default_threshold": row["default_threshold"],
        "notifications_enabled": bool(row["notifications_enabled"]),
        "dark_mode": bool(row["dark_mode"]),
        "token": row["token"],
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }
