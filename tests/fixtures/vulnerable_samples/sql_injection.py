"""Tiny in-memory 'user store' used only to give the SQL injection below a
runnable, testable context (an actual DB connection would need real
infrastructure, which is out of scope for a unit-test fixture)."""

import sqlite3


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, is_admin INTEGER)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('alice', 0)")
    conn.execute("INSERT INTO users (username, is_admin) VALUES ('admin', 1)")
    conn.commit()
    return conn


def find_user_by_name(conn: sqlite3.Connection, username: str):
    # VULNERABLE: untrusted `username` is interpolated directly into the
    # query string (CWE-89), letting a value like
    # "x' OR '1'='1" return every row instead of a single match.
    query = "SELECT id, username, is_admin FROM users WHERE username = '" + username + "'"
    cursor = conn.execute(query)
    return cursor.fetchall()
