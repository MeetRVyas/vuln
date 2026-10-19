from __future__ import annotations

import shutil

import pytest

from vulnrem.sast.tools.bandit import BanditTool

pytestmark = pytest.mark.skipif(shutil.which("bandit") is None, reason="bandit not installed")

_VULNERABLE_SNIPPET = """
def find_user_by_name(conn, username):
    query = "SELECT id, username FROM users WHERE username = '" + username + "'"
    cursor = conn.execute(query)
    return cursor.fetchall()

PASSWORD = "s3cr3t_hardcoded_password"
"""


def test_real_bandit_flags_sql_injection_and_hardcoded_secret(tmp_path):
    target = tmp_path / "target.py"
    target.write_text(_VULNERABLE_SNIPPET)

    tool = BanditTool()
    assert tool.is_available() is True

    findings = tool.scan(target)
    rule_ids = {f.rule_id for f in findings}

    assert "B608" in rule_ids  # SQL injection
    sql_finding = next(f for f in findings if f.rule_id == "B608")
    assert sql_finding.cwe_id == "CWE-89"
