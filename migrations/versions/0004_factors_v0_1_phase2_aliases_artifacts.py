"""factors v0.1 phase 2 — aliases + source_artifacts (mirror of V505)

Revision ID: 0004_factors_v0_1_phase2_aliases_artifacts
Revises: 0003_factors_v0_1_phase2_activity
Create Date: 2026-04-27

This revision is the *operator-facing* Alembic mirror of the canonical
SQL migrations under ``deployment/database/migrations/sql/``:

    upgrade()    -> reads V505__factors_v0_1_phase2_aliases_artifacts.sql
    downgrade()  -> reads V505__factors_v0_1_phase2_aliases_artifacts_DOWN.sql

The two SQL files remain the authoritative source of truth. This Python
revision exists so operators can run ``alembic upgrade head`` without
having to know the V### filename layout.

Coordination notes (TaskCreate #7 / WS7-T1):
    1. SQL slot collision: V501 was originally allocated to this
       migration (factor_aliases + source_artifacts) per the Phase 2
       brief. WS3/WS4/WS6 shipped their geography enum + seed_source
       additive migration into V501 first, so this migration was moved
       to slot V505 (next free) and the geography enum extension was
       removed from its body.
    2. Alembic chain: at commit time the only prior revision in the
       factors tree is 0001 (V500) and 0003 (WS5 V502 activity). We
       chain off 0003 directly. WS3 has not yet committed a 0002 seed
       revision; if it lands later it should be inserted between 0001
       and 0003.

Wave: Phase 2 / TaskCreate #7 / WS7-T1
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

from alembic import op
from sqlalchemy import text

try:  # pragma: no cover - sqlparse is in dev deps but be defensive
    import sqlparse  # type: ignore[import-untyped]

    _HAS_SQLPARSE = True
except ImportError:
    _HAS_SQLPARSE = False


# ---------------------------------------------------------------------------
# Alembic identifiers
# ---------------------------------------------------------------------------

revision: str = "0004_factors_v0_1_phase2_aliases_artifacts"
# down_revision chains off 0003 (WS5 V502 activity), which is the latest
# revision on disk at commit time. If WS3 lands a 0002 seed revision
# later, the integrator should insert it between 0001 and 0003 — this
# revision continues to chain off 0003 unchanged.
down_revision: Union[str, Sequence[str], None] = (
    "0003_factors_v0_1_phase2_activity"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Source-of-truth SQL files. Resolved relative to *this* file so the
# revision continues to work no matter the cwd at `alembic upgrade` time.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_FILENAME = "V505__factors_v0_1_phase2_aliases_artifacts.sql"
DOWN_SQL_FILENAME = "V505__factors_v0_1_phase2_aliases_artifacts_DOWN.sql"
UP_SQL_PATH = SQL_DIR / UP_SQL_FILENAME
DOWN_SQL_PATH = SQL_DIR / DOWN_SQL_FILENAME


# ---------------------------------------------------------------------------
# SQL splitter that respects PostgreSQL dollar-quoted strings.
#
# Mirrors the implementation in 0001_factors_v0_1_initial.py exactly so
# every Phase 2 revision behaves the same way under both the sqlparse
# fast-path and the hand-rolled fallback. V501/V503/V504 do NOT contain
# `$$ ... $$` blocks today, but keeping the splitter consistent guards
# against future edits that DO add functions.
# ---------------------------------------------------------------------------

_DOLLAR_TAG_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def _split_sql_preserving_dollar_quotes(sql: str) -> List[str]:
    """Split *sql* into top-level statements, preserving `$$ ... $$` blocks."""
    if _HAS_SQLPARSE:
        return [s.strip() for s in sqlparse.split(sql) if s and s.strip()]

    statements: List[str] = []
    buf: List[str] = []

    i = 0
    n = len(sql)
    in_single_quote = False
    in_line_comment = False
    in_block_comment = False
    dollar_tag: str | None = None

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if dollar_tag is not None:
            m = _DOLLAR_TAG_RE.match(sql, i)
            if m and (m.group(1) or "") == dollar_tag:
                buf.append(m.group(0))
                i = m.end()
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        if in_single_quote:
            buf.append(ch)
            if ch == "'":
                if nxt == "'":
                    buf.append(nxt)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        if ch == "-" and nxt == "-":
            in_line_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "'":
            in_single_quote = True
            buf.append(ch)
            i += 1
            continue
        m = _DOLLAR_TAG_RE.match(sql, i)
        if m:
            dollar_tag = m.group(1) or ""
            buf.append(m.group(0))
            i = m.end()
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _execute_sql_file(filename: str) -> None:
    """Read *filename* from the canonical SQL directory and execute it."""
    sql_path = SQL_DIR / filename
    if not sql_path.exists():
        raise FileNotFoundError(
            f"Canonical migration SQL not found: {sql_path}. "
            "Phase 2 V505 (factors aliases + artifacts) must be present "
            "before alembic upgrade can run."
        )
    sql_text = sql_path.read_text(encoding="utf-8")
    statements = _split_sql_preserving_dollar_quotes(sql_text)

    bind = op.get_bind()
    for stmt in statements:
        bind.execute(text(stmt))


# ---------------------------------------------------------------------------
# Alembic upgrade / downgrade entrypoints
# ---------------------------------------------------------------------------


def upgrade() -> None:
    """Apply the V505 Phase 2 aliases + artifacts schema."""
    _execute_sql_file(UP_SQL_FILENAME)


def downgrade() -> None:
    """Revert the V505 Phase 2 aliases + artifacts schema."""
    _execute_sql_file(DOWN_SQL_FILENAME)
