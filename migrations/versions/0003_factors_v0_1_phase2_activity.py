"""factors v0.1 alpha — Phase 2 activity taxonomy table (mirror of V502)

Revision ID: 0003_factors_v0_1_phase2_activity
Revises: 0002_factors_v0_1_phase2_seed_ontology
Create Date: 2026-04-27

This revision is the operator-facing Alembic mirror of:

    upgrade()    -> reads V502__factors_v0_1_phase2_activity.sql
    downgrade()  -> reads V502__factors_v0_1_phase2_activity_DOWN.sql

Migration order (linear chain — fork resolved 2026-04-27)
---------------------------------------------------------
    0001 (V500)  : factors_v0_1 canonical schema (FROZEN)
    0002 (V501)  : geography type extension + ontology seed (WS3+4+6)
    0003 (V502)  : activity taxonomy table             (WS5)  *<= this revision*
    0004 (V505)  : factor_aliases + source_artifacts   (WS7)
    0005 (V503)  : provenance_edges + changelog_events (WS7)
    0006 (V504)  : api_keys + entitlements + release_manifests (WS7)

This revision was originally dispatched in parallel with WS3 (0002) and
declared ``down_revision = "0001_factors_v0_1_initial"`` while WS3 was
still in flight. After both landed, the resulting two-head fork
(0002 and 0003 both rooted at 0001) was reconciled by re-chaining 0003
onto 0002. There is no real data dependency between the seed-ontology
migration and this activity-table migration — both are independent
additive changes — so the re-chain is pure ordering. V502 declares its
own DDL and never touches V500/V501/V502 objects.

Wave: Phase 2 / TaskCreate #5 / WS5-T1
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

revision: str = "0003_factors_v0_1_phase2_activity"
down_revision: Union[str, Sequence[str], None] = "0002_factors_v0_1_phase2_seed_ontology"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Source-of-truth SQL files. Resolved relative to *this* file so the
# revision continues to work no matter the cwd at `alembic upgrade` time.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_FILENAME = "V502__factors_v0_1_phase2_activity.sql"
DOWN_SQL_FILENAME = "V502__factors_v0_1_phase2_activity_DOWN.sql"
UP_SQL_PATH = SQL_DIR / UP_SQL_FILENAME
DOWN_SQL_PATH = SQL_DIR / DOWN_SQL_FILENAME


# ---------------------------------------------------------------------------
# SQL splitter that respects PostgreSQL dollar-quoted strings.
#
# V502 contains no $$-quoted bodies, but we use the same splitter as 0001
# so any future ALTER additions to V502 (or sibling V501/V503 mirrors
# pasted into this revision style) keep working without a refactor.
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
            f"Phase 2 migration SQL not found: {sql_path}. "
            "V502__factors_v0_1_phase2_activity.sql must be present "
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
    """Apply the V502 Phase 2 activity-taxonomy table.

    Reads ``deployment/database/migrations/sql/V502__factors_v0_1_phase2_activity.sql``
    relative to this revision file and executes each top-level statement.
    """
    _execute_sql_file(UP_SQL_FILENAME)


def downgrade() -> None:
    """Revert the V502 Phase 2 activity-taxonomy table.

    Reads ``deployment/database/migrations/sql/V502__factors_v0_1_phase2_activity_DOWN.sql``
    relative to this revision file and executes each top-level statement.
    """
    _execute_sql_file(DOWN_SQL_FILENAME)
