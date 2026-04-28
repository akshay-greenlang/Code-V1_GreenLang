"""factors v0.1 phase 2 - additive contract fields (mirror of V506)

Revision ID: 0007_factors_v0_1_phase2_contract_fields
Revises: 0006_factors_v0_1_phase2_releases
Create Date: 2026-04-27

Operator-facing Alembic mirror of:
    upgrade()    -> reads V506__factors_v0_1_phase2_contract_fields.sql
    downgrade()  -> reads V506__factors_v0_1_phase2_contract_fields_DOWN.sql

Strictly additive: adds five OPTIONAL columns to ``factors_v0_1.factor``
(``activity_taxonomy_urn``, ``confidence``, ``created_at_pre_publish``,
``updated_at_pre_publish``, ``superseded_by_urn``) plus two partial
indexes. The two SQL files are the source of truth.

Wave: Phase 2 / WS9-A / contract-fields amendment
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

revision: str = "0007_factors_v0_1_phase2_contract_fields"
down_revision: Union[str, Sequence[str], None] = (
    "0006_factors_v0_1_phase2_releases"
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# SQL file resolution (relative to this file).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_FILENAME = "V506__factors_v0_1_phase2_contract_fields.sql"
DOWN_SQL_FILENAME = "V506__factors_v0_1_phase2_contract_fields_DOWN.sql"
UP_SQL_PATH = SQL_DIR / UP_SQL_FILENAME
DOWN_SQL_PATH = SQL_DIR / DOWN_SQL_FILENAME


# ---------------------------------------------------------------------------
# SQL splitter (kept identical across all Phase 2 revisions).
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
            "Phase 2 V506 (factors additive contract fields) "
            "must be present before alembic upgrade can run."
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
    """Apply the V506 Phase 2 additive contract-fields columns."""
    _execute_sql_file(UP_SQL_FILENAME)


def downgrade() -> None:
    """Revert the V506 Phase 2 additive contract-fields columns."""
    _execute_sql_file(DOWN_SQL_FILENAME)
