"""factors v0.1 alpha — initial schema (mirror of V500__factors_v0_1_canonical.sql)

Revision ID: 0001_factors_v0_1_initial
Revises:
Create Date: 2026-04-25

This revision is the *operator-facing* Alembic mirror of the canonical
SQL migrations under ``deployment/database/migrations/sql/``:

    upgrade()    -> reads V500__factors_v0_1_canonical.sql
    downgrade()  -> reads V500__factors_v0_1_canonical_DOWN.sql

The two SQL files remain the authoritative source of truth for the
factors v0.1 schema. This Python revision exists so operators can run
``alembic upgrade head`` (and ``alembic downgrade -1``) without having
to know the V### filename layout. Every future V### migration in the
factors tree should land here as a new revision that simply replays
the matching SQL file.

Wave: C / TaskCreate #3 / WS1-T3
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

revision: str = "0001_factors_v0_1_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Source-of-truth SQL files. Resolved relative to *this* file so the
# revision continues to work no matter the cwd at `alembic upgrade` time.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_FILENAME = "V500__factors_v0_1_canonical.sql"
DOWN_SQL_FILENAME = "V500__factors_v0_1_canonical_DOWN.sql"
UP_SQL_PATH = SQL_DIR / UP_SQL_FILENAME
DOWN_SQL_PATH = SQL_DIR / DOWN_SQL_FILENAME


# ---------------------------------------------------------------------------
# SQL splitter that respects PostgreSQL dollar-quoted strings.
#
# The V500 file contains a CREATE OR REPLACE FUNCTION whose body is wrapped
# in `$$ ... $$` and itself contains semicolons. Naively splitting on `;`
# would corrupt the function body, so we use sqlparse when present and
# fall back to a small custom state machine that:
#   * tracks `$tag$ ... $tag$` blocks (any tag, the V500 file uses bare `$$`)
#   * tracks single-quoted string literals (with `''` escapes)
#   * ignores semicolons inside line comments (`--`) and block comments
#     (`/* ... */`)
#
# Returns the list of non-empty top-level statements with trailing semicolons
# stripped (Alembic's op.execute / connection.execute do not require them).
# ---------------------------------------------------------------------------

_DOLLAR_TAG_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def _split_sql_preserving_dollar_quotes(sql: str) -> List[str]:
    """Split *sql* into top-level statements, preserving `$$ ... $$` blocks.

    Prefer sqlparse if installed; otherwise use a hand-rolled splitter that
    respects dollar-quoted blocks, single-quoted strings, and SQL comments.
    """
    if _HAS_SQLPARSE:
        # sqlparse already understands dollar-quoted bodies.
        return [s.strip() for s in sqlparse.split(sql) if s and s.strip()]

    statements: List[str] = []
    buf: List[str] = []

    i = 0
    n = len(sql)
    in_single_quote = False
    in_line_comment = False
    in_block_comment = False
    dollar_tag: str | None = None  # current dollar-quote tag (None when outside)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # --- exit comment states ---
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

        # --- inside dollar-quoted block: only the matching $tag$ closes it ---
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

        # --- inside single-quoted string literal ---
        if in_single_quote:
            buf.append(ch)
            if ch == "'":
                # `''` is an escaped quote, stay inside the literal
                if nxt == "'":
                    buf.append(nxt)
                    i += 2
                    continue
                in_single_quote = False
            i += 1
            continue

        # --- not inside any quoted/comment context ---
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
            "Wave B (V500__factors_v0_1_canonical) must be present before "
            "alembic upgrade can run."
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
    """Apply the V500 canonical schema.

    Reads ``deployment/database/migrations/sql/V500__factors_v0_1_canonical.sql``
    relative to this revision file and executes each top-level statement,
    preserving the `$$ ... $$` block that wraps ``factor_immutable_trigger``.
    """
    _execute_sql_file(UP_SQL_FILENAME)


def downgrade() -> None:
    """Revert the V500 canonical schema.

    Reads ``deployment/database/migrations/sql/V500__factors_v0_1_canonical_DOWN.sql``
    relative to this revision file and executes each top-level statement.
    """
    _execute_sql_file(DOWN_SQL_FILENAME)
