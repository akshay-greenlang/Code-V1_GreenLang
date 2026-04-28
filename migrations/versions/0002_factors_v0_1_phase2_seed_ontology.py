"""factors v0.1 alpha - Phase 2 seed ontology (geography / unit / methodology)

Revision ID: 0002_factors_v0_1_phase2_seed_ontology
Revises: 0001_factors_v0_1_initial
Create Date: 2026-04-27

This revision is a *data* migration that loads the Phase 2 ontology seed
into the registry tables created by V500. It is the operator-facing
Alembic mirror of the canonical SQL migrations under
``deployment/database/migrations/sql/``:

    upgrade()    -> 1. replays V501__factors_v0_1_phase2_additive.sql
                       (extends geography type CHECK + URN regex to
                       admit basin/tenant; adds seed_source markers)
                    2. invokes the three idempotent loaders
                       (load_geography / load_units / load_methodologies)
                       which ingest the Phase 2 YAML seeds.
    downgrade()  -> 1. DELETEs only rows tagged with seed_source =
                       'phase2_v0_1' (production-ingested rows are not
                       touched).
                    2. replays V501__factors_v0_1_phase2_additive_DOWN.sql
                       (drops seed_source columns, restores the V500
                       narrower type CHECK + URN regex).

Wave: F / TaskCreate #3+#4+#6 / WS3+WS4+WS6
Author: GL-FormulaLibraryCurator
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence, Union

from alembic import op
from sqlalchemy import text

try:  # pragma: no cover - sqlparse is a dev dep but be defensive
    import sqlparse  # type: ignore[import-untyped]

    _HAS_SQLPARSE = True
except ImportError:
    _HAS_SQLPARSE = False


# ---------------------------------------------------------------------------
# Alembic identifiers
# ---------------------------------------------------------------------------

revision: str = "0002_factors_v0_1_phase2_seed_ontology"
down_revision: Union[str, Sequence[str], None] = "0001_factors_v0_1_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Source-of-truth SQL files. Resolved relative to *this* file so the
# revision continues to work no matter the cwd at `alembic upgrade` time.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_FILENAME = "V501__factors_v0_1_phase2_additive.sql"
DOWN_SQL_FILENAME = "V501__factors_v0_1_phase2_additive_DOWN.sql"


# ---------------------------------------------------------------------------
# SQL splitter (copied from 0001 — same dollar-quote/single-quote/comment
# state machine; V501 contains no $$-quoted blocks but we use the same
# splitter for forward-compatibility with future Phase 2 SQL files).
# ---------------------------------------------------------------------------

_DOLLAR_TAG_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def _split_sql_preserving_dollar_quotes(sql: str) -> List[str]:
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
    sql_path = SQL_DIR / filename
    if not sql_path.exists():
        raise FileNotFoundError(
            f"Phase 2 additive SQL not found: {sql_path}. Wave F V501 must "
            "be present alongside this Alembic revision."
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
    """Apply Phase 2 additive DDL then load the three ontology seeds.

    Step 1: replay V501 (extends geography CHECK enum + URN regex; adds
    seed_source columns on geography / unit / methodology).

    Step 2: invoke the three loaders against the live psycopg-style
    connection exposed by ``op.get_bind()``. SQLAlchemy's connection
    object exposes a ``connection`` attribute that is the underlying
    DB-API connection — that is what the loaders consume.
    """
    # 1. DDL bumps.
    _execute_sql_file(UP_SQL_FILENAME)

    # 2. Idempotent seed load. Imported lazily so this Alembic revision
    #    can be parsed even when the greenlang package isn't yet on the
    #    operator's PYTHONPATH (e.g., during initial DB-only bootstrap
    #    audits via the standalone V500/V501 SQL files).
    from greenlang.factors.data.ontology.loaders import (
        load_geography,
        load_methodologies,
        load_units,
    )

    bind = op.get_bind()
    raw_conn = bind.connection  # psycopg2.connection or psycopg.Connection

    geo_report = load_geography(raw_conn)
    unit_report = load_units(raw_conn)
    method_report = load_methodologies(raw_conn)

    # Surface the row counts in alembic's own log so operators can audit
    # what was inserted vs. skipped during the upgrade.
    print(
        f"[migration 0002] geography: inserted={geo_report.count_inserted} "
        f"skipped={geo_report.count_skipped} total={geo_report.total_seen}"
    )
    print(
        f"[migration 0002] unit: inserted={unit_report.count_inserted} "
        f"skipped={unit_report.count_skipped} total={unit_report.total_seen}"
    )
    print(
        f"[migration 0002] methodology: "
        f"inserted={method_report.count_inserted} "
        f"skipped={method_report.count_skipped} "
        f"total={method_report.total_seen}"
    )


def downgrade() -> None:
    """Reverse Phase 2 ontology seed.

    Step 1: DELETE only Phase-2-tagged rows (seed_source = 'phase2_v0_1').
    Production-ingested rows have ``seed_source IS NULL`` and are not
    touched. Order: methodology / unit first (no dependents), then
    geography (children before parents within Phase 2 set — but since
    we delete by tag, FK ordering inside the tagged set is handled by
    deleting non-parent rows first via type filter).

    Step 2: replay V501 DOWN (restore narrower geography CHECK + URN
    regex; drop seed_source columns).
    """
    from greenlang.factors.data.ontology.loaders import PHASE2_SEED_SOURCE

    bind = op.get_bind()

    # Methodology and unit have no inter-row FKs.
    bind.execute(
        text(
            "DELETE FROM factors_v0_1.methodology "
            "WHERE seed_source = :tag"
        ),
        {"tag": PHASE2_SEED_SOURCE},
    )
    bind.execute(
        text("DELETE FROM factors_v0_1.unit WHERE seed_source = :tag"),
        {"tag": PHASE2_SEED_SOURCE},
    )

    # Geography has self-referential parent_urn FK. Delete in
    # leaf-to-root order: basin / balancing_authority / bidding_zone /
    # grid_zone / state_or_province / subregion / country / global.
    leaf_to_root = (
        "basin",
        "balancing_authority",
        "bidding_zone",
        "grid_zone",
        "state_or_province",
        "subregion",
        "country",
        "global",
    )
    for geo_type in leaf_to_root:
        bind.execute(
            text(
                "DELETE FROM factors_v0_1.geography "
                "WHERE seed_source = :tag AND type = :type"
            ),
            {"tag": PHASE2_SEED_SOURCE, "type": geo_type},
        )

    # 2. DDL rollback.
    _execute_sql_file(DOWN_SQL_FILENAME)
