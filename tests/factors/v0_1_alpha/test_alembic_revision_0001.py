# -*- coding: utf-8 -*-
"""Static tests for the first factors Alembic revision (Wave C / WS1-T3).

These tests require neither a running Postgres nor Alembic-installed-as-a-CLI;
they import the revision module directly and assert that:

- the revision id and lineage are correct,
- the revision references the canonical V500 SQL files,
- the upgrade body loads the expected forward filename,
- the downgrade body loads the expected reverse filename,
- the dollar-quote-aware splitter actually preserves
  ``factor_immutable_trigger`` as a single statement.

The DB-apply path is gated by ``@pytest.mark.requires_postgres`` and
delegated to the existing integration test in
``test_postgres_ddl_v500.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
REV_PATH = REPO_ROOT / "migrations" / "versions" / "0001_factors_v0_1_initial.py"
SQL_DIR = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
UP_SQL_PATH = SQL_DIR / "V500__factors_v0_1_canonical.sql"
DOWN_SQL_PATH = SQL_DIR / "V500__factors_v0_1_canonical_DOWN.sql"


# --------------------------------------------------------------------------- #
# Import the revision module directly (do not depend on Alembic CLI here).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def revision_module():
    assert REV_PATH.exists(), f"Revision file missing: {REV_PATH}"
    spec = importlib.util.spec_from_file_location(
        "factors_alembic_rev_0001", REV_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def revision_source() -> str:
    return REV_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Identity / lineage
# --------------------------------------------------------------------------- #


def test_revision_id_is_pinned(revision_module) -> None:
    assert revision_module.revision == "0001_factors_v0_1_initial"


def test_down_revision_is_none(revision_module) -> None:
    # First revision in the factors tree must be a root.
    assert revision_module.down_revision is None


def test_branch_labels_and_depends_on_are_none(revision_module) -> None:
    assert revision_module.branch_labels is None
    assert revision_module.depends_on is None


def test_upgrade_and_downgrade_callables(revision_module) -> None:
    assert callable(revision_module.upgrade)
    assert callable(revision_module.downgrade)


# --------------------------------------------------------------------------- #
# References to the canonical V500 SQL files
# --------------------------------------------------------------------------- #


def test_canonical_sql_files_exist() -> None:
    assert UP_SQL_PATH.exists(), f"Missing forward DDL: {UP_SQL_PATH}"
    assert DOWN_SQL_PATH.exists(), f"Missing reverse DDL: {DOWN_SQL_PATH}"


def test_revision_module_constants_point_at_v500(revision_module) -> None:
    assert revision_module.UP_SQL_FILENAME == "V500__factors_v0_1_canonical.sql"
    assert revision_module.DOWN_SQL_FILENAME == "V500__factors_v0_1_canonical_DOWN.sql"
    # Resolved paths must match the on-disk canonical files.
    assert Path(revision_module.UP_SQL_PATH).resolve() == UP_SQL_PATH.resolve()
    assert Path(revision_module.DOWN_SQL_PATH).resolve() == DOWN_SQL_PATH.resolve()


def test_revision_source_references_v500_filenames(revision_source: str) -> None:
    assert "V500__factors_v0_1_canonical.sql" in revision_source
    assert "V500__factors_v0_1_canonical_DOWN.sql" in revision_source


def test_upgrade_body_loads_forward_sql(revision_module) -> None:
    """Parse the upgrade() body and confirm it loads the forward filename."""
    src = inspect.getsource(revision_module.upgrade)
    assert re.search(r"_execute_sql_file\(\s*UP_SQL_FILENAME\s*\)", src) or (
        "V500__factors_v0_1_canonical.sql" in src
    ), "upgrade() must execute the V500 forward SQL file"


def test_downgrade_body_loads_reverse_sql(revision_module) -> None:
    """Parse the downgrade() body and confirm it loads the reverse filename."""
    src = inspect.getsource(revision_module.downgrade)
    assert re.search(r"_execute_sql_file\(\s*DOWN_SQL_FILENAME\s*\)", src) or (
        "V500__factors_v0_1_canonical_DOWN.sql" in src
    ), "downgrade() must execute the V500 reverse SQL file"


def test_helper_resolves_relative_to_revision_file(revision_module) -> None:
    """The SQL_DIR path must be relative to the revision module, not cwd.

    Concretely, SQL_DIR must equal REPO_ROOT / deployment / database /
    migrations / sql.
    """
    expected = REPO_ROOT / "deployment" / "database" / "migrations" / "sql"
    assert Path(revision_module.SQL_DIR).resolve() == expected.resolve()


# --------------------------------------------------------------------------- #
# Dollar-quoted splitter
# --------------------------------------------------------------------------- #


def test_splitter_preserves_dollar_quoted_function_body(revision_module) -> None:
    """The V500 file contains a `$$ ... $$` block with internal `;`.

    The splitter must not break that body across statements; the
    whole CREATE FUNCTION + body + LANGUAGE plpgsql must come out as a
    single statement.
    """
    sql_text = UP_SQL_PATH.read_text(encoding="utf-8")
    statements = revision_module._split_sql_preserving_dollar_quotes(sql_text)

    assert statements, "splitter returned zero statements"

    fn_stmts = [
        s
        for s in statements
        if "CREATE OR REPLACE FUNCTION factor_immutable_trigger" in s
    ]
    assert len(fn_stmts) == 1, (
        f"Expected exactly 1 statement for factor_immutable_trigger; "
        f"got {len(fn_stmts)}"
    )
    body = fn_stmts[0]
    assert "$$" in body, "dollar-quoted body markers stripped"
    assert "LANGUAGE plpgsql" in body, "splitter cut off the LANGUAGE clause"
    # Body contains multiple internal semicolons (RAISE EXCEPTION ...; etc.)
    assert body.count(";") >= 5, (
        "splitter merged internal semicolons away — body looks corrupted"
    )


def test_splitter_handles_no_sqlparse_path(revision_module, monkeypatch) -> None:
    """The hand-rolled fallback must produce the same number of CREATE TABLE
    statements as sqlparse for the V500 file."""
    sql_text = UP_SQL_PATH.read_text(encoding="utf-8")

    # Force the fallback path.
    monkeypatch.setattr(revision_module, "_HAS_SQLPARSE", False, raising=True)
    fallback = revision_module._split_sql_preserving_dollar_quotes(sql_text)

    # The forward DDL has 7 CREATE TABLE statements (source, methodology,
    # geography, unit, factor_pack, factor, factor_publish_log). Comment
    # lines may sit in front of each `CREATE TABLE`, so use re.search.
    n_create_table_fb = sum(
        1 for s in fallback if re.search(r"\bCREATE\s+TABLE\b", s, re.IGNORECASE)
    )
    assert n_create_table_fb == 7, (
        f"Fallback splitter found {n_create_table_fb} CREATE TABLE; expected 7"
    )

    # And it must still keep the dollar-quoted function body intact.
    fn_stmts = [
        s
        for s in fallback
        if "CREATE OR REPLACE FUNCTION factor_immutable_trigger" in s
    ]
    assert len(fn_stmts) == 1


# --------------------------------------------------------------------------- #
# DB-apply test: skip in default CI; integration coverage lives in the
# existing test_postgres_ddl_v500.py (also marked `requires_postgres`).
# --------------------------------------------------------------------------- #


@pytest.mark.requires_postgres
def test_alembic_upgrade_head_smoke() -> None:  # pragma: no cover - integration
    """Placeholder for a real alembic upgrade head smoke test.

    The actual schema-correctness coverage is provided by
    ``tests/factors/v0_1_alpha/test_postgres_ddl_v500.py::test_ddl_applies_against_real_postgres``
    which executes the same SQL file directly against an isolated temp DB.
    Adding a duplicate Alembic-driven path here is gated behind
    ``requires_postgres`` and intentionally skipped in default CI runs.
    """
    pytest.skip(
        "Alembic upgrade head DB-apply is exercised via test_postgres_ddl_v500.py; "
        "skipped here to keep CI Postgres-free."
    )
