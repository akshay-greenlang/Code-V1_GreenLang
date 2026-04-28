# -*- coding: utf-8 -*-
"""Phase 2 / WS5 — activity seed loader smoke test.

Loads ``greenlang/factors/data/ontology/activity_seed_v0_1.yaml`` into
an in-memory SQLite mirror of ``factors_v0_1.activity`` and asserts:

* row count >= 100
* every row's URN survives the canonical parser
* every required CTO taxonomy is present
* the loader is idempotent (second run inserts 0)
"""
from __future__ import annotations

import sqlite3
from typing import Set

import pytest

from greenlang.factors.data.ontology.loaders.activity_loader import (
    DEFAULT_SEED_PATH,
    create_sqlite_activity_table,
    load_into_sqlite,
    load_seed_yaml,
)
from greenlang.factors.ontology.urn import (
    ALLOWED_ACTIVITY_TAXONOMIES,
    parse,
)


@pytest.fixture
def sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    # SQLite needs PRAGMA to enforce CHECK and UNIQUE; both default ON.
    conn.execute("PRAGMA foreign_keys = ON")
    create_sqlite_activity_table(conn)
    yield conn
    conn.close()


def test_seed_file_exists() -> None:
    assert DEFAULT_SEED_PATH.exists(), (
        f"activity seed not found at {DEFAULT_SEED_PATH}"
    )


def test_seed_yaml_parses_and_validates() -> None:
    rows = load_seed_yaml()
    assert len(rows) >= 100, (
        f"activity seed must contain >= 100 rows, got {len(rows)}"
    )


def test_every_seeded_urn_parses() -> None:
    rows = load_seed_yaml()
    for row in rows:
        parsed = parse(row.urn)
        assert parsed.kind == "activity"
        assert parsed.taxonomy == row.taxonomy
        # The URN-encoded code is the lowercase, dot->hyphen form of the
        # source code column.
        assert parsed.code == row.code.lower().replace(".", "-")


def test_load_into_sqlite_inserts_all(
    sqlite_conn: sqlite3.Connection,
) -> None:
    rows = load_seed_yaml()
    inserted, skipped = load_into_sqlite(sqlite_conn, rows)
    assert inserted == len(rows)
    assert skipped == 0
    cur = sqlite_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM activity")
    (count,) = cur.fetchone()
    assert count == len(rows)
    assert count >= 100


def test_load_is_idempotent(sqlite_conn: sqlite3.Connection) -> None:
    rows = load_seed_yaml()
    inserted_1, skipped_1 = load_into_sqlite(sqlite_conn, rows)
    inserted_2, skipped_2 = load_into_sqlite(sqlite_conn, rows)
    # First run inserts everything.
    assert inserted_1 == len(rows)
    assert skipped_1 == 0
    # Second run inserts nothing — every URN is already present.
    assert inserted_2 == 0
    assert skipped_2 == len(rows)


def test_every_required_taxonomy_present(
    sqlite_conn: sqlite3.Connection,
) -> None:
    rows = load_seed_yaml()
    load_into_sqlite(sqlite_conn, rows)
    cur = sqlite_conn.cursor()
    cur.execute("SELECT DISTINCT taxonomy FROM activity")
    seen: Set[str] = {r[0] for r in cur.fetchall()}
    missing = set(ALLOWED_ACTIVITY_TAXONOMIES) - seen
    assert not missing, (
        f"activity seed missing taxonomies: {sorted(missing)}"
    )


def test_no_duplicate_urns_in_seed() -> None:
    rows = load_seed_yaml()
    urns = [r.urn for r in rows]
    assert len(urns) == len(set(urns)), (
        "activity seed contains duplicate URNs"
    )


def test_no_duplicate_taxonomy_code_pairs_in_seed() -> None:
    rows = load_seed_yaml()
    pairs = [(r.taxonomy, r.code) for r in rows]
    assert len(pairs) == len(set(pairs)), (
        "activity seed contains duplicate (taxonomy, code) pairs"
    )
