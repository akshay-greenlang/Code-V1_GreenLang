# -*- coding: utf-8 -*-
"""Phase 2 / WS5 — activity table constraint tests.

Exercises the SQLite mirror of ``factors_v0_1.activity`` (used in CI
where Postgres is not available) to confirm the V502 CHECK and UNIQUE
constraints behave as specified:

* INSERT with a taxonomy outside the 15-enum list raises a CHECK
  violation.
* Inserting a duplicate ``(taxonomy, code)`` pair raises a UNIQUE
  violation (separately from the URN-level UNIQUE).
* Inserting a duplicate URN raises a UNIQUE violation.
"""
from __future__ import annotations

import sqlite3

import pytest

from greenlang.factors.data.ontology.loaders.activity_loader import (
    create_sqlite_activity_table,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("PRAGMA foreign_keys = ON")
    create_sqlite_activity_table(c)
    yield c
    c.close()


def _insert(
    conn: sqlite3.Connection,
    urn: str,
    taxonomy: str,
    code: str,
    name: str = "test row",
) -> None:
    conn.execute(
        "INSERT INTO activity (urn, taxonomy, code, name) "
        "VALUES (?, ?, ?, ?)",
        (urn, taxonomy, code, name),
    )
    conn.commit()


def test_insert_valid_row_succeeds(conn: sqlite3.Connection) -> None:
    _insert(
        conn,
        "urn:gl:activity:ipcc:1-a-1-a",
        "ipcc",
        "1.A.1.a",
        "Public electricity and heat production",
    )
    cur = conn.execute("SELECT COUNT(*) FROM activity")
    (count,) = cur.fetchone()
    assert count == 1


def test_insert_with_bad_taxonomy_raises_check(
    conn: sqlite3.Connection,
) -> None:
    # 'iso-14064' is not in the CHECK enum.
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            conn,
            "urn:gl:activity:iso-14064:foo",
            "iso-14064",
            "foo",
        )


def test_insert_with_empty_taxonomy_raises_check(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            conn,
            "urn:gl:activity:something:bar",
            "",
            "bar",
        )


def test_duplicate_urn_raises_unique(conn: sqlite3.Connection) -> None:
    _insert(
        conn,
        "urn:gl:activity:ghgp:scope1",
        "ghgp",
        "scope1",
        "Scope 1",
    )
    # Same URN, different code column — should still fail on URN UNIQUE.
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            conn,
            "urn:gl:activity:ghgp:scope1",
            "ghgp",
            "scope1-alt",
            "Scope 1 alt",
        )


def test_duplicate_taxonomy_code_pair_raises_unique(
    conn: sqlite3.Connection,
) -> None:
    _insert(
        conn,
        "urn:gl:activity:naics:11",
        "naics",
        "11",
        "Agriculture",
    )
    # Different URN, same (taxonomy, code) pair — must fail on the
    # composite UNIQUE.
    with pytest.raises(sqlite3.IntegrityError):
        _insert(
            conn,
            "urn:gl:activity:naics:11-alt",
            "naics",
            "11",
            "duplicate",
        )


def test_check_accepts_all_15_taxonomies(conn: sqlite3.Connection) -> None:
    """Smoke check that every CTO-approved taxonomy passes the CHECK."""
    samples = [
        ("ipcc", "1-a-1-a"),
        ("ghgp", "scope1"),
        ("hs-cn", "72"),
        ("cpc", "0"),
        ("nace", "a"),
        ("naics", "11"),
        ("sic", "01"),
        ("pact", "chemicals"),
        ("freight", "rail"),
        ("cbam", "cement"),
        ("pcf", "use"),
        ("refrigerants", "hfc-134a"),
        ("agriculture", "rice-cultivation"),
        ("waste", "msw-landfill"),
        ("land-use", "deforestation"),
    ]
    for tax, code in samples:
        _insert(
            conn,
            urn=f"urn:gl:activity:{tax}:{code}",
            taxonomy=tax,
            code=code,
            name=f"{tax} {code}",
        )
    cur = conn.execute("SELECT COUNT(*) FROM activity")
    (count,) = cur.fetchone()
    assert count == len(samples) == 15
