# -*- coding: utf-8 -*-
"""Phase 2 / WS3+WS4+WS6 — geography / unit / methodology seed loader tests.

Loads each Phase 2 YAML seed into an in-memory SQLite mirror of the
matching ``factors_v0_1.<table>`` (per V500 + V501) and asserts:

* Geography seed loads >= 50 rows.
* Unit seed loads >= 30 rows.
* Methodology seed loads >= 25 rows.
* Every URN survives the canonical parser
  (:func:`greenlang.factors.ontology.urn.parse`).
* The loader is idempotent (a second run inserts 0).
* The CTO Phase 2 §2.3 minimum coverage is met (specific URNs present).

Authority: docs/factors/PHASE_2_PLAN.md §2.3 (a/b/d) +
docs/factors/PHASE_2_EXIT_CHECKLIST.md Block 3.
"""
from __future__ import annotations

import sqlite3

import pytest

from greenlang.factors.data.ontology.loaders import (
    GEOGRAPHY_SEED_PATH,
    METHODOLOGY_SEED_PATH,
    PHASE2_SEED_SOURCE,
    UNIT_SEED_PATH,
    create_sqlite_geography_table,
    create_sqlite_methodology_table,
    create_sqlite_unit_table,
    load_geography,
    load_methodologies,
    load_units,
)
from greenlang.factors.data.ontology.loaders.geography_loader import (
    load_seed_yaml as load_geography_yaml,
)
from greenlang.factors.data.ontology.loaders.methodology_loader import (
    load_seed_yaml as load_methodology_yaml,
)
from greenlang.factors.data.ontology.loaders.unit_loader import (
    load_seed_yaml as load_unit_yaml,
)
from greenlang.factors.ontology.urn import parse


# ---------------------------------------------------------------------------
# SQLite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_conn() -> sqlite3.Connection:
    """In-memory SQLite with all three Phase 2 ontology tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    create_sqlite_geography_table(conn)
    create_sqlite_unit_table(conn)
    create_sqlite_methodology_table(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Seed-file presence
# ---------------------------------------------------------------------------


def test_geography_seed_file_exists() -> None:
    assert GEOGRAPHY_SEED_PATH.exists(), (
        f"geography seed not found at {GEOGRAPHY_SEED_PATH}"
    )


def test_unit_seed_file_exists() -> None:
    assert UNIT_SEED_PATH.exists(), (
        f"unit seed not found at {UNIT_SEED_PATH}"
    )


def test_methodology_seed_file_exists() -> None:
    assert METHODOLOGY_SEED_PATH.exists(), (
        f"methodology seed not found at {METHODOLOGY_SEED_PATH}"
    )


# ---------------------------------------------------------------------------
# Seed YAML parses + minimum row counts
# ---------------------------------------------------------------------------


def test_geography_seed_minimum_row_count() -> None:
    rows = load_geography_yaml()
    assert len(rows) >= 50, (
        f"geography seed must contain >= 50 rows per CTO Phase 2 §2.3, "
        f"got {len(rows)}"
    )


def test_unit_seed_minimum_row_count() -> None:
    rows = load_unit_yaml()
    assert len(rows) >= 30, (
        f"unit seed must contain >= 30 rows per CTO Phase 2 §2.3, got "
        f"{len(rows)}"
    )


def test_methodology_seed_minimum_row_count() -> None:
    rows = load_methodology_yaml()
    assert len(rows) >= 25, (
        f"methodology seed must contain >= 25 rows per CTO Phase 2 §2.3, "
        f"got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# Every URN parses canonically
# ---------------------------------------------------------------------------


def test_every_geography_urn_parses() -> None:
    for row in load_geography_yaml():
        parsed = parse(row.urn)
        assert parsed.kind == "geo", (
            f"row {row.urn!r} parsed as kind={parsed.kind!r}"
        )
        assert parsed.geo_type == row.type, (
            f"row {row.urn!r} URN type {parsed.geo_type!r} != row.type "
            f"{row.type!r}"
        )


def test_every_unit_urn_parses() -> None:
    for row in load_unit_yaml():
        parsed = parse(row.urn)
        assert parsed.kind == "unit", (
            f"row {row.urn!r} parsed as kind={parsed.kind!r}"
        )


def test_every_methodology_urn_parses() -> None:
    for row in load_methodology_yaml():
        parsed = parse(row.urn)
        assert parsed.kind == "methodology", (
            f"row {row.urn!r} parsed as kind={parsed.kind!r}"
        )


# ---------------------------------------------------------------------------
# Loader: insert + idempotency against SQLite
# ---------------------------------------------------------------------------


def test_geography_loader_inserts_then_is_idempotent(
    sqlite_conn: sqlite3.Connection,
) -> None:
    first = load_geography(sqlite_conn)
    sqlite_conn.commit()

    assert first.count_inserted >= 50, (
        f"first run must insert >= 50 rows, got {first.count_inserted}"
    )
    assert first.count_skipped == 0, (
        f"first run must skip 0 rows, got {first.count_skipped}"
    )
    assert first.count_inserted + first.count_skipped == first.total_seen

    # Re-run must be a complete no-op.
    second = load_geography(sqlite_conn)
    sqlite_conn.commit()
    assert second.count_inserted == 0, (
        f"second run must be idempotent (insert 0), got "
        f"{second.count_inserted}"
    )
    assert second.count_skipped == first.total_seen, (
        f"second run must skip every row, got skipped="
        f"{second.count_skipped}"
    )

    # Verify final row count equals the seed total.
    cur = sqlite_conn.execute("SELECT COUNT(*) FROM geography")
    (db_count,) = cur.fetchone()
    assert db_count == first.total_seen


def test_unit_loader_inserts_then_is_idempotent(
    sqlite_conn: sqlite3.Connection,
) -> None:
    first = load_units(sqlite_conn)
    sqlite_conn.commit()
    assert first.count_inserted >= 30
    assert first.count_skipped == 0

    second = load_units(sqlite_conn)
    sqlite_conn.commit()
    assert second.count_inserted == 0
    assert second.count_skipped == first.total_seen

    cur = sqlite_conn.execute("SELECT COUNT(*) FROM unit")
    (db_count,) = cur.fetchone()
    assert db_count == first.total_seen


def test_methodology_loader_inserts_then_is_idempotent(
    sqlite_conn: sqlite3.Connection,
) -> None:
    first = load_methodologies(sqlite_conn)
    sqlite_conn.commit()
    assert first.count_inserted >= 25
    assert first.count_skipped == 0

    second = load_methodologies(sqlite_conn)
    sqlite_conn.commit()
    assert second.count_inserted == 0
    assert second.count_skipped == first.total_seen

    cur = sqlite_conn.execute("SELECT COUNT(*) FROM methodology")
    (db_count,) = cur.fetchone()
    assert db_count == first.total_seen


# ---------------------------------------------------------------------------
# CTO Phase 2 §2.3 minimum coverage — specific URNs MUST be present
# ---------------------------------------------------------------------------


def test_geography_cto_minimum_coverage(
    sqlite_conn: sqlite3.Connection,
) -> None:
    load_geography(sqlite_conn)
    sqlite_conn.commit()
    rows = sqlite_conn.execute(
        "SELECT urn, type FROM geography ORDER BY urn"
    ).fetchall()
    urns = {urn for urn, _ in rows}

    # Sample of CTO-mandated URNs across every geography type.
    required = {
        "urn:gl:geo:global:world",
        "urn:gl:geo:country:in",
        "urn:gl:geo:country:us",
        "urn:gl:geo:country:gb",
        "urn:gl:geo:country:de",
        "urn:gl:geo:country:fr",
        "urn:gl:geo:subregion:eu-27",
        "urn:gl:geo:subregion:asean",
        "urn:gl:geo:subregion:oecd",
        "urn:gl:geo:subregion:latam",
        "urn:gl:geo:state_or_province:us-tx",
        "urn:gl:geo:state_or_province:us-ca",
        "urn:gl:geo:state_or_province:in-mh",
        "urn:gl:geo:grid_zone:egrid-rfcw",
        "urn:gl:geo:grid_zone:egrid-serc",
        "urn:gl:geo:grid_zone:egrid-wecc",
        "urn:gl:geo:bidding_zone:de-lu",
        "urn:gl:geo:bidding_zone:gb",
        "urn:gl:geo:balancing_authority:caiso",
        "urn:gl:geo:balancing_authority:ercot",
        "urn:gl:geo:balancing_authority:pjm",
        "urn:gl:geo:basin:amazon",
    }
    missing = required - urns
    assert not missing, (
        f"geography seed missing CTO-required URNs: {sorted(missing)}"
    )


def test_unit_cto_minimum_coverage(
    sqlite_conn: sqlite3.Connection,
) -> None:
    load_units(sqlite_conn)
    sqlite_conn.commit()
    rows = sqlite_conn.execute(
        "SELECT urn, dimension FROM unit ORDER BY urn"
    ).fetchall()
    urns = {urn for urn, _ in rows}
    dimensions = {d for _, d in rows}

    # Per CTO §2.3 (Units / WS4) — required dimensions.
    required_dimensions = {
        "mass",
        "energy",
        "volume",
        "distance",
        "freight_activity",
        "currency",
        "composite_climate",
    }
    missing_dims = required_dimensions - dimensions
    assert not missing_dims, (
        f"unit seed missing dimensions: {sorted(missing_dims)}"
    )

    # Sample of CTO-mandated URNs.
    required = {
        "urn:gl:unit:kgco2e/kwh",
        "urn:gl:unit:kgco2e/kg",
        "urn:gl:unit:kgco2e/tkm",
        "urn:gl:unit:kgco2e/passenger-km",
        "urn:gl:unit:kgco2e/usd",
        "urn:gl:unit:kgco2e/eur",
        "urn:gl:unit:kgco2e/gbp",
        "urn:gl:unit:kgco2e/inr",
        "urn:gl:unit:kgco2e/mj",
        "urn:gl:unit:kgco2e/gj",
        "urn:gl:unit:kg",
        "urn:gl:unit:t",
        "urn:gl:unit:lb",
        "urn:gl:unit:kwh",
        "urn:gl:unit:mwh",
        "urn:gl:unit:mj",
        "urn:gl:unit:gj",
        "urn:gl:unit:btu",
        "urn:gl:unit:therm",
        "urn:gl:unit:l",
        "urn:gl:unit:m3",
        "urn:gl:unit:gal-us",
        "urn:gl:unit:gal-uk",
        "urn:gl:unit:km",
        "urn:gl:unit:mi",
        "urn:gl:unit:tkm",
        "urn:gl:unit:passenger-km",
        "urn:gl:unit:usd",
        "urn:gl:unit:eur",
        "urn:gl:unit:gbp",
        "urn:gl:unit:inr",
    }
    missing = required - urns
    assert not missing, (
        f"unit seed missing CTO-required URNs: {sorted(missing)}"
    )


def test_methodology_cto_minimum_coverage(
    sqlite_conn: sqlite3.Connection,
) -> None:
    load_methodologies(sqlite_conn)
    sqlite_conn.commit()
    rows = sqlite_conn.execute(
        "SELECT urn FROM methodology ORDER BY urn"
    ).fetchall()
    urns = {urn for (urn,) in rows}

    # All 15 Scope 3 categories.
    scope3_required = {
        f"urn:gl:methodology:ghgp-corporate-scope3-cat-{i}"
        for i in range(1, 16)
    }
    required = scope3_required | {
        "urn:gl:methodology:ghgp-corporate-scope1",
        "urn:gl:methodology:ghgp-corporate-scope2-location",
        "urn:gl:methodology:ghgp-corporate-scope2-market",
        "urn:gl:methodology:ipcc-tier-1-stationary-combustion",
        "urn:gl:methodology:ipcc-tier-2-stationary",
        "urn:gl:methodology:ipcc-tier-3-stationary",
        "urn:gl:methodology:ipcc-tier-1-mobile",
        "urn:gl:methodology:ipcc-tier-1-fugitive",
        "urn:gl:methodology:eu-cbam-default",
        "urn:gl:methodology:eu-cbam-actual",
        "urn:gl:methodology:epa-egrid-subregion-2024",
        "urn:gl:methodology:india-cea-baseline",
        "urn:gl:methodology:glec-framework-v3",
        "urn:gl:methodology:iso-14083-2023",
        "urn:gl:methodology:pcaf-financed-emissions-v1",
        "urn:gl:methodology:ecoinvent-cutoff",
        "urn:gl:methodology:ecoinvent-apos",
        "urn:gl:methodology:ecoinvent-consequential",
    }
    missing = required - urns
    assert not missing, (
        f"methodology seed missing CTO-required URNs: {sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Seed-source marker — every Phase 2 row carries the canonical tag
# ---------------------------------------------------------------------------


def test_every_phase2_row_carries_seed_source_marker(
    sqlite_conn: sqlite3.Connection,
) -> None:
    """V501 adds a seed_source column; the loaders MUST set it to
    'phase2_v0_1' so the data migration's downgrade can roll back ONLY
    the rows it inserted (production-ingested rows leave seed_source
    NULL and survive the rollback)."""
    load_geography(sqlite_conn)
    load_units(sqlite_conn)
    load_methodologies(sqlite_conn)
    sqlite_conn.commit()

    for table in ("geography", "unit", "methodology"):
        cur = sqlite_conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE seed_source = ?",
            (PHASE2_SEED_SOURCE,),
        )
        (n_tagged,) = cur.fetchone()
        cur = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}")
        (n_total,) = cur.fetchone()
        assert n_tagged == n_total, (
            f"{table}: {n_tagged}/{n_total} rows carry "
            f"seed_source={PHASE2_SEED_SOURCE!r}; expected all"
        )


# ---------------------------------------------------------------------------
# Cross-table sanity: geography parent_urn FKs all resolve internally
# ---------------------------------------------------------------------------


def test_geography_parent_urns_all_resolve(
    sqlite_conn: sqlite3.Connection,
) -> None:
    load_geography(sqlite_conn)
    sqlite_conn.commit()
    rows = sqlite_conn.execute(
        "SELECT urn, parent_urn FROM geography "
        "WHERE parent_urn IS NOT NULL"
    ).fetchall()
    all_urns = {
        urn
        for (urn,) in sqlite_conn.execute(
            "SELECT urn FROM geography"
        ).fetchall()
    }
    orphans = [
        (urn, parent_urn)
        for urn, parent_urn in rows
        if parent_urn not in all_urns
    ]
    assert not orphans, (
        f"geography rows reference unknown parent URNs: {orphans[:5]}"
    )
