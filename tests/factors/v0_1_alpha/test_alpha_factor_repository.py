# -*- coding: utf-8 -*-
"""Wave D / TaskCreate #31 / WS9-T5 — :class:`AlphaFactorRepository` tests.

Covers:
  * publish() runs the AlphaProvenanceGate
  * publish() refuses to overwrite (immutability)
  * get_by_urn() round-trip equality (no coercion)
  * each filter narrows the result set
  * cursor pagination across multiple pages without overlap
  * list_sources() reads from the registry
  * list_packs() filters by source_urn
  * SQL-injection guards on filter values
  * SQLite schema present + record_jsonb column carries the verbatim blob
"""
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pytest

from greenlang.factors.quality.alpha_provenance_gate import (
    AlphaProvenanceGateError,
)
from greenlang.factors.repositories import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)

from tests.factors.v0_1_alpha._e2e_helpers import good_ipcc_ar6_factor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo() -> AlphaFactorRepository:
    """Fresh in-memory repository per test."""
    r = AlphaFactorRepository(dsn="sqlite:///:memory:")
    yield r
    r.close()


def _record(
    *,
    leaf: str,
    source: str = "ipcc-ar6",
    pack: str = "tier-1-defaults",
    category: str = "fuel",
    geo: str = "urn:gl:geo:global:world",
    vintage_start: str = "2021-01-01",
    vintage_end: str = "2099-12-31",
    published_at: str = "2026-04-25T12:00:00Z",
) -> Dict[str, Any]:
    """Mint a v0.1 factor whose URN is unique by ``leaf``."""
    base = good_ipcc_ar6_factor()
    base = copy.deepcopy(base)
    base["urn"] = f"urn:gl:factor:{source}:stationary-combustion:{leaf}:v1"
    base["factor_id_alias"] = f"EF:{source}:stationary-combustion:{leaf}:v1"
    base["source_urn"] = f"urn:gl:source:{source}"
    base["factor_pack_urn"] = f"urn:gl:pack:{source}:{pack}:2021.0"
    base["category"] = category
    base["geography_urn"] = geo
    base["vintage_start"] = vintage_start
    base["vintage_end"] = vintage_end
    base["published_at"] = published_at
    return base


# ---------------------------------------------------------------------------
# publish() — gate, immutability, persistence
# ---------------------------------------------------------------------------


def test_publish_runs_alpha_provenance_gate(repo: AlphaFactorRepository) -> None:
    """A record that fails the gate must NOT be persisted."""
    bad = good_ipcc_ar6_factor()
    bad.pop("extraction")  # required block — gate must fail
    with pytest.raises(AlphaProvenanceGateError):
        repo.publish(bad)
    assert repo.count() == 0


def test_publish_returns_urn_on_success(repo: AlphaFactorRepository) -> None:
    rec = _record(leaf="ng-residential")
    assert repo.publish(rec) == rec["urn"]
    assert repo.count() == 1


def test_publish_rejects_duplicate_urn(repo: AlphaFactorRepository) -> None:
    rec = _record(leaf="ng-residential")
    repo.publish(rec)
    with pytest.raises(FactorURNAlreadyExistsError) as excinfo:
        repo.publish(rec)
    assert excinfo.value.urn == rec["urn"]
    assert repo.count() == 1


def test_publish_rejects_record_with_blank_urn(repo: AlphaFactorRepository) -> None:
    """Schema gate already covers this; the repo defensively re-checks."""
    bad = good_ipcc_ar6_factor()
    bad["urn"] = ""
    with pytest.raises(AlphaProvenanceGateError):
        repo.publish(bad)


# ---------------------------------------------------------------------------
# get_by_urn() — strict round-trip equality
# ---------------------------------------------------------------------------


def test_get_by_urn_exact_round_trip(repo: AlphaFactorRepository) -> None:
    """The dict returned by get_by_urn() must equal the published dict."""
    rec = good_ipcc_ar6_factor()
    repo.publish(rec)
    fetched = repo.get_by_urn(rec["urn"])
    assert fetched == rec
    # Provenance / review blocks survive verbatim.
    assert fetched["extraction"] == rec["extraction"]
    assert fetched["review"] == rec["review"]


def test_get_by_urn_unknown_returns_none(repo: AlphaFactorRepository) -> None:
    assert repo.get_by_urn("urn:gl:factor:never-published") is None


def test_get_by_urn_blank_returns_none(repo: AlphaFactorRepository) -> None:
    assert repo.get_by_urn("") is None
    assert repo.get_by_urn(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# list_factors() filters — each filter narrows the set
# ---------------------------------------------------------------------------


@pytest.fixture()
def populated_repo() -> AlphaFactorRepository:
    """Repository with a varied corpus for filter tests.

    Mix of source/pack/category/geo + ascending publish times so cursors
    are deterministic.
    """
    r = AlphaFactorRepository(dsn="sqlite:///:memory:")
    rows: List[Dict[str, Any]] = [
        _record(
            leaf="ng-residential",
            source="ipcc-ar6",
            pack="tier-1-defaults",
            category="fuel",
            geo="urn:gl:geo:global:world",
            vintage_start="2021-01-01",
            vintage_end="2099-12-31",
            published_at="2026-04-25T12:00:00Z",
        ),
        _record(
            leaf="diesel-mobile",
            source="ipcc-ar6",
            pack="tier-1-defaults",
            category="fuel",
            geo="urn:gl:geo:country:gb",
            vintage_start="2022-01-01",
            vintage_end="2099-12-31",
            published_at="2026-04-25T12:01:00Z",
        ),
        _record(
            leaf="r410a",
            source="epa-hub",
            pack="default",
            category="refrigerant",
            geo="urn:gl:geo:country:us",
            vintage_start="2020-01-01",
            vintage_end="2099-12-31",
            published_at="2026-04-25T12:02:00Z",
        ),
        _record(
            leaf="grid-uk",
            source="desnz",
            pack="ghg-conversion-2024",
            category="grid_intensity",
            geo="urn:gl:geo:country:gb",
            vintage_start="2024-01-01",
            vintage_end="2024-12-31",
            published_at="2026-04-25T12:03:00Z",
        ),
        _record(
            leaf="cbam-steel",
            source="cbam-default",
            pack="default",
            category="cbam_default",
            geo="urn:gl:geo:subregion:eu-27",
            vintage_start="2023-01-01",
            vintage_end="2099-12-31",
            published_at="2026-04-25T12:04:00Z",
        ),
    ]
    for row in rows:
        r.publish(row)
    yield r
    r.close()


@pytest.mark.parametrize(
    "kwarg, value, expected_count",
    [
        ("source_urn", "urn:gl:source:ipcc-ar6", 2),
        ("pack_urn", "urn:gl:pack:ipcc-ar6:tier-1-defaults:2021.0", 2),
        ("category", "refrigerant", 1),
        ("geography_urn", "urn:gl:geo:country:gb", 2),
        ("vintage_start_after", "2023-01-01", 1),  # only grid-uk (2024)
    ],
)
def test_list_factors_filter_narrows_results(
    populated_repo: AlphaFactorRepository,
    kwarg: str,
    value: str,
    expected_count: int,
) -> None:
    rows, _ = populated_repo.list_factors(**{kwarg: value})
    assert len(rows) == expected_count, (
        f"{kwarg}={value!r} should narrow to {expected_count} rows; got "
        f"{len(rows)} -> {[r['urn'] for r in rows]}"
    )


def test_list_factors_no_filter_returns_all(
    populated_repo: AlphaFactorRepository,
) -> None:
    rows, _ = populated_repo.list_factors(limit=100)
    assert len(rows) == 5


def test_list_factors_vintage_end_before_filter(
    populated_repo: AlphaFactorRepository,
) -> None:
    rows, _ = populated_repo.list_factors(vintage_end_before="2030-01-01")
    # Only grid-uk has vintage_end < 2030-01-01 (it ends 2024-12-31).
    assert len(rows) == 1
    assert rows[0]["geography_urn"] == "urn:gl:geo:country:gb"
    assert rows[0]["category"] == "grid_intensity"


def test_list_factors_compound_filters_and_combine(
    populated_repo: AlphaFactorRepository,
) -> None:
    rows, _ = populated_repo.list_factors(
        source_urn="urn:gl:source:ipcc-ar6",
        category="fuel",
        geography_urn="urn:gl:geo:country:gb",
    )
    assert len(rows) == 1
    assert "diesel-mobile" in rows[0]["urn"]


def test_list_factors_sort_is_published_at_desc(
    populated_repo: AlphaFactorRepository,
) -> None:
    rows, _ = populated_repo.list_factors(limit=100)
    timestamps = [r["published_at"] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


def test_cursor_pagination_100_records_two_pages_no_overlap() -> None:
    """100 records / 50 per page -> 2 pages, all retrieved exactly once."""
    r = AlphaFactorRepository(dsn="sqlite:///:memory:")
    try:
        # Publish 100 unique records with strictly-monotonic publish times.
        urns_published: List[str] = []
        for i in range(100):
            rec = _record(
                leaf=f"ef-{i:03d}",
                published_at=f"2026-04-25T13:{i // 60:02d}:{i % 60:02d}Z",
            )
            urns_published.append(rec["urn"])
            r.publish(rec)
        assert r.count() == 100

        # Page 1.
        page1, cursor1 = r.list_factors(limit=50)
        assert len(page1) == 50
        assert cursor1 is not None

        # Page 2.
        page2, cursor2 = r.list_factors(limit=50, cursor=cursor1)
        assert len(page2) == 50
        # No 3rd page expected (exhausted).
        assert cursor2 is None

        # No overlap, full coverage.
        urns_seen = {row["urn"] for row in page1} | {row["urn"] for row in page2}
        assert len(urns_seen) == 100
        assert urns_seen == set(urns_published)
    finally:
        r.close()


def test_cursor_pagination_invalid_cursor_starts_from_top(
    populated_repo: AlphaFactorRepository,
) -> None:
    """An unparseable cursor is treated as start-of-list."""
    rows, _ = populated_repo.list_factors(cursor="not-a-real-cursor", limit=100)
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# list_sources / list_packs
# ---------------------------------------------------------------------------


def test_list_sources_reads_registry(repo: AlphaFactorRepository) -> None:
    """list_sources() yields the alpha-flagged registry rows."""
    rows = repo.list_sources()
    assert isinstance(rows, list)
    # The 6 canonical alpha sources are expected.
    source_ids = {str(r.get("source_id")) for r in rows}
    assert "ipcc_2006_nggi" in source_ids


def test_list_packs_filters_by_source_urn(repo: AlphaFactorRepository) -> None:
    """list_packs(source_urn=...) returns only that source's packs."""
    sources = repo.list_sources()
    if not sources:
        pytest.skip("alpha source registry unavailable in this environment")
    target = sources[0]
    target_urn = str(target.get("urn") or "")
    if not target_urn:
        pytest.skip("registry rows lack urn — env-specific")
    packs = repo.list_packs(source_urn=target_urn)
    assert all(p["source_urn"] == target_urn for p in packs)


def test_list_packs_no_filter_yields_one_per_source(
    repo: AlphaFactorRepository,
) -> None:
    sources = repo.list_sources()
    packs = repo.list_packs()
    assert len(packs) == len(sources)
    # Every pack URN starts urn:gl:pack:.
    assert all(p["urn"].startswith("urn:gl:pack:") for p in packs)


# ---------------------------------------------------------------------------
# Security / hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evil_value",
    [
        "';DROP TABLE alpha_factors_v0_1;--",
        "' OR 1=1 --",
        '"; DELETE FROM alpha_factors_v0_1; --',
        "urn:gl:source:ipcc-ar6'); --",
    ],
)
def test_list_factors_filter_values_are_bind_params(
    populated_repo: AlphaFactorRepository, evil_value: str
) -> None:
    """SQL-injection payloads in filter values must be inert.

    Each query MUST return no rows (the value doesn't match any column)
    AND the table MUST still be intact afterwards (count unchanged).
    """
    before = populated_repo.count()
    rows, _ = populated_repo.list_factors(source_urn=evil_value)
    assert rows == []
    after = populated_repo.count()
    assert before == after == 5


def test_immutability_documented_in_class_docstring() -> None:
    """The repo class docstring MUST document the immutability constraint.

    SQLite cannot prevent raw UPDATE statements at the engine level on a
    JSONB-stored blob, so the trust boundary is the repository class
    itself — :meth:`publish` is the only write path. This test is a
    living-doc guard.
    """
    docs = AlphaFactorRepository.__doc__ or ""
    # Repo module docstring carries the full statement; check both.
    import greenlang.factors.repositories.alpha_v0_1_repository as mod
    full = (mod.__doc__ or "") + "\n" + docs
    assert "immutab" in full.lower(), (
        "AlphaFactorRepository docstring must document the immutability "
        "constraint."
    )


def test_record_jsonb_column_holds_verbatim_blob() -> None:
    """The on-disk ``record_jsonb`` column equals the published dict
    after json.loads — no field-stripping or coercion.
    """
    r = AlphaFactorRepository(dsn="sqlite:///:memory:")
    try:
        rec = good_ipcc_ar6_factor()
        r.publish(rec)
        conn = r._connect()
        row = conn.execute(
            "SELECT record_jsonb FROM alpha_factors_v0_1 WHERE urn = ?",
            (rec["urn"],),
        ).fetchone()
        assert row is not None
        decoded = json.loads(row["record_jsonb"])
        assert decoded == rec
    finally:
        r.close()


def test_close_is_idempotent() -> None:
    r = AlphaFactorRepository(dsn="sqlite:///:memory:")
    r.close()
    r.close()  # Should not raise.


def test_filesystem_sqlite_dsn_creates_parent_dir(tmp_path) -> None:
    """A nested sqlite path triggers parent-dir creation."""
    nested = tmp_path / "a" / "b" / "c" / "alpha.db"
    dsn = f"sqlite:///{nested.as_posix()}"
    r = AlphaFactorRepository(dsn=dsn)
    try:
        r.publish(good_ipcc_ar6_factor())
        assert nested.exists()
        assert r.count() == 1
    finally:
        r.close()
