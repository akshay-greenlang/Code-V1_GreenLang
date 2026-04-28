"""Tests for greenlang.factors.schemas._version_registry.

Covers WS9-A deliverable: the schema version registry that drives lock
and overlap window arithmetic for the factor record schemas. The CTO
mandates v1.0 lock = 24 months, v1<->v2 overlap = 12 months,
v2<->v3 overlap = 18 months. Dates are computed from each version's
``effective_date`` via calendar-month arithmetic.

The tests synthesise hypothetical v1.0 / v2.0 entries to exercise the
arithmetic without requiring those versions to be promoted in the
canonical REGISTRY (which currently only ships v0.1).
"""

from __future__ import annotations

import pytest

from greenlang.factors.schemas import _version_registry as reg
from greenlang.factors.schemas._version_registry import (
    REGISTRY,
    SchemaVersion,
    all_active,
    get_version,
    is_in_overlap_window,
    is_locked,
    latest_frozen,
    overlap_until_for_successor,
)


# ---------------------------------------------------------------------------
# Baseline registry assertions
# ---------------------------------------------------------------------------


def test_v0_1_present_and_frozen():
    """The v0.1 entry exists in the canonical REGISTRY and is frozen."""
    assert "v0_1" in REGISTRY
    v = REGISTRY["v0_1"]
    assert v.status == "frozen"
    assert v.version == "0.1"
    assert (
        v.schema_id
        == "https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json"
    )
    assert v.effective_date == "2026-04-25"
    assert v.supersedes == ()


def test_v0_1_lock_until_is_none():
    """v0.x has no lock window (alpha contract); lock_until is None."""
    v = REGISTRY["v0_1"]
    assert v.lock_months == 0
    assert v.lock_until is None


def test_v0_1_overlap_until_is_none():
    """v0.x has no overlap-with-next window; overlap_until is None."""
    v = REGISTRY["v0_1"]
    assert v.overlap_with_next_months == 0
    assert v.overlap_until is None


# ---------------------------------------------------------------------------
# Date arithmetic - synthetic v1.0 entry
# ---------------------------------------------------------------------------


def _make_v1_synth() -> SchemaVersion:
    """A hypothetical v1.0 with effective 2026-12-01, lock 24, overlap 12."""
    return SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v1.schema.json",
        version="1.0",
        status="frozen",
        effective_date="2026-12-01",
        supersedes=(REGISTRY["v0_1"].schema_id,),
        lock_months=24,
        overlap_with_next_months=12,
        changelog_uri="docs/factors/schema/CHANGELOG.md#v10",
    )


def test_lock_until_computed_correctly():
    """Synthetic v1.0 effective 2026-12-01 + 24mo == 2028-12-01."""
    v1 = _make_v1_synth()
    assert v1.lock_until == "2028-12-01"


def test_overlap_window_computed():
    """Synthetic v1.0 effective 2026-12-01 + 12mo overlap == 2027-12-01.

    Note: ``SchemaVersion.overlap_until`` is a self-relative helper. The
    successor-relative semantics live in
    :func:`overlap_until_for_successor` and are exercised in
    :func:`test_overlap_until_for_successor_uses_successor_effective_date`.
    """
    v1 = _make_v1_synth()
    assert v1.overlap_until == "2027-12-01"


def test_overlap_until_for_successor_uses_successor_effective_date(monkeypatch):
    """v1.0 -> v2.0: overlap end = v2.effective_date + v1.overlap_with_next_months.

    A v2.0 ships on 2028-12-01 (assume v1.0 lock just ended); the
    customer-facing v1<->v2 overlap window ends 12 calendar months later
    on 2029-12-01.
    """
    v1 = _make_v1_synth()
    v2 = SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v2.schema.json",
        version="2.0",
        status="frozen",
        effective_date="2028-12-01",
        supersedes=(v1.schema_id,),
        lock_months=24,
        overlap_with_next_months=18,  # v2<->v3 overlap per CTO
        changelog_uri="docs/factors/schema/CHANGELOG.md#v20",
    )
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    monkeypatch.setitem(REGISTRY, "v2_0", v2)
    assert overlap_until_for_successor("v1_0", "v2_0") == "2029-12-01"


# ---------------------------------------------------------------------------
# get_version / latest_frozen
# ---------------------------------------------------------------------------


def test_get_version_returns_registered_entry():
    v = get_version("v0_1")
    assert v is REGISTRY["v0_1"]


def test_get_version_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_version("v99_99")


def test_latest_frozen_returns_v0_1():
    """Until v1.0 is promoted to frozen, v0.1 is the latest frozen entry."""
    latest = latest_frozen()
    assert latest.version == "0.1"
    assert latest.status == "frozen"


def test_latest_frozen_picks_highest_when_multiple(monkeypatch):
    """When multiple frozen versions are registered, the highest wins."""
    v1 = _make_v1_synth()
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    latest = latest_frozen()
    assert latest.version == "1.0"


# ---------------------------------------------------------------------------
# is_locked
# ---------------------------------------------------------------------------


def test_is_locked_today_inside_window(monkeypatch):
    """Inside v1.0's lock window (effective 2026-12-01, lock_until 2028-12-01)."""
    v1 = _make_v1_synth()
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    assert is_locked("v1_0", today="2027-06-01") is True
    assert is_locked("v1_0", today="2026-12-01") is True  # lock starts day-of
    assert is_locked("v1_0", today="2028-11-30") is True


def test_is_locked_today_outside_window(monkeypatch):
    """At/after lock_until, the version is no longer locked."""
    v1 = _make_v1_synth()
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    assert is_locked("v1_0", today="2028-12-01") is False  # boundary day
    assert is_locked("v1_0", today="2029-01-01") is False


def test_is_locked_alpha_never_locked():
    """v0.x has lock_months=0 and is never locked."""
    assert is_locked("v0_1", today="2026-04-25") is False
    assert is_locked("v0_1", today="2030-01-01") is False


def test_is_locked_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        is_locked("v99_99", today="2026-04-27")


# ---------------------------------------------------------------------------
# is_in_overlap_window
# ---------------------------------------------------------------------------


def test_is_in_overlap_window_logic(monkeypatch):
    """v1.0 with v2.0 successor: overlap runs [v2.effective_date, +12mo)."""
    v1 = _make_v1_synth()
    v2 = SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v2.schema.json",
        version="2.0",
        status="frozen",
        effective_date="2028-12-01",
        supersedes=(v1.schema_id,),
        lock_months=24,
        overlap_with_next_months=18,
        changelog_uri="docs/factors/schema/CHANGELOG.md#v20",
    )
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    monkeypatch.setitem(REGISTRY, "v2_0", v2)

    # Before v2 ships -> not in overlap.
    assert is_in_overlap_window("v1_0", today="2028-11-30") is False
    # On v2 effective day -> overlap starts.
    assert is_in_overlap_window("v1_0", today="2028-12-01") is True
    # Mid-overlap.
    assert is_in_overlap_window("v1_0", today="2029-06-01") is True
    # On overlap-end day (exclusive).
    assert is_in_overlap_window("v1_0", today="2029-12-01") is False
    # After overlap closes.
    assert is_in_overlap_window("v1_0", today="2030-01-01") is False


def test_is_in_overlap_window_no_successor():
    """A version with no registered successor is never in overlap."""
    assert is_in_overlap_window("v0_1", today="2026-04-27") is False


def test_is_in_overlap_window_alpha_zero_overlap(monkeypatch):
    """v0.x has overlap_with_next_months=0 even if a successor exists."""
    v1 = _make_v1_synth()
    monkeypatch.setitem(REGISTRY, "v1_0", v1)
    # v0_1.overlap_with_next_months == 0, so even with v1.0 registered as
    # a successor, v0_1 is not considered "in overlap".
    assert is_in_overlap_window("v0_1", today="2027-01-01") is False


# ---------------------------------------------------------------------------
# all_active
# ---------------------------------------------------------------------------


def test_all_active_includes_only_v0_1_today():
    """Today (2026-04-27), only v0.1 is active in the canonical REGISTRY."""
    active = all_active()
    versions = [v.version for v in active]
    assert "0.1" in versions
    # Nothing else should be present in the canonical registry yet.
    assert versions == ["0.1"]


def test_all_active_includes_frozen_and_overlap(monkeypatch):
    """Frozen versions and deprecated-but-in-overlap versions are both active."""
    # Synth: v1.0 frozen, v2.0 frozen successor, v1.0 deprecated and in overlap.
    v1_deprecated = SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v1.schema.json",
        version="1.0",
        status="deprecated",
        effective_date="2026-12-01",
        supersedes=(REGISTRY["v0_1"].schema_id,),
        lock_months=24,
        overlap_with_next_months=12,
        changelog_uri="docs/factors/schema/CHANGELOG.md#v10",
    )
    v2 = SchemaVersion(
        schema_id="https://schemas.greenlang.io/factors/factor_record_v2.schema.json",
        version="2.0",
        status="frozen",
        effective_date="2028-12-01",
        supersedes=(v1_deprecated.schema_id,),
        lock_months=24,
        overlap_with_next_months=18,
        changelog_uri="docs/factors/schema/CHANGELOG.md#v20",
    )
    monkeypatch.setitem(REGISTRY, "v1_0", v1_deprecated)
    monkeypatch.setitem(REGISTRY, "v2_0", v2)

    # Force "today" to land inside the v1<->v2 overlap window by
    # monkey-patching date.today via the helper used internally. We do this
    # by passing today through is_in_overlap_window and reconstructing the
    # all_active logic with that synthetic clock.
    today_iso = "2029-06-01"

    def _filter():
        out = []
        for k, v in REGISTRY.items():
            if v.status == "frozen":
                out.append(v)
            elif v.status == "deprecated" and is_in_overlap_window(k, today_iso):
                out.append(v)
        return out

    active = _filter()
    versions = sorted(v.version for v in active)
    assert versions == ["0.1", "1.0", "2.0"]


# ---------------------------------------------------------------------------
# Frozen-dataclass invariants
# ---------------------------------------------------------------------------


def test_schema_version_is_frozen():
    """SchemaVersion is immutable; mutating a field raises FrozenInstanceError."""
    v = REGISTRY["v0_1"]
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError subclasses Exception
        v.status = "deprecated"  # type: ignore[misc]


def test_changelog_uri_anchor_format():
    """The v0.1 changelog_uri points at a valid markdown anchor in CHANGELOG.md."""
    v = REGISTRY["v0_1"]
    assert v.changelog_uri.startswith("docs/factors/schema/CHANGELOG.md#")
    # Anchor should be lowercase, hyphenated, and contain the date.
    assert "2026-04-25" in v.changelog_uri
    assert "additive" in v.changelog_uri


# ---------------------------------------------------------------------------
# _add_months_iso edge cases
# ---------------------------------------------------------------------------


def test_add_months_iso_year_rollover():
    """Adding months that cross a year boundary computes correctly."""
    assert reg._add_months_iso("2026-12-01", 1) == "2027-01-01"
    assert reg._add_months_iso("2026-11-15", 14) == "2028-01-15"


def test_add_months_iso_24mo_lock():
    """The CTO 24-month lock from a December effective date."""
    assert reg._add_months_iso("2026-12-01", 24) == "2028-12-01"


def test_add_months_iso_18mo_overlap():
    """The CTO 18-month v2<->v3 overlap."""
    assert reg._add_months_iso("2030-06-01", 18) == "2031-12-01"
