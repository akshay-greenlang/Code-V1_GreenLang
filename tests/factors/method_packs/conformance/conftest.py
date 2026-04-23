# -*- coding: utf-8 -*-
"""Shared fixtures for method-pack conformance tests (MP15, Wave 4-G).

Conformance tests are the gate that blocks a pack from being promoted
to ``certified``. Each test fixture constructs a minimal candidate
record, runs it against the pack's method_profile, and asserts that the
resolver's ``chosen_factor.id`` matches an expected regex pattern (not
an exact value — that would require cooked numeric results).

Per spec: "Conformance tests must NOT cook expected results. If the
resolver returns a different factor, that's signal for methodology
review OR resolver tuning, not a test bug."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs import registry as _pack_registry
from greenlang.factors.resolution.engine import (
    ResolutionEngine,
    ResolutionError,
)
from greenlang.factors.resolution.request import ResolutionRequest


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot the global pack registry before each test; restore after."""
    snapshot = dict(_pack_registry._packs)
    try:
        yield
    finally:
        _pack_registry._packs.clear()
        _pack_registry._packs.update(snapshot)


@dataclass
class ConformanceRecord:
    """Minimal fake record used by conformance tests.

    Mirrors the attribute paths the resolver + SelectionRule.accepts()
    walk; callers supply only the fields their test needs to check.
    """

    factor_id: str
    factor_family: str = FactorFamily.EMISSIONS.value
    formula_type: str = FormulaType.DIRECT_FACTOR.value
    factor_status: str = "certified"
    activity_category: Optional[str] = None
    source_id: str = "conformance_source"
    unit: Optional[str] = None
    verification: Any = None
    primary_data_flag: str = "unknown"


def _make_candidate_source(by_step: Dict[str, List[ConformanceRecord]]):
    """Build a CandidateSource that yields records per cascade step label."""

    def _source(request: ResolutionRequest, label: str) -> List[Any]:
        return list(by_step.get(label, []))

    return _source


def resolve_case(
    method_profile: MethodProfile,
    jurisdiction: str,
    records_by_step: Dict[str, List[ConformanceRecord]],
    activity: str = "test_activity",
):
    """Execute the resolver against a canned candidate set.

    Returns ``(resolved, error)`` — exactly one is not None. The caller
    asserts its own expectation (pattern match on factor id, or that a
    ``FactorCannotResolveSafelyError`` fired).
    """
    engine = ResolutionEngine(
        candidate_source=_make_candidate_source(records_by_step)
    )
    request = ResolutionRequest(
        activity=activity,
        method_profile=method_profile,
        jurisdiction=jurisdiction,
    )
    try:
        resolved = engine.resolve(request)
        return resolved, None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def assert_chosen_matches(resolved, expected_pattern: str) -> None:
    """Assert the resolver's chosen_factor_id matches the given regex.

    Per spec: we check the *pattern* of the chosen factor id (e.g.
    ``r"^EF:DEFRA:.*:GB:\\d{4}$"``), not the exact value. A mismatch is
    signal for methodology review or resolver tuning — NOT a test bug.
    """
    assert resolved is not None, "resolver returned None without an error"
    fid = getattr(resolved, "chosen_factor_id", None) or getattr(
        getattr(resolved, "chosen_factor", None), "id", None
    )
    assert fid, f"resolved has no chosen_factor_id: {resolved!r}"
    assert re.match(expected_pattern, str(fid)), (
        f"chosen_factor_id {fid!r} does not match expected pattern "
        f"{expected_pattern!r} — methodology review / resolver tuning needed"
    )
