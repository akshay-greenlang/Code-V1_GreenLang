# -*- coding: utf-8 -*-
"""Tests for the ``cannot_resolve_safely`` contract (MP3/MP4/MP5 wave 1).

Covers:

* The default behaviour (``RAISE_NO_SAFE_MATCH``) — raises
  :class:`FactorCannotResolveSafelyError` when no candidate survives the
  selection gate.
* Blocking tier-7 global-default: with
  ``global_default_tier_allowed=False`` the engine refuses to return a
  rank-7 factor even when one exists.
* Legacy opt-in: a pack configured with
  ``ALLOW_GLOBAL_DEFAULT`` + ``global_default_tier_allowed=True`` falls
  through to the legacy :class:`ResolutionError` / tier-7 behaviour.

These tests exercise the ``ResolutionEngine`` directly with an in-memory
candidate source so the contract is verified end-to-end without any DB
or HTTP dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs import (
    CannotResolveAction,
    FactorCannotResolveSafelyError,
    MethodPack,
    SelectionRule,
    register_pack,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
)
from greenlang.factors.method_packs import registry as _pack_registry


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot the global pack registry before each test and restore it
    afterwards. This lets individual tests overwrite real packs with the
    synthetic ``_register_test_pack`` fixtures without bleeding state into
    the rest of the suite.
    """
    snapshot = dict(_pack_registry._packs)
    try:
        yield
    finally:
        _pack_registry._packs.clear()
        _pack_registry._packs.update(snapshot)
from greenlang.factors.resolution.engine import (
    ResolutionEngine,
    ResolutionError,
)
from greenlang.factors.resolution.request import ResolutionRequest


# ---------------------------------------------------------------------------
# Fixtures: fake records + candidate source + purpose-built MethodPacks
# ---------------------------------------------------------------------------


@dataclass
class _FakeRecord:
    """Minimal stand-in for ``EmissionFactorRecord`` used by the engine."""

    factor_id: str
    factor_family: str = FactorFamily.EMISSIONS.value
    formula_type: str = FormulaType.DIRECT_FACTOR.value
    factor_status: str = "certified"
    activity_category: Optional[str] = None
    source_id: str = "test_source"
    unit: Optional[str] = None


def _make_candidate_source(by_step: Dict[str, List[_FakeRecord]]):
    """Return a CandidateSource that yields records per cascade step label."""

    def _source(request: ResolutionRequest, label: str) -> List[Any]:
        return list(by_step.get(label, []))

    return _source


def _register_test_pack(
    *,
    profile: MethodProfile,
    cannot_resolve_action: CannotResolveAction,
    global_default_tier_allowed: bool,
    included_activity_categories: frozenset = frozenset(),
    excluded_activity_categories: frozenset = frozenset(),
) -> MethodPack:
    """Build and register a tiny MethodPack for the tests.

    We reuse an existing ``MethodProfile`` enum value so the registry
    accepts the registration; this also means each test must choose a
    profile that is NOT used by a real pack above OR we overwrite it.
    """
    pack = MethodPack(
        profile=profile,
        name=f"Test Pack {profile.value}",
        description="Synthetic pack for cannot_resolve_safely tests.",
        selection_rule=SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            allowed_statuses=("certified",),
            included_activity_categories=included_activity_categories,
            excluded_activity_categories=excluded_activity_categories,
        ),
        boundary_rule=BoundaryRule(
            allowed_scopes=("1",),
            allowed_boundaries=("combustion",),
            biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
            market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
        ),
        gwp_basis="IPCC_AR6_100",
        region_hierarchy=DEFAULT_FALLBACK,
        deprecation=DeprecationRule(max_age_days=365, grace_period_days=90),
        reporting_labels=("GHG_Protocol",),
        audit_text_template="test",
        cannot_resolve_action=cannot_resolve_action,
        global_default_tier_allowed=global_default_tier_allowed,
    )
    register_pack(pack)
    return pack


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCannotResolveSafely:
    """Core ``cannot_resolve_safely`` contract tests."""

    def test_strict_pack_raises_when_no_candidate_exists(self, monkeypatch):
        """With no candidates at any tier and strict mode, the engine raises
        :class:`FactorCannotResolveSafelyError` carrying a structured payload."""

        pack = _register_test_pack(
            profile=MethodProfile.CORPORATE_SCOPE1,
            cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
            global_default_tier_allowed=False,
        )
        # Sanity: the registration wiring uses CORPORATE_SCOPE1; the next
        # test uses a different profile to avoid bleed-through.
        assert pack.cannot_resolve_action == CannotResolveAction.RAISE_NO_SAFE_MATCH

        engine = ResolutionEngine(candidate_source=_make_candidate_source({}))
        request = ResolutionRequest(
            activity="test_activity",
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
        )

        with pytest.raises(FactorCannotResolveSafelyError) as exc_info:
            engine.resolve(request)

        err = exc_info.value
        assert err.pack_id == MethodProfile.CORPORATE_SCOPE1.value
        assert err.method_profile == MethodProfile.CORPORATE_SCOPE1.value
        assert err.reason in {"no_candidate", "global_default_blocked"}
        assert err.evaluated_candidates_count == 0
        # Structured context carries the step trace for forensic debugging.
        assert "evaluated_steps" in err.context
        assert any(
            "skipped_global_default_blocked" in step
            for step in err.context["evaluated_steps"]
        )

    def test_global_default_tier_blocked_even_when_candidate_exists(self):
        """Pack refuses to return a tier-7 global-default factor when
        ``global_default_tier_allowed`` is False, even if a candidate
        exists at that tier."""

        # Use CORPORATE_SCOPE2_LOCATION so the previous test's pack
        # registration doesn't interfere.
        _register_test_pack(
            profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
            global_default_tier_allowed=False,
        )

        # Expose a candidate only at the global_default step.
        global_only = _FakeRecord(factor_id="ef_global_default_1")
        engine = ResolutionEngine(
            candidate_source=_make_candidate_source({"global_default": [global_only]})
        )
        request = ResolutionRequest(
            activity="test_activity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="US",
        )

        with pytest.raises(FactorCannotResolveSafelyError) as exc_info:
            engine.resolve(request)

        err = exc_info.value
        assert err.reason == "global_default_blocked"
        # The global_default step is skipped before we see the candidate,
        # so evaluated_candidates_count stays at 0.
        assert err.evaluated_candidates_count == 0

    def test_allow_global_default_returns_tier7_factor(self):
        """Pack opts into legacy behaviour: ``ALLOW_GLOBAL_DEFAULT`` +
        ``global_default_tier_allowed=True`` lets the resolver return the
        tier-7 factor via the pre-FY27 path."""

        _register_test_pack(
            profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
            cannot_resolve_action=CannotResolveAction.ALLOW_GLOBAL_DEFAULT,
            global_default_tier_allowed=True,
        )

        global_record = _FakeRecord(factor_id="ef_global_default_legacy")
        engine = ResolutionEngine(
            candidate_source=_make_candidate_source({"global_default": [global_record]})
        )
        request = ResolutionRequest(
            activity="test_activity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
            jurisdiction="US",
        )

        resolved = engine.resolve(request)
        assert resolved.chosen_factor_id == "ef_global_default_legacy"
        assert resolved.fallback_rank == 7
        assert resolved.step_label == "global_default"

    def test_allow_global_default_empty_raises_legacy_resolution_error(self):
        """When the legacy opt-in pack has NO candidates at any tier,
        the engine raises :class:`ResolutionError` (not the new
        safe-match error) to preserve backwards compatibility."""

        _register_test_pack(
            profile=MethodProfile.CORPORATE_SCOPE3,
            cannot_resolve_action=CannotResolveAction.ALLOW_GLOBAL_DEFAULT,
            global_default_tier_allowed=True,
        )

        engine = ResolutionEngine(candidate_source=_make_candidate_source({}))
        request = ResolutionRequest(
            activity="test_activity",
            method_profile=MethodProfile.CORPORATE_SCOPE3,
            jurisdiction="US",
        )

        with pytest.raises(ResolutionError):
            engine.resolve(request)


class TestStructuredActivityCategories:
    """Independent unit tests for SelectionRule inclusion / exclusion gate."""

    def test_excluded_activity_category_rejects(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            excluded_activity_categories=frozenset({"carbon_offsets"}),
        )
        rec = _FakeRecord(factor_id="x", activity_category="carbon_offsets")
        assert rule.accepts(rec) is False

    def test_included_allow_list_rejects_other(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            included_activity_categories=frozenset({"cement", "steel"}),
        )
        rec = _FakeRecord(factor_id="x", activity_category="aluminium")
        assert rule.accepts(rec) is False

    def test_included_allow_list_accepts_listed(self):
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
            included_activity_categories=frozenset({"cement", "steel"}),
        )
        rec = _FakeRecord(factor_id="x", activity_category="cement")
        assert rule.accepts(rec) is True

    def test_empty_sets_mean_no_restriction(self):
        """The default behaviour stays permissive when both sets are empty."""
        rule = SelectionRule(
            allowed_families=(FactorFamily.EMISSIONS,),
            allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
        )
        rec = _FakeRecord(factor_id="x", activity_category="anything")
        assert rule.accepts(rec) is True


class TestMethodPackDefaults:
    """Verify certified packs ship with strict defaults."""

    def test_eu_cbam_defaults_are_strict(self):
        from greenlang.factors.method_packs.eu_policy import EU_CBAM

        assert EU_CBAM.cannot_resolve_action == CannotResolveAction.RAISE_NO_SAFE_MATCH
        assert EU_CBAM.global_default_tier_allowed is False

    def test_eu_battery_defaults_are_strict(self):
        from greenlang.factors.method_packs.eu_policy import EU_BATTERY

        assert EU_BATTERY.cannot_resolve_action == CannotResolveAction.RAISE_NO_SAFE_MATCH
        assert EU_BATTERY.global_default_tier_allowed is False

    def test_corporate_scope1_defaults_are_strict(self):
        from greenlang.factors.method_packs.corporate import CORPORATE_SCOPE1

        assert CORPORATE_SCOPE1.cannot_resolve_action == CannotResolveAction.RAISE_NO_SAFE_MATCH
        assert CORPORATE_SCOPE1.global_default_tier_allowed is False

    def test_eu_cbam_has_cn_code_allowlist(self):
        from greenlang.factors.method_packs.eu_policy import EU_CBAM

        rule = EU_CBAM.selection_rule
        # A regulation-defined CN prefix must be in the allow-list.
        assert "2523" in rule.included_activity_categories  # cement
        assert "7601" in rule.included_activity_categories  # aluminium
        assert "cement" in rule.included_activity_categories
        assert "hydrogen" in rule.included_activity_categories
        # The Annex II exempt-country slug is denied.
        assert "electricity_imports_non_cbam" in rule.excluded_activity_categories
