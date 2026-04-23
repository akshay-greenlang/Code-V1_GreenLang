# -*- coding: utf-8 -*-
"""Phase F3 — Resolution Engine tests."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, Iterable, List

import pytest

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
    RedistributionClass,
    Verification,
    VerificationStatus,
)
from greenlang.factors.resolution import (
    ResolutionEngine,
    ResolutionRequest,
    ResolvedFactor,
)
from greenlang.factors.method_packs import get_pack
from greenlang.factors.method_packs.exceptions import FactorCannotResolveSafelyError
from greenlang.factors.resolution.engine import ResolutionError  # noqa: F401  (legacy, see notes)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _mk_record(
    *,
    factor_id: str,
    factor_family: str = FactorFamily.EMISSIONS.value,
    formula_type: str = FormulaType.DIRECT_FACTOR.value,
    geography: str = "US",
    valid_to: date = date(2099, 12, 31),
    source_id: str = "epa_hub",
    redistribution_class: str = RedistributionClass.OPEN.value,
    factor_status: str = "certified",
    verification_status: str = "regulator_approved",
    ci_95: float = 0.05,
    factor_name: str = "Test factor",
) -> SimpleNamespace:
    rec = SimpleNamespace(
        factor_id=factor_id,
        factor_name=factor_name,
        factor_family=factor_family,
        formula_type=formula_type,
        geography=geography,
        valid_to=valid_to,
        source_id=source_id,
        redistribution_class=redistribution_class,
        factor_status=factor_status,
        verification=Verification(status=VerificationStatus(verification_status)),
        uncertainty_95ci=ci_95,
        unit="kWh",
        # Minimal non-negotiable fields so non-negotiables don't trip.
        vectors=SimpleNamespace(
            CO2=0.5, CH4=0.0, N2O=0.0, HFCs=0.0, PFCs=0.0, SF6=0.0, NF3=0.0,
            biogenic_CO2=0.0,
        ),
        gwp_100yr=SimpleNamespace(co2e_total=0.5),
        provenance=SimpleNamespace(source_year=2024),
        explainability=SimpleNamespace(assumptions=[], rationale=None),
        dqs=SimpleNamespace(overall_score=85.0),
        uncertainty_distribution="normal",
        replacement_factor_id=None,
    )
    return rec


def _source_returning(records_by_step: dict[str, List[Any]]):
    def source(_req: ResolutionRequest, label: str) -> Iterable[Any]:
        return list(records_by_step.get(label, []))

    return source


# --------------------------------------------------------------------------
# Non-negotiable #6 enforcement
# --------------------------------------------------------------------------


class TestMethodProfileRequired:
    def test_method_profile_cannot_be_none(self):
        with pytest.raises(Exception):  # Pydantic ValidationError
            ResolutionRequest(
                activity="diesel combustion",
                method_profile=None,  # type: ignore[arg-type]
            )

    def test_method_profile_enum_required(self):
        # String-coercion must still pass because Pydantic converts.
        req = ResolutionRequest(
            activity="diesel",
            method_profile="corporate_scope1",
        )
        assert req.method_profile == MethodProfile.CORPORATE_SCOPE1

    def test_blank_activity_rejected(self):
        with pytest.raises(Exception):
            ResolutionRequest(
                activity="   ",
                method_profile=MethodProfile.CORPORATE_SCOPE1,
            )


# --------------------------------------------------------------------------
# 7-step cascade
# --------------------------------------------------------------------------


class TestSevenStepCascade:
    def _base_request(self) -> ResolutionRequest:
        return ResolutionRequest(
            activity="diesel combustion stationary",
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
            reporting_date="2026-06-01",
        )

    def test_step1_customer_override_wins(self):
        tenant_rec = _mk_record(factor_id="CUSTOMER-OVERRIDE-1")
        engine = ResolutionEngine(
            candidate_source=_source_returning({"country_or_sector_average": [_mk_record(factor_id="EPA-1")]}),
            tenant_overlay_reader=lambda req: tenant_rec,
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "CUSTOMER-OVERRIDE-1"
        assert resolved.fallback_rank == 1
        assert resolved.step_label == "customer_override"

    def test_step2_supplier_specific(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"supplier_specific": [_mk_record(factor_id="SUPPLIER-1")]}
            )
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "SUPPLIER-1"
        assert resolved.fallback_rank == 2

    def test_step3_facility_specific(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"facility_specific": [_mk_record(factor_id="FAC-1")]}
            )
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "FAC-1"
        assert resolved.fallback_rank == 3

    def test_step4_utility_or_grid(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"utility_or_grid_subregion": [_mk_record(factor_id="GRID-1")]}
            )
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "GRID-1"
        assert resolved.fallback_rank == 4

    def test_step5_country_sector_average(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [_mk_record(factor_id="COUNTRY-1")]}
            )
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "COUNTRY-1"
        assert resolved.fallback_rank == 5

    def test_step6_method_pack_default(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"method_pack_default": [_mk_record(factor_id="MPD-1")]}
            )
        )
        resolved = engine.resolve(self._base_request())
        assert resolved.chosen_factor_id == "MPD-1"
        assert resolved.fallback_rank == 6

    def test_step7_global_default(self):
        """Tier-7 is an OPT-IN per Wave-2 cannot_resolve_safely contract.

        Every shipped pack sets ``global_default_tier_allowed=False`` so a
        regulated caller NEVER silently gets a global default. To verify
        the engine's tier-7 mechanics still work we temporarily flip the
        corporate pack's flag on this test only (MethodPack is a frozen
        dataclass, so we use ``object.__setattr__`` + finally-restore).
        The pack's own default (False) is covered by
        ``test_step7_global_default_blocked_by_default``.
        """
        pack = get_pack(MethodProfile.CORPORATE_SCOPE1)
        original = pack.global_default_tier_allowed
        object.__setattr__(pack, "global_default_tier_allowed", True)
        try:
            engine = self._tier7_engine()
            resolved = engine.resolve(self._base_request())
            assert resolved.chosen_factor_id == "GLOBAL-1"
            assert resolved.fallback_rank == 7
        finally:
            object.__setattr__(pack, "global_default_tier_allowed", original)

    def _tier7_engine(self) -> ResolutionEngine:
        return ResolutionEngine(
            candidate_source=_source_returning(
                {"global_default": [_mk_record(factor_id="GLOBAL-1", geography="GLOBAL")]}
            )
        )

    def test_step7_global_default_blocked_by_default(self):
        """Shipped packs MUST refuse the tier-7 silent fallback.

        Wave-2 contract: when a regulated pack (CBAM, Corporate, PEF,
        Battery, etc.) exhausts tiers 1-6 with no match AND has
        ``global_default_tier_allowed=False``, the engine raises
        :class:`FactorCannotResolveSafelyError` with
        ``reason="global_default_blocked"`` so the caller learns the
        exhaustion was due to policy, not missing data.
        """
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"global_default": [_mk_record(factor_id="GLOBAL-1", geography="GLOBAL")]}
            )
        )
        with pytest.raises(FactorCannotResolveSafelyError) as excinfo:
            engine.resolve(self._base_request())
        assert excinfo.value.reason == "global_default_blocked"
        assert excinfo.value.pack_id == MethodProfile.CORPORATE_SCOPE1.value

    def test_higher_step_beats_lower_even_if_lower_is_better_quality(self):
        """A supplier-specific record wins over a better country-average one."""
        weak_supplier = _mk_record(factor_id="WEAK-SUPPLIER", ci_95=0.2)
        strong_country = _mk_record(factor_id="STRONG-COUNTRY", ci_95=0.01)
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {
                    "supplier_specific": [weak_supplier],
                    "country_or_sector_average": [strong_country],
                }
            )
        )
        resolved = engine.resolve(self._base_request())
        # Supplier is earlier in the cascade — it wins regardless of quality.
        assert resolved.chosen_factor_id == "WEAK-SUPPLIER"


# --------------------------------------------------------------------------
# Tie-break inside a step
# --------------------------------------------------------------------------


class TestTieBreakInsideStep:
    def test_exact_geography_beats_global_in_same_step(self):
        exact = _mk_record(factor_id="EXACT-US", geography="US")
        global_ = _mk_record(
            factor_id="GLOBAL-FALLBACK", geography="GLOBAL",
            source_id="ghg_protocol",
        )
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [global_, exact]}
            )
        )
        req = ResolutionRequest(
            activity="diesel combustion",
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
            reporting_date="2026-01-01",
        )
        resolved = engine.resolve(req)
        assert resolved.chosen_factor_id == "EXACT-US"
        # alternate includes the global fallback.
        assert any(a.factor_id == "GLOBAL-FALLBACK" for a in resolved.alternates)

    def test_verified_beats_unverified_in_same_step(self):
        verified = _mk_record(factor_id="VERIFIED-EPA", verification_status="regulator_approved")
        unverified = _mk_record(
            factor_id="UNVERIFIED-X",
            verification_status="unverified",
            source_id="ghg_protocol",
        )
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [unverified, verified]}
            )
        )
        req = ResolutionRequest(
            activity="diesel",
            method_profile=MethodProfile.CORPORATE_SCOPE1,
            jurisdiction="US",
            reporting_date="2026-01-01",
        )
        resolved = engine.resolve(req)
        assert resolved.chosen_factor_id == "VERIFIED-EPA"


# --------------------------------------------------------------------------
# Selection-rule filtering (bridge to F2 packs)
# --------------------------------------------------------------------------


class TestSelectionRuleIntegration:
    def test_cbam_rejects_unverified_candidate_and_falls_further(self):
        unverified = _mk_record(
            factor_id="UNVERIFIED",
            factor_status="certified",
            verification_status="unverified",
        )
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [unverified]}
            )
        )
        req = ResolutionRequest(
            activity="aluminium ingot CBAM",
            method_profile=MethodProfile.EU_CBAM,
            jurisdiction="EU",
            reporting_date="2026-06-01",
        )
        # CBAM pack requires verification — unverified cannot satisfy any
        # step, so resolution fails. Wave-2 contract: since the CBAM pack
        # has ``cannot_resolve_action = RAISE_NO_SAFE_MATCH`` (the default
        # for every certified pack) the exhaustion-path exception is the
        # structured :class:`FactorCannotResolveSafelyError`, NOT the
        # legacy untyped :class:`ResolutionError`. Regulated callers
        # (CBAM Art. 4(2), PEF, Battery CFP) depend on the structured
        # payload to distinguish "no data" from "pack policy blocked a
        # low-quality fallback".
        with pytest.raises(FactorCannotResolveSafelyError) as excinfo:
            engine.resolve(req)
        assert excinfo.value.pack_id == MethodProfile.EU_CBAM.value
        assert excinfo.value.reason in (
            "no_candidate",
            "all_candidates_rejected",
            "global_default_blocked",
        )


# --------------------------------------------------------------------------
# ResolvedFactor payload + explain()
# --------------------------------------------------------------------------


class TestResolvedFactorPayload:
    def test_alternates_listed_with_why_not(self):
        winner = _mk_record(factor_id="WIN")
        loser1 = _mk_record(factor_id="LOSER-1", ci_95=0.2)
        loser2 = _mk_record(factor_id="LOSER-2", geography="GLOBAL", source_id="ghg_protocol")
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [winner, loser1, loser2]}
            )
        )
        resolved = engine.resolve(
            ResolutionRequest(
                activity="diesel",
                method_profile=MethodProfile.CORPORATE_SCOPE1,
                jurisdiction="US",
                reporting_date="2026-01-01",
            )
        )
        assert resolved.chosen_factor_id == "WIN"
        assert len(resolved.alternates) == 2
        for alt in resolved.alternates:
            assert alt.why_not_chosen

    def test_explain_has_derivation_and_emissions(self):
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [_mk_record(factor_id="X")]}
            )
        )
        resolved = engine.resolve(
            ResolutionRequest(
                activity="diesel",
                method_profile=MethodProfile.CORPORATE_SCOPE1,
                jurisdiction="US",
                reporting_date="2026-01-01",
            )
        )
        payload = resolved.explain()
        assert payload["chosen"]["factor_id"] == "X"
        assert "derivation" in payload
        assert "emissions" in payload
        assert "alternates" in payload
        assert payload["derivation"]["fallback_rank"] == 5

    def test_no_candidate_raises(self):
        """Exhausting every cascade step with zero candidates MUST raise
        the structured :class:`FactorCannotResolveSafelyError` on any pack
        whose ``cannot_resolve_action`` is ``RAISE_NO_SAFE_MATCH`` — which
        is every shipped certified pack. The reason key distinguishes
        "global_default_blocked" (tier-7 was skipped by pack policy) from
        "no_candidate" (no data at all on any tier).
        """
        engine = ResolutionEngine(candidate_source=_source_returning({}))
        with pytest.raises(FactorCannotResolveSafelyError) as excinfo:
            engine.resolve(
                ResolutionRequest(
                    activity="diesel",
                    method_profile=MethodProfile.CORPORATE_SCOPE1,
                )
            )
        # Corporate pack blocks tier-7, so the exhaustion reason is the
        # policy-block signal, not "no_candidate".
        assert excinfo.value.reason == "global_default_blocked"
        assert excinfo.value.evaluated_candidates_count == 0


# --------------------------------------------------------------------------
# License-class homogeneity (non-negotiable #4)
# --------------------------------------------------------------------------


class TestLicenseHomogeneityEnforced:
    def test_mixed_classes_in_same_step_raise(self):
        open_rec = _mk_record(
            factor_id="OPEN-1", redistribution_class=RedistributionClass.OPEN.value
        )
        licensed_rec = _mk_record(
            factor_id="LICENSED-1",
            redistribution_class=RedistributionClass.LICENSED.value,
            source_id="ghg_protocol",
        )
        engine = ResolutionEngine(
            candidate_source=_source_returning(
                {"country_or_sector_average": [open_rec, licensed_rec]}
            )
        )
        from greenlang.data.canonical_v2 import NonNegotiableViolation

        with pytest.raises(NonNegotiableViolation):
            engine.resolve(
                ResolutionRequest(
                    activity="diesel",
                    method_profile=MethodProfile.CORPORATE_SCOPE1,
                    jurisdiction="US",
                )
            )
