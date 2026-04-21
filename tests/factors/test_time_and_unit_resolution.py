# -*- coding: utf-8 -*-
"""Tests for time-granularity tie-break scoring, unit conversion in the
resolution engine, and the new product-LCA variant packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, List, Optional

import pytest

from greenlang.data.canonical_v2 import MethodProfile
from greenlang.factors.method_packs import (
    OEF,
    PAS_2050,
    PEF,
    get_pack,
    list_product_lca_variants,
)
from greenlang.factors.resolution.engine import ResolutionEngine
from greenlang.factors.resolution.request import ResolutionRequest, TimeGranularity
from greenlang.factors.resolution.tiebreak import (
    build_tiebreak,
    score_time_granularity,
)


# ---------------------------------------------------------------------------
# Product-LCA variant registration
# ---------------------------------------------------------------------------


class TestProductLcaVariants:
    def test_pas_2050_registered(self) -> None:
        pack = get_pack("pas_2050")
        assert pack is PAS_2050
        assert "PAS_2050" in pack.reporting_labels
        assert pack.gwp_basis == "IPCC_AR5_100"

    def test_pef_registered(self) -> None:
        pack = get_pack("eu_pef")
        assert pack is PEF
        assert "EU_PEF" in pack.reporting_labels
        assert pack.selection_rule.require_verification is True

    def test_oef_registered(self) -> None:
        pack = get_pack("eu_oef")
        assert pack is OEF
        assert "EU_OEF" in pack.reporting_labels
        # OEF applies to entity-level → supports all three scopes.
        assert set(pack.boundary_rule.allowed_scopes) == {"1", "2", "3"}

    def test_variant_registry_exposed(self) -> None:
        variants = list_product_lca_variants()
        assert {"pas_2050", "eu_pef", "eu_oef"} <= set(variants.keys())

    def test_unknown_variant_raises(self) -> None:
        from greenlang.factors.method_packs.product_lca_variants import (
            get_product_lca_variant,
        )

        with pytest.raises(KeyError):
            get_product_lca_variant("not_a_real_variant")


# ---------------------------------------------------------------------------
# Time-granularity scoring
# ---------------------------------------------------------------------------


class TestTimeGranularityScoring:
    @pytest.mark.parametrize(
        "record,request_,expected",
        [
            ("hourly", "hourly", 0),
            ("annual", "hourly", 8),       # 4 steps coarser * 2
            ("monthly", "hourly", 4),      # 2 steps coarser * 2
            ("quarterly", "monthly", 2),   # 1 step coarser * 2
            ("hourly", "annual", 0),       # finer than requested is free
            (None, "annual", 0),
            ("annual", None, 0),
        ],
    )
    def test_parametric(self, record: Optional[str], request_: Optional[str], expected: int) -> None:
        assert score_time_granularity(record, request_) == expected

    def test_build_tiebreak_includes_granularity(self) -> None:
        @dataclass
        class Fake:
            geography: Optional[str] = "US-CA"
            valid_to: Optional[date] = None
            source_id: Optional[str] = "egrid"
            redistribution_class: Optional[str] = "open"
            time_granularity: Optional[str] = "annual"
            uncertainty_95ci: Optional[float] = None

        tb = build_tiebreak(
            Fake(),
            request_geo="US-CA",
            request_date=date(2026, 6, 1),
            request_granularity="hourly",
        )
        assert tb.time_granularity_penalty == 8
        assert tb.score() >= 8


# ---------------------------------------------------------------------------
# Unit conversion inside the engine
# ---------------------------------------------------------------------------


@dataclass
class _Vectors:
    CO2: float = 0.0
    CH4: float = 0.0
    N2O: float = 0.0
    HFCs: float = 0.0
    PFCs: float = 0.0
    SF6: float = 0.0
    NF3: float = 0.0
    biogenic_CO2: float = 0.0


@dataclass
class _GWP100:
    co2e_total: float


@dataclass
class _FakeFactor:
    factor_id: str = "grid_fr_2025"
    factor_status: str = "certified"
    factor_family: str = "grid_intensity"
    formula_type: str = "direct_factor"
    geography: str = "FR"
    source_id: str = "eu_electricity_maps"
    unit: str = "kWh"
    vectors: _Vectors = field(default_factory=_Vectors)
    gwp_100yr: _GWP100 = field(default_factory=lambda: _GWP100(0.05))
    redistribution_class: str = "open"
    valid_to: Optional[date] = None
    verification: Any = None
    uncertainty_95ci: Optional[float] = None
    factor_version: Optional[str] = "2025.1"
    release_version: Optional[str] = "2025.1"
    source_release: Optional[str] = "2025.1"


def _candidate_source_factory(records: List[Any]):
    def _source(request: ResolutionRequest, label: str):
        # Only yield candidates at one step so the winner is deterministic.
        if label == "country_or_sector_average":
            return list(records)
        return []

    return _source


class TestEngineUnitConversion:
    def test_native_unit_matches_target(self) -> None:
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([_FakeFactor(unit="kWh")])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            target_unit="kWh",
        )
        resolved = engine.resolve(req)
        assert resolved.target_unit == "kWh"
        assert resolved.unit_conversion_factor == 1.0
        assert resolved.converted_co2e_per_unit == pytest.approx(0.05)

    def test_kwh_to_mwh_conversion(self) -> None:
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([_FakeFactor(unit="kWh")])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            target_unit="MWh",
        )
        resolved = engine.resolve(req)
        assert resolved.target_unit == "MWh"
        # 1 MWh = 1000 kWh → 500 kg CO2e / MWh (0.05 * 1000)
        assert resolved.unit_conversion_factor == pytest.approx(1000.0)
        assert resolved.converted_co2e_per_unit == pytest.approx(50.0)
        assert resolved.unit_conversion_path[0] == "MWh"
        assert resolved.unit_conversion_path[-1] == "kWh"

    def test_no_conversion_path_reports_in_note(self) -> None:
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([_FakeFactor(unit="kWh")])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            target_unit="not_a_unit",
        )
        resolved = engine.resolve(req)
        # The conversion note must explain the failure; engine must NOT throw.
        assert resolved.target_unit == "not_a_unit"
        assert resolved.converted_co2e_per_unit is None
        assert resolved.unit_conversion_note is not None
        assert "no conversion path" in resolved.unit_conversion_note.lower()

    def test_explain_payload_includes_unit_conversion_block(self) -> None:
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([_FakeFactor(unit="kWh")])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            target_unit="MWh",
        )
        resolved = engine.resolve(req)
        payload = resolved.explain()
        assert payload["unit_conversion"] is not None
        assert payload["unit_conversion"]["target_unit"] == "MWh"
        assert payload["unit_conversion"]["factor"] == pytest.approx(1000.0)

    def test_no_target_unit_leaves_explain_untouched(self) -> None:
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([_FakeFactor(unit="kWh")])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
        )
        resolved = engine.resolve(req)
        assert resolved.target_unit is None
        assert resolved.explain()["unit_conversion"] is None


# ---------------------------------------------------------------------------
# TimeGranularity plumbing through the engine
# ---------------------------------------------------------------------------


class TestEngineTimeGranularityPicker:
    def test_engine_prefers_hourly_record_for_hourly_request(self) -> None:
        hourly = _FakeFactor(factor_id="fr_hourly_2025")
        hourly.source_id = "entso_e_hourly"  # type: ignore[attr-defined]
        setattr(hourly, "time_granularity", "hourly")
        annual = _FakeFactor(factor_id="fr_annual_2025")
        setattr(annual, "time_granularity", "annual")

        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([annual, hourly])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            time_granularity=TimeGranularity.HOURLY,
            reporting_hour=14,
        )
        resolved = engine.resolve(req)
        assert resolved.chosen_factor_id == "fr_hourly_2025"

    def test_engine_falls_back_to_annual_when_no_hourly_available(self) -> None:
        annual = _FakeFactor(factor_id="fr_annual_2025")
        setattr(annual, "time_granularity", "annual")
        engine = ResolutionEngine(
            candidate_source=_candidate_source_factory([annual])
        )
        req = ResolutionRequest(
            activity="electricity",
            method_profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
            jurisdiction="FR",
            time_granularity=TimeGranularity.HOURLY,
        )
        resolved = engine.resolve(req)
        # Resolution still succeeds — penalty makes the score worse but
        # there are no alternatives, so the annual factor wins.
        assert resolved.chosen_factor_id == "fr_annual_2025"
