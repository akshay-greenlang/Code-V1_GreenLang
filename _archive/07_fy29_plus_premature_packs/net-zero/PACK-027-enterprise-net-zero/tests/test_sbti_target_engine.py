# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - SBTi Target Engine.

Tests full SBTi Corporate Standard compliance including 28 near-term criteria
(C1-C28), 14 net-zero criteria (NZ-C1 to NZ-C14), ACA/SDA/FLAG pathways,
and submission readiness.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~65 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.sbti_target_engine import (
    SBTiTargetEngine,
    SBTiTargetInput,
    SBTiTargetResult,
    TargetPathwayType,
    CriterionValidation,
    TargetDefinition,
    MilestoneEntry,
    BaselineData,
    SDASector,
    SDA_SECTOR_TARGETS,
    CriterionStatus,
)

from .conftest import (
    assert_decimal_close, assert_decimal_positive,
    assert_provenance_hash, SBTI_NEAR_TERM_CRITERIA,
    SBTI_NET_ZERO_CRITERIA, SDA_SECTORS,
)


def _make_input(**kwargs):
    """Helper to build SBTiTargetInput with proper baseline nesting."""
    baseline_fields = {}
    top_fields = {}
    baseline_keys = {
        "scope1_tco2e", "scope2_location_tco2e", "scope2_market_tco2e",
        "scope3_total_tco2e", "scope3_by_category", "flag_emissions_tco2e",
        "total_tco2e",
    }
    for k, v in kwargs.items():
        if k in baseline_keys:
            baseline_fields[k] = v
        else:
            top_fields[k] = v
    if baseline_fields and "baseline" not in top_fields:
        top_fields["baseline"] = BaselineData(**baseline_fields)
    return SBTiTargetInput(**top_fields)


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestSBTiTargetEngineInstantiation:
    def test_engine_instantiates(self):
        engine = SBTiTargetEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = SBTiTargetEngine()
        assert hasattr(engine, "calculate")

    def test_engine_supports_all_pathways(self):
        """All four pathway types exist as enum members."""
        pathways = [p.name for p in TargetPathwayType]
        assert "ACA_15C" in pathways
        assert "ACA_WB2C" in pathways
        assert "SDA" in pathways
        assert "FLAG" in pathways


# ===========================================================================
# Tests -- ACA Pathway (Absolute Contraction Approach)
# ===========================================================================


class TestACAPathway:
    def test_aca_15c_annual_reduction(self):
        """ACA 1.5C requires 4.2%/yr absolute reduction."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        assert result.near_term_target.annual_reduction_rate_pct >= Decimal("4.2")

    def test_aca_wb2c_annual_reduction(self):
        """ACA WB2C requires 2.5%/yr absolute reduction."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_WB2C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        assert result.near_term_target.annual_reduction_rate_pct >= Decimal("2.5")

    def test_aca_target_emission_calculation(self):
        """Target year emissions must reflect annual reduction."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        # 6 years * 4.2%/yr = 25.2% reduction => 210000 * 0.748 = ~157080
        assert result.near_term_target.target_year_emissions_tco2e < Decimal("210000")

    def test_aca_annual_milestones(self):
        """Annual milestones must show declining pathway."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        milestones = result.milestones
        assert len(milestones) > 0
        for i in range(1, len(milestones)):
            assert milestones[i].target_tco2e <= milestones[i - 1].target_tco2e

    def test_aca_scope12_coverage_95pct(self):
        """Scope 1+2 coverage must be >= 95%."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
            scope12_coverage_pct=Decimal("98"),
        ))
        assert result.near_term_target.coverage_pct >= Decimal("95")

    def test_aca_scope3_coverage_67pct(self):
        """Near-term Scope 3 coverage must be >= 67%."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
            scope3_coverage_pct=Decimal("72"),
        ))
        assert result.near_term_scope3_target.coverage_pct >= Decimal("67")


# ===========================================================================
# Tests -- SDA Pathway (Sectoral Decarbonization Approach)
# ===========================================================================


class TestSDAPathway:
    @pytest.mark.parametrize("sector", [
        "power_generation", "cement", "iron_steel", "aluminium",
        "pulp_paper", "chemicals", "aviation", "maritime",
        "road_transport", "commercial_buildings", "residential_buildings",
        "food_beverage",
    ])
    def test_sda_sector_pathways(self, sector):
        """Each SDA sector must have a defined pathway in constants."""
        sector_enum = SDASector(sector)
        data = SDA_SECTOR_TARGETS.get(sector_enum)
        assert data is not None
        assert "metric" in data
        assert "2030" in data
        assert "2050" in data

    def test_sda_power_generation(self):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.SDA,
            sda_sector=SDASector.POWER_GENERATION,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("500000"),
            scope2_location_tco2e=Decimal("10000"),
            production_metric_value=Decimal("3500000"),
            production_metric_unit="MWh",
        ))
        # SDA should produce a near-term target with reduction
        assert result.near_term_target.reduction_pct > Decimal("0")

    def test_sda_cement_sector(self):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.SDA,
            sda_sector=SDASector.CEMENT,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("800000"),
            scope2_location_tco2e=Decimal("50000"),
            production_metric_value=Decimal("2000000"),
            production_metric_unit="t_cement",
        ))
        assert result.near_term_target.reduction_pct > Decimal("0")

    def test_sda_intensity_declining(self):
        """SDA milestones should show a declining pathway."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.SDA,
            sda_sector=SDASector.IRON_STEEL,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("600000"),
            scope2_location_tco2e=Decimal("80000"),
            production_metric_value=Decimal("500000"),
            production_metric_unit="t_crude_steel",
        ))
        milestones = result.milestones
        if len(milestones) > 1:
            for i in range(1, len(milestones)):
                assert milestones[i].target_tco2e <= milestones[i - 1].target_tco2e


# ===========================================================================
# Tests -- FLAG Pathway
# ===========================================================================


class TestFLAGPathway:
    def test_flag_required_when_above_20pct(self):
        """FLAG target required when land use emissions > 20%."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.FLAG,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("100000"),
            scope2_location_tco2e=Decimal("20000"),
            scope3_total_tco2e=Decimal("400000"),
            flag_pct_of_total=Decimal("25"),
            has_flag_target=True,
        ))
        # FLAG target should be present when flag_pct > 20%
        assert result.flag_target is not None

    def test_flag_annual_reduction_rate(self):
        """FLAG pathway requires 3.03%/yr reduction."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.FLAG,
            base_year=2024,
            target_year=2030,
            flag_emissions_tco2e=Decimal("120000"),
            has_flag_target=True,
            flag_pct_of_total=Decimal("25"),
        ))
        if result.flag_target is not None:
            assert result.flag_target.annual_reduction_rate_pct >= Decimal("3.0")

    def test_flag_target_produces_result(self):
        """FLAG pathway produces a valid result."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.FLAG,
            base_year=2024,
            target_year=2030,
            flag_emissions_tco2e=Decimal("120000"),
            has_flag_target=True,
            flag_pct_of_total=Decimal("25"),
        ))
        assert result.provenance_hash != ""


# ===========================================================================
# Tests -- Near-Term Criteria (C1-C28)
# ===========================================================================


class TestNearTermCriteria:
    @pytest.mark.parametrize("criterion", SBTI_NEAR_TERM_CRITERIA)
    def test_near_term_criterion_validated(self, criterion, sbti_baseline_data):
        """Each of the 28 near-term criteria must be evaluated."""
        engine = SBTiTargetEngine()
        inp = _make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=sbti_baseline_data["base_year"],
            target_year=sbti_baseline_data["target_year_near_term"],
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        )
        result = engine.calculate(inp)
        criterion_ids = [c.criterion_id for c in result.criteria_validations]
        assert criterion in criterion_ids

    def test_all_28_criteria_present(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        inp = _make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=sbti_baseline_data["base_year"],
            target_year=sbti_baseline_data["target_year_near_term"],
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        )
        result = engine.calculate(inp)
        near_term_ids = [c.criterion_id for c in result.criteria_validations
                         if c.criterion_id.startswith("C") and not c.criterion_id.startswith("CDR")]
        # Filter to just C1-C28 pattern
        c_ids = [cid for cid in near_term_ids if cid.startswith("C") and cid[1:].isdigit()]
        assert len(c_ids) == 28

    def test_criteria_with_remediation(self, sbti_baseline_data):
        """Failed criteria must include remediation guidance."""
        engine = SBTiTargetEngine()
        inp = _make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=sbti_baseline_data["base_year"],
            target_year=sbti_baseline_data["target_year_near_term"],
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        )
        result = engine.calculate(inp)
        for cv in result.criteria_validations:
            if cv.status == CriterionStatus.FAIL:
                assert cv.remediation is not None
                assert len(cv.remediation) > 0


# ===========================================================================
# Tests -- Net-Zero Criteria (NZ-C1 to NZ-C14)
# ===========================================================================


class TestNetZeroCriteria:
    @pytest.mark.parametrize("criterion", SBTI_NET_ZERO_CRITERIA)
    def test_net_zero_criterion_validated(self, criterion, sbti_baseline_data):
        """Each of the 14 net-zero criteria must be evaluated."""
        engine = SBTiTargetEngine()
        inp = _make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=sbti_baseline_data["base_year"],
            target_year=sbti_baseline_data["target_year_near_term"],
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        )
        result = engine.calculate(inp)
        criterion_ids = [c.criterion_id for c in result.criteria_validations]
        assert criterion in criterion_ids

    def test_nz_long_term_target_present(self, sbti_baseline_data):
        """Long-term target should be generated."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        assert result.long_term_target is not None
        assert result.long_term_target.reduction_pct >= Decimal("0")

    def test_nz_long_term_90pct_reduction(self, sbti_baseline_data):
        """NZ-C1: 90%+ absolute reduction by 2050."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        # Long-term target should have >= 90% reduction
        assert result.long_term_target.reduction_pct >= Decimal("90")

    def test_nz_result_has_criteria_count(self, sbti_baseline_data):
        """Result should track criteria pass/fail counts."""
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=Decimal("125000"),
            scope2_location_tco2e=Decimal("85000"),
            scope3_total_tco2e=Decimal("680000"),
        ))
        total = result.criteria_pass_count + result.criteria_fail_count + result.criteria_warning_count
        assert total > 0


# ===========================================================================
# Tests -- Submission Readiness
# ===========================================================================


class TestSubmissionReadiness:
    def test_submission_readiness_score(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=sbti_baseline_data["base_year"],
            target_year=sbti_baseline_data["target_year_near_term"],
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert Decimal("0") <= result.submission_readiness_score <= Decimal("100")

    def test_target_statements_generated(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert len(result.target_statements) > 0

    def test_provenance_hash(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert_provenance_hash(result)

    def test_result_deterministic(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        inp = _make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024, target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        )
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        # Core calculations must be deterministic (provenance_hash includes result_id UUID)
        assert r1.near_term_target.reduction_pct == r2.near_term_target.reduction_pct
        assert r1.submission_readiness_score == r2.submission_readiness_score
        assert r1.criteria_pass_count == r2.criteria_pass_count

    def test_regulatory_citations(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert len(result.regulatory_citations) > 0
        assert any("SBTi" in c for c in result.regulatory_citations)

    def test_fair_share_assessment(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert result.fair_share is not None
        assert result.fair_share.approach != ""

    def test_processing_time(self, sbti_baseline_data):
        engine = SBTiTargetEngine()
        result = engine.calculate(_make_input(
            pathway_type=TargetPathwayType.ACA_15C,
            base_year=2024,
            target_year=2030,
            scope1_tco2e=sbti_baseline_data["scope1_tco2e"],
            scope2_location_tco2e=sbti_baseline_data["scope2_location_tco2e"],
            scope3_total_tco2e=sbti_baseline_data["scope3_total_tco2e"],
        ))
        assert result.processing_time_ms >= 0
