# -*- coding: utf-8 -*-
"""
Test suite for PACK-023 SBTi Alignment Pack - TargetSettingEngine.

Validates:
  - ACA (Absolute Contraction Approach) target calculations
  - SDA (Sectoral Decarbonization Approach) pathway
  - FLAG (Forest, Land & Agriculture) pathway
  - Ambition level assessment (1.5C/WB2C/2C)
  - Scope coverage validation (95% S1+2, 67% S3 near-term, 90% S3 long-term)
  - Target boundary enforcement
  - Pathway milestones and annual reduction rates
  - Provenance hashing and zero-hallucination validation

Total Tests: 50+ parametrized assertions
Author: GreenLang Test Engineering
Pack: PACK-023 SBTi Alignment
"""

import sys
from decimal import Decimal
from pathlib import Path
from datetime import datetime, timezone

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

from engines.target_setting_engine import (
    TargetSettingEngine,
    TargetSettingInput,
    TargetResult,
    TargetType,
    AmbitionLevel,
    PathwayMethod,
    TargetScope,
    Scope1Detail,
    Scope2Detail,
    Scope3Detail,
    BaselineEmissions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> TargetSettingEngine:
    """Fresh engine instance."""
    return TargetSettingEngine()


@pytest.fixture
def basic_input() -> TargetSettingInput:
    """Basic target-setting input (ACA, 1.5C, Scope 1+2)."""
    return TargetSettingInput(
        entity_name="TestCorp",
        base_year=2024,
        base_scope1_tco2e=Decimal("1000"),
        base_scope2_tco2e=Decimal("500"),
        base_scope3_tco2e=Decimal("2000"),
        sector="Technology",
        target_type=TargetType.NEAR_TERM,
        ambition_level=AmbitionLevel.CELSIUS_1_5,
        pathway_method=PathwayMethod.ACA,
        target_year=2030,
        target_scope=TargetScope.SCOPE_1_2,
    )


@pytest.fixture
def sda_input() -> TargetSettingInput:
    """SDA pathway input with sector-specific data."""
    return TargetSettingInput(
        entity_name="ManufacturerCorp",
        base_year=2024,
        base_scope1_tco2e=Decimal("5000"),
        base_scope2_tco2e=Decimal("2000"),
        base_scope3_tco2e=Decimal("8000"),
        sector="Manufacturing",
        subsector="Steel",
        target_type=TargetType.NEAR_TERM,
        ambition_level=AmbitionLevel.WELL_BELOW_2C,
        pathway_method=PathwayMethod.SDA,
        target_year=2030,
        target_scope=TargetScope.SCOPE_1_2,
        revenue_baseline_usd_millions=Decimal("1000"),
        intensity_baseline_tco2e_per_usd_m=Decimal("7"),
    )


@pytest.fixture
def flag_input() -> TargetSettingInput:
    """FLAG pathway input for agricultural/land-use entity."""
    return TargetSettingInput(
        entity_name="AgricultureCorp",
        base_year=2024,
        base_scope1_tco2e=Decimal("2000"),
        base_scope3_tco2e=Decimal("3000"),
        sector="Agriculture",
        target_type=TargetType.FLAG,
        ambition_level=AmbitionLevel.CELSIUS_2,
        pathway_method=PathwayMethod.FLAG,
        target_year=2030,
        target_scope=TargetScope.SCOPE_1_2_3,
    )


@pytest.fixture
def full_scope_input() -> TargetSettingInput:
    """Full scope input with S1+S2+S3 coverage validation."""
    return TargetSettingInput(
        entity_name="FullScopeCorp",
        base_year=2024,
        base_scope1_tco2e=Decimal("3000"),
        base_scope2_tco2e=Decimal("1500"),
        base_scope3_tco2e=Decimal("5500"),
        scope3_covered_pct=Decimal("75"),
        sector="Consumer Goods",
        target_type=TargetType.NEAR_TERM,
        ambition_level=AmbitionLevel.CELSIUS_1_5,
        pathway_method=PathwayMethod.ACA,
        target_year=2030,
        target_scope=TargetScope.SCOPE_1_2_3,
    )


# ===========================================================================
# Tests -- Engine Instantiation
# ===========================================================================


class TestEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self) -> None:
        """Engine must instantiate without arguments."""
        engine = TargetSettingEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Tests -- ACA Pathway
# ===========================================================================


class TestACAPathway:
    """Tests for Absolute Contraction Approach calculations."""

    def test_aca_1_5c_near_term(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """ACA 1.5C: annual reduction = 4.2%, over 6 years (2024->2030)."""
        basic_input.ambition_level = AmbitionLevel.CELSIUS_1_5
        basic_input.target_year = 2030
        result = engine.calculate(basic_input)

        assert isinstance(result, TargetResult)
        assert result.target_type == TargetType.NEAR_TERM
        # Base S1+2 = 1500, reduction rate = 4.2%/yr for 6 years
        # Target = 1500 * (1 - 0.042)^6 ≈ 1107 tCO2e
        expected_reduction_rate = Decimal("0.042")
        base_emissions = Decimal("1500")  # S1+2
        years = 6
        expected_target = base_emissions * ((Decimal("1") - expected_reduction_rate) ** years)
        assert float(result.target_s12_tco2e) == pytest.approx(float(expected_target), rel=0.01)

    @pytest.mark.parametrize("ambition,rate", [
        (AmbitionLevel.CELSIUS_1_5, Decimal("0.042")),
        (AmbitionLevel.WELL_BELOW_2C, Decimal("0.025")),
        (AmbitionLevel.CELSIUS_2, Decimal("0.016")),
    ])
    def test_aca_ambition_rates(
        self,
        engine: TargetSettingEngine,
        basic_input: TargetSettingInput,
        ambition: AmbitionLevel,
        rate: Decimal,
    ) -> None:
        """ACA reduction rates must match SBTi Corporate Manual tables."""
        basic_input.ambition_level = ambition
        basic_input.pathway_method = PathwayMethod.ACA
        result = engine.calculate(basic_input)

        assert result.annual_reduction_rate == rate

    def test_aca_milestone_generation(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """ACA must generate annual milestones from base year to target year."""
        basic_input.pathway_method = PathwayMethod.ACA
        basic_input.target_year = 2030
        result = engine.calculate(basic_input)

        assert len(result.pathway_milestones) == 7  # 2024-2030 inclusive
        assert result.pathway_milestones[0]["year"] == 2024
        assert result.pathway_milestones[-1]["year"] == 2030
        # Each milestone should have decreasing emissions
        for i in range(len(result.pathway_milestones) - 1):
            assert (
                result.pathway_milestones[i]["tco2e"]
                >= result.pathway_milestones[i + 1]["tco2e"]
            )

    def test_aca_long_term_target(
        self, engine: TargetSettingEngine
    ) -> None:
        """Long-term ACA target (>2030) uses same methodology."""
        inp = TargetSettingInput(
            entity_name="LongTermCorp",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            base_scope3_tco2e=Decimal("2000"),
            sector="Technology",
            target_type=TargetType.LONG_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2045,
            target_scope=TargetScope.SCOPE_1_2,
        )
        result = engine.calculate(inp)

        assert result.target_type == TargetType.LONG_TERM
        assert len(result.pathway_milestones) == 22  # 2024-2045


# ===========================================================================
# Tests -- SDA Pathway
# ===========================================================================


class TestSDAPathway:
    """Tests for Sectoral Decarbonization Approach."""

    def test_sda_calculation(
        self, engine: TargetSettingEngine, sda_input: TargetSettingInput
    ) -> None:
        """SDA convergence to sector pathway."""
        result = engine.calculate(sda_input)

        assert isinstance(result, TargetResult)
        assert result.pathway_method == PathwayMethod.SDA
        # Target should be less than baseline but > 0
        assert result.target_s12_tco2e > Decimal("0")
        assert result.target_s12_tco2e < sda_input.base_scope1_tco2e + sda_input.base_scope2_tco2e

    def test_sda_sector_mapping(
        self, engine: TargetSettingEngine, sda_input: TargetSettingInput
    ) -> None:
        """SDA must correctly map sector/subsector to pathway data."""
        result = engine.calculate(sda_input)

        assert result.sector == "Manufacturing"
        assert result.subsector == "Steel"
        assert result.sector_decarbonization_pathway is not None

    def test_sda_intensity_convergence(
        self, engine: TargetSettingEngine
    ) -> None:
        """SDA intensity convergence formula application."""
        inp = TargetSettingInput(
            entity_name="SDAIntensity",
            base_year=2024,
            base_scope1_tco2e=Decimal("2000"),
            base_scope2_tco2e=Decimal("800"),
            sector="Manufacturing",
            subsector="Cement",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.WELL_BELOW_2C,
            pathway_method=PathwayMethod.SDA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2,
            revenue_baseline_usd_millions=Decimal("500"),
            intensity_baseline_tco2e_per_usd_m=Decimal("5.6"),
        )
        result = engine.calculate(inp)

        # Intensity target should follow SDA formula
        assert result.intensity_target_tco2e_per_usd_m > Decimal("0")
        assert result.intensity_target_tco2e_per_usd_m < inp.intensity_baseline_tco2e_per_usd_m


# ===========================================================================
# Tests -- FLAG Pathway
# ===========================================================================


class TestFLAGPathway:
    """Tests for Forest, Land & Agriculture pathway."""

    def test_flag_linear_reduction(
        self, engine: TargetSettingEngine, flag_input: TargetSettingInput
    ) -> None:
        """FLAG pathway: E(t) = E(base) * (1 - 0.0303 * (t - base_year))."""
        result = engine.calculate(flag_input)

        assert isinstance(result, TargetResult)
        assert result.pathway_method == PathwayMethod.FLAG
        # FLAG uses 3.03% annual reduction (linear)
        # Base = 2000 + 3000 = 5000 (S1+S3 for agriculture)
        # Years = 6 (2024->2030)
        # Target = 5000 * (1 - 0.0303 * 6) ≈ 4090
        base_emissions = Decimal("5000")
        years = 6
        expected_target = base_emissions * (Decimal("1") - Decimal("0.0303") * Decimal(str(years)))
        assert float(result.target_tco2e) == pytest.approx(float(expected_target), rel=0.05)

    def test_flag_agricultural_sector(
        self, engine: TargetSettingEngine
    ) -> None:
        """FLAG targets apply primarily to agricultural/land-use sectors."""
        inp = TargetSettingInput(
            entity_name="Agribusiness",
            base_year=2024,
            base_scope1_tco2e=Decimal("1500"),
            base_scope3_tco2e=Decimal("2500"),
            sector="Agriculture",
            target_type=TargetType.FLAG,
            ambition_level=AmbitionLevel.CELSIUS_2,
            pathway_method=PathwayMethod.FLAG,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2_3,
        )
        result = engine.calculate(inp)

        assert result.target_type == TargetType.FLAG
        assert result.sector == "Agriculture"


# ===========================================================================
# Tests -- Scope Coverage Validation
# ===========================================================================


class TestScopeCoverageValidation:
    """Tests for emissions boundary and coverage requirements."""

    def test_scope_12_coverage_requirement_near_term(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """Near-term targets must cover >= 95% of Scope 1+2 emissions."""
        basic_input.target_type = TargetType.NEAR_TERM
        result = engine.calculate(basic_input)

        assert result.scope12_coverage_pct >= Decimal("95")

    def test_scope_3_coverage_requirement_near_term(
        self, engine: TargetSettingEngine, full_scope_input: TargetSettingInput
    ) -> None:
        """Near-term targets must cover >= 67% of Scope 3 (if material)."""
        full_scope_input.target_type = TargetType.NEAR_TERM
        full_scope_input.scope3_covered_pct = Decimal("70")
        result = engine.calculate(full_scope_input)

        assert result.scope3_coverage_pct >= Decimal("67")

    def test_scope_3_coverage_requirement_long_term(
        self, engine: TargetSettingEngine, full_scope_input: TargetSettingInput
    ) -> None:
        """Long-term targets must cover >= 90% of Scope 3."""
        full_scope_input.target_type = TargetType.LONG_TERM
        full_scope_input.target_year = 2045
        full_scope_input.scope3_covered_pct = Decimal("92")
        result = engine.calculate(full_scope_input)

        assert result.scope3_coverage_pct >= Decimal("90")

    def test_coverage_warnings_insufficient(
        self, engine: TargetSettingEngine
    ) -> None:
        """Insufficient coverage should generate warnings."""
        inp = TargetSettingInput(
            entity_name="LowCoverageCorp",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            base_scope3_tco2e=Decimal("2000"),
            sector="Technology",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2_3,
            scope3_covered_pct=Decimal("50"),  # Below 67% minimum
        )
        result = engine.calculate(inp)

        assert len(result.warnings) > 0
        assert any("coverage" in w.lower() for w in result.warnings)


# ===========================================================================
# Tests -- Ambition Level Validation
# ===========================================================================


class TestAmbitionLevelValidation:
    """Tests for temperature alignment assessment."""

    @pytest.mark.parametrize("ambition", [
        AmbitionLevel.CELSIUS_1_5,
        AmbitionLevel.WELL_BELOW_2C,
        AmbitionLevel.CELSIUS_2,
    ])
    def test_all_ambition_levels(
        self,
        engine: TargetSettingEngine,
        basic_input: TargetSettingInput,
        ambition: AmbitionLevel,
    ) -> None:
        """All ambition levels must be calculable."""
        basic_input.ambition_level = ambition
        result = engine.calculate(basic_input)

        assert result.ambition_level == ambition

    def test_1_5c_most_ambitious(
        self, engine: TargetSettingEngine
    ) -> None:
        """1.5C target should be most stringent (highest reduction)."""
        inp_1_5c = TargetSettingInput(
            entity_name="Ambitious",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            sector="Technology",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2,
        )
        inp_2c = TargetSettingInput(
            entity_name="LessAmbitious",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            sector="Technology",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_2,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2,
        )
        result_1_5c = engine.calculate(inp_1_5c)
        result_2c = engine.calculate(inp_2c)

        # 1.5C target should be lower (more aggressive reduction)
        assert result_1_5c.target_s12_tco2e < result_2c.target_s12_tco2e


# ===========================================================================
# Tests -- Net-Zero Targets
# ===========================================================================


class TestNetZeroTargets:
    """Tests for net-zero target (2050 or earlier) handling."""

    def test_net_zero_target_by_2050(
        self, engine: TargetSettingEngine
    ) -> None:
        """Net-zero target must be achievable by 2050."""
        inp = TargetSettingInput(
            entity_name="NetZeroCorp",
            base_year=2024,
            base_scope1_tco2e=Decimal("2000"),
            base_scope2_tco2e=Decimal("1000"),
            base_scope3_tco2e=Decimal("4000"),
            sector="Technology",
            target_type=TargetType.NET_ZERO,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2050,
            target_scope=TargetScope.SCOPE_1_2_3,
        )
        result = engine.calculate(inp)

        assert result.target_type == TargetType.NET_ZERO
        assert result.target_year == 2050
        # Net-zero allows max 10% residual emissions (offset)
        assert result.residual_emissions_tco2e <= (inp.base_scope1_tco2e + inp.base_scope2_tco2e + inp.base_scope3_tco2e) * Decimal("0.10")

    def test_net_zero_earlier_than_2050(
        self, engine: TargetSettingEngine
    ) -> None:
        """Early net-zero commitments (e.g., 2045) are valid and more ambitious."""
        inp = TargetSettingInput(
            entity_name="EarlyNetZero",
            base_year=2024,
            base_scope1_tco2e=Decimal("1500"),
            base_scope2_tco2e=Decimal("750"),
            base_scope3_tco2e=Decimal("3000"),
            sector="Consumer Goods",
            target_type=TargetType.NET_ZERO,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2045,
            target_scope=TargetScope.SCOPE_1_2_3,
        )
        result = engine.calculate(inp)

        assert result.target_year == 2045
        assert result.target_type == TargetType.NET_ZERO


# ===========================================================================
# Tests -- Boundary and Scope Enforcement
# ===========================================================================


class TestBoundaryEnforcement:
    """Tests for target boundary validation."""

    def test_scope_1_2_only_target(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """Scope 1+2 only target should not include Scope 3."""
        basic_input.target_scope = TargetScope.SCOPE_1_2
        result = engine.calculate(basic_input)

        assert result.target_scope == TargetScope.SCOPE_1_2
        assert result.target_s12_tco2e > Decimal("0")
        assert result.target_s3_tco2e == Decimal("0") or result.target_s3_tco2e is None

    def test_scope_1_2_3_combined_target(
        self, engine: TargetSettingEngine, full_scope_input: TargetSettingInput
    ) -> None:
        """Full scope target includes all three scopes."""
        full_scope_input.target_scope = TargetScope.SCOPE_1_2_3
        result = engine.calculate(full_scope_input)

        assert result.target_scope == TargetScope.SCOPE_1_2_3
        assert result.target_s12_tco2e > Decimal("0")
        assert result.target_s3_tco2e > Decimal("0")
        # Total target should be sum of scope targets
        assert result.target_tco2e == result.target_s12_tco2e + result.target_s3_tco2e

    def test_scope_1_only_target(
        self, engine: TargetSettingEngine
    ) -> None:
        """Scope 1-only targets are rare but valid."""
        inp = TargetSettingInput(
            entity_name="Scope1OnlyCorp",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            sector="Technology",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1,
        )
        result = engine.calculate(inp)

        assert result.target_scope == TargetScope.SCOPE_1


# ===========================================================================
# Tests -- Provenance and Zero-Hallucination
# ===========================================================================


class TestProvenanceAndValidation:
    """Tests for deterministic hashing and zero-hallucination guarantees."""

    def test_result_has_provenance_hash(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """Every result must have a SHA-256 provenance hash."""
        result = engine.calculate(basic_input)

        assert hasattr(result, "provenance_hash")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_deterministic(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """Same input must produce same provenance hash."""
        result1 = engine.calculate(basic_input)
        result2 = engine.calculate(basic_input)

        assert result1.provenance_hash == result2.provenance_hash

    def test_different_input_different_hash(
        self, engine: TargetSettingEngine, basic_input: TargetSettingInput
    ) -> None:
        """Different inputs must produce different hashes."""
        result1 = engine.calculate(basic_input)

        modified_input = basic_input.model_copy()
        modified_input.base_scope1_tco2e = Decimal("1500")
        result2 = engine.calculate(modified_input)

        assert result1.provenance_hash != result2.provenance_hash


# ===========================================================================
# Tests -- Edge Cases and Boundary Conditions
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_baseline_emissions(
        self, engine: TargetSettingEngine
    ) -> None:
        """Small baseline emissions should still calculate correctly."""
        inp = TargetSettingInput(
            entity_name="SmallCorp",
            base_year=2024,
            base_scope1_tco2e=Decimal("10"),
            base_scope2_tco2e=Decimal("5"),
            sector="Technology",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2,
        )
        result = engine.calculate(inp)

        assert result.target_s12_tco2e > Decimal("0")
        assert result.target_s12_tco2e < inp.base_scope1_tco2e + inp.base_scope2_tco2e

    def test_zero_scope_3_emissions(
        self, engine: TargetSettingEngine
    ) -> None:
        """Zero Scope 3 emissions should be handled gracefully."""
        inp = TargetSettingInput(
            entity_name="NoScope3",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            base_scope3_tco2e=Decimal("0"),
            sector="Energy",
            target_type=TargetType.NEAR_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2030,
            target_scope=TargetScope.SCOPE_1_2,
        )
        result = engine.calculate(inp)

        assert result.target_s12_tco2e > Decimal("0")
        assert result.target_s3_tco2e == Decimal("0")

    def test_very_long_time_horizon(
        self, engine: TargetSettingEngine
    ) -> None:
        """Very long time horizons (25+ years) should calculate correctly."""
        inp = TargetSettingInput(
            entity_name="LongHorizon",
            base_year=2024,
            base_scope1_tco2e=Decimal("1000"),
            base_scope2_tco2e=Decimal("500"),
            sector="Technology",
            target_type=TargetType.LONG_TERM,
            ambition_level=AmbitionLevel.CELSIUS_1_5,
            pathway_method=PathwayMethod.ACA,
            target_year=2050,
            target_scope=TargetScope.SCOPE_1_2,
        )
        result = engine.calculate(inp)

        assert result.target_s12_tco2e > Decimal("0")
        assert len(result.pathway_milestones) == 27  # 2024-2050


# ===========================================================================
# Tests -- Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_invalid_target_year_before_base(
        self, engine: TargetSettingEngine
    ) -> None:
        """Target year must be after base year."""
        with pytest.raises((ValueError, AssertionError)):
            inp = TargetSettingInput(
                entity_name="BadYear",
                base_year=2024,
                base_scope1_tco2e=Decimal("1000"),
                base_scope2_tco2e=Decimal("500"),
                sector="Technology",
                target_type=TargetType.NEAR_TERM,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
                pathway_method=PathwayMethod.ACA,
                target_year=2020,  # Before base year
                target_scope=TargetScope.SCOPE_1_2,
            )

    def test_negative_baseline_rejected(
        self, engine: TargetSettingEngine
    ) -> None:
        """Negative emissions baseline should be rejected."""
        with pytest.raises((ValueError, AssertionError)):
            inp = TargetSettingInput(
                entity_name="NegativeEmissions",
                base_year=2024,
                base_scope1_tco2e=Decimal("-500"),
                base_scope2_tco2e=Decimal("500"),
                sector="Technology",
                target_type=TargetType.NEAR_TERM,
                ambition_level=AmbitionLevel.CELSIUS_1_5,
                pathway_method=PathwayMethod.ACA,
                target_year=2030,
                target_scope=TargetScope.SCOPE_1_2,
            )
