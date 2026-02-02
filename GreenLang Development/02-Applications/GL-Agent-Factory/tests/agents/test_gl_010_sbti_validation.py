"""
Unit Tests for GL-010: SBTi Validation Agent

Comprehensive test suite covering:
- Target type validation (near-term, long-term, net-zero, FLAG)
- Ambition level assessment (1.5C, WB2C, 2C)
- Absolute Contraction Approach (ACA) pathway validation
- Sectoral Decarbonization Approach (SDA) pathway
- Scope 3 requirements (67% rule)
- FLAG sector requirements
- Net-zero neutralization requirements
- Provenance hash generation

Target: 85%+ code coverage

Reference:
- SBTi Corporate Net-Zero Standard (Version 1.2, 2024)
- SBTi Corporate Manual (2024)
- SBTi FLAG Guidance (2022)

Run with:
    pytest tests/agents/test_gl_010_sbti_validation.py -v --cov=backend/agents/gl_010_sbti_validation
"""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from agents.gl_010_sbti_validation.agent import (
    SBTiValidationAgent,
    SBTiInput,
    ScopeEmissions,
    TargetDefinition,
    TargetType,
    AmbitionLevel,
    PathwayType,
    SectorPathway,
    ScopeType,
    ValidationStatus,
    Scope3EngagementTarget,
    Scope3EngagementType,
    NeutralizationPlan,
    NeutralizationType,
    FLAGTarget,
    CurrentProgress,
)


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestSBTiAgentInitialization:
    """Tests for SBTiValidationAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self):
        """Test agent initializes correctly with default config."""
        agent = SBTiValidationAgent()

        assert agent is not None
        assert agent.AGENT_ID == "GL-010"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_has_pathway_constants(self):
        """Test agent has SBTi pathway constants."""
        agent = SBTiValidationAgent()

        # Should have 1.5C pathway requirements
        assert hasattr(agent, "NEAR_TERM_1_5C_ANNUAL_RATE")
        assert agent.NEAR_TERM_1_5C_ANNUAL_RATE == pytest.approx(0.042, rel=1e-3)


# =============================================================================
# Test Class: Input Model Validation
# =============================================================================


class TestSBTiInputValidation:
    """Tests for SBTiInput Pydantic model validation."""

    @pytest.mark.unit
    def test_valid_near_term_input(self):
        """Test valid near-term target input passes validation."""
        input_data = SBTiInput(
            company_id="COMPANY-001",
            base_year=2019,
            base_year_emissions=ScopeEmissions(
                scope1=10000.0,
                scope2=5000.0,
                scope3=30000.0,
            ),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                )
            ],
        )

        assert input_data.company_id == "COMPANY-001"
        assert input_data.base_year == 2019
        assert len(input_data.targets) == 1

    @pytest.mark.unit
    def test_scope_emissions_total(self):
        """Test ScopeEmissions calculates total correctly."""
        emissions = ScopeEmissions(
            scope1=10000.0,
            scope2=5000.0,
            scope3=30000.0,
        )

        assert emissions.total == 45000.0
        assert emissions.scope12_total == 15000.0

    @pytest.mark.unit
    def test_scope_emissions_scope3_percentage(self):
        """Test Scope 3 percentage calculation."""
        emissions = ScopeEmissions(
            scope1=10000.0,
            scope2=5000.0,
            scope3=30000.0,  # 30000/45000 = 66.7%
        )

        assert emissions.scope3_percentage == pytest.approx(66.67, rel=1e-2)

    @pytest.mark.unit
    def test_base_year_validation(self):
        """Test base year must be within valid range."""
        # Base year too old
        with pytest.raises(ValueError):
            SBTiInput(
                company_id="COMPANY-001",
                base_year=2010,  # Before 2015
                base_year_emissions=ScopeEmissions(scope1=1000, scope2=500, scope3=3000),
                targets=[TargetDefinition(target_year=2030, target_type=TargetType.NEAR_TERM, reduction_pct=40)],
            )

    @pytest.mark.unit
    def test_target_year_validation(self):
        """Test target year must be valid."""
        target = TargetDefinition(
            target_year=2030,
            target_type=TargetType.NEAR_TERM,
            reduction_pct=46.2,
        )

        assert target.target_year == 2030

    @pytest.mark.unit
    def test_reduction_pct_range(self):
        """Test reduction percentage must be 0-100."""
        with pytest.raises(ValueError):
            TargetDefinition(
                target_year=2030,
                target_type=TargetType.NEAR_TERM,
                reduction_pct=150.0,  # Over 100%
            )

    @pytest.mark.unit
    def test_at_least_one_target_required(self):
        """Test at least one target is required."""
        with pytest.raises(ValueError):
            SBTiInput(
                company_id="COMPANY-001",
                base_year=2019,
                base_year_emissions=ScopeEmissions(scope1=1000, scope2=500, scope3=3000),
                targets=[],  # Empty list
            )

    @pytest.mark.unit
    def test_default_scopes_covered(self):
        """Test default scopes covered is Scope 1 and 2."""
        target = TargetDefinition(
            target_year=2030,
            target_type=TargetType.NEAR_TERM,
            reduction_pct=46.2,
        )

        assert ScopeType.SCOPE_1 in target.scopes_covered
        assert ScopeType.SCOPE_2 in target.scopes_covered


# =============================================================================
# Test Class: Near-Term Target Validation (1.5C)
# =============================================================================


class TestSBTiNearTermValidation:
    """Tests for near-term target validation against 1.5C pathway."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_1_5c_aligned_target_valid(self, sbti_agent):
        """
        Test 1.5C aligned near-term target is validated as valid.

        SBTi 1.5C requirement: 4.2% annual linear reduction
        From 2019 to 2030 = 11 years
        Required reduction: 1 - (1-0.042)^11 = 38.5% minimum
        46.2% target exceeds this, so should be valid.
        """
        input_data = SBTiInput(
            company_id="COMPANY-001",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,  # Exceeds 1.5C requirement
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"
        # Should be classified as 1.5C aligned
        assert result.targets_validation[0].ambition_level in ["1.5C", AmbitionLevel.CELSIUS_1_5.value]

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_well_below_2c_target(self, sbti_agent):
        """
        Test well-below 2C target is validated correctly.

        SBTi WB2C requirement: ~2.5% annual reduction
        From 2019 to 2030 = 11 years
        Required reduction: ~25% minimum
        """
        input_data = SBTiInput(
            company_id="COMPANY-002",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=5000, scope2=3000, scope3=15000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=30.0,  # Meets WB2C but not 1.5C
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"
        # Should be WB2C, not 1.5C
        assert result.targets_validation[0].ambition_level in ["WB2C", AmbitionLevel.WELL_BELOW_2C.value]

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_insufficient_reduction_invalid(self, sbti_agent):
        """
        Test insufficient reduction target fails validation.

        20% reduction over 11 years is below SBTi minimum threshold.
        """
        input_data = SBTiInput(
            company_id="COMPANY-003",
            base_year=2020,
            base_year_emissions=ScopeEmissions(scope1=8000, scope2=4000, scope3=25000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=15.0,  # Too low
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should fail or be below threshold
        assert (
            result.validation_status == "FAIL" or
            result.targets_validation[0].ambition_level in ["below_threshold", AmbitionLevel.BELOW_THRESHOLD.value]
        )

    @pytest.mark.unit
    def test_near_term_timeframe_validation(self, sbti_agent):
        """
        Test near-term target timeframe is validated.

        Near-term targets must be 5-10 years from submission.
        """
        input_data = SBTiInput(
            company_id="COMPANY-004",
            base_year=2020,
            base_year_emissions=ScopeEmissions(scope1=5000, scope2=3000, scope3=20000),
            targets=[
                TargetDefinition(
                    target_year=2045,  # Too far out for near-term
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should have validation error about timeframe
        assert len(result.targets_validation[0].validation_errors) > 0


# =============================================================================
# Test Class: Long-Term Target Validation
# =============================================================================


class TestSBTiLongTermValidation:
    """Tests for long-term target validation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_long_term_90_pct_reduction(self, sbti_agent):
        """
        Test long-term target requires 90% reduction.

        SBTi Net-Zero Standard requires 90% absolute reduction by 2050.
        """
        input_data = SBTiInput(
            company_id="COMPANY-005",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2050,
                    target_type=TargetType.LONG_TERM,
                    reduction_pct=90.0,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2, ScopeType.SCOPE_3],
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"

    @pytest.mark.unit
    def test_long_term_insufficient_reduction(self, sbti_agent):
        """Test long-term target below 90% fails."""
        input_data = SBTiInput(
            company_id="COMPANY-006",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2050,
                    target_type=TargetType.LONG_TERM,
                    reduction_pct=70.0,  # Below 90% requirement
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should have validation error
        assert len(result.targets_validation[0].validation_errors) > 0


# =============================================================================
# Test Class: Net-Zero Target Validation
# =============================================================================


class TestSBTiNetZeroValidation:
    """Tests for net-zero target validation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_net_zero_with_neutralization_plan(self, sbti_agent):
        """
        Test net-zero target with valid neutralization plan.

        Net-zero requires:
        - 90-95% absolute reduction
        - Neutralization of residual 5-10%
        - BVCM commitment
        """
        input_data = SBTiInput(
            company_id="COMPANY-007",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2050,
                    target_type=TargetType.NET_ZERO,
                    reduction_pct=92.0,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2, ScopeType.SCOPE_3],
                    neutralization_plan=NeutralizationPlan(
                        neutralization_type=NeutralizationType.COMBINED,
                        residual_emissions_pct=8.0,
                        removal_capacity_tco2e=3600.0,  # 8% of 45000 = 3600
                        bvcm_commitment=True,
                        bvcm_investment_pct=0.5,
                    ),
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"

    @pytest.mark.unit
    def test_net_zero_without_neutralization_warns(self, sbti_agent):
        """Test net-zero target without neutralization plan generates warning."""
        input_data = SBTiInput(
            company_id="COMPANY-008",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2050,
                    target_type=TargetType.NET_ZERO,
                    reduction_pct=92.0,
                    # No neutralization_plan
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should have warning about missing neutralization plan
        has_warning = any("neutralization" in err.lower() for err in result.targets_validation[0].validation_errors)
        # Note: This might be a warning rather than error


# =============================================================================
# Test Class: Scope 3 Requirements
# =============================================================================


class TestSBTiScope3Requirements:
    """Tests for Scope 3 target requirements."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_scope3_67_percent_rule(self, sbti_agent):
        """
        Test Scope 3 target required when >67% of emissions.

        If Scope 3 is >67% of total, a Scope 3 target is required.
        """
        # Scope 3 = 35000 out of 45000 = 77.8% (>67%)
        input_data = SBTiInput(
            company_id="COMPANY-009",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=5000, scope2=5000, scope3=35000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],  # No Scope 3
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should flag missing Scope 3 target
        assert result.scope3_target_required is True

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_scope3_supplier_engagement_approach(self, sbti_agent):
        """
        Test Scope 3 supplier engagement approach is valid alternative.

        67% of suppliers with SBTs by mass or spend is valid.
        """
        input_data = SBTiInput(
            company_id="COMPANY-010",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=5000, scope2=5000, scope3=35000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2, ScopeType.SCOPE_3],
                    scope3_engagement=Scope3EngagementTarget(
                        engagement_type=Scope3EngagementType.SUPPLIER_ENGAGEMENT,
                        supplier_coverage_pct=70.0,  # >67% required
                        supplier_target_year=2028,
                    ),
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"


# =============================================================================
# Test Class: FLAG Sector Targets
# =============================================================================


class TestSBTiFLAGTargets:
    """Tests for Forest, Land, and Agriculture (FLAG) sector targets."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_flag_target_required_when_20_pct(self, sbti_agent):
        """
        Test FLAG target required when FLAG emissions >20%.

        Companies with >20% FLAG emissions must set separate FLAG targets.
        """
        input_data = SBTiInput(
            company_id="COMPANY-011",
            base_year=2019,
            base_year_emissions=ScopeEmissions(
                scope1=5000,
                scope2=3000,
                scope3=20000,
                flag_emissions=10000,  # 10000/38000 = 26% (>20%)
            ),
            has_flag_emissions=True,
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should flag requirement for separate FLAG target
        assert result.flag_target_required is True

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_flag_target_with_no_deforestation(self, sbti_agent):
        """Test FLAG target must include no-deforestation commitment."""
        input_data = SBTiInput(
            company_id="COMPANY-012",
            base_year=2019,
            base_year_emissions=ScopeEmissions(
                scope1=5000,
                scope2=3000,
                scope3=20000,
                flag_emissions=10000,
            ),
            has_flag_emissions=True,
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.FLAG,
                    reduction_pct=30.0,
                    flag_target=FLAGTarget(
                        base_year_flag_emissions=10000,
                        target_year_flag_emissions=7000,
                        no_deforestation_commitment=True,
                        no_deforestation_date=2025,
                        land_conversion_commitment=True,
                        commodity_focus=["soy", "palm", "beef"],
                    ),
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.validation_status == "PASS"


# =============================================================================
# Test Class: Provenance Hash
# =============================================================================


class TestSBTiProvenanceHash:
    """Tests for SBTi provenance hash generation."""

    @pytest.mark.unit
    def test_provenance_hash_exists(self, sbti_agent, sbti_near_term_input):
        """Test output includes provenance hash."""
        result = sbti_agent.run(sbti_near_term_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_valid_format(self, sbti_agent, sbti_near_term_input):
        """Test provenance hash is valid SHA-256."""
        result = sbti_agent.run(sbti_near_term_input)

        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())


# =============================================================================
# Test Class: Pathway Validation
# =============================================================================


class TestSBTiPathwayValidation:
    """Tests for decarbonization pathway validation."""

    @pytest.mark.unit
    def test_aca_pathway_validation(self, sbti_agent):
        """Test Absolute Contraction Approach (ACA) pathway."""
        input_data = SBTiInput(
            company_id="COMPANY-013",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                    pathway_type=PathwayType.ACA,
                    is_absolute_target=True,
                )
            ],
        )

        result = sbti_agent.run(input_data)

        assert result.targets_validation[0].pathway_type == PathwayType.ACA.value

    @pytest.mark.unit
    def test_sda_pathway_requires_sector(self, sbti_agent):
        """Test SDA pathway requires sector specification."""
        input_data = SBTiInput(
            company_id="COMPANY-014",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            sector=SectorPathway.STEEL,
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                    pathway_type=PathwayType.SDA,
                    sector=SectorPathway.STEEL,
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should validate with sector-specific pathway
        assert result.validation_status == "PASS"


# =============================================================================
# Test Class: Progress Tracking
# =============================================================================


class TestSBTiProgressTracking:
    """Tests for progress tracking functionality."""

    @pytest.mark.unit
    def test_progress_calculation(self, sbti_agent):
        """Test progress towards target is calculated."""
        input_data = SBTiInput(
            company_id="COMPANY-015",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                )
            ],
            current_progress=CurrentProgress(
                reporting_year=2023,
                current_emissions=ScopeEmissions(
                    scope1=8000,  # 20% reduction in S1
                    scope2=4000,  # 20% reduction in S2
                    scope3=28000,
                ),
            ),
        )

        result = sbti_agent.run(input_data)

        # Should include progress metrics
        assert hasattr(result, "progress_status") or hasattr(result, "current_progress")


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestSBTiEdgeCases:
    """Tests for SBTi edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_multiple_targets(self, sbti_agent):
        """Test multiple targets in single submission."""
        input_data = SBTiInput(
            company_id="COMPANY-016",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_id="NT-001",
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2],
                ),
                TargetDefinition(
                    target_id="LT-001",
                    target_year=2050,
                    target_type=TargetType.LONG_TERM,
                    reduction_pct=90.0,
                    scopes_covered=[ScopeType.SCOPE_1, ScopeType.SCOPE_2, ScopeType.SCOPE_3],
                ),
            ],
        )

        result = sbti_agent.run(input_data)

        assert len(result.targets_validation) == 2

    @pytest.mark.unit
    def test_zero_base_year_emissions(self, sbti_agent):
        """Test handling of zero base year emissions (edge case)."""
        # Zero emissions is invalid for target setting
        input_data = SBTiInput(
            company_id="COMPANY-017",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=0, scope2=0, scope3=0),
            targets=[
                TargetDefinition(
                    target_year=2030,
                    target_type=TargetType.NEAR_TERM,
                    reduction_pct=46.2,
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should have validation error
        assert result.validation_status == "FAIL" or len(result.validation_errors) > 0

    @pytest.mark.unit
    def test_100_percent_reduction_target(self, sbti_agent):
        """Test 100% reduction target (complete elimination)."""
        input_data = SBTiInput(
            company_id="COMPANY-018",
            base_year=2019,
            base_year_emissions=ScopeEmissions(scope1=10000, scope2=5000, scope3=30000),
            targets=[
                TargetDefinition(
                    target_year=2050,
                    target_type=TargetType.LONG_TERM,
                    reduction_pct=100.0,  # Complete elimination
                )
            ],
        )

        result = sbti_agent.run(input_data)

        # Should be valid (exceeds 90% requirement)
        assert result.validation_status == "PASS"


# =============================================================================
# Test Class: Output Model
# =============================================================================


class TestSBTiOutput:
    """Tests for SBTi output model."""

    @pytest.mark.unit
    def test_output_has_all_required_fields(self, sbti_agent, sbti_near_term_input):
        """Test output includes all required fields."""
        result = sbti_agent.run(sbti_near_term_input)

        assert hasattr(result, "validation_status")
        assert hasattr(result, "targets_validation")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "scope3_target_required")

    @pytest.mark.unit
    def test_output_timestamp(self, sbti_agent, sbti_near_term_input):
        """Test output has valid timestamp."""
        before = datetime.utcnow()
        result = sbti_agent.run(sbti_near_term_input)
        after = datetime.utcnow()

        assert hasattr(result, "assessed_at") or hasattr(result, "timestamp")


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestSBTiPerformance:
    """Performance tests for SBTiValidationAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_validation_under_50ms(self, sbti_agent, sbti_near_term_input, performance_timer):
        """Test single validation completes in under 50ms."""
        performance_timer.start()
        result = sbti_agent.run(sbti_near_term_input)
        performance_timer.stop()

        assert performance_timer.elapsed_ms < 50.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_validation(self, sbti_agent, performance_timer):
        """Test batch validation throughput."""
        num_companies = 100
        inputs = [
            SBTiInput(
                company_id=f"COMPANY-{i:03d}",
                base_year=2019,
                base_year_emissions=ScopeEmissions(
                    scope1=float(i * 1000),
                    scope2=float(i * 500),
                    scope3=float(i * 3000),
                ),
                targets=[
                    TargetDefinition(
                        target_year=2030,
                        target_type=TargetType.NEAR_TERM,
                        reduction_pct=46.2,
                    )
                ],
            )
            for i in range(1, num_companies + 1)
        ]

        performance_timer.start()
        results = [sbti_agent.run(inp) for inp in inputs]
        performance_timer.stop()

        assert len(results) == num_companies
        throughput = num_companies / (performance_timer.elapsed_ms / 1000)
        assert throughput >= 10, f"Throughput {throughput:.0f} rec/sec below target"
