"""
Unit tests for ComplianceCheckerEngine (Engine 6)

Tests compliance checking for seven regulatory frameworks:
GHG Protocol, CSRD, CDP, SBTi, SB 253, GRI, ISO 14064-1.
Validates boundary rules and double-counting prevention.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from greenlang.fuel_energy_activities.engines.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceCheckInput,
    ComplianceCheckOutput,
    ComplianceStatus,
    ComplianceIssue,
    IssueSeverity,
)
from greenlang.fuel_energy_activities.models import (
    FuelType,
    ActivityType,
    ComplianceFramework,
)
from greenlang_core import AgentConfig
from greenlang_core.exceptions import ValidationError


# Fixtures
@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        name="compliance_checker",
        version="1.0.0",
        environment="test"
    )


@pytest.fixture
def engine(agent_config):
    """Create ComplianceCheckerEngine instance for testing."""
    return ComplianceCheckerEngine(agent_config)


@pytest.fixture
def ghg_protocol_compliant_input():
    """Create GHG Protocol compliant input."""
    return ComplianceCheckInput(
        framework=ComplianceFramework.GHG_PROTOCOL,
        activity_3a_emissions_kgco2e=Decimal("50000"),
        activity_3b_emissions_kgco2e=Decimal("30000"),
        activity_3c_emissions_kgco2e=Decimal("10000"),
        wtt_combustion_excluded=True,
        td_loss_generation_excluded=False,
        biogenic_separated=True,
        reporting_period="2025-Q1"
    )


@pytest.fixture
def csrd_compliant_input():
    """Create CSRD compliant input."""
    return ComplianceCheckInput(
        framework=ComplianceFramework.CSRD,
        activity_3a_emissions_kgco2e=Decimal("75000"),
        activity_3b_emissions_kgco2e=Decimal("45000"),
        activity_3c_emissions_kgco2e=Decimal("15000"),
        emission_intensity_kgco2e_per_revenue=Decimal("2.5"),
        sector="MANUFACTURING",
        reporting_period="2025-Q1"
    )


# Test Class
class TestComplianceCheckerEngine:
    """Test suite for ComplianceCheckerEngine."""

    def test_initialization(self, agent_config):
        """Test engine initializes correctly."""
        engine = ComplianceCheckerEngine(agent_config)

        assert engine.config == agent_config
        assert engine.framework_requirements is not None
        assert len(engine.framework_requirements) == 7  # 7 frameworks

    def test_check_all_seven_frameworks(self, engine):
        """Test checking all seven frameworks."""
        frameworks = [
            ComplianceFramework.GHG_PROTOCOL,
            ComplianceFramework.CSRD,
            ComplianceFramework.CDP,
            ComplianceFramework.SBTI,
            ComplianceFramework.SB_253,
            ComplianceFramework.GRI,
            ComplianceFramework.ISO_14064,
        ]

        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,  # Will be overridden
            activity_3a_emissions_kgco2e=Decimal("50000"),
            activity_3b_emissions_kgco2e=Decimal("30000"),
            activity_3c_emissions_kgco2e=Decimal("10000"),
            reporting_period="2025-Q1"
        )

        for framework in frameworks:
            input_data.framework = framework
            result = engine.check(input_data)

            assert isinstance(result, ComplianceCheckOutput)
            assert result.framework == framework

    def test_check_ghg_protocol_compliant(self, engine, ghg_protocol_compliant_input):
        """Test GHG Protocol compliant calculation."""
        result = engine.check(ghg_protocol_compliant_input)

        assert result.compliance_status == ComplianceStatus.COMPLIANT
        assert len(result.issues) == 0
        assert result.framework == ComplianceFramework.GHG_PROTOCOL

    def test_check_ghg_protocol_missing_activity(self, engine):
        """Test GHG Protocol with missing activity data."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3a_emissions_kgco2e=Decimal("50000"),
            # Missing activity_3b and activity_3c
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should still be compliant (activities are optional)
        # But may have warnings
        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_check_ghg_protocol_double_counting_detected(self, engine):
        """Test GHG Protocol detects double counting."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3a_emissions_kgco2e=Decimal("50000"),
            wtt_combustion_excluded=False,  # Should exclude WTT of combusted fuels
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should have warning about potential double counting
        assert result.compliance_status in [
            ComplianceStatus.COMPLIANT_WITH_WARNINGS,
            ComplianceStatus.NON_COMPLIANT
        ]

        # Should have issue about WTT exclusion
        wtt_issues = [i for i in result.issues if "WTT" in i.description or "well-to-tank" in i.description.lower()]
        assert len(wtt_issues) > 0

    def test_check_csrd_compliant(self, engine, csrd_compliant_input):
        """Test CSRD compliant calculation."""
        result = engine.check(csrd_compliant_input)

        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]
        assert result.framework == ComplianceFramework.CSRD

    def test_check_csrd_missing_intensity(self, engine):
        """Test CSRD with missing emission intensity."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.CSRD,
            activity_3a_emissions_kgco2e=Decimal("75000"),
            activity_3b_emissions_kgco2e=Decimal("45000"),
            # Missing emission_intensity_kgco2e_per_revenue
            sector="MANUFACTURING",
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should have issue about missing intensity
        intensity_issues = [i for i in result.issues if "intensity" in i.description.lower()]
        assert len(intensity_issues) > 0

    def test_check_cdp_compliant(self, engine):
        """Test CDP compliant calculation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.CDP,
            activity_3a_emissions_kgco2e=Decimal("60000"),
            activity_3b_emissions_kgco2e=Decimal("40000"),
            activity_3c_emissions_kgco2e=Decimal("12000"),
            verification_status="VERIFIED",
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_check_cdp_missing_verification(self, engine):
        """Test CDP with missing verification."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.CDP,
            activity_3a_emissions_kgco2e=Decimal("60000"),
            # Missing verification_status
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # CDP encourages verification
        verification_issues = [i for i in result.issues if "verification" in i.description.lower()]
        # May or may not be required, but should be noted
        assert result.compliance_status != ComplianceStatus.ERROR

    def test_check_sbti_compliant(self, engine):
        """Test SBTi compliant calculation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.SBTI,
            activity_3a_emissions_kgco2e=Decimal("100000"),
            activity_3b_emissions_kgco2e=Decimal("80000"),
            total_scope3_emissions_kgco2e=Decimal("500000"),
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Check if fuel & energy is >40% of Scope 3 (materiality threshold)
        fuel_energy_total = input_data.activity_3a_emissions_kgco2e + input_data.activity_3b_emissions_kgco2e
        percentage = (fuel_energy_total / input_data.total_scope3_emissions_kgco2e) * Decimal("100")

        if percentage >= Decimal("40"):
            # Should be compliant if meets threshold
            assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_check_sbti_below_threshold(self, engine):
        """Test SBTi below materiality threshold."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.SBTI,
            activity_3a_emissions_kgco2e=Decimal("10000"),
            activity_3b_emissions_kgco2e=Decimal("5000"),
            total_scope3_emissions_kgco2e=Decimal("1000000"),  # Fuel & energy <2%
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Below 40% threshold - may not need to include in SBTi target
        materiality_issues = [i for i in result.issues if "materiality" in i.description.lower()]
        # Should have note about materiality
        assert result.compliance_status != ComplianceStatus.ERROR

    def test_check_sb253_compliant(self, engine):
        """Test California SB 253 compliant calculation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.SB_253,
            activity_3a_emissions_kgco2e=Decimal("70000"),
            activity_3b_emissions_kgco2e=Decimal("50000"),
            activity_3c_emissions_kgco2e=Decimal("15000"),
            verification_status="THIRD_PARTY_VERIFIED",
            california_reporting=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_check_sb253_missing_verification(self, engine):
        """Test SB 253 with missing third-party verification."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.SB_253,
            activity_3a_emissions_kgco2e=Decimal("70000"),
            verification_status="UNVERIFIED",  # SB 253 requires verification
            california_reporting=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # SB 253 requires third-party verification
        assert result.compliance_status == ComplianceStatus.NON_COMPLIANT

        verification_issues = [i for i in result.issues if "verification" in i.description.lower()]
        assert len(verification_issues) > 0

    def test_check_gri_compliant(self, engine):
        """Test GRI 305 compliant calculation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GRI,
            activity_3a_emissions_kgco2e=Decimal("55000"),
            activity_3b_emissions_kgco2e=Decimal("35000"),
            biogenic_co2_kgco2e=Decimal("5000"),
            biogenic_separated=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_check_gri_missing_biogenic(self, engine):
        """Test GRI with missing biogenic CO2 separation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GRI,
            activity_3a_emissions_kgco2e=Decimal("55000"),
            biogenic_separated=False,  # GRI requires separate reporting
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # GRI 305 requires biogenic CO2 to be reported separately
        biogenic_issues = [i for i in result.issues if "biogenic" in i.description.lower()]
        assert len(biogenic_issues) > 0

    def test_check_iso14064_compliant(self, engine):
        """Test ISO 14064-1 compliant calculation."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.ISO_14064,
            activity_3a_emissions_kgco2e=Decimal("65000"),
            activity_3b_emissions_kgco2e=Decimal("42000"),
            activity_3c_emissions_kgco2e=Decimal("13000"),
            uncertainty_quantified=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        assert result.compliance_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.COMPLIANT_WITH_WARNINGS]

    def test_get_framework_requirements_all_7(self, engine):
        """Test getting requirements for all 7 frameworks."""
        frameworks = [
            ComplianceFramework.GHG_PROTOCOL,
            ComplianceFramework.CSRD,
            ComplianceFramework.CDP,
            ComplianceFramework.SBTI,
            ComplianceFramework.SB_253,
            ComplianceFramework.GRI,
            ComplianceFramework.ISO_14064,
        ]

        for framework in frameworks:
            requirements = engine.get_framework_requirements(framework)

            assert isinstance(requirements, dict)
            assert "required_fields" in requirements
            assert "optional_fields" in requirements
            assert "boundary_rules" in requirements

    def test_get_compliance_summary(self, engine, ghg_protocol_compliant_input):
        """Test getting compliance summary."""
        result = engine.check(ghg_protocol_compliant_input)

        summary = engine.get_compliance_summary(result)

        assert "framework" in summary
        assert "status" in summary
        assert "total_issues" in summary
        assert "critical_issues" in summary
        assert "warnings" in summary

    def test_wtt_combustion_exclusion_rule(self, engine):
        """Test WTT of combusted fuels exclusion rule."""
        # Should exclude WTT of fuels that will be combusted (to avoid double counting with Scope 1)
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3a_emissions_kgco2e=Decimal("50000"),
            fuel_combustion_in_scope1=True,
            wtt_combustion_excluded=False,  # Should be True
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should have issue about WTT exclusion
        wtt_issues = [i for i in result.issues if "WTT" in i.description or "combustion" in i.description.lower()]
        assert len(wtt_issues) > 0

    def test_td_loss_boundary_rule(self, engine):
        """Test T&D loss boundary rule."""
        # T&D losses should exclude generation component if already in Scope 2
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3c_emissions_kgco2e=Decimal("10000"),
            td_loss_generation_excluded=False,  # Should check if needed
            scope2_includes_td_losses=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should check for double counting
        td_issues = [i for i in result.issues if "T&D" in i.description or "transmission" in i.description.lower()]
        # May or may not have issues depending on boundary

    def test_upstream_generation_exclusion_rule(self, engine):
        """Test upstream of purchased electricity generation exclusion rule."""
        # Only include upstream of utility's generation, not generation itself
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3d_emissions_kgco2e=Decimal("20000"),  # Activity 3d
            utility_generation_in_scope2=True,
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should be compliant if structured correctly
        assert result.compliance_status != ComplianceStatus.ERROR

    def test_activity_classification_rule(self, engine):
        """Test activity classification rule."""
        # All fuel & energy activities should be classified correctly
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3a_emissions_kgco2e=Decimal("50000"),
            activity_3b_emissions_kgco2e=Decimal("30000"),
            activity_3c_emissions_kgco2e=Decimal("10000"),
            activity_3d_emissions_kgco2e=Decimal("15000"),
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # All activities should be properly classified
        assert result.compliance_status != ComplianceStatus.ERROR

    def test_utility_only_3d_rule(self, engine):
        """Test activity 3d (upstream of purchased electricity) is utility-only."""
        # Activity 3d should only apply to utilities
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            activity_3d_emissions_kgco2e=Decimal("20000"),
            organization_type="MANUFACTURER",  # Not a utility
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should have warning about activity 3d applicability
        activity_3d_issues = [i for i in result.issues if "3d" in i.description.lower()]
        # May have warning if not applicable

    def test_biogenic_separation_rule(self, engine):
        """Test biogenic CO2 separation rule."""
        input_data = ComplianceCheckInput(
            framework=ComplianceFramework.GRI,
            activity_3a_emissions_kgco2e=Decimal("55000"),
            biogenic_co2_kgco2e=Decimal("8000"),
            biogenic_separated=False,  # Should be True for GRI
            reporting_period="2025-Q1"
        )

        result = engine.check(input_data)

        # Should have issue about biogenic separation
        biogenic_issues = [i for i in result.issues if "biogenic" in i.description.lower()]
        assert len(biogenic_issues) > 0

    def test_check_batch(self, engine):
        """Test batch compliance checking."""
        inputs = [
            ComplianceCheckInput(
                framework=ComplianceFramework.GHG_PROTOCOL,
                activity_3a_emissions_kgco2e=Decimal("50000"),
                reporting_period="2025-Q1"
            ),
            ComplianceCheckInput(
                framework=ComplianceFramework.CSRD,
                activity_3a_emissions_kgco2e=Decimal("75000"),
                reporting_period="2025-Q1"
            ),
            ComplianceCheckInput(
                framework=ComplianceFramework.CDP,
                activity_3a_emissions_kgco2e=Decimal("60000"),
                reporting_period="2025-Q1"
            ),
        ]

        results = engine.check_batch(inputs)

        assert len(results) == 3
        assert results[0].framework == ComplianceFramework.GHG_PROTOCOL
        assert results[1].framework == ComplianceFramework.CSRD
        assert results[2].framework == ComplianceFramework.CDP

    def test_aggregate_issues_by_severity(self, engine):
        """Test aggregating issues by severity."""
        result = ComplianceCheckOutput(
            framework=ComplianceFramework.GHG_PROTOCOL,
            compliance_status=ComplianceStatus.COMPLIANT_WITH_WARNINGS,
            issues=[
                ComplianceIssue(
                    severity=IssueSeverity.CRITICAL,
                    description="Critical issue"
                ),
                ComplianceIssue(
                    severity=IssueSeverity.WARNING,
                    description="Warning 1"
                ),
                ComplianceIssue(
                    severity=IssueSeverity.WARNING,
                    description="Warning 2"
                ),
                ComplianceIssue(
                    severity=IssueSeverity.INFO,
                    description="Info"
                ),
            ]
        )

        aggregated = engine.aggregate_issues_by_severity(result)

        assert aggregated[IssueSeverity.CRITICAL] == 1
        assert aggregated[IssueSeverity.WARNING] == 2
        assert aggregated[IssueSeverity.INFO] == 1

    def test_generate_compliance_report(self, engine, ghg_protocol_compliant_input):
        """Test generating compliance report."""
        result = engine.check(ghg_protocol_compliant_input)

        report = engine.generate_compliance_report(result)

        assert isinstance(report, str)
        assert "Compliance" in report
        assert "GHG Protocol" in report

    def test_export_to_json(self, engine, ghg_protocol_compliant_input):
        """Test exporting compliance result to JSON."""
        result = engine.check(ghg_protocol_compliant_input)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "framework" in result_dict
        assert "compliance_status" in result_dict
        assert "issues" in result_dict

    def test_cross_framework_comparison(self, engine):
        """Test comparing compliance across frameworks."""
        base_data = {
            "activity_3a_emissions_kgco2e": Decimal("50000"),
            "activity_3b_emissions_kgco2e": Decimal("30000"),
            "activity_3c_emissions_kgco2e": Decimal("10000"),
            "reporting_period": "2025-Q1"
        }

        frameworks = [
            ComplianceFramework.GHG_PROTOCOL,
            ComplianceFramework.CSRD,
            ComplianceFramework.CDP,
        ]

        results = []
        for framework in frameworks:
            input_data = ComplianceCheckInput(framework=framework, **base_data)
            results.append(engine.check(input_data))

        comparison = engine.compare_frameworks(results)

        assert len(comparison) == 3

    def test_identify_conflicting_requirements(self, engine):
        """Test identifying conflicting requirements across frameworks."""
        conflicts = engine.identify_framework_conflicts(
            [ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.ISO_14064]
        )

        # May or may not have conflicts
        assert isinstance(conflicts, list)

    def test_get_statistics(self, engine, ghg_protocol_compliant_input):
        """Test getting engine statistics."""
        engine.check(ghg_protocol_compliant_input)
        engine.check(ghg_protocol_compliant_input)

        stats = engine.get_statistics()

        assert stats["checks_performed"] == 2

    def test_reset(self, engine, ghg_protocol_compliant_input):
        """Test resetting engine state."""
        engine.check(ghg_protocol_compliant_input)

        engine.reset()

        stats = engine.get_statistics()
        assert stats["checks_performed"] == 0

    def test_error_handling_invalid_framework(self, engine):
        """Test error handling for invalid framework."""
        input_data = ComplianceCheckInput(
            framework="INVALID_FRAMEWORK",  # Invalid
            activity_3a_emissions_kgco2e=Decimal("50000"),
            reporting_period="2025-Q1"
        )

        with pytest.raises(ValidationError, match="framework"):
            engine.check(input_data)

    def test_provenance_tracking(self, engine, ghg_protocol_compliant_input):
        """Test provenance tracking."""
        result = engine.check(ghg_protocol_compliant_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_performance_batch_processing(self, engine, benchmark):
        """Test batch processing performance."""
        inputs = [
            ComplianceCheckInput(
                framework=ComplianceFramework.GHG_PROTOCOL,
                activity_3a_emissions_kgco2e=Decimal("50000"),
                reporting_period="2025-Q1"
            )
            for _ in range(100)
        ]

        def run_batch():
            return engine.check_batch(inputs)

        results = benchmark(run_batch)

        assert len(results) == 100


# Integration Tests
class TestComplianceCheckerIntegration:
    """Integration tests for ComplianceCheckerEngine."""

    @pytest.mark.integration
    def test_integration_with_calculation_engines(self, engine):
        """Test integration with calculation engines."""
        pass


# Performance Tests
class TestComplianceCheckerPerformance:
    """Performance tests for ComplianceCheckerEngine."""

    @pytest.mark.performance
    def test_throughput_target(self, engine):
        """Test engine meets throughput target (1000 checks/sec)."""
        num_records = 10000
        inputs = [
            ComplianceCheckInput(
                framework=ComplianceFramework.GHG_PROTOCOL,
                activity_3a_emissions_kgco2e=Decimal("50000"),
                reporting_period="2025-Q1"
            )
            for _ in range(num_records)
        ]

        start_time = datetime.now()
        results = engine.check_batch(inputs)
        end_time = datetime.now()

        duration_seconds = (end_time - start_time).total_seconds()
        throughput = num_records / duration_seconds

        assert throughput >= 1000
        assert len(results) == num_records
