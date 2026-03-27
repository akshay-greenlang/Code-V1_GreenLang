"""
Unit tests for ComplianceCheckerEngine (AGENT-MRV-014).

Tests cover:
- Singleton pattern
- GHG Protocol compliance
- CSRD/ESRS compliance
- CDP compliance
- SBTi compliance
- SB 253 compliance
- GRI compliance
- ISO 14064 compliance
- Cross-framework checks
- Threshold validation
- Health checks
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

try:
    from greenlang.agents.mrv.purchased_goods_services.compliance_checker import (
        ComplianceCheckerEngine,
        ComplianceInput,
        ComplianceOutput,
        ComplianceStatus,
        FrameworkRequirement,
        ComplianceReport,
    )
except ImportError:
    pytest.skip("ComplianceCheckerEngine not available", allow_module_level=True)


class TestComplianceCheckerSingleton:
    """Test singleton pattern for ComplianceCheckerEngine."""

    def test_singleton_same_instance(self):
        """Test that get_instance returns same instance."""
        engine1 = ComplianceCheckerEngine.get_instance()
        engine2 = ComplianceCheckerEngine.get_instance()
        assert engine1 is engine2

    def test_singleton_thread_safe(self):
        """Test thread-safe singleton creation."""
        import threading
        instances = []

        def get_instance():
            instances.append(ComplianceCheckerEngine.get_instance())

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_singleton_reset(self):
        """Test singleton reset functionality."""
        engine1 = ComplianceCheckerEngine.get_instance()
        ComplianceCheckerEngine.reset_instance()
        engine2 = ComplianceCheckerEngine.get_instance()
        assert engine1 is not engine2


class TestGHGProtocolCompliance:
    """Test GHG Protocol Corporate Standard compliance."""

    def test_all_requirements_met(self):
        """Test when all GHG Protocol requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = engine.check_ghg_protocol(data)

        assert result["compliant"] is True
        assert len(result["missing"]) == 0

    def test_missing_dqi_requirement(self):
        """Test when DQI requirement is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = engine.check_ghg_protocol(data)

        assert result["compliant"] is False
        assert "dqi" in [req["field"] for req in result["missing"]]

    def test_low_coverage_fails(self):
        """Test when coverage is below threshold."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("60.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = engine.check_ghg_protocol(data)

        assert result["compliant"] is False
        assert any("coverage" in req["field"] for req in result["missing"])

    def test_boundary_incomplete_flagged(self):
        """Test when boundary definition is incomplete."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": False,
            "documentation_complete": True,
            "base_year_set": True,
        }

        result = engine.check_ghg_protocol(data)

        assert result["compliant"] is False
        assert any("boundary" in req["field"] for req in result["missing"])

    def test_missing_base_year(self):
        """Test when base year is not set."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": False,
        }

        result = engine.check_ghg_protocol(data)

        assert result["compliant"] is False
        assert any("base_year" in req["field"] for req in result["missing"])


class TestCSRDESRSCompliance:
    """Test CSRD/ESRS E1 compliance."""

    def test_all_requirements_met(self):
        """Test when all CSRD/ESRS requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.5"),
            "supplier_engagement": True,
            "value_chain_scope": True,
            "materiality_assessed": True,
        }

        result = engine.check_csrd_esrs(data)

        assert result["compliant"] is True
        assert len(result["missing"]) == 0

    def test_missing_supplier_engagement(self):
        """Test when supplier engagement is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.5"),
            "supplier_engagement": False,
            "value_chain_scope": True,
            "materiality_assessed": True,
        }

        result = engine.check_csrd_esrs(data)

        assert result["compliant"] is False
        assert any("engagement" in req["field"] for req in result["missing"])

    def test_materiality_not_assessed(self):
        """Test when materiality assessment is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.5"),
            "supplier_engagement": True,
            "value_chain_scope": True,
            "materiality_assessed": False,
        }

        result = engine.check_csrd_esrs(data)

        assert result["compliant"] is False

    def test_value_chain_incomplete(self):
        """Test when value chain scope is incomplete."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.5"),
            "supplier_engagement": True,
            "value_chain_scope": False,
            "materiality_assessed": True,
        }

        result = engine.check_csrd_esrs(data)

        assert result["compliant"] is False


class TestCDPCompliance:
    """Test CDP Climate Change questionnaire compliance."""

    def test_all_requirements_met(self):
        """Test when all CDP requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("67.0"),
            "verification_status": "third_party",
            "supplier_engagement": True,
            "emissions_reduction_targets": True,
        }

        result = engine.check_cdp(data)

        assert result["compliant"] is True
        assert len(result["missing"]) == 0

    def test_missing_verification(self):
        """Test when third-party verification is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("67.0"),
            "verification_status": "internal",
            "supplier_engagement": True,
            "emissions_reduction_targets": True,
        }

        result = engine.check_cdp(data)

        assert result["compliant"] is False
        assert any("verification" in req["field"] for req in result["missing"])

    def test_no_reduction_targets(self):
        """Test when reduction targets are missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("67.0"),
            "verification_status": "third_party",
            "supplier_engagement": True,
            "emissions_reduction_targets": False,
        }

        result = engine.check_cdp(data)

        assert result["compliant"] is False

    def test_insufficient_coverage(self):
        """Test when coverage is below CDP threshold."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("50.0"),
            "verification_status": "third_party",
            "supplier_engagement": True,
            "emissions_reduction_targets": True,
        }

        result = engine.check_cdp(data)

        assert result["compliant"] is False


class TestSBTiCompliance:
    """Test Science Based Targets initiative compliance."""

    def test_coverage_above_67_percent(self):
        """Test when coverage meets SBTi 67% threshold."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("70.0"),
            "supplier_engagement": True,
            "reduction_pathway": "1.5C",
            "base_year_set": True,
        }

        result = engine.check_sbti(data)

        assert result["compliant"] is True

    def test_coverage_below_67_percent(self):
        """Test when coverage is below SBTi 67% threshold."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("60.0"),
            "supplier_engagement": True,
            "reduction_pathway": "1.5C",
            "base_year_set": True,
        }

        result = engine.check_sbti(data)

        assert result["compliant"] is False
        assert any("coverage" in req["field"] for req in result["missing"])

    def test_missing_reduction_pathway(self):
        """Test when reduction pathway is not defined."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("70.0"),
            "supplier_engagement": True,
            "base_year_set": True,
        }

        result = engine.check_sbti(data)

        assert result["compliant"] is False

    def test_no_supplier_engagement_plan(self):
        """Test when supplier engagement plan is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("70.0"),
            "supplier_engagement": False,
            "reduction_pathway": "1.5C",
            "base_year_set": True,
        }

        result = engine.check_sbti(data)

        assert result["compliant"] is False


class TestSB253Compliance:
    """Test California SB 253 compliance."""

    def test_all_requirements_met(self):
        """Test when all SB 253 requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "assurance_level": "limited",
            "reporting_year": 2024,
            "scope3_complete": True,
        }

        result = engine.check_sb253(data)

        assert result["compliant"] is True

    def test_missing_assurance(self):
        """Test when assurance is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "assurance_level": "none",
            "reporting_year": 2024,
            "scope3_complete": True,
        }

        result = engine.check_sb253(data)

        assert result["compliant"] is False
        assert any("assurance" in req["field"] for req in result["missing"])

    def test_scope3_incomplete(self):
        """Test when Scope 3 reporting is incomplete."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "assurance_level": "limited",
            "reporting_year": 2024,
            "scope3_complete": False,
        }

        result = engine.check_sb253(data)

        assert result["compliant"] is False


class TestGRICompliance:
    """Test GRI 305 (Emissions) compliance."""

    def test_all_requirements_met(self):
        """Test when all GRI requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "scope3_categories_reported": 15,
            "base_year_set": True,
            "methodology_disclosed": True,
            "emission_factors_disclosed": True,
        }

        result = engine.check_gri(data)

        assert result["compliant"] is True

    def test_missing_base_year(self):
        """Test when base year is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "scope3_categories_reported": 15,
            "base_year_set": False,
            "methodology_disclosed": True,
            "emission_factors_disclosed": True,
        }

        result = engine.check_gri(data)

        assert result["compliant"] is False

    def test_insufficient_categories_reported(self):
        """Test when insufficient Scope 3 categories reported."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "scope3_categories_reported": 10,
            "base_year_set": True,
            "methodology_disclosed": True,
            "emission_factors_disclosed": True,
        }

        result = engine.check_gri(data)

        assert result["compliant"] is False


class TestISO14064Compliance:
    """Test ISO 14064-1:2018 compliance."""

    def test_all_requirements_met(self):
        """Test when all ISO 14064 requirements are met."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "uncertainty_quantified": True,
            "by_gas_reporting": True,
            "verification_status": "third_party",
        }

        result = engine.check_iso14064(data)

        assert result["compliant"] is True

    def test_missing_by_gas_reporting(self):
        """Test when by-gas reporting is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "uncertainty_quantified": True,
            "by_gas_reporting": False,
            "verification_status": "third_party",
        }

        result = engine.check_iso14064(data)

        assert result["compliant"] is False
        assert any("by_gas" in req["field"] for req in result["missing"])

    def test_missing_uncertainty_quantification(self):
        """Test when uncertainty quantification is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "uncertainty_quantified": False,
            "by_gas_reporting": True,
            "verification_status": "third_party",
        }

        result = engine.check_iso14064(data)

        assert result["compliant"] is False

    def test_no_third_party_verification(self):
        """Test when third-party verification is missing."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "uncertainty_quantified": True,
            "by_gas_reporting": True,
            "verification_status": "internal",
        }

        result = engine.check_iso14064(data)

        assert result["compliant"] is False


class TestCrossFramework:
    """Test cross-framework compliance checks."""

    def test_check_all_frameworks(self):
        """Test checking all frameworks at once."""
        engine = ComplianceCheckerEngine.get_instance()

        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.5"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
            "supplier_engagement": True,
            "value_chain_scope": True,
            "materiality_assessed": True,
            "verification_status": "third_party",
            "emissions_reduction_targets": True,
            "assurance_level": "limited",
            "scope3_complete": True,
            "scope3_categories_reported": 15,
            "methodology_disclosed": True,
            "emission_factors_disclosed": True,
            "uncertainty_quantified": True,
            "by_gas_reporting": True,
            "reduction_pathway": "1.5C",
        }

        results = engine.check_all_frameworks(data)

        assert "ghg_protocol" in results
        assert "csrd_esrs" in results
        assert "cdp" in results
        assert "sbti" in results
        assert "sb253" in results
        assert "gri" in results
        assert "iso14064" in results

    def test_compliance_summary(self):
        """Test compliance summary generation."""
        engine = ComplianceCheckerEngine.get_instance()

        framework_results = {
            "ghg_protocol": {"compliant": True, "missing": []},
            "csrd_esrs": {"compliant": False, "missing": [{"field": "supplier_engagement"}]},
            "cdp": {"compliant": True, "missing": []},
        }

        summary = engine.generate_compliance_summary(framework_results)

        assert summary["total_frameworks"] == 3
        assert summary["compliant_count"] == 2
        assert summary["non_compliant_count"] == 1

    def test_recommendations_generation(self):
        """Test recommendations for non-compliant frameworks."""
        engine = ComplianceCheckerEngine.get_instance()

        framework_results = {
            "ghg_protocol": {"compliant": False, "missing": [{"field": "dqi"}]},
            "sbti": {"compliant": False, "missing": [{"field": "coverage"}]},
        }

        recommendations = engine.generate_recommendations(framework_results)

        assert len(recommendations) > 0
        assert any("dqi" in rec.lower() for rec in recommendations)
        assert any("coverage" in rec.lower() for rec in recommendations)

    def test_priority_ranking_of_gaps(self):
        """Test priority ranking of compliance gaps."""
        engine = ComplianceCheckerEngine.get_instance()

        gaps = [
            {"framework": "ghg_protocol", "field": "coverage", "impact": "high"},
            {"framework": "cdp", "field": "verification", "impact": "medium"},
            {"framework": "gri", "field": "methodology", "impact": "low"},
        ]

        ranked = engine.rank_gaps_by_priority(gaps)

        assert ranked[0]["impact"] == "high"
        assert ranked[-1]["impact"] == "low"

    def test_compliance_percentage_calculation(self):
        """Test overall compliance percentage calculation."""
        engine = ComplianceCheckerEngine.get_instance()

        framework_results = {
            "ghg_protocol": {"compliant": True},
            "csrd_esrs": {"compliant": True},
            "cdp": {"compliant": False},
            "sbti": {"compliant": True},
        }

        percentage = engine.calculate_compliance_percentage(framework_results)

        assert percentage == Decimal("75.0")  # 3 out of 4


class TestThresholds:
    """Test threshold validation."""

    def test_coverage_threshold_validation(self):
        """Test coverage threshold validation."""
        engine = ComplianceCheckerEngine.get_instance()

        assert engine.validate_coverage_threshold(Decimal("95.0"), "ghg_protocol") is True
        assert engine.validate_coverage_threshold(Decimal("60.0"), "ghg_protocol") is False

    def test_dqi_threshold_validation(self):
        """Test DQI threshold validation."""
        engine = ComplianceCheckerEngine.get_instance()

        assert engine.validate_dqi_threshold(Decimal("1.5"), "csrd_esrs") is True
        assert engine.validate_dqi_threshold(Decimal("3.5"), "csrd_esrs") is False

    def test_framework_specific_thresholds(self):
        """Test framework-specific thresholds."""
        engine = ComplianceCheckerEngine.get_instance()

        # SBTi requires 67% coverage
        assert engine.get_coverage_threshold("sbti") == Decimal("67.0")

        # GHG Protocol typically requires higher coverage
        assert engine.get_coverage_threshold("ghg_protocol") >= Decimal("95.0")


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(self):
        """Test health check returns healthy status."""
        engine = ComplianceCheckerEngine.get_instance()

        health = engine.health_check()

        assert health["status"] == "healthy"
        assert "engine" in health
        assert health["engine"] == "ComplianceCheckerEngine"

    def test_health_check_includes_stats(self):
        """Test health check includes statistics."""
        engine = ComplianceCheckerEngine.get_instance()

        # Perform some checks
        data = {
            "coverage": Decimal("95.0"),
            "dqi": Decimal("1.8"),
            "boundary_complete": True,
            "documentation_complete": True,
            "base_year_set": True,
        }
        engine.check_ghg_protocol(data)

        health = engine.health_check()

        assert "checks_performed" in health
        assert health["checks_performed"] > 0
