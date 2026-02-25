"""
Unit tests for ComplianceCheckerEngine.

Tests compliance checking across 7 regulatory frameworks with
capital goods-specific rules and boundary validation.
"""

import pytest
from datetime import date
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from greenlang.mrv.capital_goods.engines.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceInput,
    ComplianceResult,
    ComplianceStatus,
    ComplianceGap,
    RegulatoryFramework,
    CapitalGoodsRule,
)


class TestComplianceCheckerEngineSingleton:
    """Test singleton pattern."""

    def test_singleton_same_instance(self):
        """Test that multiple calls return same instance."""
        engine1 = ComplianceCheckerEngine()
        engine2 = ComplianceCheckerEngine()
        assert engine1 is engine2

    def test_singleton_with_reset(self):
        """Test singleton reset for testing."""
        engine1 = ComplianceCheckerEngine()
        ComplianceCheckerEngine._instance = None
        engine2 = ComplianceCheckerEngine()
        assert engine1 is not engine2


class TestCheckAll:
    """Test check_all() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    @pytest.fixture
    def sample_input(self):
        """Create sample compliance input."""
        return ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "iso_14064"],
            calculation_results=[
                Mock(
                    asset_id=f"A{i:03d}",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                    depreciation_method="straight_line",
                    useful_life_years=5,
                )
                for i in range(5)
            ],
        )

    def test_check_all_success(self, engine, sample_input):
        """Test successful compliance check across all frameworks."""
        result = engine.check_all(sample_input)

        assert isinstance(result, ComplianceResult)
        assert result.overall_status in [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.PARTIAL,
            ComplianceStatus.NON_COMPLIANT,
        ]
        assert len(result.framework_results) > 0

    def test_check_all_multiple_frameworks(self, engine, sample_input):
        """Test checking multiple frameworks."""
        result = engine.check_all(sample_input)

        assert "ghg_protocol" in result.framework_results
        assert "iso_14064" in result.framework_results

    def test_check_all_empty_results(self, engine):
        """Test compliance check with empty calculation results."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[],
        )

        result = engine.check_all(input_data)

        assert result.overall_status == ComplianceStatus.COMPLIANT  # No violations


class TestGHGProtocolValidator:
    """Test GHG Protocol framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_ghg_protocol_compliant(self, engine):
        """Test GHG Protocol compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                    category="capital_goods",
                ),
            ],
        )

        result = engine.validate_ghg_protocol(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT
        assert len(result["gaps"]) == 0

    def test_ghg_protocol_no_depreciation_rule(self, engine):
        """Test GHG Protocol NO_DEPRECIATION_RULE enforcement."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    depreciation_applied=True,  # Violation
                ),
            ],
        )

        result = engine.validate_ghg_protocol(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert len(result["gaps"]) > 0
        assert any("depreciation" in gap.description.lower() for gap in result["gaps"])

    def test_ghg_protocol_capitalization_classification(self, engine):
        """Test GHG Protocol CAPITALIZATION_CLASSIFICATION check."""
        input_data = ComplianceInput(
            organization_id="ORG003",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A003",
                    method="supplier_specific",
                    emissions=Decimal("50.0"),
                    purchase_value=Decimal("500.00"),  # Below capitalization threshold
                    asset_type="Office Supplies",
                    is_capitalized=False,
                ),
            ],
        )

        result = engine.validate_ghg_protocol(input_data)

        # Should flag non-capitalized items
        assert any("capitalization" in gap.description.lower() for gap in result["gaps"])

    def test_ghg_protocol_category_boundary(self, engine):
        """Test GHG Protocol CATEGORY_BOUNDARY check (no Category 1 overlap)."""
        input_data = ComplianceInput(
            organization_id="ORG004",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A004",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Raw Material",  # Category 1 item
                    category="purchased_goods",  # Violation
                ),
            ],
        )

        result = engine.validate_ghg_protocol(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert any("Category 1" in gap.description for gap in result["gaps"])

    def test_ghg_protocol_scope_boundary(self, engine):
        """Test GHG Protocol SCOPE_BOUNDARY check (no Scope 1/2 overlap)."""
        input_data = ComplianceInput(
            organization_id="ORG005",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A005",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Company Vehicle",
                    owned_operated=True,  # Scope 1/2 violation
                ),
            ],
        )

        result = engine.validate_ghg_protocol(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert any("Scope 1" in gap.description or "Scope 2" in gap.description for gap in result["gaps"])


class TestISO14064Validator:
    """Test ISO 14064 framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_iso14064_compliant(self, engine):
        """Test ISO 14064 compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["iso_14064"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    uncertainty=10.0,
                    data_quality_score=4.5,
                ),
            ],
        )

        result = engine.validate_iso_14064(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT

    def test_iso14064_uncertainty_required(self, engine):
        """Test ISO 14064 requires uncertainty quantification."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["iso_14064"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    uncertainty=None,  # Missing
                ),
            ],
        )

        result = engine.validate_iso_14064(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert any("uncertainty" in gap.description.lower() for gap in result["gaps"])


class TestCSRDValidator:
    """Test CSRD framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_csrd_compliant(self, engine):
        """Test CSRD compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["csrd"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    data_quality_score=4.5,
                    supplier_engagement=True,
                ),
            ],
        )

        result = engine.validate_csrd(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT

    def test_csrd_dqi_required(self, engine):
        """Test CSRD requires data quality indicators."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["csrd"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="spend_based",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    data_quality_score=None,  # Missing
                ),
            ],
        )

        result = engine.validate_csrd(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert any("data quality" in gap.description.lower() for gap in result["gaps"])


class TestTCFDValidator:
    """Test TCFD framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_tcfd_compliant(self, engine):
        """Test TCFD compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["tcfd"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    climate_scenario_tested=True,
                ),
            ],
            scenario_analysis_performed=True,
        )

        result = engine.validate_tcfd(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT

    def test_tcfd_scenario_analysis_required(self, engine):
        """Test TCFD requires scenario analysis."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["tcfd"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                ),
            ],
            scenario_analysis_performed=False,  # Missing
        )

        result = engine.validate_tcfd(input_data)

        assert result["status"] == ComplianceStatus.NON_COMPLIANT
        assert any("scenario" in gap.description.lower() for gap in result["gaps"])


class TestSBTiValidator:
    """Test SBTi framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_sbti_compliant(self, engine):
        """Test SBTi compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["sbti"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    supplier_id="SUP001",
                    supplier_has_sbti_target=True,
                ),
            ],
            supplier_specific_coverage_pct=75.0,
        )

        result = engine.validate_sbti(input_data)

        assert result["status"] in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIAL]

    def test_sbti_coverage_threshold(self, engine):
        """Test SBTi 67% supplier-specific coverage threshold."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["sbti"],
            calculation_results=[
                Mock(
                    asset_id=f"A{i:03d}",
                    method="supplier_specific" if i < 5 else "spend_based",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                )
                for i in range(10)
            ],
            supplier_specific_coverage_pct=50.0,  # Below 67%
        )

        result = engine.validate_sbti(input_data)

        assert result["status"] == ComplianceStatus.PARTIAL
        assert any("67%" in gap.description for gap in result["gaps"])


class TestCDPValidator:
    """Test CDP framework validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_cdp_compliant(self, engine):
        """Test CDP compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["cdp"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    verification_level="third_party",
                ),
            ],
        )

        result = engine.validate_cdp(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT

    def test_cdp_verification_recommended(self, engine):
        """Test CDP recommends third-party verification."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["cdp"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="spend_based",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    verification_level="unverified",
                ),
            ],
        )

        result = engine.validate_cdp(input_data)

        # May be partial compliance without verification
        assert result["status"] in [ComplianceStatus.PARTIAL, ComplianceStatus.COMPLIANT]


class TestSEC1505Validator:
    """Test SEC Climate Rule (17 CFR 1505) validator."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_sec1505_compliant(self, engine):
        """Test SEC 1505 compliance for valid data."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["sec_climate"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    attestation_level="limited",
                    material_to_business=False,
                ),
            ],
            total_scope3_emissions=Decimal("10000.0"),
        )

        result = engine.validate_sec_1505(input_data)

        assert result["status"] == ComplianceStatus.COMPLIANT

    def test_sec1505_materiality_threshold(self, engine):
        """Test SEC 1505 40% materiality threshold."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["sec_climate"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("5000.0"),  # 50% of total
                    purchase_value=Decimal("500000.00"),
                    asset_type="Equipment",
                ),
            ],
            total_scope3_emissions=Decimal("10000.0"),
        )

        result = engine.validate_sec_1505(input_data)

        # Capital goods >40% requires attestation
        assert any("40%" in gap.description for gap in result["gaps"]) or result["status"] == ComplianceStatus.PARTIAL


class TestCapexVolatilityContext:
    """Test CAPEX_VOLATILITY_CONTEXT check."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_capex_volatility_documented(self, engine):
        """Test CAPEX volatility context check with documentation."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("1000.0"),
                    purchase_value=Decimal("100000.00"),
                    asset_type="Equipment",
                ),
            ],
            historical_data=[
                {"period": "2023", "total_emissions": 500.0},
            ],
            volatility_explanation="Large equipment purchase in 2024",
        )

        result = engine.check_capex_volatility_context(input_data)

        # With explanation, should pass
        assert result["has_context"] is True

    def test_capex_volatility_missing_context(self, engine):
        """Test CAPEX volatility check flags missing context."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("1000.0"),
                    purchase_value=Decimal("100000.00"),
                    asset_type="Equipment",
                ),
            ],
            historical_data=[
                {"period": "2023", "total_emissions": 200.0},  # 5x increase
            ],
            volatility_explanation=None,  # Missing
        )

        result = engine.check_capex_volatility_context(input_data)

        # Should flag missing explanation for high volatility
        assert result["has_context"] is False
        assert result["volatility_ratio"] == 5.0


class TestGetComplianceSummary:
    """Test get_compliance_summary() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_get_summary_single_framework(self, engine):
        """Test compliance summary for single framework."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                ),
            ],
        )

        result = engine.check_all(input_data)
        summary = engine.get_compliance_summary(result)

        assert "total_frameworks_checked" in summary
        assert "compliant_frameworks" in summary
        assert "non_compliant_frameworks" in summary
        assert summary["total_frameworks_checked"] == 1

    def test_get_summary_multiple_frameworks(self, engine):
        """Test compliance summary for multiple frameworks."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "iso_14064", "csrd"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    uncertainty=10.0,
                    data_quality_score=4.5,
                ),
            ],
        )

        result = engine.check_all(input_data)
        summary = engine.get_compliance_summary(result)

        assert summary["total_frameworks_checked"] == 3


class TestGetGaps:
    """Test get_gaps() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_get_gaps_by_framework(self, engine):
        """Test retrieving gaps by framework."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "iso_14064"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    uncertainty=None,  # ISO gap
                ),
            ],
        )

        result = engine.check_all(input_data)
        iso_gaps = engine.get_gaps(result, framework="iso_14064")

        assert isinstance(iso_gaps, list)
        assert len(iso_gaps) > 0
        assert all(isinstance(gap, ComplianceGap) for gap in iso_gaps)

    def test_get_gaps_all_frameworks(self, engine):
        """Test retrieving all gaps across frameworks."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "iso_14064"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    depreciation_applied=True,  # GHG gap
                    uncertainty=None,  # ISO gap
                ),
            ],
        )

        result = engine.check_all(input_data)
        all_gaps = engine.get_gaps(result)

        assert len(all_gaps) > 0


class TestGetRecommendations:
    """Test get_recommendations() method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_get_recommendations_for_gaps(self, engine):
        """Test generating recommendations for compliance gaps."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="spend_based",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    data_quality_score=2.0,  # Low quality
                ),
            ],
        )

        result = engine.check_all(input_data)
        recommendations = engine.get_recommendations(result)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_get_recommendations_compliant(self, engine):
        """Test recommendations for compliant data."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                    data_quality_score=4.5,
                ),
            ],
        )

        result = engine.check_all(input_data)
        recommendations = engine.get_recommendations(result)

        # May have improvement recommendations even if compliant
        assert isinstance(recommendations, list)


class TestThreeTierStatusAssignment:
    """Test three-tier status assignment logic."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_status_compliant(self, engine):
        """Test COMPLIANT status assignment."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                ),
            ],
        )

        result = engine.check_all(input_data)

        assert result.overall_status == ComplianceStatus.COMPLIANT

    def test_status_partial(self, engine):
        """Test PARTIAL status assignment."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["sbti"],
            calculation_results=[
                Mock(
                    asset_id=f"A{i:03d}",
                    method="supplier_specific" if i < 6 else "spend_based",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                )
                for i in range(10)
            ],
            supplier_specific_coverage_pct=60.0,  # Below 67%, but some coverage
        )

        result = engine.check_all(input_data)

        # Should be PARTIAL (some but not full compliance)
        assert result.overall_status in [ComplianceStatus.PARTIAL, ComplianceStatus.NON_COMPLIANT]

    def test_status_non_compliant(self, engine):
        """Test NON_COMPLIANT status assignment."""
        input_data = ComplianceInput(
            organization_id="ORG003",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol"],
            calculation_results=[
                Mock(
                    asset_id="A003",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Company Vehicle",
                    owned_operated=True,  # Critical violation
                ),
            ],
        )

        result = engine.check_all(input_data)

        assert result.overall_status == ComplianceStatus.NON_COMPLIANT


class TestFrameworkPriority:
    """Test framework-specific priority and rule enforcement."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        ComplianceCheckerEngine._instance = None
        return ComplianceCheckerEngine()

    def test_ghg_protocol_highest_priority_rules(self, engine):
        """Test GHG Protocol has highest priority rules."""
        input_data = ComplianceInput(
            organization_id="ORG001",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "cdp"],
            calculation_results=[
                Mock(
                    asset_id="A001",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=True,  # Scope boundary violation
                ),
            ],
        )

        result = engine.check_all(input_data)

        # GHG Protocol violation should make overall non-compliant
        assert result.overall_status == ComplianceStatus.NON_COMPLIANT

    def test_multiple_framework_aggregation(self, engine):
        """Test aggregation of multiple framework results."""
        input_data = ComplianceInput(
            organization_id="ORG002",
            reporting_period_start=date(2024, 1, 1),
            reporting_period_end=date(2024, 12, 31),
            frameworks=["ghg_protocol", "iso_14064", "csrd"],
            calculation_results=[
                Mock(
                    asset_id="A002",
                    method="supplier_specific",
                    emissions=Decimal("100.0"),
                    purchase_value=Decimal("10000.00"),
                    asset_type="Equipment",
                    owned_operated=False,
                    uncertainty=10.0,
                    data_quality_score=4.5,
                ),
            ],
        )

        result = engine.check_all(input_data)

        # All frameworks should pass
        assert all(
            fw_result["status"] == ComplianceStatus.COMPLIANT
            for fw_result in result.framework_results.values()
        )
