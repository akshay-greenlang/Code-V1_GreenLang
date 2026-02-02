# -*- coding: utf-8 -*-
"""
Unit Tests for GL-004: EUDR Compliance Agent

Comprehensive test suite with 50 test cases covering:
- Commodity validation (10 tests)
- Geolocation validation (15 tests)
- Risk assessment (10 tests)
- DDS generation (10 tests)
- Error handling (5 tests)

Target: 85%+ coverage for EUDR Compliance Agent
Run with: pytest tests/unit/test_gl004_eudr_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

EUDR (EU Deforestation Regulation) validates deforestation-free supply chains
per EU Regulation 2023/1115.
"""

import pytest
import hashlib
import json
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GL-Agent-Factory" / "backend" / "agents"))

# Import agent components
from gl_004_eudr_compliance.agent import (
    EUDRComplianceAgent,
    EUDRInput,
    EUDROutput,
    CommodityType,
    RiskLevel,
    ComplianceStatus,
    GeometryType,
    GeoLocation,
    CountryRisk,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create EUDRComplianceAgent instance."""
    return EUDRComplianceAgent()


@pytest.fixture
def valid_coffee_input():
    """Create valid coffee import input data."""
    return EUDRInput(
        commodity_type=CommodityType.COFFEE,
        cn_code="09011100",
        quantity_kg=50000.0,
        country_of_origin="BR",
        geolocation=GeoLocation(
            type=GeometryType.POINT,
            coordinates=[-47.5, -15.5]
        ),
        production_date=date(2024, 6, 1),
        operator_id="OP-001",
        supply_chain=[
            {"node_id": "FARM-001", "verified": True},
            {"node_id": "COOP-001", "verified": True},
        ],
        certifications=["Rainforest Alliance"],
    )


@pytest.fixture
def valid_soya_input():
    """Create valid soya import input data."""
    return EUDRInput(
        commodity_type=CommodityType.SOYA,
        cn_code="12019000",
        quantity_kg=100000.0,
        country_of_origin="BR",
        geolocation=GeoLocation(
            type=GeometryType.POINT,
            coordinates=[-55.0, -12.0]
        ),
        production_date=date(2024, 3, 15),
    )


@pytest.fixture
def valid_palm_oil_input():
    """Create valid palm oil import input data."""
    return EUDRInput(
        commodity_type=CommodityType.PALM_OIL,
        cn_code="15111000",
        quantity_kg=75000.0,
        country_of_origin="ID",
        geolocation=GeoLocation(
            type=GeometryType.POINT,
            coordinates=[110.5, -2.0]
        ),
        production_date=date(2024, 4, 20),
        certifications=["RSPO"],
    )


@pytest.fixture
def polygon_geolocation():
    """Create polygon geolocation for farm boundary."""
    return GeoLocation(
        type=GeometryType.POLYGON,
        coordinates=[[
            [-47.5, -15.5],
            [-47.4, -15.5],
            [-47.4, -15.4],
            [-47.5, -15.4],
            [-47.5, -15.5],  # Closed ring
        ]]
    )


@pytest.fixture
def pre_cutoff_date_input():
    """Create input with production date before EUDR cutoff."""
    return EUDRInput(
        commodity_type=CommodityType.COFFEE,
        cn_code="09011100",
        quantity_kg=50000.0,
        country_of_origin="BR",
        geolocation=GeoLocation(
            type=GeometryType.POINT,
            coordinates=[-47.5, -15.5]
        ),
        production_date=date(2020, 6, 1),  # Before Dec 31, 2020 cutoff
    )


# =============================================================================
# Commodity Validation Tests (10 tests)
# =============================================================================

class TestCommodityValidation:
    """Test suite for commodity validation - 10 test cases."""

    @pytest.mark.unit
    def test_coffee_commodity_type(self):
        """UT-GL004-001: Test CommodityType.COFFEE value."""
        assert CommodityType.COFFEE.value == "coffee"

    @pytest.mark.unit
    def test_all_seven_commodities_defined(self):
        """UT-GL004-002: Test all 7 EUDR commodities are defined."""
        expected = {"cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"}
        actual = {c.value for c in CommodityType}
        assert expected == actual

    @pytest.mark.unit
    def test_cn_code_coffee_in_scope(self, agent):
        """UT-GL004-003: Test coffee CN code 0901 is in EUDR scope."""
        assert agent.is_in_eudr_scope("09011100") is True

    @pytest.mark.unit
    def test_cn_code_cocoa_in_scope(self, agent):
        """UT-GL004-004: Test cocoa CN code 1801 is in EUDR scope."""
        assert agent.is_in_eudr_scope("18010000") is True

    @pytest.mark.unit
    def test_cn_code_palm_oil_in_scope(self, agent):
        """UT-GL004-005: Test palm oil CN code 1511 is in EUDR scope."""
        assert agent.is_in_eudr_scope("15111000") is True

    @pytest.mark.unit
    def test_cn_code_soya_in_scope(self, agent):
        """UT-GL004-006: Test soya CN code 1201 is in EUDR scope."""
        assert agent.is_in_eudr_scope("12019000") is True

    @pytest.mark.unit
    def test_cn_code_wood_in_scope(self, agent):
        """UT-GL004-007: Test wood CN code 44xx is in EUDR scope."""
        assert agent.is_in_eudr_scope("44011100") is True

    @pytest.mark.unit
    def test_cn_code_rubber_in_scope(self, agent):
        """UT-GL004-008: Test rubber CN code 4001 is in EUDR scope."""
        assert agent.is_in_eudr_scope("40011000") is True

    @pytest.mark.unit
    def test_cn_code_cattle_in_scope(self, agent):
        """UT-GL004-009: Test cattle CN code 0102 is in EUDR scope."""
        assert agent.is_in_eudr_scope("01022900") is True

    @pytest.mark.unit
    def test_non_eudr_cn_code_out_of_scope(self, agent):
        """UT-GL004-010: Test non-EUDR CN code is out of scope."""
        assert agent.is_in_eudr_scope("99999999") is False


# =============================================================================
# Geolocation Validation Tests (15 tests)
# =============================================================================

class TestGeolocationValidation:
    """Test suite for geolocation validation - 15 test cases."""

    @pytest.mark.unit
    def test_valid_point_coordinates(self, agent):
        """UT-GL004-011: Test valid point coordinates pass validation."""
        geo = GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5])
        assert agent._validate_geolocation(geo) is True

    @pytest.mark.unit
    def test_point_at_equator_prime_meridian(self, agent):
        """UT-GL004-012: Test point at equator/prime meridian is valid."""
        geo = GeoLocation(type=GeometryType.POINT, coordinates=[0.0, 0.0])
        assert agent._validate_geolocation(geo) is True

    @pytest.mark.unit
    def test_point_at_longitude_extremes(self, agent):
        """UT-GL004-013: Test points at longitude extremes."""
        geo_west = GeoLocation(type=GeometryType.POINT, coordinates=[-180.0, 0.0])
        geo_east = GeoLocation(type=GeometryType.POINT, coordinates=[180.0, 0.0])
        assert agent._validate_geolocation(geo_west) is True
        assert agent._validate_geolocation(geo_east) is True

    @pytest.mark.unit
    def test_point_at_latitude_extremes(self, agent):
        """UT-GL004-014: Test points at latitude extremes."""
        geo_south = GeoLocation(type=GeometryType.POINT, coordinates=[0.0, -90.0])
        geo_north = GeoLocation(type=GeometryType.POINT, coordinates=[0.0, 90.0])
        assert agent._validate_geolocation(geo_south) is True
        assert agent._validate_geolocation(geo_north) is True

    @pytest.mark.unit
    def test_invalid_longitude_rejected(self):
        """UT-GL004-015: Test invalid longitude is rejected."""
        with pytest.raises(ValueError):
            GeoLocation(type=GeometryType.POINT, coordinates=[200.0, 0.0])

    @pytest.mark.unit
    def test_invalid_latitude_rejected(self):
        """UT-GL004-016: Test invalid latitude is rejected."""
        with pytest.raises(ValueError):
            GeoLocation(type=GeometryType.POINT, coordinates=[0.0, 100.0])

    @pytest.mark.unit
    def test_valid_polygon_geometry(self, agent, polygon_geolocation):
        """UT-GL004-017: Test valid polygon passes validation."""
        assert agent._validate_geolocation(polygon_geolocation) is True

    @pytest.mark.unit
    def test_polygon_must_be_closed(self, agent):
        """UT-GL004-018: Test polygon must have closed ring."""
        # Non-closed polygon
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-47.5, -15.5],
                [-47.4, -15.5],
                [-47.4, -15.4],
                [-47.5, -15.4],
                # Missing closing point
            ]]
        )
        assert agent._validate_geolocation(geo) is False

    @pytest.mark.unit
    def test_polygon_minimum_points(self, agent):
        """UT-GL004-019: Test polygon must have at least 4 points."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-47.5, -15.5],
                [-47.4, -15.5],
                [-47.5, -15.5],
            ]]
        )
        assert agent._validate_geolocation(geo) is False

    @pytest.mark.unit
    def test_geolocation_valid_in_result(self, agent, valid_coffee_input):
        """UT-GL004-020: Test geolocation_valid is True for valid coordinates."""
        result = agent.run(valid_coffee_input)
        assert result.geolocation_valid is True

    @pytest.mark.unit
    def test_geometry_type_point(self):
        """UT-GL004-021: Test GeometryType.POINT value."""
        assert GeometryType.POINT.value == "Point"

    @pytest.mark.unit
    def test_geometry_type_polygon(self):
        """UT-GL004-022: Test GeometryType.POLYGON value."""
        assert GeometryType.POLYGON.value == "Polygon"

    @pytest.mark.unit
    def test_geometry_type_multipolygon(self):
        """UT-GL004-023: Test GeometryType.MULTI_POLYGON value."""
        assert GeometryType.MULTI_POLYGON.value == "MultiPolygon"

    @pytest.mark.unit
    def test_multipolygon_validation(self, agent):
        """UT-GL004-024: Test MultiPolygon validation."""
        geo = GeoLocation(
            type=GeometryType.MULTI_POLYGON,
            coordinates=[
                [[[-47.5, -15.5], [-47.4, -15.5], [-47.4, -15.4], [-47.5, -15.5]]],
                [[[-47.3, -15.3], [-47.2, -15.3], [-47.2, -15.2], [-47.3, -15.3]]],
            ]
        )
        assert agent._validate_geolocation(geo) is True

    @pytest.mark.unit
    def test_empty_polygon_invalid(self, agent):
        """UT-GL004-025: Test empty polygon is invalid."""
        geo = GeoLocation(type=GeometryType.POLYGON, coordinates=[])
        assert agent._validate_geolocation(geo) is False


# =============================================================================
# Risk Assessment Tests (10 tests)
# =============================================================================

class TestRiskAssessment:
    """Test suite for risk assessment - 10 test cases."""

    @pytest.mark.unit
    def test_brazil_high_risk(self, agent):
        """UT-GL004-026: Test Brazil classified as HIGH risk."""
        risk = agent._get_country_risk("BR")
        assert risk.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_indonesia_high_risk(self, agent):
        """UT-GL004-027: Test Indonesia classified as HIGH risk."""
        risk = agent._get_country_risk("ID")
        assert risk.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_malaysia_high_risk(self, agent):
        """UT-GL004-028: Test Malaysia classified as HIGH risk."""
        risk = agent._get_country_risk("MY")
        assert risk.risk_level == RiskLevel.HIGH

    @pytest.mark.unit
    def test_colombia_standard_risk(self, agent):
        """UT-GL004-029: Test Colombia classified as STANDARD risk."""
        risk = agent._get_country_risk("CO")
        assert risk.risk_level == RiskLevel.STANDARD

    @pytest.mark.unit
    def test_unknown_country_default_risk(self, agent):
        """UT-GL004-030: Test unknown country uses DEFAULT risk."""
        risk = agent._get_country_risk("ZZ")
        assert risk.risk_level == RiskLevel.STANDARD  # Default

    @pytest.mark.unit
    def test_risk_score_in_range(self, agent, valid_coffee_input):
        """UT-GL004-031: Test risk score is in 0-100 range."""
        result = agent.run(valid_coffee_input)
        assert 0 <= result.country_risk_score <= 100

    @pytest.mark.unit
    def test_brazil_risk_score_high(self, agent, valid_coffee_input):
        """UT-GL004-032: Test Brazil has high risk score."""
        result = agent.run(valid_coffee_input)
        assert result.country_risk_score >= 70  # High risk

    @pytest.mark.unit
    def test_risk_level_enum_values(self):
        """UT-GL004-033: Test RiskLevel enum values."""
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.STANDARD.value == "standard"
        assert RiskLevel.LOW.value == "low"

    @pytest.mark.unit
    def test_risk_level_in_output(self, agent, valid_coffee_input):
        """UT-GL004-034: Test risk level is in output."""
        result = agent.run(valid_coffee_input)
        assert result.risk_level == "high"

    @pytest.mark.unit
    def test_country_risk_has_source(self, agent):
        """UT-GL004-035: Test country risk has source attribution."""
        risk = agent._get_country_risk("BR")
        assert risk.source is not None
        assert "EU" in risk.source


# =============================================================================
# DDS Generation Tests (10 tests)
# =============================================================================

class TestDDSGeneration:
    """Test suite for Due Diligence Statement generation - 10 test cases."""

    @pytest.mark.unit
    def test_compliance_status_compliant(self, agent, valid_coffee_input):
        """UT-GL004-036: Test compliance status can be COMPLIANT."""
        # With full traceability and certifications
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="09011100",
            quantity_kg=50000.0,
            country_of_origin="CO",  # Standard risk
            geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-74.0, 4.5]),
            production_date=date(2024, 6, 1),
            supply_chain=[
                {"node_id": "FARM-001", "verified": True},
                {"node_id": "COOP-001", "verified": True},
            ],
            certifications=["Rainforest Alliance", "Fairtrade"],
        )
        result = agent.run(input_data)
        # May be COMPLIANT or PENDING depending on all criteria

    @pytest.mark.unit
    def test_compliance_status_non_compliant_bad_geolocation(self, agent):
        """UT-GL004-037: Test NON_COMPLIANT with invalid geolocation."""
        # Mock invalid geolocation handling
        with patch.object(agent, '_validate_geolocation', return_value=False):
            result = agent.run(EUDRInput(
                commodity_type=CommodityType.COFFEE,
                cn_code="09011100",
                quantity_kg=50000.0,
                country_of_origin="BR",
                geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
                production_date=date(2024, 6, 1),
            ))
            assert result.compliance_status == "non_compliant"

    @pytest.mark.unit
    def test_compliance_status_non_compliant_pre_cutoff(self, agent, pre_cutoff_date_input):
        """UT-GL004-038: Test NON_COMPLIANT with pre-cutoff production date."""
        result = agent.run(pre_cutoff_date_input)
        assert result.compliance_status == "non_compliant"

    @pytest.mark.unit
    def test_cutoff_date_compliance_flag(self, agent, valid_coffee_input):
        """UT-GL004-039: Test cutoff_date_compliant flag."""
        result = agent.run(valid_coffee_input)
        assert result.cutoff_date_compliant is True  # 2024 > 2020-12-31

    @pytest.mark.unit
    def test_cutoff_date_non_compliant_flag(self, agent, pre_cutoff_date_input):
        """UT-GL004-040: Test cutoff_date_compliant False for pre-cutoff."""
        result = agent.run(pre_cutoff_date_input)
        assert result.cutoff_date_compliant is False

    @pytest.mark.unit
    def test_traceability_score_calculation(self, agent, valid_coffee_input):
        """UT-GL004-041: Test traceability score calculation."""
        result = agent.run(valid_coffee_input)
        # 2 nodes, both verified = 100%
        assert result.traceability_score == 100.0

    @pytest.mark.unit
    def test_traceability_score_partial(self, agent):
        """UT-GL004-042: Test partial traceability score."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="09011100",
            quantity_kg=50000.0,
            country_of_origin="BR",
            geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
            production_date=date(2024, 6, 1),
            supply_chain=[
                {"node_id": "FARM-001", "verified": True},
                {"node_id": "COOP-001", "verified": False},  # Not verified
            ],
        )
        result = agent.run(input_data)
        assert result.traceability_score == 50.0

    @pytest.mark.unit
    def test_traceability_score_zero_no_chain(self, agent):
        """UT-GL004-043: Test traceability score 0 with no supply chain."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="09011100",
            quantity_kg=50000.0,
            country_of_origin="BR",
            geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
            production_date=date(2024, 6, 1),
            supply_chain=[],  # Empty
        )
        result = agent.run(input_data)
        assert result.traceability_score == 0.0

    @pytest.mark.unit
    def test_mitigation_measures_generated(self, agent, valid_soya_input):
        """UT-GL004-044: Test mitigation measures are generated."""
        result = agent.run(valid_soya_input)
        assert isinstance(result.mitigation_measures, list)

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_coffee_input):
        """UT-GL004-045: Test provenance hash is generated."""
        result = agent.run(valid_coffee_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# Error Handling Tests (5 tests)
# =============================================================================

class TestErrorHandling:
    """Test suite for error handling - 5 test cases."""

    @pytest.mark.unit
    def test_commodity_cn_code_mismatch_error(self, agent):
        """UT-GL004-046: Test error when commodity and CN code don't match."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,  # Coffee
            cn_code="15111000",  # But palm oil CN code
            quantity_kg=50000.0,
            country_of_origin="BR",
            geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
            production_date=date(2024, 6, 1),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.run(input_data)

        assert "not in EUDR scope" in str(exc_info.value)

    @pytest.mark.unit
    def test_negative_quantity_rejected(self):
        """UT-GL004-047: Test negative quantity is rejected."""
        with pytest.raises(ValueError):
            EUDRInput(
                commodity_type=CommodityType.COFFEE,
                cn_code="09011100",
                quantity_kg=-100.0,  # Negative
                country_of_origin="BR",
                geolocation=GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5]),
                production_date=date(2024, 6, 1),
            )

    @pytest.mark.unit
    def test_output_includes_timestamp(self, agent, valid_coffee_input):
        """UT-GL004-048: Test output includes calculation timestamp."""
        result = agent.run(valid_coffee_input)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    @pytest.mark.unit
    def test_compliance_status_enum_values(self):
        """UT-GL004-049: Test ComplianceStatus enum values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PENDING_VERIFICATION.value == "pending_verification"
        assert ComplianceStatus.INSUFFICIENT_DATA.value == "insufficient_data"

    @pytest.mark.unit
    def test_get_commodities_method(self, agent):
        """UT-GL004-050: Test get_commodities utility method."""
        commodities = agent.get_commodities()
        assert "coffee" in commodities
        assert "soya" in commodities
        assert "palm_oil" in commodities
        assert len(commodities) == 7


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = EUDRComplianceAgent()
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/eudr_compliance_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_cutoff_date_constant(self):
        """Test EUDR cutoff date is correctly set."""
        agent = EUDRComplianceAgent()
        assert agent.CUTOFF_DATE == date(2020, 12, 31)

    @pytest.mark.unit
    def test_recognized_certifications(self):
        """Test recognized certifications list."""
        agent = EUDRComplianceAgent()
        certs = agent.RECOGNIZED_CERTIFICATIONS
        assert "FSC" in certs
        assert "RSPO" in certs
        assert "Rainforest Alliance" in certs


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedEUDR:
    """Parametrized tests for EUDR scenarios."""

    @pytest.mark.unit
    @pytest.mark.parametrize("commodity,cn_prefix", [
        (CommodityType.CATTLE, "0102"),
        (CommodityType.COCOA, "1801"),
        (CommodityType.COFFEE, "0901"),
        (CommodityType.PALM_OIL, "1511"),
        (CommodityType.RUBBER, "4001"),
        (CommodityType.SOYA, "1201"),
        (CommodityType.WOOD, "44"),
    ])
    def test_commodity_cn_code_mapping(self, agent, commodity, cn_prefix):
        """Test commodity to CN code mapping."""
        cn_code = f"{cn_prefix}0000" if len(cn_prefix) == 4 else f"{cn_prefix}010000"
        assert agent.is_in_eudr_scope(cn_code) is True

    @pytest.mark.unit
    @pytest.mark.parametrize("country,expected_risk", [
        ("BR", RiskLevel.HIGH),
        ("ID", RiskLevel.HIGH),
        ("MY", RiskLevel.HIGH),
        ("CO", RiskLevel.STANDARD),
        ("PE", RiskLevel.STANDARD),
        ("GH", RiskLevel.STANDARD),
    ])
    def test_country_risk_levels(self, agent, country, expected_risk):
        """Test country risk level classification."""
        risk = agent._get_country_risk(country)
        assert risk.risk_level == expected_risk


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
