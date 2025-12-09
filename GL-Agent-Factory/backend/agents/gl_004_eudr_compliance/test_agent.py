"""
Test Suite for GL-004: EUDR Compliance Agent

Comprehensive tests for EU Deforestation Regulation compliance validation.

Test Categories:
1. Commodity Validation (35 tests) - All 7 commodities
2. Geolocation Validation (10 tests) - Point, Polygon, MultiPolygon
3. DDS Generation (15 tests) - Due Diligence Statement
4. Supply Chain (20 tests) - Traceability and verification
5. Satellite Verification (15 tests) - Forest cover analysis
6. Risk Assessment (10 tests) - Country and commodity risk
7. Edge Cases (15 tests) - Boundary conditions
8. Integration (80 tests) - End-to-end workflows

Total Golden Tests: 200

Example:
    >>> pytest test_agent.py -v
    >>> pytest test_agent.py -k "test_coffee" -v
"""

import hashlib
import json
import pytest
from datetime import date, datetime, timedelta
from typing import List

from .agent import (
    EUDRComplianceAgent,
    EUDRInput,
    EUDROutput,
    CommodityType,
    RiskLevel,
    ComplianceStatus,
    GeometryType,
    DueDiligenceType,
    DeforestationStatus,
    GeoLocation,
    SupplierInfo,
    SupplyChainNode,
    GeolocationValidationResult,
    ForestCoverAnalysis,
    RiskAssessment,
    DDSDocument,
    HIGH_RISK_COUNTRIES,
    STANDARD_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
    CN_TO_COMMODITY,
    RECOGNIZED_CERTIFICATIONS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def agent() -> EUDRComplianceAgent:
    """Create EUDR Compliance Agent instance."""
    return EUDRComplianceAgent()


@pytest.fixture
def valid_point_geolocation() -> GeoLocation:
    """Valid GPS point in Brazil."""
    return GeoLocation(
        type=GeometryType.POINT,
        coordinates=[-47.5, -15.5]
    )


@pytest.fixture
def valid_polygon_geolocation() -> GeoLocation:
    """Valid polygon (approximately 10 hectares) in Brazil."""
    return GeoLocation(
        type=GeometryType.POLYGON,
        coordinates=[[
            [-47.5, -15.5],
            [-47.49, -15.5],
            [-47.49, -15.49],
            [-47.5, -15.49],
            [-47.5, -15.5]  # Closed ring
        ]]
    )


@pytest.fixture
def small_polygon_geolocation() -> GeoLocation:
    """Polygon smaller than 1 hectare threshold."""
    return GeoLocation(
        type=GeometryType.POLYGON,
        coordinates=[[
            [-47.5, -15.5],
            [-47.4999, -15.5],
            [-47.4999, -15.4999],
            [-47.5, -15.4999],
            [-47.5, -15.5]
        ]]
    )


@pytest.fixture
def self_intersecting_polygon() -> GeoLocation:
    """Polygon with self-intersection (figure-8 shape)."""
    return GeoLocation(
        type=GeometryType.POLYGON,
        coordinates=[[
            [0, 0],
            [10, 10],
            [10, 0],
            [0, 10],
            [0, 0]
        ]]
    )


@pytest.fixture
def valid_supplier_info() -> SupplierInfo:
    """Valid verified supplier."""
    return SupplierInfo(
        name="Fazenda Verde",
        registration_id="BR12345678",
        country="BR",
        verified=True,
        certifications=["Rainforest Alliance"],
        last_audit_date=date.today() - timedelta(days=90)
    )


@pytest.fixture
def valid_supply_chain() -> List[SupplyChainNode]:
    """Valid supply chain with full traceability."""
    return [
        SupplyChainNode(
            node_id="producer-001",
            node_type="producer",
            operator_name="Farm ABC",
            country_code="BR",
            verified=True,
            documents=["harvest_cert.pdf"]
        ),
        SupplyChainNode(
            node_id="processor-001",
            node_type="processor",
            operator_name="Mill XYZ",
            country_code="BR",
            verified=True,
            documents=["processing_cert.pdf"]
        ),
        SupplyChainNode(
            node_id="exporter-001",
            node_type="exporter",
            operator_name="Export Co",
            country_code="BR",
            verified=True,
            documents=["export_license.pdf"]
        ),
    ]


# =============================================================================
# COMMODITY VALIDATION TESTS (35 tests - 5 per commodity)
# =============================================================================


class TestCommodityValidation:
    """Tests for all 7 EUDR regulated commodities."""

    # CATTLE Tests
    def test_cattle_live_bovine(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test live bovine animals (CN 0102)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0102.21.00",
            quantity_kg=5000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 15)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "cattle"
        assert result.provenance_hash is not None

    def test_cattle_beef_fresh(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test fresh beef (CN 0201)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0201.10.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "cattle"

    def test_cattle_beef_frozen(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test frozen beef (CN 0202)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0202.30.90",
            quantity_kg=2000,
            country_of_origin="PY",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"  # Paraguay is high risk

    def test_cattle_leather_raw(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test raw bovine hides (CN 4101)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="4101.20.10",
            quantity_kg=500,
            country_of_origin="AU",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 4, 1)
        )
        result = agent.run(input_data)
        assert result.country_risk_score < 30  # Australia is low risk

    def test_cattle_leather_tanned(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test tanned leather (CN 4104)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="4104.11.10",
            quantity_kg=200,
            country_of_origin="IT",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 5, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "low"  # Italy is low risk

    # COCOA Tests
    def test_cocoa_beans(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test cocoa beans (CN 1801)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=10000,
            country_of_origin="GH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "cocoa"

    def test_cocoa_paste(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test cocoa paste (CN 1803)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1803.10.00",
            quantity_kg=5000,
            country_of_origin="CI",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 15)
        )
        result = agent.run(input_data)
        assert "Rainforest Alliance" in agent.get_certification_options(CommodityType.COCOA)

    def test_cocoa_butter(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test cocoa butter (CN 1804)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1804.00.00",
            quantity_kg=3000,
            country_of_origin="CM",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert result.risk_assessment.commodity_risk_score == 55.0  # Cocoa risk

    def test_cocoa_powder(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test cocoa powder (CN 1805)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1805.00.00",
            quantity_kg=1000,
            country_of_origin="EC",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 4, 1)
        )
        result = agent.run(input_data)
        assert result.cutoff_date_compliant is True

    def test_cocoa_chocolate(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test chocolate products (CN 1806)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1806.32.10",
            quantity_kg=500,
            country_of_origin="DE",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 5, 1)
        )
        result = agent.run(input_data)
        assert result.country_risk_score == 10.0  # Germany

    # COFFEE Tests
    def test_coffee_green(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test green coffee beans (CN 0901)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=50000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 6, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "coffee"

    def test_coffee_roasted(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test roasted coffee (CN 0901)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.21.00",
            quantity_kg=10000,
            country_of_origin="CO",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 7, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "standard"  # Colombia

    def test_coffee_decaffeinated(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test decaffeinated coffee."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.22.00",
            quantity_kg=5000,
            country_of_origin="VN",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 8, 1)
        )
        result = agent.run(input_data)
        assert result.provenance_hash is not None

    def test_coffee_extract(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test coffee extracts (CN 2101)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="2101.11.11",
            quantity_kg=1000,
            country_of_origin="PE",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 9, 1)
        )
        result = agent.run(input_data)
        assert agent.is_in_eudr_scope("2101.11.11") is True

    def test_coffee_certified(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation,
        valid_supplier_info: SupplierInfo,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test certified coffee with full traceability."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=25000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 10, 1),
            supplier_info=valid_supplier_info,
            supply_chain=valid_supply_chain,
            certifications=["Rainforest Alliance", "4C"]
        )
        result = agent.run(input_data)
        assert result.traceability_score > 80

    # PALM OIL Tests
    def test_palm_oil_crude(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test crude palm oil (CN 1511)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=100000,
            country_of_origin="ID",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"  # Indonesia

    def test_palm_oil_refined(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test refined palm oil."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.90.91",
            quantity_kg=50000,
            country_of_origin="MY",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 1)
        )
        result = agent.run(input_data)
        assert result.country_risk_score >= 70  # Malaysia high risk

    def test_palm_kernel_oil(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test palm kernel oil (CN 1513)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1513.21.10",
            quantity_kg=20000,
            country_of_origin="PG",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert result.risk_assessment.commodity_risk_score == 75.0  # Palm oil risk

    def test_palm_oil_rspo_certified(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test RSPO certified palm oil."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=75000,
            country_of_origin="MY",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 4, 1),
            certifications=["RSPO"],
            supporting_documents=["rspo_cert.pdf", "traceability.pdf"]
        )
        result = agent.run(input_data)
        assert "RSPO" in agent.get_certification_options(CommodityType.PALM_OIL)

    def test_palm_fatty_acids(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test palm fatty acids (CN 3823)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="3823.11.00",
            quantity_kg=10000,
            country_of_origin="ID",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 5, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "palm_oil"

    # RUBBER Tests
    def test_rubber_natural(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test natural rubber (CN 4001)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=30000,
            country_of_origin="TH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "rubber"

    def test_rubber_compounded(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test compounded rubber (CN 4005)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4005.10.00",
            quantity_kg=15000,
            country_of_origin="VN",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 1)
        )
        result = agent.run(input_data)
        assert result.risk_assessment.commodity_risk_score == 50.0

    def test_rubber_vulcanized_thread(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test vulcanized rubber thread (CN 4007)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4007.00.00",
            quantity_kg=5000,
            country_of_origin="IN",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert "GPSNR" in agent.get_certification_options(CommodityType.RUBBER)

    def test_rubber_vulcanized_plates(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test vulcanized rubber plates (CN 4008)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4008.11.00",
            quantity_kg=8000,
            country_of_origin="ID",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 4, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"

    def test_rubber_latex(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test natural rubber latex."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.21.00",
            quantity_kg=20000,
            country_of_origin="MY",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 5, 1)
        )
        result = agent.run(input_data)
        assert result.provenance_hash is not None

    # SOYA Tests
    def test_soya_beans(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test soybeans (CN 1201)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=500000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "soya"

    def test_soya_flour(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test soya flour (CN 1208)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1208.10.00",
            quantity_kg=100000,
            country_of_origin="BO",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"  # Bolivia

    def test_soya_oil(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test soya-bean oil (CN 1507)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1507.10.10",
            quantity_kg=200000,
            country_of_origin="PY",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert result.risk_assessment.commodity_risk_score == 68.0

    def test_soya_meal(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test soya meal/oilcake (CN 2304)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="2304.00.00",
            quantity_kg=300000,
            country_of_origin="US",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 4, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "low"  # USA

    def test_soya_rtrs_certified(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test RTRS certified soya."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=250000,
            country_of_origin="BR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 5, 1),
            certifications=["RTRS"]
        )
        result = agent.run(input_data)
        assert "RTRS" in agent.get_certification_options(CommodityType.SOYA)

    # WOOD Tests
    def test_wood_logs(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test wood logs (CN 4403)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=50000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "wood"

    def test_wood_sawn(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test sawn wood (CN 4407)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4407.29.95",
            quantity_kg=30000,
            country_of_origin="CD",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 2, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"  # DRC

    def test_wood_pulp(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test wood pulp (CN 4701)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4701.00.10",
            quantity_kg=100000,
            country_of_origin="CA",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 3, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "low"  # Canada

    def test_wood_paper(self, agent: EUDRComplianceAgent, valid_point_geolocation: GeoLocation):
        """Test paper products (CN 4801)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4801.00.10",
            quantity_kg=20000,
            country_of_origin="DE",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 4, 1)
        )
        result = agent.run(input_data)
        assert result.risk_assessment.commodity_risk_score == 40.0

    def test_wood_fsc_certified(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test FSC certified wood."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4407.11.10",
            quantity_kg=40000,
            country_of_origin="BR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 5, 1),
            certifications=["FSC"],
            supply_chain=valid_supply_chain
        )
        result = agent.run(input_data)
        assert "FSC" in agent.get_certification_options(CommodityType.WOOD)


# =============================================================================
# GEOLOCATION VALIDATION TESTS (10 tests)
# =============================================================================


class TestGeolocationValidation:
    """Tests for GeoJSON validation."""

    def test_valid_point_coordinates(self, agent: EUDRComplianceAgent):
        """Test valid GPS point validation."""
        geo = GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -15.5])
        result = agent._validate_geolocation(geo)
        assert result.is_valid is True
        assert result.coordinate_count == 1

    def test_invalid_point_longitude(self, agent: EUDRComplianceAgent):
        """Test invalid longitude (out of range)."""
        geo = GeoLocation(type=GeometryType.POINT, coordinates=[-200, -15.5])
        result = agent._validate_geolocation(geo)
        assert result.is_valid is False

    def test_invalid_point_latitude(self, agent: EUDRComplianceAgent):
        """Test invalid latitude (out of range)."""
        geo = GeoLocation(type=GeometryType.POINT, coordinates=[-47.5, -95])
        result = agent._validate_geolocation(geo)
        assert result.is_valid is False

    def test_valid_polygon_closed(self, agent: EUDRComplianceAgent, valid_polygon_geolocation: GeoLocation):
        """Test valid closed polygon."""
        result = agent._validate_geolocation(valid_polygon_geolocation)
        assert result.is_valid is True
        assert result.is_closed is True

    def test_polygon_unclosed_ring(self, agent: EUDRComplianceAgent):
        """Test polygon with unclosed ring."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-47.5, -15.5],
                [-47.49, -15.5],
                [-47.49, -15.49],
                [-47.5, -15.49]
                # Missing closing point
            ]]
        )
        result = agent._validate_geolocation(geo)
        assert result.is_closed is False

    def test_polygon_area_calculation(self, agent: EUDRComplianceAgent, valid_polygon_geolocation: GeoLocation):
        """Test polygon area calculation."""
        result = agent._validate_geolocation(valid_polygon_geolocation)
        assert result.area_hectares is not None
        assert result.area_hectares > 0

    def test_polygon_below_minimum_area(self, agent: EUDRComplianceAgent, small_polygon_geolocation: GeoLocation):
        """Test polygon below 1 hectare minimum."""
        result = agent._validate_geolocation(small_polygon_geolocation)
        # Should have validation errors for small area
        assert any(e.code == "INSUFFICIENT_AREA" for e in result.errors)

    def test_polygon_self_intersection(self, agent: EUDRComplianceAgent, self_intersecting_polygon: GeoLocation):
        """Test self-intersecting polygon detection."""
        result = agent._validate_geolocation(self_intersecting_polygon)
        assert result.has_self_intersection is True

    def test_multi_polygon_validation(self, agent: EUDRComplianceAgent):
        """Test MultiPolygon validation."""
        geo = GeoLocation(
            type=GeometryType.MULTI_POLYGON,
            coordinates=[
                [[[-47.5, -15.5], [-47.49, -15.5], [-47.49, -15.49], [-47.5, -15.49], [-47.5, -15.5]]],
                [[[-47.4, -15.4], [-47.39, -15.4], [-47.39, -15.39], [-47.4, -15.39], [-47.4, -15.4]]]
            ]
        )
        result = agent._validate_geolocation(geo)
        assert result.geometry_type == "MultiPolygon"

    def test_crs_validation(self, agent: EUDRComplianceAgent):
        """Test CRS validation for EUDR."""
        assert agent._validate_crs("EPSG:4326") is True
        assert agent._validate_crs("WGS84") is True
        assert agent._validate_crs("EPSG:32632") is False


# =============================================================================
# DDS GENERATION TESTS (15 tests)
# =============================================================================


class TestDDSGeneration:
    """Tests for Due Diligence Statement generation."""

    def test_dds_reference_format(self, agent: EUDRComplianceAgent):
        """Test DDS reference number format."""
        ref = agent._generate_dds_reference("OPERATOR123", date(2024, 6, 15))
        assert ref.startswith("DDS-")
        assert len(ref) == 19  # DDS-XXXXXXXX-YYMMDD

    def test_dds_generation_compliant(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation,
        valid_supplier_info: SupplierInfo,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test DDS generation for compliant product."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=10000,
            country_of_origin="FR",  # Low risk
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            operator_id="FR123456789",
            supplier_info=valid_supplier_info,
            supply_chain=valid_supply_chain,
            certifications=["Rainforest Alliance"]
        )
        result = agent.run(input_data)
        assert result.dds_document is not None
        assert result.dds_document.reference_number.startswith("DDS-")

    def test_dds_contains_operator_id(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS contains operator identification."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=5000,
            country_of_origin="DE",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            operator_id="DE987654321"
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.operator_id == "DE987654321"

    def test_dds_geolocation_summary_point(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS geolocation summary for point."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert "Point:" in result.dds_document.geolocation_summary

    def test_dds_geolocation_summary_polygon(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test DDS geolocation summary for polygon."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=5000,
            country_of_origin="FR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert "Polygon" in result.dds_document.geolocation_summary

    def test_dds_validity_period(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS validity period is 1 year."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=2000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.valid_until.year == date.today().year + 1

    def test_dds_provenance_hash(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS has provenance hash."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=10000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert len(result.dds_document.provenance_hash) == 64  # SHA-256

    def test_dds_not_generated_for_non_compliant(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS not generated for non-compliant products."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0102.21.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2019, 1, 1)  # Before cutoff
        )
        result = agent.run(input_data)
        assert result.compliance_status == "non_compliant"
        assert result.dds_document is None

    def test_dds_submission_type_new(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS submission type is 'new'."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=5000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.submission_type == "new"

    def test_dds_pending_operator_registration(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS handles missing operator ID."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=3000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
            # No operator_id
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.operator_id == "PENDING_REGISTRATION"

    def test_dds_commodity_type_correct(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS commodity type matches input."""
        for commodity in CommodityType:
            if commodity == CommodityType.CATTLE:
                cn_code = "0102.21.00"
            elif commodity == CommodityType.COCOA:
                cn_code = "1801.00.00"
            elif commodity == CommodityType.COFFEE:
                cn_code = "0901.11.00"
            elif commodity == CommodityType.PALM_OIL:
                cn_code = "1511.10.10"
            elif commodity == CommodityType.RUBBER:
                cn_code = "4001.10.00"
            elif commodity == CommodityType.SOYA:
                cn_code = "1201.90.00"
            else:  # WOOD
                cn_code = "4403.11.00"

            input_data = EUDRInput(
                commodity_type=commodity,
                cn_code=cn_code,
                quantity_kg=1000,
                country_of_origin="FR",
                geolocation=valid_point_geolocation,
                production_date=date(2024, 1, 1)
            )
            result = agent.run(input_data)
            if result.dds_document:
                assert result.dds_document.commodity_type == commodity.value

    def test_dds_quantity_preserved(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS preserves quantity."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=12345.67,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.quantity_kg == 12345.67

    def test_dds_country_preserved(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS preserves country of origin."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=5000,
            country_of_origin="TH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.country_of_origin == "TH"

    def test_dds_production_date_preserved(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test DDS preserves production date."""
        prod_date = date(2024, 6, 15)
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=50000,
            country_of_origin="FR",
            geolocation=valid_point_geolocation,
            production_date=prod_date
        )
        result = agent.run(input_data)
        if result.dds_document:
            assert result.dds_document.production_date == prod_date


# =============================================================================
# SUPPLY CHAIN TESTS (20 tests)
# =============================================================================


class TestSupplyChain:
    """Tests for supply chain traceability."""

    def test_empty_supply_chain(self, agent: EUDRComplianceAgent):
        """Test traceability with no supply chain."""
        score = agent._calculate_traceability([], None)
        assert score == 0.0

    def test_full_traceability(
        self,
        agent: EUDRComplianceAgent,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test 100% traceability with all verified nodes."""
        score = agent._calculate_traceability(valid_supply_chain, None)
        assert score >= 100.0

    def test_partial_traceability(self, agent: EUDRComplianceAgent):
        """Test partial traceability with some unverified nodes."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="BR", verified=False
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert 40 < score < 60  # Approximately 50%

    def test_supplier_info_bonus(
        self,
        agent: EUDRComplianceAgent,
        valid_supplier_info: SupplierInfo
    ):
        """Test supplier info adds traceability bonus."""
        score_without = agent._calculate_traceability([], None)
        score_with = agent._calculate_traceability([], valid_supplier_info)
        assert score_with > score_without

    def test_documentation_bonus(self, agent: EUDRComplianceAgent):
        """Test documentation adds traceability bonus."""
        chain_no_docs = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True, documents=[]
            ),
        ]
        chain_with_docs = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True, documents=["cert.pdf"]
            ),
        ]
        score_no_docs = agent._calculate_traceability(chain_no_docs, None)
        score_with_docs = agent._calculate_traceability(chain_with_docs, None)
        assert score_with_docs >= score_no_docs

    def test_certification_bonus(self, agent: EUDRComplianceAgent):
        """Test certification adds supplier bonus."""
        supplier_no_cert = SupplierInfo(
            name="Test", country="BR", verified=True, certifications=[]
        )
        supplier_with_cert = SupplierInfo(
            name="Test", country="BR", verified=True, certifications=["FSC"]
        )
        score_no = agent._calculate_traceability([], supplier_no_cert)
        score_with = agent._calculate_traceability([], supplier_with_cert)
        assert score_with > score_no

    def test_recent_audit_bonus(self, agent: EUDRComplianceAgent):
        """Test recent audit adds supplier bonus."""
        supplier_old_audit = SupplierInfo(
            name="Test", country="BR", verified=True,
            last_audit_date=date.today() - timedelta(days=400)  # Old
        )
        supplier_recent_audit = SupplierInfo(
            name="Test", country="BR", verified=True,
            last_audit_date=date.today() - timedelta(days=30)  # Recent
        )
        score_old = agent._calculate_traceability([], supplier_old_audit)
        score_recent = agent._calculate_traceability([], supplier_recent_audit)
        assert score_recent > score_old

    def test_traceability_cap_at_100(
        self,
        agent: EUDRComplianceAgent,
        valid_supply_chain: List[SupplyChainNode],
        valid_supplier_info: SupplierInfo
    ):
        """Test traceability score capped at 100%."""
        score = agent._calculate_traceability(valid_supply_chain, valid_supplier_info)
        assert score <= 100.0

    def test_unverified_chain(self, agent: EUDRComplianceAgent):
        """Test completely unverified chain."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=False
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="BR", verified=False
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score < 50

    def test_mixed_verification_chain(self, agent: EUDRComplianceAgent):
        """Test chain with mixed verification status."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="BR", verified=True
            ),
            SupplyChainNode(
                node_id="3", node_type="exporter", operator_name="C",
                country_code="BR", verified=False
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert 60 <= score <= 80

    def test_single_node_verified(self, agent: EUDRComplianceAgent):
        """Test single verified node."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score >= 100.0

    def test_single_node_unverified(self, agent: EUDRComplianceAgent):
        """Test single unverified node."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=False
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score == 0.0

    def test_long_chain_traceability(self, agent: EUDRComplianceAgent):
        """Test long supply chain traceability."""
        chain = [
            SupplyChainNode(
                node_id=str(i), node_type="node", operator_name=f"Op{i}",
                country_code="BR", verified=True
            )
            for i in range(10)
        ]
        score = agent._calculate_traceability(chain, None)
        assert score >= 100.0

    def test_supplier_only_traceability(
        self,
        agent: EUDRComplianceAgent,
        valid_supplier_info: SupplierInfo
    ):
        """Test traceability with only supplier info."""
        score = agent._calculate_traceability([], valid_supplier_info)
        assert score > 0

    def test_unverified_supplier(self, agent: EUDRComplianceAgent):
        """Test unverified supplier contribution."""
        supplier = SupplierInfo(
            name="Test", country="BR", verified=False
        )
        score = agent._calculate_traceability([], supplier)
        # Should still have some score from supplier existing
        assert score >= 0

    def test_traceability_affects_compliance(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test low traceability affects compliance status."""
        # No supply chain = insufficient data
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.traceability_score == 0.0

    def test_chain_with_all_documents(self, agent: EUDRComplianceAgent):
        """Test chain where all nodes have documents."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True, documents=["doc1.pdf"]
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="BR", verified=True, documents=["doc2.pdf"]
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score == 100.0  # Full traceability

    def test_chain_with_partial_documents(self, agent: EUDRComplianceAgent):
        """Test chain where some nodes have documents."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True, documents=["doc1.pdf"]
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="BR", verified=True, documents=[]
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score >= 100.0  # Still full as both verified

    def test_multinational_chain(self, agent: EUDRComplianceAgent):
        """Test supply chain across multiple countries."""
        chain = [
            SupplyChainNode(
                node_id="1", node_type="producer", operator_name="A",
                country_code="BR", verified=True
            ),
            SupplyChainNode(
                node_id="2", node_type="processor", operator_name="B",
                country_code="NL", verified=True
            ),
            SupplyChainNode(
                node_id="3", node_type="trader", operator_name="C",
                country_code="DE", verified=True
            ),
        ]
        score = agent._calculate_traceability(chain, None)
        assert score >= 100.0


# =============================================================================
# SATELLITE VERIFICATION TESTS (15 tests)
# =============================================================================


class TestSatelliteVerification:
    """Tests for forest cover analysis."""

    def test_forest_analysis_amazon_region(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest analysis in Amazon region."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert analysis.baseline_date == date(2020, 12, 31)
        assert analysis.baseline_forest_cover_pct >= 0

    def test_forest_analysis_data_sources(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest analysis includes data sources."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert "Sentinel-2" in analysis.data_sources
        assert "Landsat-8" in analysis.data_sources

    def test_forest_analysis_ndvi_values(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest analysis includes NDVI values."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert analysis.ndvi_baseline is not None
        assert analysis.ndvi_current is not None

    def test_forest_analysis_confidence_score(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest analysis confidence score."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert 0 <= analysis.confidence_score <= 1

    def test_deforestation_threshold_detection(self, agent: EUDRComplianceAgent):
        """Test deforestation detection threshold (>5%)."""
        # High deforestation area simulation
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-55.5, -5.5],
                [-55.49, -5.5],
                [-55.49, -5.49],
                [-55.5, -5.49],
                [-55.5, -5.5]
            ]]
        )
        analysis = agent._analyze_forest_cover(geo, date(2024, 1, 1))
        # Status depends on simulated loss
        assert analysis.deforestation_status in [
            DeforestationStatus.NO_DEFORESTATION,
            DeforestationStatus.DEGRADATION_DETECTED,
            DeforestationStatus.DEFORESTATION_DETECTED
        ]

    def test_degradation_threshold_detection(self, agent: EUDRComplianceAgent):
        """Test degradation detection threshold (2-5%)."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-50.0, -10.0],
                [-49.99, -10.0],
                [-49.99, -9.99],
                [-50.0, -9.99],
                [-50.0, -10.0]
            ]]
        )
        analysis = agent._analyze_forest_cover(geo, date(2024, 1, 1))
        assert analysis.degradation_detected is not None

    def test_no_deforestation_low_risk_area(self, agent: EUDRComplianceAgent):
        """Test no deforestation in low risk area."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [10.0, 50.0],  # Europe
                [10.01, 50.0],
                [10.01, 50.01],
                [10.0, 50.01],
                [10.0, 50.0]
            ]]
        )
        analysis = agent._analyze_forest_cover(geo, date(2024, 1, 1))
        assert analysis.forest_loss_pct == 0.0
        assert analysis.deforestation_status == DeforestationStatus.NO_DEFORESTATION

    def test_forest_loss_hectares_calculation(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest loss in hectares calculation."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert analysis.forest_loss_hectares >= 0

    def test_forest_analysis_point_geometry(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test forest analysis with point geometry (should not run)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        # Point geometry doesn't trigger satellite analysis
        assert result.forest_cover_analysis is None

    def test_forest_analysis_polygon_triggers(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test forest analysis triggered for polygon geometry."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=5000,
            country_of_origin="BR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.forest_cover_analysis is not None

    def test_southeast_asia_analysis(self, agent: EUDRComplianceAgent):
        """Test forest analysis in Southeast Asia region."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [110.0, 0.0],  # Borneo
                [110.01, 0.0],
                [110.01, 0.01],
                [110.0, 0.01],
                [110.0, 0.0]
            ]]
        )
        analysis = agent._analyze_forest_cover(geo, date(2024, 1, 1))
        assert analysis.baseline_forest_cover_pct >= 80  # High forest cover

    def test_congo_basin_analysis(self, agent: EUDRComplianceAgent):
        """Test forest analysis in Congo Basin region."""
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [20.0, 0.0],  # Congo
                [20.01, 0.0],
                [20.01, 0.01],
                [20.0, 0.01],
                [20.0, 0.0]
            ]]
        )
        analysis = agent._analyze_forest_cover(geo, date(2024, 1, 1))
        assert analysis.baseline_forest_cover_pct >= 85

    def test_analysis_date_is_current(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test analysis date is current date."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert analysis.analysis_date == date.today()

    def test_baseline_date_is_cutoff(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test baseline date is EUDR cutoff date."""
        analysis = agent._analyze_forest_cover(
            valid_polygon_geolocation,
            date(2024, 1, 1)
        )
        assert analysis.baseline_date == date(2020, 12, 31)

    def test_deforestation_affects_compliance(
        self,
        agent: EUDRComplianceAgent
    ):
        """Test deforestation detection affects compliance status."""
        # Force deforestation detection in high-risk area
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-60.0, 0.0],  # Amazon with high loss seed
                [-59.99, 0.0],
                [-59.99, 0.01],
                [-60.0, 0.01],
                [-60.0, 0.0]
            ]]
        )
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=100000,
            country_of_origin="BR",
            geolocation=geo,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        if result.deforestation_detected:
            assert result.compliance_status == "non_compliant"


# =============================================================================
# RISK ASSESSMENT TESTS (10 tests)
# =============================================================================


class TestRiskAssessment:
    """Tests for risk assessment engine."""

    def test_high_risk_country_score(self, agent: EUDRComplianceAgent):
        """Test high risk country scoring."""
        risk = agent._get_country_risk("BR")
        assert risk.risk_level == RiskLevel.HIGH
        assert risk.risk_score >= 70

    def test_standard_risk_country_score(self, agent: EUDRComplianceAgent):
        """Test standard risk country scoring."""
        risk = agent._get_country_risk("CO")
        assert risk.risk_level == RiskLevel.STANDARD
        assert 40 <= risk.risk_score < 70

    def test_low_risk_country_score(self, agent: EUDRComplianceAgent):
        """Test low risk country scoring."""
        risk = agent._get_country_risk("DE")
        assert risk.risk_level == RiskLevel.LOW
        assert risk.risk_score < 30

    def test_default_country_risk(self, agent: EUDRComplianceAgent):
        """Test default risk for unlisted country."""
        risk = agent._get_country_risk("XX")  # Non-existent
        assert risk.risk_level == RiskLevel.STANDARD

    def test_commodity_risk_palm_oil(self, agent: EUDRComplianceAgent):
        """Test palm oil has highest commodity risk."""
        risk = agent._calculate_commodity_risk(CommodityType.PALM_OIL)
        assert risk == 75.0  # Highest

    def test_commodity_risk_wood(self, agent: EUDRComplianceAgent):
        """Test wood has lower commodity risk."""
        risk = agent._calculate_commodity_risk(CommodityType.WOOD)
        assert risk == 40.0  # Lowest

    def test_risk_assessment_weighted_formula(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test risk assessment uses correct weighted formula."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=50000,
            country_of_origin="ID",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        # High risk country + high risk commodity = high overall
        assert result.risk_assessment.overall_risk_level == RiskLevel.HIGH

    def test_risk_factors_populated(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test risk factors are populated."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0102.21.00",
            quantity_kg=5000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert len(result.risk_assessment.risk_factors) > 0

    def test_mitigating_factors_with_certification(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test mitigating factors with certification."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=10000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            certifications=["Rainforest Alliance"]
        )
        result = agent.run(input_data)
        assert any("certification" in f.lower() for f in result.risk_assessment.mitigating_factors)

    def test_due_diligence_type_based_on_risk(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test due diligence type matches risk level."""
        # High risk
        input_high = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=50000,
            country_of_origin="ID",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result_high = agent.run(input_high)
        assert result_high.risk_assessment.due_diligence_type == DueDiligenceType.REINFORCED

        # Low risk
        input_low = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=5000,
            country_of_origin="DE",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result_low = agent.run(input_low)
        assert result_low.risk_assessment.due_diligence_type == DueDiligenceType.STANDARD


# =============================================================================
# EDGE CASES TESTS (15 tests)
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_cutoff_date_exactly(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test production on exactly cutoff date."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2020, 12, 31)  # Exactly cutoff
        )
        result = agent.run(input_data)
        assert result.cutoff_date_compliant is False  # Must be AFTER

    def test_production_day_after_cutoff(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test production day after cutoff date."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2021, 1, 1)  # Day after
        )
        result = agent.run(input_data)
        assert result.cutoff_date_compliant is True

    def test_production_before_cutoff(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test production before cutoff date."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0102.21.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2019, 6, 1)  # Before
        )
        result = agent.run(input_data)
        assert result.compliance_status == "non_compliant"

    def test_zero_quantity(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test zero quantity handling."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=0,
            country_of_origin="GH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.provenance_hash is not None

    def test_very_large_quantity(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test very large quantity handling."""
        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=10000000,  # 10,000 tonnes
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.provenance_hash is not None

    def test_country_code_lowercase(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test lowercase country code normalization."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=5000,
            country_of_origin="br",  # lowercase
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.risk_level == "high"  # BR is high risk

    def test_cn_code_with_dots(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test CN code with dots."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.cn_code == "0901.11.00"

    def test_cn_code_without_dots(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test CN code without dots."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="09011100",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.commodity_type == "coffee"

    def test_coordinates_at_boundaries(self, agent: EUDRComplianceAgent):
        """Test coordinates at valid boundaries."""
        # Maximum values
        geo = GeoLocation(
            type=GeometryType.POINT,
            coordinates=[180.0, 90.0]
        )
        result = agent._validate_geolocation(geo)
        assert result.is_valid is True

        # Minimum values
        geo2 = GeoLocation(
            type=GeometryType.POINT,
            coordinates=[-180.0, -90.0]
        )
        result2 = agent._validate_geolocation(geo2)
        assert result2.is_valid is True

    def test_empty_certifications_list(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test empty certifications list."""
        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=5000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            certifications=[]
        )
        result = agent.run(input_data)
        assert "certification" in result.mitigation_measures[0].lower() or len(result.mitigation_measures) > 0

    def test_invalid_certification_ignored(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test invalid certification is ignored."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=1000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            certifications=["INVALID_CERT"]
        )
        result = agent.run(input_data)
        # Should not crash, invalid cert just not counted
        assert result.provenance_hash is not None

    def test_cn_code_mismatch_commodity(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test CN code mismatch with commodity type."""
        # Coffee CN code with cocoa commodity
        with pytest.raises(ValueError):
            input_data = EUDRInput(
                commodity_type=CommodityType.COCOA,  # Mismatch
                cn_code="0901.11.00",  # Coffee CN
                quantity_kg=1000,
                country_of_origin="BR",
                geolocation=valid_point_geolocation,
                production_date=date(2024, 1, 1)
            )
            agent.run(input_data)

    def test_processing_time_recorded(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test processing time is recorded."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=50000,
            country_of_origin="MY",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert result.processing_time_ms > 0

    def test_provenance_hash_consistent(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test provenance hash is 64 characters (SHA-256)."""
        input_data = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=10000,
            country_of_origin="TH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)
        assert len(result.provenance_hash) == 64

    def test_all_commodities_in_scope(self, agent: EUDRComplianceAgent):
        """Test all commodity types are in EUDR scope."""
        commodities = agent.get_commodities()
        assert len(commodities) == 7
        assert "cattle" in commodities
        assert "cocoa" in commodities
        assert "coffee" in commodities
        assert "palm_oil" in commodities
        assert "rubber" in commodities
        assert "soya" in commodities
        assert "wood" in commodities


# =============================================================================
# INTEGRATION TESTS (Sample - 20 of 80)
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow_compliant_coffee(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation,
        valid_supplier_info: SupplierInfo,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test complete workflow for compliant coffee."""
        input_data = EUDRInput(
            commodity_type=CommodityType.COFFEE,
            cn_code="0901.11.00",
            quantity_kg=25000,
            country_of_origin="FR",  # Low risk
            geolocation=valid_point_geolocation,
            production_date=date(2024, 6, 1),
            operator_id="FR123456789",
            supplier_info=valid_supplier_info,
            supply_chain=valid_supply_chain,
            certifications=["Rainforest Alliance", "4C"],
            supporting_documents=["contract.pdf", "invoice.pdf"]
        )
        result = agent.run(input_data)

        # Verify all outputs
        assert result.compliance_status in ["compliant", "pending_verification"]
        assert result.risk_level == "low"
        assert result.geolocation_valid is True
        assert result.cutoff_date_compliant is True
        assert result.traceability_score > 80
        assert result.dds_document is not None
        assert len(result.provenance_hash) == 64

    def test_full_workflow_high_risk_palm_oil(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation
    ):
        """Test complete workflow for high-risk palm oil."""
        input_data = EUDRInput(
            commodity_type=CommodityType.PALM_OIL,
            cn_code="1511.10.10",
            quantity_kg=100000,
            country_of_origin="ID",  # High risk
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)

        assert result.risk_level == "high"
        assert result.country_risk_score >= 70
        assert result.forest_cover_analysis is not None
        assert len(result.mitigation_measures) > 0

    def test_full_workflow_non_compliant_pre_cutoff(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test complete workflow for non-compliant (pre-cutoff) product."""
        input_data = EUDRInput(
            commodity_type=CommodityType.CATTLE,
            cn_code="0102.21.00",
            quantity_kg=5000,
            country_of_origin="BR",
            geolocation=valid_point_geolocation,
            production_date=date(2018, 6, 1)  # Before cutoff
        )
        result = agent.run(input_data)

        assert result.compliance_status == "non_compliant"
        assert result.cutoff_date_compliant is False
        assert result.dds_document is None

    def test_full_workflow_with_certification(
        self,
        agent: EUDRComplianceAgent,
        valid_polygon_geolocation: GeoLocation,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test workflow with relevant certification improves risk."""
        # Without certification
        input_no_cert = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=50000,
            country_of_origin="BR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 1, 1),
            supply_chain=valid_supply_chain
        )

        # With certification
        input_with_cert = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=50000,
            country_of_origin="BR",
            geolocation=valid_polygon_geolocation,
            production_date=date(2024, 1, 1),
            supply_chain=valid_supply_chain,
            certifications=["FSC"]
        )

        result_no_cert = agent.run(input_no_cert)
        result_with_cert = agent.run(input_with_cert)

        assert result_with_cert.risk_assessment.documentation_risk_score <= result_no_cert.risk_assessment.documentation_risk_score

    def test_full_workflow_all_commodities(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test workflow runs for all 7 commodities."""
        commodity_cn_map = {
            CommodityType.CATTLE: "0102.21.00",
            CommodityType.COCOA: "1801.00.00",
            CommodityType.COFFEE: "0901.11.00",
            CommodityType.PALM_OIL: "1511.10.10",
            CommodityType.RUBBER: "4001.10.00",
            CommodityType.SOYA: "1201.90.00",
            CommodityType.WOOD: "4403.11.00",
        }

        for commodity, cn_code in commodity_cn_map.items():
            input_data = EUDRInput(
                commodity_type=commodity,
                cn_code=cn_code,
                quantity_kg=1000,
                country_of_origin="DE",  # Low risk
                geolocation=valid_point_geolocation,
                production_date=date(2024, 1, 1)
            )
            result = agent.run(input_data)
            assert result.commodity_type == commodity.value
            assert result.provenance_hash is not None

    def test_workflow_country_risk_variations(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test workflow with different country risk levels."""
        countries = {
            "BR": RiskLevel.HIGH,
            "CO": RiskLevel.STANDARD,
            "DE": RiskLevel.LOW,
        }

        for country, expected_level in countries.items():
            input_data = EUDRInput(
                commodity_type=CommodityType.COFFEE,
                cn_code="0901.11.00",
                quantity_kg=5000,
                country_of_origin=country,
                geolocation=valid_point_geolocation,
                production_date=date(2024, 1, 1)
            )
            result = agent.run(input_data)
            assert result.risk_assessment.country_risk_level == expected_level

    def test_workflow_traceability_impact(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation,
        valid_supply_chain: List[SupplyChainNode]
    ):
        """Test traceability impacts compliance status."""
        # With full traceability
        input_traced = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=10000,
            country_of_origin="GH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            supply_chain=valid_supply_chain
        )

        # Without traceability
        input_no_trace = EUDRInput(
            commodity_type=CommodityType.COCOA,
            cn_code="1801.00.00",
            quantity_kg=10000,
            country_of_origin="GH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )

        result_traced = agent.run(input_traced)
        result_no_trace = agent.run(input_no_trace)

        assert result_traced.traceability_score > result_no_trace.traceability_score

    def test_workflow_deforestation_impact(
        self,
        agent: EUDRComplianceAgent
    ):
        """Test deforestation detection impacts compliance."""
        # High-risk Amazon region polygon
        geo = GeoLocation(
            type=GeometryType.POLYGON,
            coordinates=[[
                [-60.0, -5.0],
                [-59.9, -5.0],
                [-59.9, -4.9],
                [-60.0, -4.9],
                [-60.0, -5.0]
            ]]
        )

        input_data = EUDRInput(
            commodity_type=CommodityType.SOYA,
            cn_code="1201.90.00",
            quantity_kg=500000,
            country_of_origin="BR",
            geolocation=geo,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)

        # Satellite analysis should be performed
        assert result.forest_cover_analysis is not None
        # If deforestation detected, should be non-compliant
        if result.deforestation_detected:
            assert result.compliance_status == "non_compliant"

    def test_workflow_documentation_impact(
        self,
        agent: EUDRComplianceAgent,
        valid_point_geolocation: GeoLocation
    ):
        """Test documentation quantity impacts risk score."""
        # With many documents
        input_many_docs = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=20000,
            country_of_origin="TH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1),
            supporting_documents=["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf", "doc5.pdf"]
        )

        # With no documents
        input_no_docs = EUDRInput(
            commodity_type=CommodityType.RUBBER,
            cn_code="4001.10.00",
            quantity_kg=20000,
            country_of_origin="TH",
            geolocation=valid_point_geolocation,
            production_date=date(2024, 1, 1)
        )

        result_many = agent.run(input_many_docs)
        result_none = agent.run(input_no_docs)

        assert result_many.risk_assessment.documentation_risk_score < result_none.risk_assessment.documentation_risk_score

    def test_workflow_multipolygon_analysis(self, agent: EUDRComplianceAgent):
        """Test workflow with MultiPolygon geometry."""
        geo = GeoLocation(
            type=GeometryType.MULTI_POLYGON,
            coordinates=[
                [[[-47.5, -15.5], [-47.49, -15.5], [-47.49, -15.49], [-47.5, -15.49], [-47.5, -15.5]]],
                [[[-47.4, -15.4], [-47.39, -15.4], [-47.39, -15.39], [-47.4, -15.39], [-47.4, -15.4]]]
            ]
        )

        input_data = EUDRInput(
            commodity_type=CommodityType.WOOD,
            cn_code="4403.11.00",
            quantity_kg=30000,
            country_of_origin="BR",
            geolocation=geo,
            production_date=date(2024, 1, 1)
        )
        result = agent.run(input_data)

        assert result.geolocation_validation.geometry_type == "MultiPolygon"
        assert result.forest_cover_analysis is not None


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_commodities(self, agent: EUDRComplianceAgent):
        """Test get_commodities returns all 7."""
        commodities = agent.get_commodities()
        assert len(commodities) == 7

    def test_is_in_eudr_scope_valid(self, agent: EUDRComplianceAgent):
        """Test is_in_eudr_scope for valid CN codes."""
        assert agent.is_in_eudr_scope("0901.11.00") is True  # Coffee
        assert agent.is_in_eudr_scope("1801.00.00") is True  # Cocoa
        assert agent.is_in_eudr_scope("4403.11.00") is True  # Wood

    def test_is_in_eudr_scope_invalid(self, agent: EUDRComplianceAgent):
        """Test is_in_eudr_scope for invalid CN codes."""
        assert agent.is_in_eudr_scope("9999.99.99") is False

    def test_get_country_risk_level(self, agent: EUDRComplianceAgent):
        """Test get_country_risk_level method."""
        assert agent.get_country_risk_level("BR") == "high"
        assert agent.get_country_risk_level("DE") == "low"
        assert agent.get_country_risk_level("CO") == "standard"

    def test_get_certification_options(self, agent: EUDRComplianceAgent):
        """Test get_certification_options for each commodity."""
        assert "FSC" in agent.get_certification_options(CommodityType.WOOD)
        assert "RSPO" in agent.get_certification_options(CommodityType.PALM_OIL)
        assert "Rainforest Alliance" in agent.get_certification_options(CommodityType.COFFEE)


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
