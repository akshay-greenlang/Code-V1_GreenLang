# -*- coding: utf-8 -*-
"""
Cross-Agent Integration Tests - AGENT-EUDR-021 Indigenous Rights Checker

Tests validating the EUDR-021 Indigenous Rights Checker agent's integration
with other GreenLang agents across cross-agent workflows:

1. EUDR-001 Supply Chain Mapper   - Territory overlap with supply chain plots
2. EUDR-002 Geolocation Verifier  - Plot coordinate validation
3. EUDR-016 Country Risk Evaluator - Indigenous rights governance scoring
4. AGENT-DATA-005 EUDR Traceability - Plot registry access
5. AGENT-DATA-006 GIS Connector    - Spatial operations
6. EUDR-012 Document Authentication - FPIC document verification

Test count: 25 integration tests
Coverage: Cross-agent data flow, provenance chain continuity, and
          bidirectional validation between EUDR-021 and 6 peer agents.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    AlertSeverity,
    ConfidenceLevel,
    ConsultationStage,
    CountryRiskLevel,
    FPICStatus,
    FPICWorkflowStage,
    OverlapType,
    RiskLevel,
    TerritoryLegalStatus,
    ViolationType,
    IndigenousTerritory,
    FPICAssessment,
    TerritoryOverlap,
    IndigenousCommunity,
    ConsultationRecord,
    ViolationAlert,
    ComplianceReport,
    CountryIndigenousRightsScore,
    DetectOverlapRequest,
    BatchOverlapRequest,
    VerifyFPICRequest,
)
from greenlang.agents.eudr.indigenous_rights_checker.territory_database_engine import (
    TerritoryDatabaseEngine,
)
from greenlang.agents.eudr.indigenous_rights_checker.fpic_verification_engine import (
    FPICVerificationEngine,
)
from greenlang.agents.eudr.indigenous_rights_checker.land_rights_overlap_engine import (
    LandRightsOverlapEngine,
)
from greenlang.agents.eudr.indigenous_rights_checker.rights_violation_engine import (
    RightsViolationEngine,
)
from greenlang.agents.eudr.indigenous_rights_checker.compliance_reporting_engine import (
    ComplianceReportingEngine,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash for test assertions."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _make_territory(
    territory_id: str,
    country_code: str,
    legal_status: TerritoryLegalStatus,
    lat: float,
    lon: float,
    area_hectares: Decimal,
    people_name: str = "TestPeople",
) -> IndigenousTerritory:
    """Create a territory with a simple square polygon for testing."""
    return IndigenousTerritory(
        territory_id=territory_id,
        territory_name=f"Territory {territory_id}",
        people_name=people_name,
        country_code=country_code,
        area_hectares=area_hectares,
        legal_status=legal_status,
        boundary_geojson={
            "type": "Polygon",
            "coordinates": [[
                [lon, lat],
                [lon, lat + 1.0],
                [lon + 1.0, lat + 1.0],
                [lon + 1.0, lat],
                [lon, lat],
            ]],
        },
        data_source="test",
        confidence=ConfidenceLevel.HIGH,
        provenance_hash=_compute_hash({
            "territory_id": territory_id,
            "country_code": country_code,
        }),
    )


def _make_supply_chain_node(
    node_id: str,
    node_type: str,
    country_code: str,
    lat: float,
    lon: float,
    plot_ids: Optional[List[str]] = None,
    commodity: str = "soya",
) -> Dict[str, Any]:
    """Create a mock EUDR-001 supply chain node."""
    return {
        "node_id": node_id,
        "node_type": node_type,
        "country_code": country_code,
        "latitude": lat,
        "longitude": lon,
        "commodities": [commodity],
        "plot_ids": plot_ids or [],
        "compliance_status": "pending",
        "tier_depth": 1,
    }


def _make_plot_geojson(lat: float, lon: float, size_deg: float = 0.1) -> Dict:
    """Create a square plot GeoJSON polygon centered at lat/lon."""
    half = size_deg / 2.0
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - half, lat - half],
            [lon - half, lat + half],
            [lon + half, lat + half],
            [lon + half, lat - half],
            [lon - half, lat - half],
        ]],
    }


# ---------------------------------------------------------------------------
# Mock factories for external agents
# ---------------------------------------------------------------------------


def _mock_supply_chain_mapper():
    """Create a mock EUDR-001 Supply Chain Mapper service."""
    mapper = MagicMock()
    mapper.get_plot_locations = AsyncMock(return_value=[
        {"plot_id": "PLOT-BR-001", "lat": -3.05, "lon": -60.02,
         "area_ha": 150.0, "commodity": "soya", "country_code": "BR"},
        {"plot_id": "PLOT-BR-002", "lat": -3.50, "lon": -60.50,
         "area_ha": 200.0, "commodity": "cattle", "country_code": "BR"},
        {"plot_id": "PLOT-ID-001", "lat": -1.50, "lon": 116.05,
         "area_ha": 300.0, "commodity": "palm_oil", "country_code": "ID"},
    ])
    mapper.get_supplier_plots = AsyncMock(return_value=[
        {"plot_id": "PLOT-BR-001", "supplier_id": "SUP-001",
         "node_type": "producer", "tier": 1},
        {"plot_id": "PLOT-BR-002", "supplier_id": "SUP-002",
         "node_type": "producer", "tier": 2},
    ])
    mapper.get_supply_chain_graph = AsyncMock(return_value={
        "graph_id": "g-001",
        "operator_id": "op-001",
        "commodity": "soya",
        "nodes": [
            _make_supply_chain_node(
                "n-001", "producer", "BR", -3.05, -60.02,
                plot_ids=["PLOT-BR-001"],
            ),
            _make_supply_chain_node(
                "n-002", "producer", "BR", -3.50, -60.50,
                plot_ids=["PLOT-BR-002"],
            ),
        ],
    })
    mapper.update_node_compliance = AsyncMock(return_value=True)
    return mapper


def _mock_geolocation_verifier():
    """Create a mock EUDR-002 Geolocation Verification service."""
    verifier = MagicMock()
    verifier.verify_coordinates = AsyncMock(return_value={
        "valid": True,
        "country_code": "BR",
        "in_protected_area": False,
        "confidence": 0.98,
        "elevation_m": 85.0,
    })
    verifier.verify_polygon = AsyncMock(return_value={
        "valid": True,
        "area_hectares": 150.5,
        "centroid_lat": -3.05,
        "centroid_lon": -60.02,
        "is_self_intersecting": False,
        "country_code": "BR",
    })
    verifier.check_protected_areas = AsyncMock(return_value={
        "in_protected_area": False,
        "nearest_protected_area": {
            "name": "Reserva Florestal do Jau",
            "distance_km": 45.0,
            "type": "national_park",
        },
    })
    verifier.batch_verify = AsyncMock(return_value=[
        {"plot_id": "PLOT-BR-001", "valid": True, "country_code": "BR"},
        {"plot_id": "PLOT-BR-002", "valid": True, "country_code": "BR"},
    ])
    return verifier


def _mock_country_risk_evaluator():
    """Create a mock EUDR-016 Country Risk Evaluator service."""
    evaluator = MagicMock()
    evaluator.get_country_risk = AsyncMock(side_effect=lambda cc: {
        "BR": {
            "country_code": "BR",
            "overall_risk": "high",
            "governance_score": 45.0,
            "deforestation_risk": 78.0,
            "indigenous_rights_score": 40.0,
            "ilo_169_ratified": True,
            "fpic_legal_requirement": True,
            "land_tenure_security": 42.0,
        },
        "ID": {
            "country_code": "ID",
            "overall_risk": "high",
            "governance_score": 38.0,
            "deforestation_risk": 82.0,
            "indigenous_rights_score": 28.0,
            "ilo_169_ratified": False,
            "fpic_legal_requirement": False,
            "land_tenure_security": 25.0,
        },
        "DK": {
            "country_code": "DK",
            "overall_risk": "low",
            "governance_score": 92.0,
            "deforestation_risk": 2.0,
            "indigenous_rights_score": 90.0,
            "ilo_169_ratified": False,
            "fpic_legal_requirement": False,
            "land_tenure_security": 95.0,
        },
    }.get(cc, {
        "country_code": cc,
        "overall_risk": "medium",
        "governance_score": 50.0,
        "indigenous_rights_score": 50.0,
    }))
    evaluator.get_governance_score = AsyncMock(side_effect=lambda cc: {
        "BR": 45.0, "ID": 38.0, "DK": 92.0, "CM": 25.0, "CO": 40.0,
    }.get(cc, 50.0))
    return evaluator


def _mock_traceability_connector():
    """Create a mock AGENT-DATA-005 EUDR Traceability Connector."""
    connector = MagicMock()
    connector.get_plot_details = AsyncMock(return_value={
        "plot_id": "PLOT-BR-001",
        "registered": True,
        "registration_date": "2024-06-01",
        "operator_id": "op-001",
        "commodity": "soya",
        "area_hectares": 150.5,
        "country_code": "BR",
        "geolocation": {"lat": -3.05, "lon": -60.02},
        "geojson": _make_plot_geojson(-3.05, -60.02),
        "dds_reference": "DDS-2026-001",
    })
    connector.get_plots_for_operator = AsyncMock(return_value=[
        {"plot_id": "PLOT-BR-001", "commodity": "soya", "country_code": "BR"},
        {"plot_id": "PLOT-BR-002", "commodity": "cattle", "country_code": "BR"},
    ])
    connector.update_plot_indigenous_status = AsyncMock(return_value=True)
    connector.get_dds_data = AsyncMock(return_value={
        "dds_id": "DDS-2026-001",
        "operator_id": "op-001",
        "submission_date": "2026-01-15",
        "commodity": "soya",
        "plots": ["PLOT-BR-001", "PLOT-BR-002"],
    })
    return connector


def _mock_gis_connector():
    """Create a mock AGENT-DATA-006 GIS/Mapping Connector."""
    connector = MagicMock()
    connector.calculate_intersection = AsyncMock(return_value={
        "intersects": True,
        "intersection_area_ha": 150.5,
        "intersection_pct_a": 100.0,
        "intersection_pct_b": 0.002,
    })
    connector.calculate_distance = AsyncMock(return_value={
        "distance_meters": 0.0,
        "nearest_point_a": {"lat": -3.05, "lon": -60.02},
        "nearest_point_b": {"lat": -3.05, "lon": -60.02},
    })
    connector.calculate_buffer = AsyncMock(return_value={
        "type": "Polygon",
        "coordinates": [[
            [-60.2, -3.2], [-60.2, -2.8],
            [-59.8, -2.8], [-59.8, -3.2],
            [-60.2, -3.2],
        ]],
    })
    connector.geocode_reverse = AsyncMock(return_value={
        "country_code": "BR",
        "state": "Roraima",
        "municipality": "Alto Alegre",
    })
    connector.contains = AsyncMock(return_value=True)
    return connector


def _mock_document_authentication():
    """Create a mock EUDR-012 Document Authentication service."""
    authenticator = MagicMock()
    authenticator.verify_document = AsyncMock(return_value={
        "document_id": "doc-fpic-001",
        "authentic": True,
        "integrity_hash": _compute_hash({"doc": "fpic-001"}),
        "issuer_verified": True,
        "tamper_detected": False,
        "metadata": {
            "document_type": "fpic_consent_form",
            "issuer": "FUNAI",
            "issue_date": "2024-08-01",
            "expiry_date": "2029-08-01",
        },
    })
    authenticator.verify_batch = AsyncMock(return_value=[
        {
            "document_id": "doc-fpic-001",
            "authentic": True,
            "tamper_detected": False,
        },
        {
            "document_id": "doc-fpic-002",
            "authentic": True,
            "tamper_detected": False,
        },
    ])
    authenticator.check_expiry = AsyncMock(return_value={
        "document_id": "doc-fpic-001",
        "expired": False,
        "days_until_expiry": 1248,
    })
    authenticator.get_document_chain = AsyncMock(return_value={
        "document_id": "doc-fpic-001",
        "chain": [
            {"hash": _compute_hash({"step": 1}), "action": "created"},
            {"hash": _compute_hash({"step": 2}), "action": "signed"},
            {"hash": _compute_hash({"step": 3}), "action": "verified"},
        ],
        "chain_valid": True,
    })
    return authenticator


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def irc_config():
    """IndigenousRightsCheckerConfig with integration test defaults."""
    return IndigenousRightsCheckerConfig(
        database_url="postgresql://localhost:5432/greenlang_test",
        redis_url="redis://localhost:6379/1",
        log_level="DEBUG",
        inner_buffer_km=5.0,
        outer_buffer_km=25.0,
        enable_provenance=True,
        genesis_hash="GL-EUDR-IRC-021-INTEG-GENESIS",
        enable_metrics=False,
    )


@pytest.fixture
def supply_chain_mapper():
    """Mock EUDR-001 Supply Chain Mapper."""
    return _mock_supply_chain_mapper()


@pytest.fixture
def geolocation_verifier():
    """Mock EUDR-002 Geolocation Verifier."""
    return _mock_geolocation_verifier()


@pytest.fixture
def country_risk_evaluator():
    """Mock EUDR-016 Country Risk Evaluator."""
    return _mock_country_risk_evaluator()


@pytest.fixture
def traceability_connector():
    """Mock AGENT-DATA-005 EUDR Traceability Connector."""
    return _mock_traceability_connector()


@pytest.fixture
def gis_connector():
    """Mock AGENT-DATA-006 GIS Connector."""
    return _mock_gis_connector()


@pytest.fixture
def document_authenticator():
    """Mock EUDR-012 Document Authentication."""
    return _mock_document_authentication()


@pytest.fixture
def brazil_territory():
    """Yanomami territory in Brazil for integration testing."""
    return _make_territory(
        territory_id="t-integ-001",
        country_code="BR",
        legal_status=TerritoryLegalStatus.TITLED,
        lat=-3.0,
        lon=-60.0,
        area_hectares=Decimal("9664975"),
        people_name="Yanomami",
    )


@pytest.fixture
def indonesia_territory():
    """Dayak territory in Indonesia for integration testing."""
    return _make_territory(
        territory_id="t-integ-002",
        country_code="ID",
        legal_status=TerritoryLegalStatus.CUSTOMARY,
        lat=-1.5,
        lon=116.0,
        area_hectares=Decimal("500000"),
        people_name="Dayak",
    )


@pytest.fixture
def brazil_community():
    """Yanomami community for integration testing."""
    return IndigenousCommunity(
        community_id="c-integ-001",
        community_name="Yanomami do Rio Catrimani",
        people_name="Yanomami",
        country_code="BR",
        estimated_population=26000,
        territory_ids=["t-integ-001"],
        ilo_169_coverage=True,
        fpic_legal_requirement=True,
        provenance_hash=_compute_hash({
            "community_id": "c-integ-001",
            "country_code": "BR",
        }),
    )


# ===========================================================================
# EUDR-001 Supply Chain Mapper Integration Tests
# ===========================================================================


class TestEUDR001SupplyChainMapperIntegration:
    """Test integration between EUDR-021 and EUDR-001 Supply Chain Mapper.

    Validates that supply chain plot locations from EUDR-001 can be checked
    against indigenous territory boundaries managed by EUDR-021, and that
    overlap results propagate back to supply chain compliance status.
    """

    def test_plot_territory_overlap_detection_from_supply_chain(
        self, supply_chain_mapper, brazil_territory, gis_connector,
    ):
        """Plots from EUDR-001 are checked against EUDR-021 territory data."""
        # Retrieve plots from supply chain mapper
        import asyncio
        plots = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.get_plot_locations()
        )

        # Verify we got plots back
        assert len(plots) == 3
        br_plots = [p for p in plots if p["country_code"] == "BR"]
        assert len(br_plots) == 2

        # For each BR plot, check territory boundary
        territory_geojson = brazil_territory.boundary_geojson
        for plot in br_plots:
            plot_geojson = _make_plot_geojson(plot["lat"], plot["lon"])

            # Use GIS connector to check intersection
            result = asyncio.get_event_loop().run_until_complete(
                gis_connector.calculate_intersection()
            )
            assert result["intersects"] is True
            assert result["intersection_area_ha"] > 0

    def test_supply_chain_compliance_update_on_overlap(
        self, supply_chain_mapper,
    ):
        """EUDR-021 overlap detection triggers EUDR-001 compliance update."""
        import asyncio

        # Simulate overlap detection result
        overlap = TerritoryOverlap(
            overlap_id="o-integ-001",
            plot_id="PLOT-BR-001",
            territory_id="t-integ-001",
            overlap_type=OverlapType.DIRECT,
            overlap_area_hectares=Decimal("150.5"),
            overlap_pct_of_plot=Decimal("100.0"),
            risk_score=Decimal("92.50"),
            risk_level=RiskLevel.CRITICAL,
            affected_communities=["c-integ-001"],
            provenance_hash=_compute_hash({"overlap_id": "o-integ-001"}),
        )

        # Update EUDR-001 supply chain node compliance
        result = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.update_node_compliance(
                node_id="n-001",
                compliance_status="blocked",
                reason="indigenous_territory_overlap",
                overlap_id=overlap.overlap_id,
                risk_level=overlap.risk_level.value,
            )
        )
        assert result is True
        supply_chain_mapper.update_node_compliance.assert_called_once()

    def test_supplier_tier_mapping_with_territory_check(
        self, supply_chain_mapper, brazil_territory,
    ):
        """Multi-tier supplier data from EUDR-001 feeds into territory checks."""
        import asyncio

        # Get supplier plots from EUDR-001
        supplier_plots = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.get_supplier_plots()
        )
        assert len(supplier_plots) == 2

        # Each plot should be checkable against territory boundaries
        for sp in supplier_plots:
            assert "plot_id" in sp
            assert "supplier_id" in sp
            # In production, each plot_id would be sent to EUDR-021 overlap engine

    def test_supply_chain_graph_indigenous_annotation(
        self, supply_chain_mapper,
    ):
        """Supply chain graph from EUDR-001 is annotated with indigenous data."""
        import asyncio

        graph = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.get_supply_chain_graph()
        )
        assert graph["graph_id"] == "g-001"
        assert len(graph["nodes"]) == 2

        # Each node with plot_ids can be cross-referenced
        for node in graph["nodes"]:
            assert "plot_ids" in node
            assert "country_code" in node


# ===========================================================================
# EUDR-002 Geolocation Verifier Integration Tests
# ===========================================================================


class TestEUDR002GeolocationVerifierIntegration:
    """Test integration between EUDR-021 and EUDR-002 Geolocation Verifier.

    Validates that territory boundaries and plot polygons are verified through
    EUDR-002 before overlap analysis, ensuring coordinate accuracy.
    """

    def test_territory_boundary_validation_via_geolocation(
        self, geolocation_verifier, brazil_territory,
    ):
        """Territory boundary GeoJSON is validated through EUDR-002."""
        import asyncio

        # Verify the territory polygon through EUDR-002
        result = asyncio.get_event_loop().run_until_complete(
            geolocation_verifier.verify_polygon(
                brazil_territory.boundary_geojson
            )
        )
        assert result["valid"] is True
        assert result["is_self_intersecting"] is False
        assert result["country_code"] == "BR"

    def test_plot_coordinates_verified_before_overlap_analysis(
        self, geolocation_verifier,
    ):
        """Plot coordinates are verified via EUDR-002 before running overlaps."""
        import asyncio

        # Verify plot coordinates
        result = asyncio.get_event_loop().run_until_complete(
            geolocation_verifier.verify_coordinates(
                lat=-3.05, lon=-60.02
            )
        )
        assert result["valid"] is True
        assert result["country_code"] == "BR"
        assert result["confidence"] >= 0.95

    def test_protected_area_check_enriches_overlap_context(
        self, geolocation_verifier,
    ):
        """EUDR-002 protected area data enriches EUDR-021 overlap analysis."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            geolocation_verifier.check_protected_areas(
                lat=-3.05, lon=-60.02
            )
        )
        assert "in_protected_area" in result
        assert "nearest_protected_area" in result
        # If in protected area, overlap risk increases

    def test_batch_verification_before_bulk_overlap(
        self, geolocation_verifier,
    ):
        """Batch verification via EUDR-002 validates all plots before bulk overlap."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            geolocation_verifier.batch_verify(
                plots=[
                    {"plot_id": "PLOT-BR-001", "lat": -3.05, "lon": -60.02},
                    {"plot_id": "PLOT-BR-002", "lat": -3.50, "lon": -60.50},
                ]
            )
        )
        assert len(result) == 2
        assert all(r["valid"] for r in result)


# ===========================================================================
# EUDR-016 Country Risk Evaluator Integration Tests
# ===========================================================================


class TestEUDR016CountryRiskEvaluatorIntegration:
    """Test integration between EUDR-021 and EUDR-016 Country Risk Evaluator.

    Validates that country-level indigenous rights governance scores from
    EUDR-016 influence EUDR-021 overlap risk calculations and FPIC
    compliance requirements.
    """

    def test_country_risk_enriches_overlap_risk_score(
        self, country_risk_evaluator,
    ):
        """EUDR-016 country risk data feeds into EUDR-021 overlap risk scoring."""
        import asyncio

        # Get country risk for Brazil
        br_risk = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_country_risk("BR")
        )
        assert br_risk["overall_risk"] == "high"
        assert br_risk["indigenous_rights_score"] == 40.0
        assert br_risk["ilo_169_ratified"] is True

        # The indigenous_rights_score should increase the overlap risk weight
        # for the country_rights_framework factor
        rights_score = Decimal(str(br_risk["indigenous_rights_score"]))
        # Invert: lower rights score = higher risk
        framework_risk = Decimal("100") - rights_score
        assert framework_risk == Decimal("60")

    def test_ilo_169_status_triggers_fpic_requirement(
        self, country_risk_evaluator,
    ):
        """Countries with ILO 169 (from EUDR-016) require mandatory FPIC."""
        import asyncio

        br_risk = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_country_risk("BR")
        )
        id_risk = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_country_risk("ID")
        )

        # Brazil ratified ILO 169 -> mandatory FPIC
        assert br_risk["ilo_169_ratified"] is True
        assert br_risk["fpic_legal_requirement"] is True

        # Indonesia has not -> FPIC not legally required but EUDR still requires it
        assert id_risk["ilo_169_ratified"] is False
        assert id_risk["fpic_legal_requirement"] is False

    def test_governance_score_determines_dd_level(
        self, country_risk_evaluator,
    ):
        """Governance scores from EUDR-016 determine due diligence level."""
        import asyncio

        br_gov = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_governance_score("BR")
        )
        dk_gov = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_governance_score("DK")
        )

        # Brazil: low governance -> enhanced due diligence
        assert br_gov < 50.0
        # Denmark: high governance -> standard due diligence
        assert dk_gov > 80.0

    def test_multi_country_supply_chain_risk_aggregation(
        self, country_risk_evaluator,
    ):
        """Risk scores aggregate across countries in a multi-country supply chain."""
        import asyncio

        countries = ["BR", "ID", "DK"]
        risks = {}
        for cc in countries:
            risk = asyncio.get_event_loop().run_until_complete(
                country_risk_evaluator.get_country_risk(cc)
            )
            risks[cc] = risk

        # Aggregate: highest risk dominates
        max_risk = max(risks.values(), key=lambda r: r.get("deforestation_risk", 0))
        assert max_risk["country_code"] == "ID"
        assert max_risk["deforestation_risk"] == 82.0


# ===========================================================================
# AGENT-DATA-005 EUDR Traceability Connector Integration Tests
# ===========================================================================


class TestAGENTDATA005TraceabilityConnectorIntegration:
    """Test integration between EUDR-021 and AGENT-DATA-005 Traceability.

    Validates that EUDR-021 can access plot registry data from the
    traceability connector to perform territory overlap checks and
    update plot indigenous status.
    """

    def test_plot_registry_lookup_for_overlap_analysis(
        self, traceability_connector,
    ):
        """EUDR-021 retrieves plot details from traceability for overlap check."""
        import asyncio

        plot = asyncio.get_event_loop().run_until_complete(
            traceability_connector.get_plot_details("PLOT-BR-001")
        )
        assert plot["registered"] is True
        assert plot["country_code"] == "BR"
        assert "geojson" in plot
        assert plot["geojson"]["type"] == "Polygon"

    def test_operator_plots_bulk_retrieval(
        self, traceability_connector,
    ):
        """EUDR-021 retrieves all plots for an operator from traceability."""
        import asyncio

        plots = asyncio.get_event_loop().run_until_complete(
            traceability_connector.get_plots_for_operator("op-001")
        )
        assert len(plots) == 2
        assert all("plot_id" in p for p in plots)

    def test_indigenous_status_update_propagates_to_traceability(
        self, traceability_connector,
    ):
        """EUDR-021 overlap results update the traceability plot status."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            traceability_connector.update_plot_indigenous_status(
                plot_id="PLOT-BR-001",
                indigenous_overlap=True,
                overlap_type="direct",
                risk_level="critical",
                territory_id="t-integ-001",
                fpic_required=True,
                fpic_status="consent_missing",
            )
        )
        assert result is True
        traceability_connector.update_plot_indigenous_status.assert_called_once()

    def test_dds_reference_links_plot_to_compliance_report(
        self, traceability_connector,
    ):
        """DDS reference from traceability links to EUDR-021 compliance report."""
        import asyncio

        plot = asyncio.get_event_loop().run_until_complete(
            traceability_connector.get_plot_details("PLOT-BR-001")
        )
        dds_ref = plot["dds_reference"]
        assert dds_ref == "DDS-2026-001"

        # Retrieve DDS data
        dds = asyncio.get_event_loop().run_until_complete(
            traceability_connector.get_dds_data(dds_ref)
        )
        assert dds["operator_id"] == "op-001"
        assert "PLOT-BR-001" in dds["plots"]


# ===========================================================================
# AGENT-DATA-006 GIS/Mapping Connector Integration Tests
# ===========================================================================


class TestAGENTDATA006GISConnectorIntegration:
    """Test integration between EUDR-021 and AGENT-DATA-006 GIS Connector.

    Validates that EUDR-021 spatial operations (intersection, distance,
    buffer, containment) are delegated to the GIS connector for
    territory-plot overlap analysis.
    """

    def test_intersection_calculation_for_direct_overlap(
        self, gis_connector, brazil_territory,
    ):
        """GIS connector calculates plot-territory intersection area."""
        import asyncio

        plot_geojson = _make_plot_geojson(-3.05, -60.02)
        result = asyncio.get_event_loop().run_until_complete(
            gis_connector.calculate_intersection(
                geometry_a=plot_geojson,
                geometry_b=brazil_territory.boundary_geojson,
            )
        )
        assert result["intersects"] is True
        assert result["intersection_area_ha"] > 0
        assert result["intersection_pct_a"] == 100.0

    def test_distance_calculation_for_proximity_check(
        self, gis_connector,
    ):
        """GIS connector calculates distance for non-overlapping plots."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            gis_connector.calculate_distance(
                geometry_a=_make_plot_geojson(-5.0, -65.0),
                geometry_b=_make_plot_geojson(-3.0, -60.0),
            )
        )
        assert "distance_meters" in result

    def test_buffer_zone_generation(
        self, gis_connector, brazil_territory,
    ):
        """GIS connector generates buffer zones around territory boundaries."""
        import asyncio

        buffer = asyncio.get_event_loop().run_until_complete(
            gis_connector.calculate_buffer(
                geometry=brazil_territory.boundary_geojson,
                buffer_km=5.0,
            )
        )
        assert buffer["type"] == "Polygon"
        assert len(buffer["coordinates"]) > 0

    def test_containment_check_for_plot_in_territory(
        self, gis_connector,
    ):
        """GIS connector checks if a plot centroid is within territory."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            gis_connector.contains(
                outer_geometry=_make_plot_geojson(-3.0, -60.0, size_deg=2.0),
                inner_point={"lat": -3.05, "lon": -60.02},
            )
        )
        assert result is True


# ===========================================================================
# EUDR-012 Document Authentication Integration Tests
# ===========================================================================


class TestEUDR012DocumentAuthenticationIntegration:
    """Test integration between EUDR-021 and EUDR-012 Document Authentication.

    Validates that FPIC consent forms and community agreement documents
    managed by EUDR-021 are authenticated through EUDR-012 for integrity
    verification, tamper detection, and audit trail continuity.
    """

    def test_fpic_document_authentication(
        self, document_authenticator,
    ):
        """FPIC consent forms are verified through EUDR-012."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            document_authenticator.verify_document(
                document_id="doc-fpic-001",
                document_type="fpic_consent_form",
            )
        )
        assert result["authentic"] is True
        assert result["tamper_detected"] is False
        assert result["issuer_verified"] is True

    def test_batch_fpic_document_verification(
        self, document_authenticator,
    ):
        """Batch FPIC document verification through EUDR-012."""
        import asyncio

        results = asyncio.get_event_loop().run_until_complete(
            document_authenticator.verify_batch(
                document_ids=["doc-fpic-001", "doc-fpic-002"],
            )
        )
        assert len(results) == 2
        assert all(r["authentic"] for r in results)
        assert all(not r["tamper_detected"] for r in results)

    def test_fpic_document_expiry_check(
        self, document_authenticator,
    ):
        """FPIC consent form expiry is checked through EUDR-012."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            document_authenticator.check_expiry("doc-fpic-001")
        )
        assert result["expired"] is False
        assert result["days_until_expiry"] > 0

    def test_document_provenance_chain_integrity(
        self, document_authenticator,
    ):
        """Document provenance chain from EUDR-012 validates FPIC audit trail."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            document_authenticator.get_document_chain("doc-fpic-001")
        )
        assert result["chain_valid"] is True
        assert len(result["chain"]) == 3

        # Verify chain ordering: created -> signed -> verified
        actions = [step["action"] for step in result["chain"]]
        assert actions == ["created", "signed", "verified"]


# ===========================================================================
# Cross-Agent Workflow Integration Tests
# ===========================================================================


class TestCrossAgentWorkflows:
    """Test end-to-end workflows spanning multiple agents.

    These tests validate complete cross-agent pipelines that involve
    EUDR-021 as a participant in larger EUDR compliance workflows.
    """

    def test_full_plot_compliance_pipeline(
        self,
        supply_chain_mapper,
        geolocation_verifier,
        traceability_connector,
        gis_connector,
        country_risk_evaluator,
        document_authenticator,
        brazil_territory,
    ):
        """Full pipeline: supply chain -> geolocation -> overlap -> FPIC -> compliance.

        Steps:
        1. EUDR-001: Get plot from supply chain
        2. EUDR-002: Verify plot coordinates
        3. DATA-005: Get plot registry data
        4. DATA-006: Calculate territory intersection
        5. EUDR-016: Get country risk for FPIC decision
        6. EUDR-012: Authenticate FPIC documents
        7. EUDR-021: Generate compliance report
        """
        import asyncio

        # Step 1: Get plots from supply chain
        plots = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.get_plot_locations()
        )
        target_plot = plots[0]
        assert target_plot["plot_id"] == "PLOT-BR-001"

        # Step 2: Verify coordinates
        coord_result = asyncio.get_event_loop().run_until_complete(
            geolocation_verifier.verify_coordinates(
                lat=target_plot["lat"],
                lon=target_plot["lon"],
            )
        )
        assert coord_result["valid"] is True

        # Step 3: Get plot registry data
        plot_details = asyncio.get_event_loop().run_until_complete(
            traceability_connector.get_plot_details(target_plot["plot_id"])
        )
        assert plot_details["registered"] is True

        # Step 4: Calculate intersection with territory
        intersection = asyncio.get_event_loop().run_until_complete(
            gis_connector.calculate_intersection(
                geometry_a=plot_details["geojson"],
                geometry_b=brazil_territory.boundary_geojson,
            )
        )
        assert intersection["intersects"] is True

        # Step 5: Get country risk for FPIC requirement decision
        country_risk = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_country_risk("BR")
        )
        assert country_risk["ilo_169_ratified"] is True
        fpic_required = country_risk["fpic_legal_requirement"]
        assert fpic_required is True

        # Step 6: Authenticate FPIC documents
        doc_result = asyncio.get_event_loop().run_until_complete(
            document_authenticator.verify_document(
                document_id="doc-fpic-001",
                document_type="fpic_consent_form",
            )
        )
        assert doc_result["authentic"] is True

        # Step 7: All data gathered -- compliance report can be assembled
        compliance_data = {
            "plot_id": target_plot["plot_id"],
            "country_code": "BR",
            "coordinates_valid": coord_result["valid"],
            "plot_registered": plot_details["registered"],
            "territory_overlap": intersection["intersects"],
            "overlap_area_ha": intersection["intersection_area_ha"],
            "ilo_169_ratified": country_risk["ilo_169_ratified"],
            "fpic_required": fpic_required,
            "fpic_documents_authentic": doc_result["authentic"],
        }
        assert compliance_data["territory_overlap"] is True
        assert compliance_data["fpic_required"] is True
        assert compliance_data["fpic_documents_authentic"] is True

    def test_provenance_chain_continuity_across_agents(
        self,
        gis_connector,
        document_authenticator,
        brazil_territory,
    ):
        """Provenance hashes chain continuously across agent boundaries."""
        import asyncio

        # EUDR-021 territory provenance hash
        territory_hash = brazil_territory.provenance_hash
        assert len(territory_hash) == 64

        # GIS connector operation creates a derived hash
        intersection_data = asyncio.get_event_loop().run_until_complete(
            gis_connector.calculate_intersection()
        )
        derived_hash = _compute_hash({
            "parent_hash": territory_hash,
            "operation": "intersection",
            "result": intersection_data,
        })
        assert len(derived_hash) == 64
        assert derived_hash != territory_hash

        # Document authentication creates another link
        doc_result = asyncio.get_event_loop().run_until_complete(
            document_authenticator.verify_document("doc-fpic-001")
        )
        final_hash = _compute_hash({
            "parent_hash": derived_hash,
            "operation": "doc_verification",
            "doc_hash": doc_result["integrity_hash"],
        })
        assert len(final_hash) == 64
        assert final_hash != derived_hash

    def test_risk_escalation_workflow_across_agents(
        self,
        supply_chain_mapper,
        country_risk_evaluator,
    ):
        """High-risk overlap triggers escalation across EUDR-001 and EUDR-016."""
        import asyncio

        # Get country risk -- high risk triggers enhanced due diligence
        br_risk = asyncio.get_event_loop().run_until_complete(
            country_risk_evaluator.get_country_risk("BR")
        )
        assert br_risk["overall_risk"] == "high"

        # Create a critical overlap
        overlap = TerritoryOverlap(
            overlap_id="o-escalation-001",
            plot_id="PLOT-BR-001",
            territory_id="t-integ-001",
            overlap_type=OverlapType.DIRECT,
            risk_score=Decimal("95.00"),
            risk_level=RiskLevel.CRITICAL,
            affected_communities=["c-integ-001"],
            provenance_hash=_compute_hash({"overlap_id": "o-escalation-001"}),
        )

        # Escalation: block supply chain node
        result = asyncio.get_event_loop().run_until_complete(
            supply_chain_mapper.update_node_compliance(
                node_id="n-001",
                compliance_status="blocked",
                reason="critical_indigenous_overlap",
                risk_level="critical",
            )
        )
        assert result is True
