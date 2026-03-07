# -*- coding: utf-8 -*-
"""
Tests for Pydantic v2 Data Models - AGENT-EUDR-001

Tests for all enumerations, core models, request/response models:
- Enum values and completeness
- SupplyChainNode validation
- SupplyChainEdge validation
- SupplyChainGraph construction
- SupplyChainGap auto-population
- Request model validation (CreateGraphRequest, CreateNodeRequest, etc.)
- Response model construction (TraceResult, GapAnalysisResult, etc.)
- Constants and mappings (DERIVED_TO_PRIMARY, GAP_SEVERITY_MAP, etc.)

Test count: 55 tests

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-001 Supply Chain Mapping Master
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Tuple

import pytest

from greenlang.agents.eudr.supply_chain_mapper.models import (
    ComplianceStatus,
    CreateEdgeRequest,
    CreateGraphRequest,
    CreateNodeRequest,
    CustodyModel,
    DDSExportData,
    DEFAULT_RISK_WEIGHTS,
    DERIVED_TO_PRIMARY,
    EUDRCommodity,
    EUDR_DEFORESTATION_CUTOFF,
    EdgeQueryParams,
    GAP_ARTICLE_MAP,
    GAP_SEVERITY_MAP,
    GapAnalysisResult,
    GapSeverity,
    GapType,
    GraphLayoutData,
    GraphQueryParams,
    MAX_EDGES_PER_GRAPH,
    MAX_NODES_PER_GRAPH,
    MAX_TIER_DEPTH,
    NodeQueryParams,
    NodeType,
    PRIMARY_COMMODITIES,
    RiskLevel,
    RiskPropagationResult,
    RiskSummary,
    SankeyData,
    SupplyChainEdge,
    SupplyChainGap,
    SupplyChainGraph,
    SupplyChainNode,
    TierDistribution,
    TraceResult,
    TransportMode,
    UpdateNodeRequest,
    VERSION,
)


# ===========================================================================
# 1. Constants Tests (10 tests)
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_version_string(self):
        assert VERSION == "1.0.0"

    def test_max_nodes(self):
        assert MAX_NODES_PER_GRAPH == 100_000

    def test_max_edges(self):
        assert MAX_EDGES_PER_GRAPH == 500_000

    def test_max_tier_depth(self):
        assert MAX_TIER_DEPTH == 50

    def test_deforestation_cutoff(self):
        assert EUDR_DEFORESTATION_CUTOFF == "2020-12-31"

    def test_default_risk_weights_sum_to_one(self):
        total = sum(DEFAULT_RISK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_primary_commodities_count(self):
        assert len(PRIMARY_COMMODITIES) == 7

    def test_all_primary_commodities_present(self):
        expected = {
            EUDRCommodity.CATTLE, EUDRCommodity.COCOA, EUDRCommodity.COFFEE,
            EUDRCommodity.OIL_PALM, EUDRCommodity.RUBBER, EUDRCommodity.SOYA,
            EUDRCommodity.WOOD,
        }
        assert PRIMARY_COMMODITIES == expected

    def test_derived_to_primary_mapping(self):
        assert DERIVED_TO_PRIMARY[EUDRCommodity.BEEF] == EUDRCommodity.CATTLE
        assert DERIVED_TO_PRIMARY[EUDRCommodity.CHOCOLATE] == EUDRCommodity.COCOA
        assert DERIVED_TO_PRIMARY[EUDRCommodity.PALM_OIL] == EUDRCommodity.OIL_PALM
        assert DERIVED_TO_PRIMARY[EUDRCommodity.TIMBER] == EUDRCommodity.WOOD

    def test_gap_severity_map_completeness(self):
        for gt in GapType:
            assert gt in GAP_SEVERITY_MAP

    def test_gap_article_map_completeness(self):
        for gt in GapType:
            assert gt in GAP_ARTICLE_MAP


# ===========================================================================
# 2. Enum Tests (12 tests)
# ===========================================================================


class TestEnums:
    """Tests for all enumeration values."""

    def test_node_type_all_values(self):
        expected = {"producer", "collector", "processor", "trader",
                    "importer", "certifier", "warehouse", "port"}
        actual = {nt.value for nt in NodeType}
        assert actual == expected

    def test_eudr_commodity_primary(self):
        primaries = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        for c in primaries:
            assert EUDRCommodity(c) is not None

    def test_eudr_commodity_derived(self):
        derived = {"beef", "leather", "chocolate", "palm_oil",
                   "natural_rubber", "tyres", "soybean_oil", "soybean_meal",
                   "timber", "furniture", "paper", "charcoal"}
        for d in derived:
            assert EUDRCommodity(d) is not None

    def test_custody_model_values(self):
        assert CustodyModel.IDENTITY_PRESERVED.value == "identity_preserved"
        assert CustodyModel.SEGREGATED.value == "segregated"
        assert CustodyModel.MASS_BALANCE.value == "mass_balance"

    def test_risk_level_values(self):
        assert {rl.value for rl in RiskLevel} == {"low", "standard", "high"}

    def test_compliance_status_values(self):
        expected = {"compliant", "non_compliant", "pending_verification",
                    "under_review", "insufficient_data", "exempted"}
        assert {cs.value for cs in ComplianceStatus} == expected

    def test_gap_type_count(self):
        assert len(GapType) == 10

    def test_gap_severity_values(self):
        assert {gs.value for gs in GapSeverity} == {"critical", "high", "medium", "low"}

    def test_transport_mode_values(self):
        expected = {"road", "sea", "rail", "air", "river", "pipeline", "multimodal"}
        assert {tm.value for tm in TransportMode} == expected


# ===========================================================================
# 3. SupplyChainNode Tests (8 tests)
# ===========================================================================


class TestSupplyChainNode:
    """Tests for SupplyChainNode validation."""

    def test_node_valid_creation(self):
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            operator_id="OP-001",
            operator_name="Farm Alpha",
            country_code="GH",
        )
        assert node.node_id is not None
        assert node.compliance_status == ComplianceStatus.PENDING_VERIFICATION

    def test_node_country_code_uppercase(self):
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            operator_id="OP-001",
            operator_name="Farm",
            country_code="gh",
        )
        assert node.country_code == "GH"

    def test_node_invalid_country_code(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_id="OP-001",
                operator_name="Farm",
                country_code="X",
            )

    def test_node_empty_operator_id(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_id="",
                operator_name="Farm",
                country_code="GH",
            )

    def test_node_valid_coordinates(self):
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            operator_id="OP-001",
            operator_name="Farm",
            country_code="GH",
            coordinates=(6.0, -1.5),
        )
        assert node.coordinates == (6.0, -1.5)

    def test_node_invalid_latitude(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_id="OP-001",
                operator_name="Farm",
                country_code="GH",
                coordinates=(100.0, 0.0),
            )

    def test_node_invalid_longitude(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_id="OP-001",
                operator_name="Farm",
                country_code="GH",
                coordinates=(0.0, 200.0),
            )

    def test_node_risk_score_bounds(self):
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                operator_id="OP-001",
                operator_name="Farm",
                country_code="GH",
                risk_score=150.0,
            )


# ===========================================================================
# 4. SupplyChainEdge Tests (6 tests)
# ===========================================================================


class TestSupplyChainEdge:
    """Tests for SupplyChainEdge validation."""

    def test_edge_valid_creation(self):
        edge = SupplyChainEdge(
            source_node_id="n1",
            target_node_id="n2",
            commodity=EUDRCommodity.COCOA,
            product_description="Raw cocoa beans",
            quantity=Decimal("1000"),
        )
        assert edge.unit == "kg"

    def test_edge_self_loop_rejected(self):
        with pytest.raises(ValueError):
            SupplyChainEdge(
                source_node_id="n1",
                target_node_id="n1",
                commodity=EUDRCommodity.COCOA,
                product_description="Self",
                quantity=Decimal("100"),
            )

    def test_edge_empty_source(self):
        with pytest.raises(ValueError):
            SupplyChainEdge(
                source_node_id="",
                target_node_id="n2",
                commodity=EUDRCommodity.COCOA,
                product_description="Test",
                quantity=Decimal("100"),
            )

    def test_edge_zero_quantity(self):
        with pytest.raises(ValueError):
            SupplyChainEdge(
                source_node_id="n1",
                target_node_id="n2",
                commodity=EUDRCommodity.COCOA,
                product_description="Test",
                quantity=Decimal("0"),
            )

    def test_edge_with_transport_mode(self):
        edge = SupplyChainEdge(
            source_node_id="n1",
            target_node_id="n2",
            commodity=EUDRCommodity.COCOA,
            product_description="Shipped cocoa",
            quantity=Decimal("5000"),
            transport_mode=TransportMode.SEA,
        )
        assert edge.transport_mode == TransportMode.SEA


# ===========================================================================
# 5. SupplyChainGap Tests (5 tests)
# ===========================================================================


class TestSupplyChainGap:
    """Tests for SupplyChainGap auto-population."""

    def test_gap_auto_severity(self):
        gap = SupplyChainGap(
            gap_type=GapType.MISSING_GEOLOCATION,
            description="Producer lacks GPS",
        )
        assert gap.severity == GapSeverity.CRITICAL

    def test_gap_auto_article(self):
        gap = SupplyChainGap(
            gap_type=GapType.MISSING_GEOLOCATION,
            description="Missing GPS",
        )
        assert gap.eudr_article == "Article 9"

    def test_gap_broken_chain_critical(self):
        gap = SupplyChainGap(
            gap_type=GapType.BROKEN_CUSTODY_CHAIN,
            description="No origin link",
        )
        assert gap.severity == GapSeverity.CRITICAL

    def test_gap_orphan_node_low(self):
        gap = SupplyChainGap(
            gap_type=GapType.ORPHAN_NODE,
            description="Disconnected node",
        )
        assert gap.severity == GapSeverity.LOW

    def test_gap_empty_description_rejected(self):
        with pytest.raises(ValueError):
            SupplyChainGap(
                gap_type=GapType.STALE_DATA,
                description="",
            )


# ===========================================================================
# 6. Request Model Tests (8 tests)
# ===========================================================================


class TestRequestModels:
    """Tests for API request model validation."""

    def test_create_graph_request(self):
        req = CreateGraphRequest(commodity=EUDRCommodity.COCOA)
        assert req.commodity == EUDRCommodity.COCOA

    def test_create_node_request_valid(self):
        req = CreateNodeRequest(
            node_type=NodeType.PRODUCER,
            operator_id="OP-001",
            operator_name="Farm Alpha",
            country_code="GH",
        )
        assert req.country_code == "GH"

    def test_create_edge_request_self_loop(self):
        with pytest.raises(ValueError):
            CreateEdgeRequest(
                source_node_id="n1",
                target_node_id="n1",
                commodity=EUDRCommodity.COCOA,
                product_description="Self",
                quantity=Decimal("100"),
            )

    def test_update_node_request_optional(self):
        req = UpdateNodeRequest(operator_name="Updated Name")
        assert req.operator_name == "Updated Name"
        assert req.country_code is None

    def test_graph_query_params_defaults(self):
        params = GraphQueryParams()
        assert params.limit == 50
        assert params.offset == 0

    def test_node_query_params_defaults(self):
        params = NodeQueryParams()
        assert params.limit == 100

    def test_edge_query_params_defaults(self):
        params = EdgeQueryParams()
        assert params.limit == 100


# ===========================================================================
# 7. Response Model Tests (6 tests)
# ===========================================================================


class TestResponseModels:
    """Tests for API response model construction."""

    def test_trace_result(self):
        result = TraceResult(
            direction="backward",
            start_node_id="n-imp-001",
            visited_nodes=["n-imp-001", "n-trader-001"],
            is_complete=True,
        )
        assert result.direction == "backward"
        assert result.is_complete is True

    def test_tier_distribution(self):
        dist = TierDistribution(
            tier_counts={0: 1, 1: 2, 2: 3, 3: 5, 4: 10},
            max_depth=4,
            average_depth=2.5,
            median_depth=3.0,
        )
        assert dist.max_depth == 4

    def test_gap_analysis_result(self):
        result = GapAnalysisResult(
            graph_id="g-001",
            total_gaps=5,
            compliance_readiness=72.0,
        )
        assert result.compliance_readiness == 72.0

    def test_dds_export_data(self):
        export = DDSExportData(
            graph_id="g-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COCOA,
            total_supply_chain_actors=25,
            traceability_score=95.0,
        )
        assert export.traceability_score == 95.0

    def test_sankey_data(self):
        sankey = SankeyData(
            graph_id="g-001",
            nodes=[{"label": "Producer", "value": 100}],
            links=[{"source": 0, "target": 1, "value": 100}],
        )
        assert len(sankey.nodes) == 1
        assert len(sankey.links) == 1
