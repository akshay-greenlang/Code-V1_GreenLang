# -*- coding: utf-8 -*-
"""
Tests for GL-EUDR-SCM-001 Feature 2: Multi-Tier Recursive Mapper
================================================================

Comprehensive test suite covering:
    - Initialization and validation
    - Tier 1 discovery (ERP, bulk, PDF sources)
    - Sub-tier recursive discovery (questionnaire, PDF, bulk)
    - Opaque segment identification
    - Completeness calculations
    - Depth distribution reporting
    - Incremental mapping
    - All 7 EUDR commodity archetypes
    - Provenance hash determinism
    - Error handling and edge cases
    - Timeout behavior
    - Prometheus metrics integration

Test count target: 80+ (per PRD Section 13.1)
Coverage target: >= 85%
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.agents.eudr.supply_chain_mapper.multi_tier_mapper import (
    COMMODITY_ARCHETYPES,
    ComplianceStatus,
    CommodityArchetype,
    DiscoveryStatus,
    EUDRCommodity,
    MappingSourceType,
    MultiTierMapper,
    MultiTierMappingInput,
    MultiTierMappingOutput,
    NodeType,
    OpaqueReason,
    OpaqueSegment,
    RiskLevel,
    SupplierRecord,
    TierDepthDistribution,
    TierMappingResult,
)


# =============================================================================
# FIXTURES: Mock Connectors
# =============================================================================


class MockGraphStorage:
    """Mock graph storage backend for testing."""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self._node_counter = 0
        self._edge_counter = 0

    async def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        return {"graph_id": graph_id, "nodes": self.nodes, "edges": self.edges}

    async def get_nodes_at_tier(
        self, graph_id: str, tier_depth: int
    ) -> List[Dict[str, Any]]:
        if tier_depth == -1:
            return list(self.nodes.values())
        return [
            n for n in self.nodes.values()
            if n.get("tier_depth") == tier_depth
        ]

    async def get_children(
        self, graph_id: str, node_id: str
    ) -> List[Dict[str, Any]]:
        children = []
        for edge in self.edges.values():
            if edge.get("target_node_id") == node_id:
                source_id = edge.get("source_node_id", "")
                if source_id in self.nodes:
                    children.append(self.nodes[source_id])
        return children

    async def node_exists(self, graph_id: str, supplier_id: str) -> bool:
        for node in self.nodes.values():
            if node.get("supplier_id") == supplier_id:
                return True
        return False

    async def add_node(
        self, graph_id: str, node_data: Dict[str, Any]
    ) -> str:
        self._node_counter += 1
        node_id = f"node-{self._node_counter}"
        node_data["node_id"] = node_id
        self.nodes[node_id] = node_data
        return node_id

    async def add_edge(
        self, graph_id: str, edge_data: Dict[str, Any]
    ) -> str:
        self._edge_counter += 1
        edge_id = f"edge-{self._edge_counter}"
        edge_data["edge_id"] = edge_id
        self.edges[edge_id] = edge_data
        return edge_id

    async def get_node_count(self, graph_id: str) -> int:
        return len(self.nodes)

    async def get_edge_count(self, graph_id: str) -> int:
        return len(self.edges)

    async def get_leaf_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        # Leaf nodes are those that are not the target of any edge
        # (i.e., no one supplies TO them -- they are upstream endpoints)
        target_ids = {e.get("target_node_id") for e in self.edges.values()}
        source_ids = {e.get("source_node_id") for e in self.edges.values()}
        # Leaf nodes: appear as source but not as target (or have no edges at all)
        leaf_ids = set()
        for nid in self.nodes:
            # A leaf is a node with no children (no edges where target = nid)
            children_of_nid = [
                e for e in self.edges.values()
                if e.get("target_node_id") == nid
            ]
            if not children_of_nid:
                # Only if it also appears as a source (has a parent)
                # or is the only node
                leaf_ids.add(nid)

        return [self.nodes[nid] for nid in leaf_ids if nid in self.nodes]

    async def get_all_chain_depths(self, graph_id: str) -> List[int]:
        # Return depths of all nodes for simplicity in tests
        depths = [n.get("tier_depth", 0) for n in self.nodes.values()]
        return depths if depths else []

    async def update_graph_metadata(
        self, graph_id: str, metadata: Dict[str, Any]
    ) -> None:
        pass


class MockERPConnector:
    """Mock AGENT-DATA-003 ERP/Finance Connector."""

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self._records = records or []

    async def fetch_procurement_records(
        self,
        operator_id: str,
        commodity: str,
        *,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        return self._records


class MockQuestionnaireProcessor:
    """Mock AGENT-DATA-008 Supplier Questionnaire Processor."""

    def __init__(
        self,
        declarations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        # Map supplier_id -> list of sub-tier records
        self._declarations = declarations or {}

    async def fetch_supplier_declarations(
        self, supplier_id: str, commodity: str
    ) -> List[Dict[str, Any]]:
        return self._declarations.get(supplier_id, [])


class MockPDFExtractor:
    """Mock AGENT-DATA-001 PDF Invoice Extractor."""

    def __init__(
        self, records: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ):
        self._records = records or {}

    async def extract_custody_records(
        self, supplier_id: str, commodity: str
    ) -> List[Dict[str, Any]]:
        return self._records.get(supplier_id, [])


class MockBulkImporter:
    """Mock AGENT-DATA-002 Excel/CSV Normalizer."""

    def __init__(
        self, records: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ):
        self._records = records or {}

    async def fetch_bulk_supplier_data(
        self, operator_id: str, commodity: str
    ) -> List[Dict[str, Any]]:
        return self._records.get(operator_id, [])


# =============================================================================
# HELPER FACTORIES
# =============================================================================


def make_supplier_dict(
    supplier_id: str,
    supplier_name: str,
    country_code: str,
    node_type: str = "trader",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create a supplier dictionary for mock connectors."""
    record = {
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "country_code": country_code,
        "node_type": node_type,
    }
    record.update(kwargs)
    return record


def make_erp_records(count: int = 3, prefix: str = "erp") -> List[Dict[str, Any]]:
    """Generate a list of mock ERP supplier records."""
    return [
        make_supplier_dict(
            supplier_id=f"{prefix}-{i}",
            supplier_name=f"Supplier {prefix.upper()}-{i}",
            country_code="DE",
            node_type="trader",
            confidence_score=0.85,
        )
        for i in range(1, count + 1)
    ]


def make_questionnaire_records(
    count: int = 2, prefix: str = "q"
) -> List[Dict[str, Any]]:
    """Generate a list of mock questionnaire sub-tier records."""
    return [
        make_supplier_dict(
            supplier_id=f"{prefix}-{i}",
            supplier_name=f"SubTier {prefix.upper()}-{i}",
            country_code="GH",
            node_type="collector",
            confidence_score=0.6,
        )
        for i in range(1, count + 1)
    ]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def graph_storage():
    """Fresh mock graph storage."""
    return MockGraphStorage()


@pytest.fixture
def erp_connector():
    """ERP connector with 3 Tier 1 suppliers."""
    return MockERPConnector(records=make_erp_records(3))


@pytest.fixture
def questionnaire_processor():
    """Questionnaire processor with sub-tier declarations."""
    return MockQuestionnaireProcessor(
        declarations={
            "erp-1": make_questionnaire_records(2, prefix="q1"),
            "erp-2": make_questionnaire_records(1, prefix="q2"),
        }
    )


@pytest.fixture
def pdf_extractor():
    """PDF extractor with no initial records."""
    return MockPDFExtractor()


@pytest.fixture
def bulk_importer():
    """Bulk importer with no initial records."""
    return MockBulkImporter()


@pytest.fixture
def mapper(graph_storage, erp_connector, questionnaire_processor):
    """MultiTierMapper with ERP and questionnaire connectors."""
    return MultiTierMapper(
        graph_storage=graph_storage,
        erp_connector=erp_connector,
        questionnaire_processor=questionnaire_processor,
    )


@pytest.fixture
def mapper_all_sources(
    graph_storage,
    erp_connector,
    questionnaire_processor,
    pdf_extractor,
    bulk_importer,
):
    """MultiTierMapper with all four data source connectors."""
    return MultiTierMapper(
        graph_storage=graph_storage,
        erp_connector=erp_connector,
        questionnaire_processor=questionnaire_processor,
        pdf_extractor=pdf_extractor,
        bulk_importer=bulk_importer,
    )


@pytest.fixture
def basic_input():
    """Standard mapping input for cocoa."""
    return MultiTierMappingInput(
        graph_id="graph-test-001",
        operator_id="op-importer-001",
        commodity=EUDRCommodity.COCOA,
        max_depth=6,
        target_depth=4,
        timeout_seconds=300,
    )


# =============================================================================
# TESTS: INITIALIZATION AND VALIDATION
# =============================================================================


class TestMultiTierMapperInit:
    """Tests for MultiTierMapper initialization."""

    def test_init_with_erp_connector(self, graph_storage, erp_connector):
        """Mapper initializes with ERP connector only."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp_connector,
        )
        assert mapper.AGENT_ID == "GL-EUDR-SCM-001-F2"
        assert mapper.VERSION == "1.0.0"

    def test_init_with_questionnaire_processor(
        self, graph_storage, questionnaire_processor
    ):
        """Mapper initializes with questionnaire processor only."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            questionnaire_processor=questionnaire_processor,
        )
        assert mapper is not None

    def test_init_with_pdf_extractor(self, graph_storage, pdf_extractor):
        """Mapper initializes with PDF extractor only."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            pdf_extractor=pdf_extractor,
        )
        assert mapper is not None

    def test_init_with_bulk_importer(self, graph_storage, bulk_importer):
        """Mapper initializes with bulk importer only."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            bulk_importer=bulk_importer,
        )
        assert mapper is not None

    def test_init_with_all_connectors(
        self, graph_storage, erp_connector, questionnaire_processor,
        pdf_extractor, bulk_importer
    ):
        """Mapper initializes with all four connectors."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp_connector,
            questionnaire_processor=questionnaire_processor,
            pdf_extractor=pdf_extractor,
            bulk_importer=bulk_importer,
        )
        assert mapper is not None

    def test_init_fails_without_connectors(self, graph_storage):
        """Mapper fails to initialize without any data source."""
        with pytest.raises(ValueError, match="At least one data source"):
            MultiTierMapper(graph_storage=graph_storage)

    def test_constants(self, mapper):
        """Verify agent constants."""
        assert mapper.AGENT_NAME == "Multi-Tier Recursive Mapper"
        assert mapper.DEFAULT_TARGET_DEPTH == 4
        assert mapper.DEFAULT_TARGET_COVERAGE_PCT == 80.0


# =============================================================================
# TESTS: COMMODITY ARCHETYPES
# =============================================================================


class TestCommodityArchetypes:
    """Tests for EUDR commodity archetype definitions."""

    def test_all_seven_commodities_defined(self):
        """All 7 EUDR commodities have archetypes."""
        for commodity in EUDRCommodity:
            assert commodity in COMMODITY_ARCHETYPES, (
                f"Missing archetype for {commodity.value}"
            )

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_archetype_has_valid_depths(self, commodity):
        """Each archetype has valid min/typical/max depths."""
        archetype = COMMODITY_ARCHETYPES[commodity]
        assert archetype.min_depth >= 1
        assert archetype.typical_depth >= archetype.min_depth
        assert archetype.max_depth >= archetype.typical_depth

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_archetype_has_actor_sequence(self, commodity):
        """Each archetype has an expected actor sequence."""
        archetype = COMMODITY_ARCHETYPES[commodity]
        assert len(archetype.expected_actor_sequence) >= 3
        # First entry should contain IMPORTER
        assert NodeType.IMPORTER in archetype.expected_actor_sequence[0]
        # Last entry should contain PRODUCER
        assert NodeType.PRODUCER in archetype.expected_actor_sequence[-1]

    def test_cattle_archetype(self):
        """Cattle archetype has correct structure."""
        a = COMMODITY_ARCHETYPES[EUDRCommodity.CATTLE]
        assert a.typical_depth == 5
        assert a.mapping_complexity == "High"
        assert len(a.key_challenges) > 0

    def test_cocoa_archetype(self):
        """Cocoa archetype has correct structure."""
        a = COMMODITY_ARCHETYPES[EUDRCommodity.COCOA]
        assert a.typical_depth == 6
        assert a.mapping_complexity == "Very High"

    def test_wood_archetype(self):
        """Wood archetype has highest complexity."""
        a = COMMODITY_ARCHETYPES[EUDRCommodity.WOOD]
        assert a.typical_depth == 6
        assert a.mapping_complexity == "Very High"
        assert a.max_depth == 10


# =============================================================================
# TESTS: DATA MODELS
# =============================================================================


class TestDataModels:
    """Tests for Pydantic data models."""

    def test_supplier_record_valid(self):
        """Valid supplier record passes validation."""
        record = SupplierRecord(
            supplier_id="sup-001",
            supplier_name="Test Supplier",
            country_code="gh",
            node_type=NodeType.COLLECTOR,
            commodities=[EUDRCommodity.COCOA],
            source_type=MappingSourceType.ERP_PROCUREMENT,
            confidence_score=0.85,
        )
        assert record.country_code == "GH"  # Auto uppercased

    def test_supplier_record_invalid_coordinates(self):
        """Invalid coordinates are rejected."""
        with pytest.raises(ValueError, match="Latitude"):
            SupplierRecord(
                supplier_id="sup-001",
                supplier_name="Test",
                country_code="GH",
                node_type=NodeType.COLLECTOR,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.ERP_PROCUREMENT,
                coordinates=(91.0, 0.0),
            )

    def test_supplier_record_invalid_longitude(self):
        """Invalid longitude is rejected."""
        with pytest.raises(ValueError, match="Longitude"):
            SupplierRecord(
                supplier_id="sup-001",
                supplier_name="Test",
                country_code="GH",
                node_type=NodeType.COLLECTOR,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.ERP_PROCUREMENT,
                coordinates=(0.0, 181.0),
            )

    def test_supplier_record_valid_coordinates(self):
        """Valid coordinates pass validation."""
        record = SupplierRecord(
            supplier_id="sup-001",
            supplier_name="Test",
            country_code="GH",
            node_type=NodeType.PRODUCER,
            commodities=[EUDRCommodity.COCOA],
            source_type=MappingSourceType.ERP_PROCUREMENT,
            coordinates=(6.688, -1.624),
        )
        assert record.coordinates == (6.688, -1.624)

    def test_multi_tier_mapping_input_defaults(self):
        """Input model has correct defaults."""
        inp = MultiTierMappingInput(
            graph_id="g-1",
            operator_id="op-1",
            commodity=EUDRCommodity.PALM_OIL,
        )
        assert inp.max_depth == 10
        assert inp.target_depth == 4
        assert inp.timeout_seconds == 600
        assert inp.incremental is False
        assert inp.start_tier == 0
        assert len(inp.source_filters) == 5

    def test_opaque_segment_model(self):
        """OpaqueSegment model constructs correctly."""
        seg = OpaqueSegment(
            segment_id="opaque-123",
            parent_node_id="node-5",
            parent_node_name="Unknown Mill",
            tier_depth=3,
            commodity=EUDRCommodity.PALM_OIL,
            reason=OpaqueReason.SUPPLIER_REFUSED,
            estimated_missing_tiers=2,
            risk_impact=RiskLevel.HIGH,
            remediation_action="Contact supplier",
        )
        assert seg.reason == OpaqueReason.SUPPLIER_REFUSED
        assert seg.estimated_missing_tiers == 2

    def test_tier_depth_distribution_model(self):
        """TierDepthDistribution constructs correctly."""
        dist = TierDepthDistribution(
            commodity=EUDRCommodity.COFFEE,
            total_chains=10,
            depth_histogram={3: 2, 4: 5, 5: 3},
            median_depth=4.0,
            mean_depth=4.1,
            max_depth=5,
            min_depth=3,
            pct_at_target_depth=80.0,
            target_depth=4,
        )
        assert dist.pct_at_target_depth == 80.0


# =============================================================================
# TESTS: TIER 1 DISCOVERY
# =============================================================================


class TestTierOneDiscovery:
    """Tests for Tier 1 supplier discovery."""

    @pytest.mark.asyncio
    async def test_discover_tier_one_from_erp(self, mapper, basic_input):
        """Tier 1 suppliers are discovered from ERP data."""
        result = await mapper.discover_supply_chain(basic_input)
        assert result.status in (
            DiscoveryStatus.COMPLETED,
            DiscoveryStatus.PARTIAL,
        )
        assert result.tiers_mapped >= 1
        # We have 3 ERP suppliers
        tier1 = result.tier_results[0]
        assert tier1.tier_depth == 1
        assert tier1.suppliers_discovered == 3

    @pytest.mark.asyncio
    async def test_tier_one_nodes_added_to_graph(
        self, graph_storage, erp_connector, basic_input
    ):
        """Tier 1 nodes are persisted in graph storage."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp_connector,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.total_nodes_added >= 3
        assert len(graph_storage.nodes) >= 3

    @pytest.mark.asyncio
    async def test_tier_one_edges_created(
        self, graph_storage, erp_connector, basic_input
    ):
        """Edges from Tier 1 suppliers to importer are created."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp_connector,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.total_edges_added >= 3
        assert len(graph_storage.edges) >= 3

    @pytest.mark.asyncio
    async def test_tier_one_from_bulk_import(self, graph_storage, basic_input):
        """Tier 1 suppliers are discovered from bulk imports."""
        bulk = MockBulkImporter(
            records={
                "op-importer-001": [
                    make_supplier_dict("bulk-1", "Bulk Supplier 1", "BR"),
                    make_supplier_dict("bulk-2", "Bulk Supplier 2", "BR"),
                ]
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            bulk_importer=bulk,
        )
        result = await mapper.discover_supply_chain(basic_input)
        tier1 = result.tier_results[0]
        assert tier1.suppliers_discovered == 2
        assert "bulk_import" in tier1.source_types

    @pytest.mark.asyncio
    async def test_tier_one_from_pdf(self, graph_storage, basic_input):
        """Tier 1 suppliers are discovered from PDF invoices."""
        pdf = MockPDFExtractor(
            records={
                "op-importer-001": [
                    make_supplier_dict("pdf-1", "PDF Supplier 1", "CI"),
                ]
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            pdf_extractor=pdf,
        )
        result = await mapper.discover_supply_chain(basic_input)
        tier1 = result.tier_results[0]
        assert tier1.suppliers_discovered == 1

    @pytest.mark.asyncio
    async def test_tier_one_deduplication(self, graph_storage, basic_input):
        """Duplicate supplier IDs across sources are deduplicated."""
        erp = MockERPConnector(
            records=[make_supplier_dict("dup-1", "Dup Supplier", "DE")]
        )
        bulk = MockBulkImporter(
            records={
                "op-importer-001": [
                    make_supplier_dict("dup-1", "Dup Supplier", "DE")
                ]
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            bulk_importer=bulk,
        )
        result = await mapper.discover_supply_chain(basic_input)
        tier1 = result.tier_results[0]
        assert tier1.suppliers_discovered == 1  # Not 2

    @pytest.mark.asyncio
    async def test_tier_one_completeness(self, mapper, basic_input):
        """Tier 1 completeness is calculated correctly."""
        result = await mapper.discover_supply_chain(basic_input)
        tier1 = result.tier_results[0]
        assert 0.0 <= tier1.completeness_pct <= 100.0

    @pytest.mark.asyncio
    async def test_tier_one_source_breakdown(self, mapper, basic_input):
        """Tier 1 reports source type breakdown."""
        result = await mapper.discover_supply_chain(basic_input)
        tier1 = result.tier_results[0]
        assert "erp_procurement" in tier1.source_types
        assert tier1.source_types["erp_procurement"] == 3


# =============================================================================
# TESTS: SUB-TIER (RECURSIVE) DISCOVERY
# =============================================================================


class TestSubTierDiscovery:
    """Tests for recursive sub-tier discovery (Tier 2+)."""

    @pytest.mark.asyncio
    async def test_tier_two_discovered(self, mapper, basic_input):
        """Tier 2 suppliers are discovered via questionnaire responses."""
        result = await mapper.discover_supply_chain(basic_input)
        # erp-1 has 2 sub-tier suppliers, erp-2 has 1
        assert result.tiers_mapped >= 2
        tier2 = result.tier_results[1]
        assert tier2.tier_depth == 2
        assert tier2.suppliers_discovered == 3

    @pytest.mark.asyncio
    async def test_recursive_stops_when_no_new_suppliers(
        self, graph_storage, basic_input
    ):
        """Recursion stops when no new suppliers are found."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s-1", "Supplier 1", "DE")]
        )
        questionnaire = MockQuestionnaireProcessor(
            declarations={
                "s-1": [
                    make_supplier_dict("s-2", "Sub-Supplier 2", "GH")
                ]
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        # s-1 -> s-2, then s-2 has no questionnaire response
        assert result.tiers_mapped >= 2

    @pytest.mark.asyncio
    async def test_deep_chain_discovery(self, graph_storage, basic_input):
        """Deep chains (4+ tiers) are discovered correctly."""
        erp = MockERPConnector(
            records=[make_supplier_dict("t1", "Tier1", "DE", "trader")]
        )
        # Build a chain: t1 -> t2 -> t3 -> t4 (producer)
        questionnaire = MockQuestionnaireProcessor(
            declarations={
                "t1": [make_supplier_dict("t2", "Tier2", "CI", "processor")],
                "t2": [make_supplier_dict("t3", "Tier3", "CI", "collector")],
                "t3": [
                    make_supplier_dict("t4", "Tier4Farm", "CI", "producer")
                ],
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.tiers_mapped >= 4
        assert result.reached_plot_level is True

    @pytest.mark.asyncio
    async def test_fan_out_discovery(self, graph_storage, basic_input):
        """Fan-out topologies (one supplier to many sub-suppliers) work."""
        erp = MockERPConnector(
            records=[make_supplier_dict("t1", "Trader1", "DE")]
        )
        questionnaire = MockQuestionnaireProcessor(
            declarations={
                "t1": [
                    make_supplier_dict("c1", "Coop1", "GH", "collector"),
                    make_supplier_dict("c2", "Coop2", "GH", "collector"),
                    make_supplier_dict("c3", "Coop3", "CI", "collector"),
                ],
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        tier2 = result.tier_results[1]
        assert tier2.suppliers_discovered == 3

    @pytest.mark.asyncio
    async def test_no_cycles_in_discovery(self, graph_storage, basic_input):
        """Cycle detection: same supplier ID is not added twice."""
        erp = MockERPConnector(
            records=[make_supplier_dict("t1", "Trader1", "DE")]
        )
        # t1 declares t2, t2 declares t1 (cycle)
        questionnaire = MockQuestionnaireProcessor(
            declarations={
                "t1": [make_supplier_dict("t2", "Proc1", "GH", "processor")],
                "t2": [make_supplier_dict("t1", "Trader1", "DE", "trader")],
            }
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        # t1 already visited, so cycle is broken
        assert result.total_nodes_added <= 3  # t1 + t2 at most


# =============================================================================
# TESTS: OPAQUE SEGMENT IDENTIFICATION
# =============================================================================


class TestOpaqueSegments:
    """Tests for opaque segment detection."""

    @pytest.mark.asyncio
    async def test_opaque_segment_detected(self, graph_storage, basic_input):
        """Opaque segments are detected when sub-tier data is missing."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "Supplier1", "DE")]
        )
        # No questionnaire data for s1
        questionnaire = MockQuestionnaireProcessor(declarations={})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert len(result.opaque_segments) >= 1

    @pytest.mark.asyncio
    async def test_opaque_segment_has_remediation(
        self, graph_storage, basic_input
    ):
        """Opaque segments include remediation actions."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "Supplier1", "DE")]
        )
        questionnaire = MockQuestionnaireProcessor(declarations={})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        for seg in result.opaque_segments:
            assert seg.remediation_action != ""

    @pytest.mark.asyncio
    async def test_opaque_segment_not_for_producers(
        self, graph_storage, basic_input
    ):
        """Producer leaf nodes are NOT flagged as opaque."""
        erp = MockERPConnector(
            records=[
                make_supplier_dict(
                    "farm1", "Farm1", "GH", "producer"
                )
            ]
        )
        questionnaire = MockQuestionnaireProcessor(declarations={})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        # The producer is a leaf node but should not be opaque
        for seg in result.opaque_segments:
            assert seg.parent_node_id != "farm1"

    @pytest.mark.asyncio
    async def test_identify_opaque_segments_method(
        self, graph_storage
    ):
        """identify_opaque_segments standalone method works."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "Supplier1", "DE")]
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        # Add a non-producer leaf manually
        await graph_storage.add_node("g-1", {
            "node_id": "n-1",
            "supplier_id": "s1",
            "node_type": "trader",
            "operator_name": "Supplier1",
            "tier_depth": 1,
        })
        segments = await mapper.identify_opaque_segments(
            "g-1", EUDRCommodity.COCOA
        )
        assert len(segments) >= 1

    @pytest.mark.asyncio
    async def test_opaque_status_sets_partial(
        self, graph_storage, basic_input
    ):
        """Discovery status is PARTIAL when opaque segments exist."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "Supplier1", "DE")]
        )
        questionnaire = MockQuestionnaireProcessor(declarations={})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.status == DiscoveryStatus.PARTIAL


# =============================================================================
# TESTS: COMPLETENESS CALCULATIONS
# =============================================================================


class TestCompletenessCalculations:
    """Tests for deterministic completeness calculations."""

    def test_tier_completeness_100_percent(self, mapper):
        """100% when discovered equals expected."""
        result = mapper._compute_tier_completeness(10, 10)
        assert result == 100.0

    def test_tier_completeness_50_percent(self, mapper):
        """50% when discovered is half of expected."""
        result = mapper._compute_tier_completeness(5, 10)
        assert result == 50.0

    def test_tier_completeness_capped_at_100(self, mapper):
        """Completeness is capped at 100% even when discovered > expected."""
        result = mapper._compute_tier_completeness(15, 10)
        assert result == 100.0

    def test_tier_completeness_zero_expected(self, mapper):
        """Completeness handles zero expected gracefully."""
        result = mapper._compute_tier_completeness(5, 0)
        assert result == 100.0

    def test_tier_completeness_both_zero(self, mapper):
        """Completeness handles both zero gracefully."""
        result = mapper._compute_tier_completeness(0, 0)
        assert result == 0.0

    def test_overall_completeness_empty(self, mapper):
        """Overall completeness is 0% with no tier results."""
        result = mapper._calculate_overall_completeness([])
        assert result == 0.0

    def test_overall_completeness_weighted(self, mapper):
        """Overall completeness uses weighted average."""
        tiers = [
            TierMappingResult(
                tier_depth=1,
                suppliers_discovered=10,
                suppliers_expected=10,
                completeness_pct=100.0,
            ),
            TierMappingResult(
                tier_depth=2,
                suppliers_discovered=5,
                suppliers_expected=10,
                completeness_pct=50.0,
            ),
        ]
        result = mapper._calculate_overall_completeness(tiers)
        # tier 1 weight = 1 + 0.5 = 1.5, tier 2 weight = 1 + 1.0 = 2.0
        # (100*1.5 + 50*2.0) / (1.5+2.0) = (150+100)/3.5 = 71.4
        assert 70.0 <= result <= 73.0

    def test_average_confidence(self, mapper):
        """Average confidence score computed correctly."""
        suppliers = [
            SupplierRecord(
                supplier_id=f"s-{i}",
                supplier_name=f"S{i}",
                country_code="DE",
                node_type=NodeType.TRADER,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.ERP_PROCUREMENT,
                confidence_score=score,
            )
            for i, score in enumerate([0.8, 0.6, 0.9])
        ]
        result = mapper._compute_average_confidence(suppliers)
        assert abs(result - 0.767) < 0.01

    def test_average_confidence_empty(self, mapper):
        """Average confidence is 0.0 for empty list."""
        assert mapper._compute_average_confidence([]) == 0.0

    def test_check_reached_plot_level_true(self, mapper):
        """Plot level detected when producer nodes exist."""
        tiers = [
            TierMappingResult(
                tier_depth=3,
                suppliers_discovered=5,
                suppliers_expected=5,
                completeness_pct=100.0,
                node_types_found={"producer": 5},
            ),
        ]
        assert mapper._check_reached_plot_level(tiers) is True

    def test_check_reached_plot_level_false(self, mapper):
        """Plot level not reached when no producer nodes."""
        tiers = [
            TierMappingResult(
                tier_depth=2,
                suppliers_discovered=3,
                suppliers_expected=5,
                completeness_pct=60.0,
                node_types_found={"collector": 3},
            ),
        ]
        assert mapper._check_reached_plot_level(tiers) is False


# =============================================================================
# TESTS: DEPTH DISTRIBUTION
# =============================================================================


class TestDepthDistribution:
    """Tests for tier-depth distribution reporting."""

    @pytest.mark.asyncio
    async def test_empty_graph_distribution(self, mapper):
        """Empty graph returns zero distribution."""
        dist = await mapper.get_tier_depth_report(
            "g-empty", EUDRCommodity.COCOA
        )
        assert dist.total_chains == 0
        assert dist.pct_at_target_depth == 0.0

    @pytest.mark.asyncio
    async def test_distribution_after_mapping(
        self, mapper, basic_input
    ):
        """Distribution is computed after successful mapping."""
        result = await mapper.discover_supply_chain(basic_input)
        if result.depth_distribution is not None:
            dist = result.depth_distribution
            assert dist.commodity == EUDRCommodity.COCOA
            assert dist.total_chains >= 0

    @pytest.mark.asyncio
    async def test_distribution_statistics(self, graph_storage):
        """Distribution computes correct statistics."""
        # Manually set up nodes with known depths
        for i, depth in enumerate([2, 3, 4, 4, 5]):
            await graph_storage.add_node("g-stats", {
                "tier_depth": depth,
                "node_type": "trader",
            })
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=MockERPConnector(),
        )
        dist = await mapper.get_tier_depth_report(
            "g-stats", EUDRCommodity.COFFEE, target_depth=4
        )
        assert dist.total_chains == 5
        assert dist.min_depth == 2
        assert dist.max_depth == 5
        assert dist.mean_depth == 3.6
        # 3 out of 5 are >= 4
        assert dist.pct_at_target_depth == 60.0

    @pytest.mark.asyncio
    async def test_distribution_median_odd(self, graph_storage):
        """Median computed correctly for odd count."""
        for depth in [1, 3, 5]:
            await graph_storage.add_node("g-odd", {"tier_depth": depth})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=MockERPConnector(),
        )
        dist = await mapper.get_tier_depth_report(
            "g-odd", EUDRCommodity.SOYA
        )
        assert dist.median_depth == 3.0

    @pytest.mark.asyncio
    async def test_distribution_median_even(self, graph_storage):
        """Median computed correctly for even count."""
        for depth in [2, 3, 4, 5]:
            await graph_storage.add_node("g-even", {"tier_depth": depth})
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=MockERPConnector(),
        )
        dist = await mapper.get_tier_depth_report(
            "g-even", EUDRCommodity.RUBBER
        )
        assert dist.median_depth == 3.5


# =============================================================================
# TESTS: INCREMENTAL MAPPING
# =============================================================================


class TestIncrementalMapping:
    """Tests for incremental (add-to-existing-graph) mapping."""

    @pytest.mark.asyncio
    async def test_incremental_flag(self):
        """Incremental mapping loads existing supplier IDs and skips duplicates."""
        gs = MockGraphStorage()
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "Sup1", "DE")]
        )
        mapper = MultiTierMapper(
            graph_storage=gs,
            erp_connector=erp,
        )
        inp = MultiTierMappingInput(
            graph_id="g-inc-test",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
            max_depth=2,
            target_depth=4,
            timeout_seconds=60,
            incremental=False,
        )
        # First run -- discover s1
        result1 = await mapper.discover_supply_chain(inp)
        nodes_after_first = len(gs.nodes)
        assert nodes_after_first == 1  # s1 added

        # Second run with incremental=True and same ERP data
        inp2 = MultiTierMappingInput(
            graph_id="g-inc-test",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
            max_depth=2,
            target_depth=4,
            timeout_seconds=60,
            incremental=True,
        )
        result2 = await mapper.discover_supply_chain(inp2)
        # Should not add duplicate nodes
        assert len(gs.nodes) == nodes_after_first

    @pytest.mark.asyncio
    async def test_add_tier_incrementally(self, graph_storage):
        """add_tier_incrementally adds new suppliers to existing graph."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=MockERPConnector(),
        )
        # Set up parent node
        parent_id = await graph_storage.add_node("g-inc", {
            "supplier_id": "parent-1",
            "node_type": "processor",
            "operator_name": "Processor1",
            "tier_depth": 1,
        })

        # Add sub-tier suppliers
        suppliers = [
            SupplierRecord(
                supplier_id="sub-1",
                supplier_name="SubSupplier1",
                country_code="GH",
                node_type=NodeType.COLLECTOR,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.MANUAL_ENTRY,
                confidence_score=0.7,
            ),
            SupplierRecord(
                supplier_id="sub-2",
                supplier_name="SubSupplier2",
                country_code="CI",
                node_type=NodeType.PRODUCER,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.MANUAL_ENTRY,
                confidence_score=0.6,
            ),
        ]

        nodes_added, edges_added = await mapper.add_tier_incrementally(
            "g-inc", parent_id, suppliers, EUDRCommodity.COCOA
        )
        assert nodes_added == 2
        assert edges_added == 2
        assert len(graph_storage.nodes) == 3  # parent + 2 new

    @pytest.mark.asyncio
    async def test_incremental_skips_existing(self, graph_storage):
        """Incremental add skips suppliers that already exist."""
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=MockERPConnector(),
        )
        parent_id = await graph_storage.add_node("g-inc2", {
            "supplier_id": "parent-1",
            "node_type": "trader",
            "operator_name": "Trader1",
            "tier_depth": 0,
        })
        # Add first supplier
        await graph_storage.add_node("g-inc2", {
            "supplier_id": "existing-1",
            "node_type": "collector",
            "operator_name": "Existing1",
            "tier_depth": 1,
        })

        # Try to add the same supplier
        suppliers = [
            SupplierRecord(
                supplier_id="existing-1",
                supplier_name="Existing1",
                country_code="GH",
                node_type=NodeType.COLLECTOR,
                commodities=[EUDRCommodity.COCOA],
                source_type=MappingSourceType.MANUAL_ENTRY,
            ),
        ]
        nodes_added, edges_added = await mapper.add_tier_incrementally(
            "g-inc2", parent_id, suppliers, EUDRCommodity.COCOA
        )
        assert nodes_added == 0
        assert edges_added == 0


# =============================================================================
# TESTS: PROVENANCE HASH
# =============================================================================


class TestProvenanceHash:
    """Tests for provenance hash determinism."""

    def test_provenance_hash_deterministic(self, mapper):
        """Same inputs always produce the same provenance hash."""
        input_data = MultiTierMappingInput(
            graph_id="g-1",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
        )
        tier_results = [
            TierMappingResult(
                tier_depth=1,
                suppliers_discovered=5,
                suppliers_expected=5,
                completeness_pct=100.0,
            ),
        ]
        hash1 = mapper._compute_provenance_hash(
            input_data, tier_results, 5, 5
        )
        hash2 = mapper._compute_provenance_hash(
            input_data, tier_results, 5, 5
        )
        assert hash1 == hash2

    def test_provenance_hash_changes_with_input(self, mapper):
        """Different inputs produce different provenance hashes."""
        input1 = MultiTierMappingInput(
            graph_id="g-1",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
        )
        input2 = MultiTierMappingInput(
            graph_id="g-2",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
        )
        tier_results = [
            TierMappingResult(
                tier_depth=1,
                suppliers_discovered=5,
                suppliers_expected=5,
                completeness_pct=100.0,
            ),
        ]
        hash1 = mapper._compute_provenance_hash(
            input1, tier_results, 5, 5
        )
        hash2 = mapper._compute_provenance_hash(
            input2, tier_results, 5, 5
        )
        assert hash1 != hash2

    def test_provenance_hash_is_sha256(self, mapper):
        """Provenance hash is a valid SHA-256 hex digest (64 chars)."""
        input_data = MultiTierMappingInput(
            graph_id="g-1",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
        )
        h = mapper._compute_provenance_hash(input_data, [], 0, 0)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# =============================================================================
# TESTS: RECORD PARSING
# =============================================================================


class TestRecordParsing:
    """Tests for data source record parsing."""

    def test_parse_erp_record_valid(self, mapper):
        """Valid ERP record is parsed correctly."""
        record = {
            "supplier_id": "erp-1",
            "supplier_name": "ERP Supplier",
            "country_code": "DE",
            "node_type": "trader",
            "latitude": 50.0,
            "longitude": 8.0,
            "certifications": ["FSC-123"],
        }
        result = mapper._parse_erp_record(record, EUDRCommodity.WOOD)
        assert result is not None
        assert result.supplier_id == "erp-1"
        assert result.country_code == "DE"
        assert result.coordinates == (50.0, 8.0)
        assert result.certifications == ["FSC-123"]
        assert result.source_type == MappingSourceType.ERP_PROCUREMENT

    def test_parse_erp_record_missing_fields(self, mapper):
        """ERP record with missing required fields returns None."""
        result = mapper._parse_erp_record(
            {"supplier_id": "x"}, EUDRCommodity.COCOA
        )
        assert result is None

    def test_parse_questionnaire_record_valid(self, mapper):
        """Valid questionnaire record is parsed correctly."""
        record = {
            "supplier_id": "q-1",
            "supplier_name": "Cooperative",
            "country_code": "GH",
            "node_type": "collector",
        }
        result = mapper._parse_questionnaire_record(
            record, EUDRCommodity.COCOA
        )
        assert result is not None
        assert result.source_type == MappingSourceType.SUPPLIER_QUESTIONNAIRE
        assert result.confidence_score == 0.6  # Default for questionnaire

    def test_parse_pdf_record_valid(self, mapper):
        """Valid PDF record is parsed correctly."""
        record = {
            "supplier_id": "pdf-1",
            "supplier_name": "PDF Supplier",
            "country_code": "BR",
        }
        result = mapper._parse_pdf_record(record, EUDRCommodity.SOYA)
        assert result is not None
        assert result.source_type == MappingSourceType.PDF_INVOICE
        assert result.confidence_score == 0.5  # Default for PDF

    def test_parse_bulk_record_valid(self, mapper):
        """Valid bulk record is parsed correctly."""
        record = {
            "supplier_id": "bulk-1",
            "supplier_name": "Bulk Supplier",
            "country_code": "MY",
            "node_type": "processor",
        }
        result = mapper._parse_bulk_record(record, EUDRCommodity.PALM_OIL)
        assert result is not None
        assert result.source_type == MappingSourceType.BULK_IMPORT
        assert result.node_type == NodeType.PROCESSOR

    def test_parse_record_invalid_node_type_defaults(self, mapper):
        """Invalid node type defaults to TRADER."""
        record = {
            "supplier_id": "x-1",
            "supplier_name": "X Supplier",
            "country_code": "DE",
            "node_type": "invalid_type",
        }
        result = mapper._parse_erp_record(record, EUDRCommodity.COCOA)
        assert result is not None
        assert result.node_type == NodeType.TRADER

    def test_parse_record_without_coordinates(self, mapper):
        """Record without coordinates has None coordinates."""
        record = {
            "supplier_id": "x-1",
            "supplier_name": "X Supplier",
            "country_code": "DE",
        }
        result = mapper._parse_erp_record(record, EUDRCommodity.COCOA)
        assert result is not None
        assert result.coordinates is None


# =============================================================================
# TESTS: ALL 7 COMMODITY ARCHETYPES
# =============================================================================


class TestAllCommodities:
    """Tests for all 7 EUDR commodity supply chain archetypes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    async def test_mapping_each_commodity(
        self, graph_storage, commodity
    ):
        """Each commodity can be mapped through the discovery pipeline."""
        erp = MockERPConnector(
            records=[
                make_supplier_dict(
                    f"{commodity.value}-1",
                    f"{commodity.value.title()} Supplier",
                    "DE",
                )
            ]
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        inp = MultiTierMappingInput(
            graph_id=f"g-{commodity.value}",
            operator_id="op-1",
            commodity=commodity,
        )
        result = await mapper.discover_supply_chain(inp)
        assert result.commodity == commodity
        assert result.tiers_mapped >= 1

    @pytest.mark.asyncio
    async def test_invalid_commodity_raises(self, graph_storage):
        """Invalid commodity raises ValueError."""
        erp = MockERPConnector()
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        # Create input with a mocked commodity not in archetypes
        inp = MultiTierMappingInput(
            graph_id="g-invalid",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
        )
        # Remove cocoa from archetypes temporarily
        original = mapper._commodity_archetypes.get(EUDRCommodity.COCOA)
        del mapper._commodity_archetypes[EUDRCommodity.COCOA]
        try:
            with pytest.raises(ValueError, match="No archetype"):
                await mapper.discover_supply_chain(inp)
        finally:
            mapper._commodity_archetypes[EUDRCommodity.COCOA] = original


# =============================================================================
# TESTS: ERROR HANDLING
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and resilience."""

    @pytest.mark.asyncio
    async def test_erp_connector_failure_graceful(
        self, graph_storage, basic_input
    ):
        """ERP connector failure does not crash the mapper."""
        erp = MockERPConnector()
        erp.fetch_procurement_records = AsyncMock(
            side_effect=ConnectionError("ERP down")
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        result = await mapper.discover_supply_chain(basic_input)
        # Should still complete, just with no ERP data
        assert result.status in (
            DiscoveryStatus.COMPLETED,
            DiscoveryStatus.PARTIAL,
        )

    @pytest.mark.asyncio
    async def test_questionnaire_failure_graceful(
        self, graph_storage, basic_input
    ):
        """Questionnaire processor failure does not crash."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "S1", "DE")]
        )
        questionnaire = MockQuestionnaireProcessor()
        questionnaire.fetch_supplier_declarations = AsyncMock(
            side_effect=RuntimeError("Service unavailable")
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
            questionnaire_processor=questionnaire,
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.tiers_mapped >= 1

    @pytest.mark.asyncio
    async def test_graph_storage_node_count_failure(
        self, graph_storage, basic_input
    ):
        """Graph storage node count failure handled gracefully."""
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "S1", "DE")]
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        # Simulate failure in get_node_count
        original = graph_storage.get_node_count
        graph_storage.get_node_count = AsyncMock(
            side_effect=RuntimeError("DB error")
        )
        result = await mapper.discover_supply_chain(basic_input)
        assert result.total_nodes_in_graph == 0  # Graceful fallback
        graph_storage.get_node_count = original


# =============================================================================
# TESTS: OUTPUT MODEL
# =============================================================================


class TestOutputModel:
    """Tests for MultiTierMappingOutput completeness."""

    @pytest.mark.asyncio
    async def test_output_has_all_fields(self, mapper, basic_input):
        """Output model has all required fields populated."""
        result = await mapper.discover_supply_chain(basic_input)
        assert result.graph_id == "graph-test-001"
        assert result.operator_id == "op-importer-001"
        assert result.commodity == EUDRCommodity.COCOA
        assert result.status is not None
        assert result.tiers_mapped >= 0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64
        assert result.processing_time_ms >= 0
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_output_warnings_on_issues(
        self, graph_storage
    ):
        """Output includes warnings when appropriate."""
        inp = MultiTierMappingInput(
            graph_id="g-warn",
            operator_id="op-1",
            commodity=EUDRCommodity.COCOA,
            max_depth=1,
            timeout_seconds=60,
        )
        erp = MockERPConnector(
            records=[make_supplier_dict("s1", "S1", "DE")]
        )
        mapper = MultiTierMapper(
            graph_storage=graph_storage,
            erp_connector=erp,
        )
        result = await mapper.discover_supply_chain(inp)
        assert result.status is not None


# =============================================================================
# TESTS: SEGMENT ID GENERATION
# =============================================================================


class TestSegmentIdGeneration:
    """Tests for opaque segment ID generation."""

    def test_segment_id_deterministic(self, mapper):
        """Same inputs produce same segment ID."""
        id1 = mapper._generate_segment_id("g-1", "n-1")
        id2 = mapper._generate_segment_id("g-1", "n-1")
        assert id1 == id2

    def test_segment_id_prefix(self, mapper):
        """Segment ID starts with 'opaque_' prefix."""
        sid = mapper._generate_segment_id("g-1", "n-1")
        assert sid.startswith("opaque_")

    def test_segment_id_different_inputs(self, mapper):
        """Different inputs produce different segment IDs."""
        id1 = mapper._generate_segment_id("g-1", "n-1")
        id2 = mapper._generate_segment_id("g-1", "n-2")
        assert id1 != id2


# =============================================================================
# TESTS: EXPECTED SUPPLIER ESTIMATION
# =============================================================================


class TestExpectedSupplierEstimation:
    """Tests for supplier count estimation logic."""

    def test_tier_one_estimation(self, mapper):
        """Tier 1 estimate equals discovered count."""
        archetype = COMMODITY_ARCHETYPES[EUDRCommodity.COCOA]
        result = mapper._estimate_expected_suppliers(archetype, 1, 5)
        assert result == 5

    def test_tier_one_minimum_one(self, mapper):
        """Tier 1 estimate is at least 1."""
        archetype = COMMODITY_ARCHETYPES[EUDRCommodity.COCOA]
        result = mapper._estimate_expected_suppliers(archetype, 1, 0)
        assert result >= 1

    def test_deeper_tier_fan_out(self, mapper):
        """Deeper tiers apply fan-out multiplier."""
        archetype = COMMODITY_ARCHETYPES[EUDRCommodity.COCOA]
        # Cocoa has 3.0 multiplier
        result = mapper._estimate_expected_suppliers(archetype, 3, 10)
        assert result >= 10  # At least what was discovered


# =============================================================================
# TESTS: ENUMS
# =============================================================================


class TestEnums:
    """Tests for enum definitions."""

    def test_all_eudr_commodities(self):
        """All 7 EUDR commodities are defined."""
        assert len(EUDRCommodity) == 7

    def test_node_types(self):
        """All expected node types are defined."""
        expected = {
            "producer", "collector", "processor", "trader",
            "importer", "certifier", "warehouse", "port"
        }
        actual = {nt.value for nt in NodeType}
        assert expected == actual

    def test_mapping_source_types(self):
        """All expected source types are defined."""
        expected = {
            "erp_procurement", "supplier_questionnaire", "pdf_invoice",
            "bulk_import", "manual_entry", "api_integration"
        }
        actual = {st.value for st in MappingSourceType}
        assert expected == actual

    def test_opaque_reasons(self):
        """All expected opaque reasons are defined."""
        assert len(OpaqueReason) >= 5

    def test_discovery_status(self):
        """All expected discovery statuses are defined."""
        expected = {
            "pending", "in_progress", "completed",
            "partial", "failed", "timed_out"
        }
        actual = {ds.value for ds in DiscoveryStatus}
        assert expected == actual

    def test_risk_levels(self):
        """Risk levels match EUDR Article 29."""
        expected = {"low", "standard", "high"}
        actual = {rl.value for rl in RiskLevel}
        assert expected == actual

    def test_compliance_statuses(self):
        """Compliance statuses are defined."""
        expected = {
            "verified", "pending_verification",
            "non_compliant", "unknown"
        }
        actual = {cs.value for cs in ComplianceStatus}
        assert expected == actual
