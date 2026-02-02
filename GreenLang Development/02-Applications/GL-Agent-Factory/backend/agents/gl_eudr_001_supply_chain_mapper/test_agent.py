"""
GL-EUDR-001: Supply Chain Mapper Agent Tests

Comprehensive test suite covering:
- Supply chain mapping and graph construction
- Entity resolution with scoring and merging
- Coverage calculation and gate checks
- Snapshot creation and diffing
- Natural language queries
- Edge cases and error handling

Run with: pytest test_agent.py -v
"""

import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from .agent import (
    # Agent
    SupplyChainMapperAgent,
    # Input/Output
    SupplyChainMapperInput,
    SupplyChainMapperOutput,
    # Enums
    CommodityType,
    NodeType,
    EdgeType,
    DataSource,
    VerificationStatus,
    DisclosureStatus,
    GapType,
    GapSeverity,
    ResolutionStatus,
    ResolutionDecision,
    SnapshotTrigger,
    RiskLevel,
    OperationType,
    # Models
    Address,
    Certification,
    SupplyChainNode,
    SupplyChainEdge,
    OriginPlot,
    PlotGeometry,
    SupplyChainGraph,
    CoverageReport,
    CoverageGateResult,
    SupplyChainSnapshot,
    EntityResolutionCandidate,
    MatchResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh agent instance."""
    return SupplyChainMapperAgent()


@pytest.fixture
def sample_importer():
    """Create a sample importer node."""
    return SupplyChainNode(
        node_id=uuid.uuid4(),
        node_type=NodeType.IMPORTER,
        name="EU Coffee Imports GmbH",
        country_code="DE",
        commodities=[CommodityType.COFFEE],
        verification_status=VerificationStatus.VERIFIED,
        operator_size="LARGE",
        eori_number="DE123456789012345"
    )


@pytest.fixture
def sample_trader():
    """Create a sample trader node."""
    return SupplyChainNode(
        node_id=uuid.uuid4(),
        node_type=NodeType.TRADER,
        name="Global Coffee Trading Ltd",
        country_code="CH",
        commodities=[CommodityType.COFFEE],
        verification_status=VerificationStatus.VERIFIED,
        tax_id="CHE-123.456.789"
    )


@pytest.fixture
def sample_processor():
    """Create a sample processor node."""
    return SupplyChainNode(
        node_id=uuid.uuid4(),
        node_type=NodeType.PROCESSOR,
        name="Colombia Coffee Processing SA",
        country_code="CO",
        commodities=[CommodityType.COFFEE],
        verification_status=VerificationStatus.VERIFIED
    )


@pytest.fixture
def sample_producer():
    """Create a sample producer node."""
    return SupplyChainNode(
        node_id=uuid.uuid4(),
        node_type=NodeType.PRODUCER,
        name="Finca La Esperanza",
        country_code="CO",
        commodities=[CommodityType.COFFEE],
        verification_status=VerificationStatus.VERIFIED,
        certifications=[
            Certification(
                name="Rainforest Alliance",
                issuer="RA",
                valid_from=date(2023, 1, 1),
                valid_to=date(2025, 12, 31),
                commodities=[CommodityType.COFFEE]
            )
        ]
    )


@pytest.fixture
def sample_plot(sample_producer):
    """Create a sample origin plot."""
    return OriginPlot(
        plot_id=uuid.uuid4(),
        producer_node_id=sample_producer.node_id,
        plot_identifier="FINCA-001",
        geometry=PlotGeometry(
            type="Point",
            coordinates=[-75.5, 4.5]  # Colombia
        ),
        area_hectares=Decimal("25.5"),
        commodity=CommodityType.COFFEE,
        country_code="CO",
        validation_status="VALIDATED"
    )


@pytest.fixture
def populated_agent(agent, sample_importer, sample_trader, sample_processor, sample_producer, sample_plot):
    """Create an agent with a complete supply chain."""
    # Add nodes
    agent.add_node(sample_importer)
    agent.add_node(sample_trader)
    agent.add_node(sample_processor)
    agent.add_node(sample_producer)

    # Add plot
    agent.add_plot(sample_plot)

    # Add edges (producer -> processor -> trader -> importer)
    agent.add_edge(SupplyChainEdge(
        source_node_id=sample_producer.node_id,
        target_node_id=sample_processor.node_id,
        edge_type=EdgeType.SUPPLIES,
        commodity=CommodityType.COFFEE,
        quantity=Decimal("1000"),
        quantity_unit="kg",
        verified=True,
        confidence_score=Decimal("0.95")
    ))

    agent.add_edge(SupplyChainEdge(
        source_node_id=sample_processor.node_id,
        target_node_id=sample_trader.node_id,
        edge_type=EdgeType.PROCESSES,
        commodity=CommodityType.COFFEE,
        quantity=Decimal("800"),
        quantity_unit="kg",
        verified=True
    ))

    agent.add_edge(SupplyChainEdge(
        source_node_id=sample_trader.node_id,
        target_node_id=sample_importer.node_id,
        edge_type=EdgeType.TRADES,
        commodity=CommodityType.COFFEE,
        quantity=Decimal("800"),
        quantity_unit="kg",
        verified=True
    ))

    return agent, sample_importer


# =============================================================================
# BASIC AGENT TESTS
# =============================================================================

class TestAgentBasics:
    """Test basic agent functionality."""

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent._nodes == {}
        assert agent._edges == {}
        assert agent._plots == {}

    def test_add_node(self, agent, sample_importer):
        """Test adding a node."""
        result = agent.add_node(sample_importer)
        assert result.node_id == sample_importer.node_id
        assert sample_importer.node_id in agent._nodes

    def test_add_edge(self, agent, sample_importer, sample_trader):
        """Test adding an edge."""
        agent.add_node(sample_importer)
        agent.add_node(sample_trader)

        edge = SupplyChainEdge(
            source_node_id=sample_trader.node_id,
            target_node_id=sample_importer.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        )
        result = agent.add_edge(edge)
        assert result.edge_id == edge.edge_id
        assert edge.edge_id in agent._edges

    def test_add_plot(self, agent, sample_producer, sample_plot):
        """Test adding a plot."""
        agent.add_node(sample_producer)
        result = agent.add_plot(sample_plot)
        assert result.plot_id == sample_plot.plot_id
        assert sample_plot.plot_id in agent._plots


# =============================================================================
# SUPPLY CHAIN MAPPING TESTS
# =============================================================================

class TestSupplyChainMapping:
    """Test supply chain mapping operations."""

    def test_map_supply_chain(self, populated_agent):
        """Test mapping a complete supply chain."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN,
            depth=10
        )

        result = agent.run(input_data)

        assert result.success is True
        assert result.operation == OperationType.MAP_SUPPLY_CHAIN
        assert result.node_count == 4
        assert result.edge_count == 3
        assert result.graph is not None
        assert result.provenance_hash is not None

    def test_tier_calculation(self, populated_agent):
        """Test tier calculation from importer."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN
        )

        result = agent.run(input_data)
        graph = result.graph

        # Check tiers
        node_tiers = {n.name: n.tier for n in graph.nodes}
        assert node_tiers["EU Coffee Imports GmbH"] == 0  # Importer
        assert node_tiers["Global Coffee Trading Ltd"] == 1  # Trader
        assert node_tiers["Colombia Coffee Processing SA"] == 2  # Processor
        assert node_tiers["Finca La Esperanza"] == 3  # Producer

    def test_cycle_detection(self, agent):
        """Test cycle detection in supply chain."""
        # Create nodes that form a cycle
        node_a = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Trader A",
            country_code="CH",
            commodities=[CommodityType.COFFEE]
        )
        node_b = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Trader B",
            country_code="NL",
            commodities=[CommodityType.COFFEE]
        )
        node_c = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.IMPORTER,
            name="Importer C",
            country_code="DE",
            commodities=[CommodityType.COFFEE]
        )

        agent.add_node(node_a)
        agent.add_node(node_b)
        agent.add_node(node_c)

        # A -> B -> C and B -> A (cycle)
        agent.add_edge(SupplyChainEdge(
            source_node_id=node_a.node_id,
            target_node_id=node_b.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        ))
        agent.add_edge(SupplyChainEdge(
            source_node_id=node_b.node_id,
            target_node_id=node_c.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        ))
        agent.add_edge(SupplyChainEdge(
            source_node_id=node_b.node_id,
            target_node_id=node_a.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        ))

        input_data = SupplyChainMapperInput(
            importer_id=node_c.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN
        )

        result = agent.run(input_data)
        assert result.graph.has_cycles is True

    def test_include_inferred_edges(self, agent, sample_importer, sample_trader):
        """Test filtering inferred edges."""
        agent.add_node(sample_importer)
        agent.add_node(sample_trader)

        # Add declared edge
        declared_edge = SupplyChainEdge(
            source_node_id=sample_trader.node_id,
            target_node_id=sample_importer.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE,
            data_source=DataSource.SUPPLIER_DECLARED
        )
        agent.add_edge(declared_edge)

        # Add inferred edge (should be excluded when include_inferred=False)
        inferred_edge = SupplyChainEdge(
            source_node_id=sample_trader.node_id,
            target_node_id=sample_importer.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE,
            data_source=DataSource.INFERRED_CUSTOMS
        )
        agent.add_edge(inferred_edge)

        # With inferred
        input_with = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN,
            include_inferred=True
        )
        result_with = agent.run(input_with)
        assert result_with.edge_count == 2

        # Without inferred
        input_without = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN,
            include_inferred=False
        )
        result_without = agent.run(input_without)
        assert result_without.edge_count == 1


# =============================================================================
# COVERAGE TESTS
# =============================================================================

class TestCoverageCalculation:
    """Test coverage calculation and gaps."""

    def test_calculate_coverage(self, populated_agent):
        """Test coverage calculation."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CALCULATE_COVERAGE
        )

        result = agent.run(input_data)

        assert result.success is True
        assert result.coverage_report is not None
        assert result.coverage_report.overall_coverage >= 0
        assert result.coverage_report.mapping_completeness >= 0
        assert result.coverage_report.plot_coverage >= 0

    def test_identify_gaps_unverified(self, agent, sample_importer):
        """Test gap identification for unverified suppliers."""
        # Add unverified supplier
        unverified = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Unverified Trader",
            country_code="CH",
            commodities=[CommodityType.COFFEE],
            verification_status=VerificationStatus.UNVERIFIED
        )

        agent.add_node(sample_importer)
        agent.add_node(unverified)
        agent.add_edge(SupplyChainEdge(
            source_node_id=unverified.node_id,
            target_node_id=sample_importer.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        ))

        input_data = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CALCULATE_COVERAGE
        )

        result = agent.run(input_data)
        gaps = result.coverage_report.gaps

        unverified_gaps = [g for g in gaps if g.gap_type == GapType.UNVERIFIED_SUPPLIER]
        assert len(unverified_gaps) > 0

    def test_identify_gaps_missing_plot(self, agent, sample_importer):
        """Test gap identification for missing plot data."""
        # Add producer without plot
        producer_no_plot = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Producer Without Plot",
            country_code="CO",
            commodities=[CommodityType.COFFEE],
            verification_status=VerificationStatus.VERIFIED
        )

        agent.add_node(sample_importer)
        agent.add_node(producer_no_plot)
        agent.add_edge(SupplyChainEdge(
            source_node_id=producer_no_plot.node_id,
            target_node_id=sample_importer.node_id,
            edge_type=EdgeType.SUPPLIES,
            commodity=CommodityType.COFFEE
        ))

        input_data = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CALCULATE_COVERAGE
        )

        result = agent.run(input_data)
        gaps = result.coverage_report.gaps

        missing_plot_gaps = [g for g in gaps if g.gap_type == GapType.MISSING_PLOT_DATA]
        assert len(missing_plot_gaps) == 1
        assert missing_plot_gaps[0].severity == GapSeverity.CRITICAL

    def test_gap_summary(self, populated_agent):
        """Test gap summary counts."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CALCULATE_COVERAGE
        )

        result = agent.run(input_data)
        summary = result.coverage_report.gap_summary

        assert 'critical' in summary
        assert 'high' in summary
        assert 'medium' in summary
        assert 'low' in summary


# =============================================================================
# COVERAGE GATES TESTS
# =============================================================================

class TestCoverageGates:
    """Test coverage gate checks."""

    def test_check_gates_pass(self, populated_agent):
        """Test gates pass with good coverage."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CHECK_GATES,
            risk_level=RiskLevel.STANDARD
        )

        result = agent.run(input_data)

        assert result.success is True
        assert result.gate_result is not None
        assert result.gate_result.risk_level_applied == RiskLevel.STANDARD

    def test_check_gates_high_risk(self, populated_agent):
        """Test gates with high risk level (stricter thresholds)."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CHECK_GATES,
            risk_level=RiskLevel.HIGH
        )

        result = agent.run(input_data)

        assert result.gate_result.required_mapping == 98.0  # HIGH threshold
        assert result.gate_result.required_plot == 95.0


# =============================================================================
# ENTITY RESOLUTION TESTS
# =============================================================================

class TestEntityResolution:
    """Test entity resolution functionality."""

    def test_find_duplicate_by_tax_id(self, agent):
        """Test finding duplicates by tax ID match."""
        # Two nodes with same tax ID
        node1 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="ABC Trading Company",
            country_code="DE",
            commodities=[CommodityType.COFFEE],
            tax_id="DE123456789"
        )
        node2 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="ABC Trading GmbH",  # Slightly different name
            country_code="DE",
            commodities=[CommodityType.COFFEE],
            tax_id="DE-123-456-789"  # Same but different format
        )

        agent.add_node(node1)
        agent.add_node(node2)

        input_data = SupplyChainMapperInput(
            importer_id=uuid.uuid4(),  # Dummy
            commodity=CommodityType.COFFEE,
            operation=OperationType.RUN_ENTITY_RESOLUTION
        )

        result = agent.run(input_data)

        # Should find duplicate
        assert result.success is True
        # Either auto-merged or in review queue
        assert result.auto_merged_count > 0 or result.review_queue_count > 0

    def test_scoring_with_strong_features(self, agent):
        """Test scoring with strong features (DUNS, EORI)."""
        node1 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Test Company",
            country_code="NL",
            commodities=[CommodityType.COCOA],
            duns_number="123456789"
        )
        node2 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Test Company NL",
            country_code="NL",
            commodities=[CommodityType.COCOA],
            duns_number="123456789"  # Same DUNS
        )

        agent.add_node(node1)
        agent.add_node(node2)

        match_result = agent._score_entity_pair(node1.node_id, node2.node_id)

        assert match_result.strong_feature_matched is True
        assert match_result.overall_score >= 0.85

    def test_no_merge_different_entities(self, agent):
        """Test entities that should not merge."""
        node1 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Farm Alpha",
            country_code="BR",
            commodities=[CommodityType.COFFEE]
        )
        node2 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Plantation Beta",
            country_code="CO",
            commodities=[CommodityType.COFFEE]
        )

        agent.add_node(node1)
        agent.add_node(node2)

        match_result = agent._score_entity_pair(node1.node_id, node2.node_id)

        assert match_result.decision == ResolutionDecision.NO_MERGE


# =============================================================================
# SNAPSHOT TESTS
# =============================================================================

class TestSnapshots:
    """Test snapshot creation and diffing."""

    def test_create_snapshot(self, populated_agent):
        """Test creating a snapshot."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CREATE_SNAPSHOT,
            trigger_type=SnapshotTrigger.MANUAL
        )

        result = agent.run(input_data)

        assert result.success is True
        assert result.snapshot is not None
        assert result.snapshot.node_count == 4
        assert result.snapshot.edge_count == 3
        assert result.snapshot.trigger_type == SnapshotTrigger.MANUAL
        assert result.provenance_hash is not None

    def test_snapshot_immutability(self, populated_agent):
        """Test that snapshots capture immutable state."""
        agent, importer = populated_agent

        # Create first snapshot
        input1 = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CREATE_SNAPSHOT
        )
        result1 = agent.run(input1)
        hash1 = result1.provenance_hash

        # Add another node
        new_node = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="New Trader",
            country_code="GB",
            commodities=[CommodityType.COFFEE]
        )
        agent.add_node(new_node)

        # Create second snapshot
        result2 = agent.run(input1)
        hash2 = result2.provenance_hash

        # Hashes should be different
        assert hash1 != hash2

    def test_snapshot_diff(self, populated_agent):
        """Test diffing two snapshots."""
        agent, importer = populated_agent

        # Create first snapshot
        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CREATE_SNAPSHOT
        )
        result1 = agent.run(input_data)
        snapshot1_id = result1.snapshot.snapshot_id

        # Add a node
        new_node = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Additional Trader",
            country_code="BE",
            commodities=[CommodityType.COFFEE]
        )
        agent.add_node(new_node)
        agent.add_edge(SupplyChainEdge(
            source_node_id=new_node.node_id,
            target_node_id=importer.node_id,
            edge_type=EdgeType.TRADES,
            commodity=CommodityType.COFFEE
        ))

        # Create second snapshot
        result2 = agent.run(input_data)
        snapshot2_id = result2.snapshot.snapshot_id

        # Get diff
        diff = agent._diff_snapshots(snapshot1_id, snapshot2_id)

        assert len(diff.nodes_added) > 0
        assert len(diff.edges_added) > 0


# =============================================================================
# NATURAL LANGUAGE QUERY TESTS
# =============================================================================

class TestNaturalLanguageQueries:
    """Test NL query parsing and execution."""

    def test_nl_query_by_country(self, populated_agent):
        """Test NL query filtering by country."""
        agent, importer = populated_agent

        input_data = SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.NATURAL_LANGUAGE_QUERY,
            query="Show me all suppliers in Colombia"
        )

        result = agent.run(input_data)

        assert result.success is True
        assert result.nl_generated_filter.get('country_code') == 'CO'

    def test_nl_query_by_verification(self, agent, sample_importer):
        """Test NL query filtering by verification status."""
        agent.add_node(sample_importer)

        unverified = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Unverified Trader",
            country_code="CH",
            commodities=[CommodityType.COFFEE],
            verification_status=VerificationStatus.UNVERIFIED
        )
        agent.add_node(unverified)

        input_data = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.NATURAL_LANGUAGE_QUERY,
            query="Find all unverified suppliers"
        )

        result = agent.run(input_data)

        assert result.nl_generated_filter.get('verification_status') == 'UNVERIFIED'

    def test_nl_query_expired_certs(self, agent, sample_importer):
        """Test NL query for expired certifications."""
        # Add supplier with expired cert
        expired_cert_supplier = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Farm with Expired Cert",
            country_code="CO",
            commodities=[CommodityType.COFFEE],
            certifications=[
                Certification(
                    name="Rainforest Alliance",
                    issuer="RA",
                    valid_from=date(2020, 1, 1),
                    valid_to=date(2022, 12, 31),  # Expired
                    commodities=[CommodityType.COFFEE]
                )
            ]
        )

        agent.add_node(sample_importer)
        agent.add_node(expired_cert_supplier)

        input_data = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.NATURAL_LANGUAGE_QUERY,
            query="Show suppliers with expired certifications"
        )

        result = agent.run(input_data)

        assert result.nl_generated_filter.get('expired_certifications') is True
        assert len(result.nl_results) > 0


# =============================================================================
# MODEL VALIDATION TESTS
# =============================================================================

class TestModelValidation:
    """Test Pydantic model validation."""

    def test_node_validation(self):
        """Test node model validation."""
        # Valid node
        node = SupplyChainNode(
            node_type=NodeType.PRODUCER,
            name="Test Farm",
            country_code="BR",
            commodities=[CommodityType.COFFEE]
        )
        assert node.node_id is not None

        # Invalid country code
        with pytest.raises(ValueError):
            SupplyChainNode(
                node_type=NodeType.PRODUCER,
                name="Test Farm",
                country_code="BRAZIL",  # Should be 2 chars
                commodities=[CommodityType.COFFEE]
            )

    def test_edge_validation(self):
        """Test edge model validation."""
        edge = SupplyChainEdge(
            source_node_id=uuid.uuid4(),
            target_node_id=uuid.uuid4(),
            edge_type=EdgeType.SUPPLIES,
            commodity=CommodityType.COFFEE
        )
        assert edge.edge_id is not None
        assert edge.data_source == DataSource.SUPPLIER_DECLARED

    def test_plot_validation(self):
        """Test plot geometry validation."""
        # Valid point
        plot = OriginPlot(
            producer_node_id=uuid.uuid4(),
            geometry=PlotGeometry(type="Point", coordinates=[-75.5, 4.5]),
            commodity=CommodityType.COFFEE,
            country_code="CO"
        )
        assert plot.plot_id is not None

        # Invalid point (wrong number of coordinates)
        with pytest.raises(ValueError):
            OriginPlot(
                producer_node_id=uuid.uuid4(),
                geometry=PlotGeometry(type="Point", coordinates=[-75.5]),  # Missing lat
                commodity=CommodityType.COFFEE,
                country_code="CO"
            )

    def test_certification_validity(self):
        """Test certification validity check."""
        valid_cert = Certification(
            name="Test Cert",
            issuer="Test Issuer",
            valid_from=date(2023, 1, 1),
            valid_to=date(2030, 12, 31)
        )
        assert valid_cert.is_valid is True

        expired_cert = Certification(
            name="Expired Cert",
            issuer="Test Issuer",
            valid_from=date(2020, 1, 1),
            valid_to=date(2021, 12, 31)
        )
        assert expired_cert.is_valid is False

        no_expiry_cert = Certification(
            name="Perpetual Cert",
            issuer="Test Issuer",
            valid_from=date(2020, 1, 1),
            valid_to=None
        )
        assert no_expiry_cert.is_valid is True


# =============================================================================
# GRAPH UTILITY TESTS
# =============================================================================

class TestGraphUtilities:
    """Test graph utility methods."""

    def test_graph_hash(self):
        """Test graph hash computation."""
        node1 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Test",
            country_code="BR",
            commodities=[CommodityType.COFFEE]
        )

        graph = SupplyChainGraph(nodes=[node1], edges=[])
        hash1 = graph.compute_hash()

        # Same graph should produce same hash
        hash2 = graph.compute_hash()
        assert hash1 == hash2

        # Adding node should change hash
        node2 = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.TRADER,
            name="Test 2",
            country_code="CH",
            commodities=[CommodityType.COFFEE]
        )
        graph.nodes.append(node2)
        hash3 = graph.compute_hash()
        assert hash1 != hash3

    def test_normalize_name(self):
        """Test company name normalization."""
        assert SupplyChainMapperAgent._normalize_name("ABC Trading Ltd.") == "ABC TRADING"
        assert SupplyChainMapperAgent._normalize_name("XYZ Corp.") == "XYZ"
        assert SupplyChainMapperAgent._normalize_name("Test GmbH") == "TEST"

    def test_normalize_tax_id(self):
        """Test tax ID normalization."""
        assert SupplyChainMapperAgent._normalize_tax_id("DE-123-456-789") == "DE123456789"
        assert SupplyChainMapperAgent._normalize_tax_id("DE 123 456 789") == "DE123456789"
        assert SupplyChainMapperAgent._normalize_tax_id("de.123.456.789") == "DE123456789"

    def test_string_similarity(self):
        """Test string similarity calculation."""
        sim = SupplyChainMapperAgent._string_similarity
        assert sim("ABC", "ABC") == 1.0
        assert sim("ABC", "ABD") > 0.5
        assert sim("ABC", "XYZ") < 0.5


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_invalid_operation(self, agent):
        """Test handling of unknown operations."""
        # This should be caught by Pydantic validation
        with pytest.raises(ValueError):
            SupplyChainMapperInput(
                importer_id=uuid.uuid4(),
                commodity=CommodityType.COFFEE,
                operation="INVALID_OP"
            )

    def test_nl_query_without_query(self, agent, sample_importer):
        """Test NL query without query string."""
        agent.add_node(sample_importer)

        input_data = SupplyChainMapperInput(
            importer_id=sample_importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.NATURAL_LANGUAGE_QUERY,
            query=None
        )

        result = agent.run(input_data)

        assert result.success is False
        assert "Query is required" in result.errors[0]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, agent):
        """Test complete workflow: map -> coverage -> gates -> snapshot."""
        # Setup supply chain
        importer = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.IMPORTER,
            name="Coffee Import EU",
            country_code="DE",
            commodities=[CommodityType.COFFEE],
            verification_status=VerificationStatus.VERIFIED
        )
        producer = SupplyChainNode(
            node_id=uuid.uuid4(),
            node_type=NodeType.PRODUCER,
            name="Brazil Coffee Farm",
            country_code="BR",
            commodities=[CommodityType.COFFEE],
            verification_status=VerificationStatus.VERIFIED
        )

        agent.add_node(importer)
        agent.add_node(producer)
        agent.add_edge(SupplyChainEdge(
            source_node_id=producer.node_id,
            target_node_id=importer.node_id,
            edge_type=EdgeType.SUPPLIES,
            commodity=CommodityType.COFFEE,
            verified=True
        ))
        agent.add_plot(OriginPlot(
            producer_node_id=producer.node_id,
            geometry=PlotGeometry(type="Point", coordinates=[-47.5, -15.5]),
            commodity=CommodityType.COFFEE,
            country_code="BR"
        ))

        # Step 1: Map supply chain
        map_result = agent.run(SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.MAP_SUPPLY_CHAIN
        ))
        assert map_result.success is True
        assert map_result.node_count == 2

        # Step 2: Calculate coverage
        coverage_result = agent.run(SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CALCULATE_COVERAGE
        ))
        assert coverage_result.success is True
        assert coverage_result.coverage_report.plot_coverage == 100.0

        # Step 3: Check gates
        gates_result = agent.run(SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CHECK_GATES
        ))
        assert gates_result.success is True

        # Step 4: Create snapshot
        snapshot_result = agent.run(SupplyChainMapperInput(
            importer_id=importer.node_id,
            commodity=CommodityType.COFFEE,
            operation=OperationType.CREATE_SNAPSHOT,
            trigger_type=SnapshotTrigger.DDS_SUBMISSION
        ))
        assert snapshot_result.success is True
        assert snapshot_result.snapshot.trigger_type == SnapshotTrigger.DDS_SUBMISSION


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
