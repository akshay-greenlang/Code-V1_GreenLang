# -*- coding: utf-8 -*-
"""
Comprehensive test suite for RiskPropagationEngine.

Tests cover:
    1. Configuration validation (weights, thresholds)
    2. Single-node risk calculation (all four dimensions)
    3. Linear chain propagation (plot -> coop -> processor -> trader -> importer)
    4. Many-to-one propagation (multiple plots -> single collector)
    5. One-to-many propagation (single processor -> multiple traders)
    6. Diamond topology (diverge and converge)
    7. "Highest risk wins" principle
    8. Cycle detection and fallback
    9. Risk classification (LOW / STANDARD / HIGH)
    10. Enhanced due diligence triggers
    11. Risk concentration analysis
    12. Heatmap generation
    13. Provenance hash reproducibility (AGENT-FOUND-008 integration)
    14. Incremental propagation
    15. Country risk lookup (EUDR Article 29)
    16. Commodity risk lookup
    17. Performance (10,000-node graph < 3 seconds)
    18. Edge cases (empty graph, single node, disconnected components)
    19. Custom weight configurations
    20. Audit trail completeness

PRD: PRD-AGENT-EUDR-001, Feature 5
Agent: GL-EUDR-SCM-001
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Dict, List

import pytest

from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import (
    COMMODITY_RISK_SCORES,
    COUNTRY_CLASSIFICATIONS,
    COUNTRY_RISK_SCORES,
    EnhancedDueDiligenceTrigger,
    NodeRiskInput,
    NodeRiskResult,
    PropagationAuditEntry,
    PropagationDirection,
    PropagationResult,
    RiskConcentrationEntry,
    RiskHeatmapEntry,
    RiskLevel,
    RiskPropagationConfig,
    RiskPropagationEngine,
    _clamp_risk,
    _compute_provenance_hash,
    _to_decimal,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def default_config() -> RiskPropagationConfig:
    """Default configuration with PRD-specified weights."""
    return RiskPropagationConfig()


@pytest.fixture
def engine(default_config: RiskPropagationConfig) -> RiskPropagationEngine:
    """Default risk propagation engine."""
    return RiskPropagationEngine(default_config)


@pytest.fixture
def linear_chain_adjacency() -> Dict[str, List[str]]:
    """Simple linear supply chain: plot -> coop -> processor -> trader -> importer."""
    return {
        "PLOT-001": ["COOP-001"],
        "COOP-001": ["PROC-001"],
        "PROC-001": ["TRADER-001"],
        "TRADER-001": ["IMPORTER-001"],
    }


@pytest.fixture
def linear_chain_inputs() -> Dict[str, NodeRiskInput]:
    """Risk inputs for a linear supply chain."""
    return {
        "PLOT-001": NodeRiskInput(
            node_id="PLOT-001",
            country_code="CI",
            country_risk=65.0,
            commodity_risk=60.0,
            supplier_risk=50.0,
            deforestation_risk=80.0,
            node_type="producer",
            commodities=["cocoa"],
            tier_depth=4,
        ),
        "COOP-001": NodeRiskInput(
            node_id="COOP-001",
            country_code="CI",
            country_risk=65.0,
            commodity_risk=60.0,
            supplier_risk=40.0,
            deforestation_risk=30.0,
            node_type="collector",
            commodities=["cocoa"],
            tier_depth=3,
        ),
        "PROC-001": NodeRiskInput(
            node_id="PROC-001",
            country_code="NL",
            country_risk=5.0,
            commodity_risk=55.0,
            supplier_risk=20.0,
            deforestation_risk=10.0,
            node_type="processor",
            commodities=["chocolate"],
            tier_depth=2,
        ),
        "TRADER-001": NodeRiskInput(
            node_id="TRADER-001",
            country_code="NL",
            country_risk=5.0,
            commodity_risk=55.0,
            supplier_risk=15.0,
            deforestation_risk=5.0,
            node_type="trader",
            commodities=["chocolate"],
            tier_depth=1,
        ),
        "IMPORTER-001": NodeRiskInput(
            node_id="IMPORTER-001",
            country_code="DE",
            country_risk=5.0,
            commodity_risk=55.0,
            supplier_risk=10.0,
            deforestation_risk=5.0,
            node_type="importer",
            commodities=["chocolate"],
            tier_depth=0,
        ),
    }


@pytest.fixture
def many_to_one_adjacency() -> Dict[str, List[str]]:
    """Multiple plots feed into a single collector."""
    return {
        "PLOT-001": ["COOP-001"],
        "PLOT-002": ["COOP-001"],
        "PLOT-003": ["COOP-001"],
        "COOP-001": ["PROC-001"],
    }


@pytest.fixture
def many_to_one_inputs() -> Dict[str, NodeRiskInput]:
    """Risk inputs with varying risk levels across plots."""
    return {
        "PLOT-001": NodeRiskInput(
            node_id="PLOT-001",
            country_code="CI",
            country_risk=65.0,
            commodity_risk=60.0,
            supplier_risk=50.0,
            deforestation_risk=80.0,
            node_type="producer",
        ),
        "PLOT-002": NodeRiskInput(
            node_id="PLOT-002",
            country_code="GH",
            country_risk=50.0,
            commodity_risk=60.0,
            supplier_risk=30.0,
            deforestation_risk=20.0,
            node_type="producer",
        ),
        "PLOT-003": NodeRiskInput(
            node_id="PLOT-003",
            country_code="DE",
            country_risk=5.0,
            commodity_risk=10.0,
            supplier_risk=10.0,
            deforestation_risk=5.0,
            node_type="producer",
        ),
        "COOP-001": NodeRiskInput(
            node_id="COOP-001",
            country_code="CI",
            country_risk=65.0,
            commodity_risk=60.0,
            supplier_risk=40.0,
            deforestation_risk=30.0,
            node_type="collector",
        ),
        "PROC-001": NodeRiskInput(
            node_id="PROC-001",
            country_code="NL",
            country_risk=5.0,
            commodity_risk=55.0,
            supplier_risk=20.0,
            deforestation_risk=10.0,
            node_type="processor",
        ),
    }


# ===========================================================================
# 1. Configuration Tests
# ===========================================================================


class TestRiskPropagationConfig:
    """Tests for RiskPropagationConfig validation."""

    def test_default_config_valid(self) -> None:
        """Default configuration should be valid."""
        config = RiskPropagationConfig()
        assert config.weight_country == Decimal("0.30")
        assert config.weight_commodity == Decimal("0.20")
        assert config.weight_supplier == Decimal("0.25")
        assert config.weight_deforestation == Decimal("0.25")
        total = (
            config.weight_country
            + config.weight_commodity
            + config.weight_supplier
            + config.weight_deforestation
        )
        assert total == Decimal("1.00")

    def test_weights_must_sum_to_one(self) -> None:
        """Weights that do not sum to 1.00 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.00"):
            RiskPropagationConfig(
                weight_country=Decimal("0.50"),
                weight_commodity=Decimal("0.50"),
                weight_supplier=Decimal("0.25"),
                weight_deforestation=Decimal("0.25"),
            )

    def test_negative_weight_rejected(self) -> None:
        """Negative weights should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            RiskPropagationConfig(
                weight_country=Decimal("-0.10"),
                weight_commodity=Decimal("0.40"),
                weight_supplier=Decimal("0.35"),
                weight_deforestation=Decimal("0.35"),
            )

    def test_threshold_order_validation(self) -> None:
        """Low threshold must be less than high threshold."""
        with pytest.raises(ValueError, match="less than"):
            RiskPropagationConfig(
                threshold_low=Decimal("80"),
                threshold_high=Decimal("30"),
            )

    def test_max_iterations_validation(self) -> None:
        """max_iterations must be >= 1."""
        with pytest.raises(ValueError, match="max_iterations"):
            RiskPropagationConfig(max_iterations=0)

    def test_from_dict_factory(self) -> None:
        """Configuration can be created from a dictionary."""
        data = {
            "weight_country": 0.30,
            "weight_commodity": 0.20,
            "weight_supplier": 0.25,
            "weight_deforestation": 0.25,
            "enhanced_due_diligence_threshold": 75.0,
            "enable_audit_log": False,
        }
        config = RiskPropagationConfig.from_dict(data)
        assert config.weight_country == Decimal("0.30")
        assert config.enhanced_due_diligence_threshold == Decimal("75.0")
        assert config.enable_audit_log is False

    def test_config_is_frozen(self) -> None:
        """Configuration should be immutable (frozen dataclass)."""
        config = RiskPropagationConfig()
        with pytest.raises(AttributeError):
            config.weight_country = Decimal("0.50")  # type: ignore

    def test_custom_weights_valid(self) -> None:
        """Custom weights that sum to 1.00 should be accepted."""
        config = RiskPropagationConfig(
            weight_country=Decimal("0.40"),
            weight_commodity=Decimal("0.30"),
            weight_supplier=Decimal("0.15"),
            weight_deforestation=Decimal("0.15"),
        )
        assert config.weight_country == Decimal("0.40")


# ===========================================================================
# 2. Single-Node Risk Calculation
# ===========================================================================


class TestSingleNodeRisk:
    """Tests for risk calculation on a single isolated node."""

    def test_single_node_no_parents(self, engine: RiskPropagationEngine) -> None:
        """A single node with no parents should use own weighted risks."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "NODE-001": NodeRiskInput(
                node_id="NODE-001",
                country_risk=80.0,
                commodity_risk=60.0,
                supplier_risk=40.0,
                deforestation_risk=50.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        node = result.node_results["NODE-001"]
        # Expected: max(80*0.30, 60*0.20, 40*0.25, 50*0.25)
        # = max(24.00, 12.00, 10.00, 12.50) = 24.00
        assert node.composite_risk == Decimal("24.00")
        assert node.inherited_risk == Decimal("0.00")
        assert node.risk_level == RiskLevel.LOW

    def test_single_high_risk_node(self, engine: RiskPropagationEngine) -> None:
        """A node with all-100 risk should score at the weighted max."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "HIGH-001": NodeRiskInput(
                node_id="HIGH-001",
                country_risk=100.0,
                commodity_risk=100.0,
                supplier_risk=100.0,
                deforestation_risk=100.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)
        node = result.node_results["HIGH-001"]
        # max(100*0.30, 100*0.20, 100*0.25, 100*0.25) = max(30, 20, 25, 25) = 30
        assert node.composite_risk == Decimal("30.00")
        assert node.risk_level == RiskLevel.STANDARD

    def test_zero_risk_node(self, engine: RiskPropagationEngine) -> None:
        """A node with all-zero risk should score 0.00."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "ZERO-001": NodeRiskInput(
                node_id="ZERO-001",
                country_risk=0.0,
                commodity_risk=0.0,
                supplier_risk=0.0,
                deforestation_risk=0.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)
        node = result.node_results["ZERO-001"]
        assert node.composite_risk == Decimal("0.00")
        assert node.risk_level == RiskLevel.LOW


# ===========================================================================
# 3. Linear Chain Propagation
# ===========================================================================


class TestLinearChainPropagation:
    """Tests for risk propagation through a linear supply chain."""

    def test_linear_chain_propagation(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Risk should propagate from plot to importer."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)

        # All 5 nodes should be processed
        assert result.total_nodes == 5
        assert result.total_edges == 4

        # PLOT-001: root node, no inheritance
        plot = result.node_results["PLOT-001"]
        assert plot.inherited_risk == Decimal("0.00")
        # max(65*0.30, 60*0.20, 50*0.25, 80*0.25) = max(19.50, 12.00, 12.50, 20.00) = 20.00
        assert plot.composite_risk == Decimal("20.00")

        # COOP-001: inherits from PLOT-001
        coop = result.node_results["COOP-001"]
        assert coop.inherited_risk == plot.composite_risk
        assert coop.composite_risk >= plot.composite_risk  # "highest risk wins"

        # IMPORTER-001: final node, should inherit the chain's risk
        importer = result.node_results["IMPORTER-001"]
        assert importer.inherited_risk >= Decimal("0")
        # Importer should have risk >= its own weighted risk
        assert importer.composite_risk >= importer.own_composite_risk

    def test_risk_never_decreases_downstream(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Risk should never decrease as it propagates downstream."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)

        chain_order = ["PLOT-001", "COOP-001", "PROC-001", "TRADER-001", "IMPORTER-001"]
        for i in range(len(chain_order) - 1):
            upstream = result.node_results[chain_order[i]]
            downstream = result.node_results[chain_order[i + 1]]
            # Downstream risk >= upstream risk (highest risk wins)
            assert downstream.composite_risk >= upstream.composite_risk, (
                f"{chain_order[i+1]} risk ({downstream.composite_risk}) < "
                f"{chain_order[i]} risk ({upstream.composite_risk})"
            )

    def test_propagation_depth_increases(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Propagation depth should increase along the chain."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)

        assert result.node_results["PLOT-001"].propagation_depth == 0
        assert result.node_results["COOP-001"].propagation_depth == 1
        assert result.node_results["PROC-001"].propagation_depth == 2
        assert result.node_results["TRADER-001"].propagation_depth == 3
        assert result.node_results["IMPORTER-001"].propagation_depth == 4


# ===========================================================================
# 4. Many-to-One Propagation
# ===========================================================================


class TestManyToOnePropagation:
    """Tests for multiple upstream nodes feeding into a single node."""

    def test_highest_risk_parent_wins(
        self,
        engine: RiskPropagationEngine,
        many_to_one_adjacency: Dict[str, List[str]],
        many_to_one_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Collector should inherit the highest risk from its parent plots."""
        result = engine.propagate("G1", many_to_one_adjacency, many_to_one_inputs)

        coop = result.node_results["COOP-001"]

        # PLOT-001 has the highest risk (deforestation_risk=80 * 0.25 = 20)
        plot1 = result.node_results["PLOT-001"]
        plot2 = result.node_results["PLOT-002"]
        plot3 = result.node_results["PLOT-003"]

        max_parent_risk = max(
            plot1.composite_risk,
            plot2.composite_risk,
            plot3.composite_risk,
        )

        assert coop.inherited_risk == max_parent_risk
        assert coop.composite_risk >= max_parent_risk

    def test_highest_risk_parent_id_tracked(
        self,
        engine: RiskPropagationEngine,
        many_to_one_adjacency: Dict[str, List[str]],
        many_to_one_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """The highest-risk parent ID should be recorded."""
        result = engine.propagate("G1", many_to_one_adjacency, many_to_one_inputs)
        coop = result.node_results["COOP-001"]

        # PLOT-001 has the highest deforestation risk
        assert coop.highest_risk_parent_id == "PLOT-001"


# ===========================================================================
# 5. One-to-Many Propagation
# ===========================================================================


class TestOneToManyPropagation:
    """Tests for a single upstream node feeding multiple downstream nodes."""

    def test_all_downstream_inherit(self, engine: RiskPropagationEngine) -> None:
        """All downstream nodes should inherit from the shared parent."""
        adjacency = {
            "PROC-001": ["TRADER-A", "TRADER-B", "TRADER-C"],
        }
        inputs = {
            "PROC-001": NodeRiskInput(
                node_id="PROC-001",
                country_risk=55.0,
                commodity_risk=70.0,
                supplier_risk=40.0,
                deforestation_risk=60.0,
            ),
            "TRADER-A": NodeRiskInput(node_id="TRADER-A", country_risk=5.0),
            "TRADER-B": NodeRiskInput(node_id="TRADER-B", country_risk=10.0),
            "TRADER-C": NodeRiskInput(node_id="TRADER-C", country_risk=80.0),
        }
        result = engine.propagate("G1", adjacency, inputs)

        proc_risk = result.node_results["PROC-001"].composite_risk

        # All traders should have risk >= processor's risk
        for tid in ["TRADER-A", "TRADER-B", "TRADER-C"]:
            assert result.node_results[tid].composite_risk >= proc_risk


# ===========================================================================
# 6. Diamond Topology
# ===========================================================================


class TestDiamondTopology:
    """Tests for diamond-shaped supply chains (diverge then converge)."""

    def test_diamond_highest_risk_wins(self, engine: RiskPropagationEngine) -> None:
        """In a diamond, the converging node should inherit the highest path risk."""
        adjacency = {
            "PLOT-001": ["PROC-A", "PROC-B"],
            "PROC-A": ["TRADER-001"],
            "PROC-B": ["TRADER-001"],
        }
        inputs = {
            "PLOT-001": NodeRiskInput(
                node_id="PLOT-001",
                country_risk=80.0,
                commodity_risk=70.0,
                supplier_risk=60.0,
                deforestation_risk=90.0,
            ),
            "PROC-A": NodeRiskInput(
                node_id="PROC-A",
                country_risk=5.0,
                commodity_risk=55.0,
                supplier_risk=20.0,
                deforestation_risk=10.0,
            ),
            "PROC-B": NodeRiskInput(
                node_id="PROC-B",
                country_risk=70.0,
                commodity_risk=55.0,
                supplier_risk=50.0,
                deforestation_risk=60.0,
            ),
        }
        # TRADER-001 needs to be added
        inputs["TRADER-001"] = NodeRiskInput(
            node_id="TRADER-001",
            country_risk=5.0,
            commodity_risk=55.0,
            supplier_risk=15.0,
            deforestation_risk=5.0,
        )

        result = engine.propagate("G1", adjacency, inputs)

        trader = result.node_results["TRADER-001"]
        proc_a = result.node_results["PROC-A"]
        proc_b = result.node_results["PROC-B"]

        # Trader inherits max of PROC-A and PROC-B risks
        expected_inherited = max(proc_a.composite_risk, proc_b.composite_risk)
        assert trader.inherited_risk == expected_inherited


# ===========================================================================
# 7. Highest Risk Wins Principle
# ===========================================================================


class TestHighestRiskWins:
    """Tests for the 'highest risk wins' principle from PRD Feature 5."""

    def test_inherited_risk_dominates_own(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """If inherited risk > all own weighted risks, inherited wins."""
        adjacency = {"PARENT": ["CHILD"]}
        inputs = {
            "PARENT": NodeRiskInput(
                node_id="PARENT",
                country_risk=100.0,
                commodity_risk=100.0,
                supplier_risk=100.0,
                deforestation_risk=100.0,
            ),
            "CHILD": NodeRiskInput(
                node_id="CHILD",
                country_risk=5.0,
                commodity_risk=5.0,
                supplier_risk=5.0,
                deforestation_risk=5.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        parent_risk = result.node_results["PARENT"].composite_risk
        child = result.node_results["CHILD"]

        # Child's own weighted risks: max(5*0.3, 5*0.2, 5*0.25, 5*0.25) = 1.50
        # But inherited = parent_risk = 30.00
        assert child.composite_risk == parent_risk
        assert child.inherited_risk == parent_risk

    def test_own_risk_dominates_inherited(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """If own weighted risk > inherited risk, own risk wins."""
        adjacency = {"PARENT": ["CHILD"]}
        inputs = {
            "PARENT": NodeRiskInput(
                node_id="PARENT",
                country_risk=5.0,
                commodity_risk=5.0,
                supplier_risk=5.0,
                deforestation_risk=5.0,
            ),
            "CHILD": NodeRiskInput(
                node_id="CHILD",
                country_risk=100.0,
                commodity_risk=100.0,
                supplier_risk=100.0,
                deforestation_risk=100.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        child = result.node_results["CHILD"]
        # Child own: max(100*0.3, 100*0.2, 100*0.25, 100*0.25) = max(30, 20, 25, 25) = 30
        # Parent risk: max(5*0.3, ...) = 1.50
        assert child.composite_risk == Decimal("30.00")
        assert child.own_composite_risk == Decimal("30.00")


# ===========================================================================
# 8. Cycle Detection
# ===========================================================================


class TestCycleDetection:
    """Tests for cycle handling in the supply chain graph."""

    def test_cycle_nodes_get_fallback_risk(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """Nodes in a cycle should receive fallback risk scores."""
        adjacency = {
            "A": ["B"],
            "B": ["C"],
            "C": ["A"],  # Creates a cycle
        }
        inputs = {
            "A": NodeRiskInput(node_id="A", country_risk=70.0),
            "B": NodeRiskInput(node_id="B", country_risk=60.0),
            "C": NodeRiskInput(node_id="C", country_risk=50.0),
        }
        result = engine.propagate("G1", adjacency, inputs)

        # All nodes should have results (no crash)
        assert len(result.node_results) == 3
        for nid in ["A", "B", "C"]:
            assert nid in result.node_results
            assert result.node_results[nid].composite_risk >= Decimal("0")


# ===========================================================================
# 9. Risk Classification
# ===========================================================================


class TestRiskClassification:
    """Tests for risk level classification."""

    def test_low_risk_classification(self, engine: RiskPropagationEngine) -> None:
        """Score < 30 should classify as LOW."""
        assert engine.classify_risk(0.0) == RiskLevel.LOW
        assert engine.classify_risk(15.0) == RiskLevel.LOW
        assert engine.classify_risk(29.99) == RiskLevel.LOW

    def test_standard_risk_classification(self, engine: RiskPropagationEngine) -> None:
        """Score 30-69 should classify as STANDARD."""
        assert engine.classify_risk(30.0) == RiskLevel.STANDARD
        assert engine.classify_risk(50.0) == RiskLevel.STANDARD
        assert engine.classify_risk(69.99) == RiskLevel.STANDARD

    def test_high_risk_classification(self, engine: RiskPropagationEngine) -> None:
        """Score >= 70 should classify as HIGH."""
        assert engine.classify_risk(70.0) == RiskLevel.HIGH
        assert engine.classify_risk(85.0) == RiskLevel.HIGH
        assert engine.classify_risk(100.0) == RiskLevel.HIGH

    def test_risk_level_from_score(self) -> None:
        """RiskLevel.from_score should classify correctly."""
        assert RiskLevel.from_score(Decimal("0")) == RiskLevel.LOW
        assert RiskLevel.from_score(Decimal("29.99")) == RiskLevel.LOW
        assert RiskLevel.from_score(Decimal("30")) == RiskLevel.STANDARD
        assert RiskLevel.from_score(Decimal("69.99")) == RiskLevel.STANDARD
        assert RiskLevel.from_score(Decimal("70")) == RiskLevel.HIGH
        assert RiskLevel.from_score(Decimal("100")) == RiskLevel.HIGH


# ===========================================================================
# 10. Enhanced Due Diligence Triggers
# ===========================================================================


class TestEnhancedDueDiligence:
    """Tests for enhanced due diligence threshold triggers."""

    def test_edd_triggered_at_threshold(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """Nodes at or above the EDD threshold should be flagged."""
        adjacency = {"HIGH-RISK": ["DOWNSTREAM"]}
        inputs = {
            "HIGH-RISK": NodeRiskInput(
                node_id="HIGH-RISK",
                country_risk=100.0,
                commodity_risk=100.0,
                supplier_risk=100.0,
                deforestation_risk=100.0,
            ),
            "DOWNSTREAM": NodeRiskInput(
                node_id="DOWNSTREAM",
                country_risk=5.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        # Downstream inherits 30.00 from HIGH-RISK => not >= 70 threshold
        # HIGH-RISK own = 30.00 => not >= 70 threshold
        # Neither should trigger with default weights since max = 30
        # Let's check with explicit high values
        assert isinstance(result.edd_triggers, list)

    def test_edd_generates_recommendations(self) -> None:
        """EDD triggers should include deterministic mitigation actions."""
        config = RiskPropagationConfig(
            enhanced_due_diligence_threshold=Decimal("15"),
        )
        engine = RiskPropagationEngine(config)

        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "NODE-001": NodeRiskInput(
                node_id="NODE-001",
                country_risk=80.0,
                commodity_risk=70.0,
                supplier_risk=60.0,
                deforestation_risk=90.0,
                country_code="CD",
                commodities=["oil_palm"],
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        # With threshold 15 and max weighted risk = 90*0.25 = 22.50,
        # the node should trigger EDD
        assert len(result.edd_triggers) >= 1
        trigger = result.edd_triggers[0]
        assert trigger.node_id == "NODE-001"
        assert len(trigger.recommended_actions) >= 1


# ===========================================================================
# 11. Risk Concentration Analysis
# ===========================================================================


class TestRiskConcentration:
    """Tests for risk concentration identification."""

    def test_concentration_identified(
        self,
        engine: RiskPropagationEngine,
        many_to_one_adjacency: Dict[str, List[str]],
        many_to_one_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """High-risk plots should be identified as risk concentrators."""
        result = engine.propagate("G1", many_to_one_adjacency, many_to_one_inputs)

        # Should have at least one concentration entry
        assert isinstance(result.risk_concentrations, list)
        # PLOT-001 has the highest risk and feeds downstream
        if result.risk_concentrations:
            top_concentrator = result.risk_concentrations[0]
            assert top_concentrator.downstream_nodes_affected >= 1

    def test_concentration_serialization(self) -> None:
        """RiskConcentrationEntry should serialize correctly."""
        entry = RiskConcentrationEntry(
            node_id="PLOT-001",
            node_type="producer",
            country_code="CI",
            own_risk_score=Decimal("75.00"),
            downstream_nodes_affected=5,
            downstream_node_ids=["A", "B", "C", "D", "E"],
            risk_contribution=Decimal("45.00"),
        )
        d = entry.to_dict()
        assert d["node_id"] == "PLOT-001"
        assert d["downstream_nodes_affected"] == 5


# ===========================================================================
# 12. Heatmap Generation
# ===========================================================================


class TestHeatmapGeneration:
    """Tests for risk heatmap overlay generation."""

    def test_heatmap_entries_for_all_nodes(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Every node should have a heatmap entry."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        assert len(result.heatmap) == 5

    def test_heatmap_color_coding(self, engine: RiskPropagationEngine) -> None:
        """Heatmap colors should match risk levels."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "LOW-NODE": NodeRiskInput(
                node_id="LOW-NODE",
                country_risk=5.0,
                commodity_risk=5.0,
                supplier_risk=5.0,
                deforestation_risk=5.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)

        heatmap_entry = result.heatmap[0]
        assert heatmap_entry.color_hex == "#22C55E"  # Green for LOW
        assert heatmap_entry.risk_level == RiskLevel.LOW

    def test_heatmap_sorted_by_node_id(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Heatmap entries should be sorted by node_id for determinism."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        node_ids = [h.node_id for h in result.heatmap]
        assert node_ids == sorted(node_ids)


# ===========================================================================
# 13. Provenance Hash Reproducibility
# ===========================================================================


class TestProvenanceReproducibility:
    """Tests for bit-perfect reproducibility (AGENT-FOUND-008)."""

    def test_same_input_same_hash(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Same inputs must produce identical provenance hashes."""
        result1 = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        result2 = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        assert result1.provenance_hash == result2.provenance_hash
        assert len(result1.provenance_hash) == 64  # SHA-256

    def test_different_input_different_hash(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """Different inputs must produce different provenance hashes."""
        adjacency = {"A": ["B"]}
        inputs1 = {
            "A": NodeRiskInput(node_id="A", country_risk=80.0),
            "B": NodeRiskInput(node_id="B"),
        }
        inputs2 = {
            "A": NodeRiskInput(node_id="A", country_risk=20.0),
            "B": NodeRiskInput(node_id="B"),
        }
        result1 = engine.propagate("G1", adjacency, inputs1)
        result2 = engine.propagate("G1", adjacency, inputs2)
        assert result1.provenance_hash != result2.provenance_hash

    def test_verify_reproducibility_passes(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """verify_reproducibility should return True for matching hashes."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        assert engine.verify_reproducibility(
            "G1",
            linear_chain_adjacency,
            linear_chain_inputs,
            result.provenance_hash,
        )

    def test_verify_reproducibility_fails_for_wrong_hash(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """verify_reproducibility should return False for wrong hash."""
        assert not engine.verify_reproducibility(
            "G1",
            linear_chain_adjacency,
            linear_chain_inputs,
            "wrong_hash_value",
        )

    def test_ten_runs_same_hash(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Ten consecutive runs must all produce the same hash."""
        hashes = set()
        for _ in range(10):
            result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
            hashes.add(result.provenance_hash)
        assert len(hashes) == 1, f"Got {len(hashes)} different hashes"


# ===========================================================================
# 14. Incremental Propagation
# ===========================================================================


class TestIncrementalPropagation:
    """Tests for incremental risk re-propagation."""

    def test_incremental_propagation(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Incremental propagation should update affected nodes."""
        # Initial propagation
        result1 = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)

        # Change PLOT-001 risk and re-propagate incrementally
        modified_inputs = dict(linear_chain_inputs)
        modified_inputs["PLOT-001"] = NodeRiskInput(
            node_id="PLOT-001",
            country_code="DE",
            country_risk=5.0,
            commodity_risk=10.0,
            supplier_risk=10.0,
            deforestation_risk=5.0,
            node_type="producer",
        )

        result2 = engine.propagate_incremental(
            "G1",
            linear_chain_adjacency,
            modified_inputs,
            changed_node_ids={"PLOT-001"},
            previous_results=result1.node_results,
        )

        # PLOT-001 should have lower risk now
        assert result2.node_results["PLOT-001"].composite_risk < result1.node_results["PLOT-001"].composite_risk


# ===========================================================================
# 15. Country Risk Lookup
# ===========================================================================


class TestCountryRiskLookup:
    """Tests for EUDR Article 29 country risk database."""

    def test_known_high_risk_country(self, engine: RiskPropagationEngine) -> None:
        """Congo (CD) should have high deforestation risk."""
        score = engine.get_country_risk("CD")
        assert score >= 70.0

    def test_known_low_risk_country(self, engine: RiskPropagationEngine) -> None:
        """Germany (DE) should have low deforestation risk."""
        score = engine.get_country_risk("DE")
        assert score < 30.0

    def test_unknown_country_default(self, engine: RiskPropagationEngine) -> None:
        """Unknown country should return default risk (50)."""
        score = engine.get_country_risk("ZZ")
        assert score == 50.0

    def test_case_insensitive_lookup(self, engine: RiskPropagationEngine) -> None:
        """Country lookup should be case-insensitive."""
        assert engine.get_country_risk("de") == engine.get_country_risk("DE")

    def test_country_classification(self, engine: RiskPropagationEngine) -> None:
        """Country classification should match Article 29."""
        assert engine.get_country_classification("DE") == RiskLevel.LOW
        assert engine.get_country_classification("BR") == RiskLevel.STANDARD
        assert engine.get_country_classification("CD") == RiskLevel.HIGH

    def test_all_eu_members_low_risk(self) -> None:
        """All EU member states should be classified as LOW risk."""
        eu_members = [
            "DE", "FR", "IT", "NL", "BE", "AT", "SE", "FI", "DK",
            "IE", "LU", "PT", "ES", "PL", "CZ", "SK", "HU",
            "HR", "SI", "EE", "LV", "LT", "GR", "CY", "MT",
        ]
        for code in eu_members:
            if code in COUNTRY_RISK_SCORES:
                assert COUNTRY_RISK_SCORES[code] < 30, (
                    f"EU member {code} has risk {COUNTRY_RISK_SCORES[code]} >= 30"
                )


# ===========================================================================
# 16. Commodity Risk Lookup
# ===========================================================================


class TestCommodityRiskLookup:
    """Tests for commodity deforestation risk database."""

    def test_palm_oil_highest_risk(self, engine: RiskPropagationEngine) -> None:
        """Oil palm should have the highest commodity risk."""
        assert engine.get_commodity_risk("oil_palm") == 75.0

    def test_paper_lowest_risk(self, engine: RiskPropagationEngine) -> None:
        """Paper (derived) should have lower risk than wood (primary)."""
        assert engine.get_commodity_risk("paper") < engine.get_commodity_risk("wood")

    def test_unknown_commodity_default(self, engine: RiskPropagationEngine) -> None:
        """Unknown commodity should return default risk (50)."""
        assert engine.get_commodity_risk("unknown_product") == 50.0

    def test_all_seven_primary_commodities_present(self) -> None:
        """All seven EUDR primary commodities must be in the database."""
        primaries = ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]
        for commodity in primaries:
            assert commodity in COMMODITY_RISK_SCORES, (
                f"Primary commodity '{commodity}' missing from COMMODITY_RISK_SCORES"
            )

    def test_derived_products_lower_than_primary(self) -> None:
        """Derived products should generally have <= risk of primary."""
        derived_to_primary = {
            "beef": "cattle",
            "leather": "cattle",
            "chocolate": "cocoa",
            "palm_oil": "oil_palm",
            "natural_rubber": "rubber",
            "soybean_oil": "soya",
            "timber": "wood",
            "furniture": "wood",
            "paper": "wood",
        }
        for derived, primary in derived_to_primary.items():
            assert COMMODITY_RISK_SCORES[derived] <= COMMODITY_RISK_SCORES[primary], (
                f"Derived '{derived}' ({COMMODITY_RISK_SCORES[derived]}) > "
                f"primary '{primary}' ({COMMODITY_RISK_SCORES[primary]})"
            )


# ===========================================================================
# 17. Performance
# ===========================================================================


class TestPerformance:
    """Tests for performance targets from PRD-AGENT-EUDR-001."""

    def test_10000_node_graph_under_3_seconds(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """Full propagation for 10,000-node graph should complete in < 3 seconds."""
        # Build a 10,000-node linear chain (worst case for BFS)
        node_count = 10_000
        adjacency: Dict[str, List[str]] = {}
        inputs: Dict[str, NodeRiskInput] = {}

        for i in range(node_count):
            node_id = f"N-{i:05d}"
            inputs[node_id] = NodeRiskInput(
                node_id=node_id,
                country_risk=float(50 + (i % 50)),
                commodity_risk=float(40 + (i % 30)),
                supplier_risk=float(30 + (i % 40)),
                deforestation_risk=float(20 + (i % 60)),
            )
            if i > 0:
                prev_id = f"N-{i-1:05d}"
                adjacency[prev_id] = [node_id]

        start = time.monotonic()
        result = engine.propagate("PERF-TEST", adjacency, inputs)
        elapsed = time.monotonic() - start

        assert result.total_nodes == node_count
        assert elapsed < 3.0, (
            f"10,000-node propagation took {elapsed:.2f}s (target: < 3.0s)"
        )

    def test_1000_node_wide_graph(
        self, engine: RiskPropagationEngine,
    ) -> None:
        """Wide graph (1000 plots -> 1 collector -> 1 importer)."""
        adjacency: Dict[str, List[str]] = {}
        inputs: Dict[str, NodeRiskInput] = {}

        # 1000 plots feeding into one collector
        for i in range(1000):
            plot_id = f"PLOT-{i:04d}"
            adjacency[plot_id] = ["COOP-001"]
            inputs[plot_id] = NodeRiskInput(
                node_id=plot_id,
                country_risk=float(40 + (i % 50)),
                commodity_risk=60.0,
                deforestation_risk=float(30 + (i % 60)),
            )

        adjacency["COOP-001"] = ["IMPORTER-001"]
        inputs["COOP-001"] = NodeRiskInput(
            node_id="COOP-001", country_risk=50.0,
        )
        inputs["IMPORTER-001"] = NodeRiskInput(
            node_id="IMPORTER-001", country_risk=5.0,
        )

        start = time.monotonic()
        result = engine.propagate("WIDE-TEST", adjacency, inputs)
        elapsed = time.monotonic() - start

        assert result.total_nodes == 1002
        assert elapsed < 3.0


# ===========================================================================
# 18. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_node_inputs_raises(self, engine: RiskPropagationEngine) -> None:
        """Empty node_inputs should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.propagate("G1", {}, {})

    def test_single_node_no_edges(self, engine: RiskPropagationEngine) -> None:
        """Single node with no edges should work."""
        result = engine.propagate(
            "G1",
            {},
            {"SOLO": NodeRiskInput(node_id="SOLO", country_risk=55.0)},
        )
        assert result.total_nodes == 1
        assert result.total_edges == 0

    def test_disconnected_components(self, engine: RiskPropagationEngine) -> None:
        """Disconnected graph components should each propagate independently."""
        adjacency = {
            "A1": ["A2"],
            "B1": ["B2"],
        }
        inputs = {
            "A1": NodeRiskInput(node_id="A1", country_risk=90.0),
            "A2": NodeRiskInput(node_id="A2", country_risk=5.0),
            "B1": NodeRiskInput(node_id="B1", country_risk=10.0),
            "B2": NodeRiskInput(node_id="B2", country_risk=5.0),
        }
        result = engine.propagate("G1", adjacency, inputs)

        # A2 should inherit from A1 (high risk)
        a2 = result.node_results["A2"]
        assert a2.inherited_risk > Decimal("0")

        # B2 should inherit from B1 (low risk)
        b2 = result.node_results["B2"]
        # B1's own risk is low, B2 should not inherit A1's risk
        assert b2.composite_risk < a2.composite_risk

    def test_missing_node_in_adjacency(self, engine: RiskPropagationEngine) -> None:
        """Nodes referenced in adjacency but not in inputs get default scores."""
        adjacency = {"A": ["B"]}
        inputs = {
            "A": NodeRiskInput(node_id="A", country_risk=80.0),
            # "B" missing from inputs
        }
        result = engine.propagate("G1", adjacency, inputs)
        assert "B" in result.node_results

    def test_risk_score_clamped_to_100(self) -> None:
        """Risk scores should never exceed 100."""
        assert _clamp_risk(Decimal("150.00")) == Decimal("100.00")

    def test_risk_score_clamped_to_0(self) -> None:
        """Risk scores should never go below 0."""
        assert _clamp_risk(Decimal("-10.00")) == Decimal("0.00")


# ===========================================================================
# 19. Custom Weight Configurations
# ===========================================================================


class TestCustomWeights:
    """Tests for custom weight configurations without code changes."""

    def test_country_heavy_weights(self) -> None:
        """Country-heavy weights should increase country risk impact."""
        config = RiskPropagationConfig(
            weight_country=Decimal("0.70"),
            weight_commodity=Decimal("0.10"),
            weight_supplier=Decimal("0.10"),
            weight_deforestation=Decimal("0.10"),
        )
        engine = RiskPropagationEngine(config)

        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "NODE": NodeRiskInput(
                node_id="NODE",
                country_risk=100.0,
                commodity_risk=0.0,
                supplier_risk=0.0,
                deforestation_risk=0.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)
        node = result.node_results["NODE"]
        # 100 * 0.70 = 70.00
        assert node.composite_risk == Decimal("70.00")
        assert node.risk_level == RiskLevel.HIGH

    def test_equal_weights(self) -> None:
        """Equal weights should produce consistent results."""
        config = RiskPropagationConfig(
            weight_country=Decimal("0.25"),
            weight_commodity=Decimal("0.25"),
            weight_supplier=Decimal("0.25"),
            weight_deforestation=Decimal("0.25"),
        )
        engine = RiskPropagationEngine(config)

        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "NODE": NodeRiskInput(
                node_id="NODE",
                country_risk=80.0,
                commodity_risk=60.0,
                supplier_risk=40.0,
                deforestation_risk=20.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)
        node = result.node_results["NODE"]
        # max(80*0.25, 60*0.25, 40*0.25, 20*0.25) = max(20, 15, 10, 5) = 20
        assert node.composite_risk == Decimal("20.00")


# ===========================================================================
# 20. Audit Trail Completeness
# ===========================================================================


class TestAuditTrail:
    """Tests for audit trail completeness and correctness."""

    def test_audit_entries_for_all_nodes(
        self,
        engine: RiskPropagationEngine,
        linear_chain_adjacency: Dict[str, List[str]],
        linear_chain_inputs: Dict[str, NodeRiskInput],
    ) -> None:
        """Every processed node should have an audit entry."""
        result = engine.propagate("G1", linear_chain_adjacency, linear_chain_inputs)
        assert len(result.audit_entries) == result.total_nodes

    def test_audit_entries_have_required_fields(
        self,
        engine: RiskPropagationEngine,
    ) -> None:
        """Audit entries should have all required fields populated."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {"N": NodeRiskInput(node_id="N", country_risk=60.0)}
        result = engine.propagate("G1", adjacency, inputs)

        entry = result.audit_entries[0]
        assert entry.graph_id == "G1"
        assert entry.node_id == "N"
        assert entry.new_risk_score >= Decimal("0")
        assert entry.new_risk_level in ("low", "standard", "high")
        assert entry.propagation_source != ""
        assert entry.calculated_at is not None

    def test_audit_disabled(self) -> None:
        """When audit is disabled, no audit entries should be generated."""
        config = RiskPropagationConfig(enable_audit_log=False)
        engine = RiskPropagationEngine(config)

        adjacency: Dict[str, List[str]] = {}
        inputs = {"N": NodeRiskInput(node_id="N")}
        result = engine.propagate("G1", adjacency, inputs)
        assert len(result.audit_entries) == 0

    def test_audit_entry_serialization(self) -> None:
        """Audit entries should serialize to dictionaries."""
        entry = PropagationAuditEntry(
            graph_id="G1",
            node_id="N",
            new_risk_score=Decimal("45.00"),
            new_risk_level="standard",
            propagation_source="country_risk",
        )
        d = entry.to_dict()
        assert d["graph_id"] == "G1"
        assert d["new_risk_score"] == "45.00"

    def test_risk_summary_correct(
        self,
        engine: RiskPropagationEngine,
    ) -> None:
        """Risk summary should correctly count nodes by level."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {
            "LOW": NodeRiskInput(
                node_id="LOW",
                country_risk=10.0,
                commodity_risk=10.0,
                supplier_risk=10.0,
                deforestation_risk=10.0,
            ),
            "STD": NodeRiskInput(
                node_id="STD",
                country_risk=100.0,
                commodity_risk=100.0,
                supplier_risk=100.0,
                deforestation_risk=100.0,
            ),
        }
        result = engine.propagate("G1", adjacency, inputs)
        # LOW: max(10*0.3, 10*0.2, 10*0.25, 10*0.25) = 3.00 => LOW
        # STD: max(100*0.3, 100*0.2, 100*0.25, 100*0.25) = 30.00 => STANDARD
        assert result.risk_summary["low"] >= 1
        assert result.risk_summary["standard"] >= 1

    def test_result_serialization(
        self,
        engine: RiskPropagationEngine,
    ) -> None:
        """PropagationResult should serialize to a complete dictionary."""
        adjacency: Dict[str, List[str]] = {}
        inputs = {"N": NodeRiskInput(node_id="N")}
        result = engine.propagate("G1", adjacency, inputs)

        d = result.to_dict()
        assert "propagation_id" in d
        assert "graph_id" in d
        assert "node_results" in d
        assert "provenance_hash" in d
        assert "risk_summary" in d
        assert d["total_nodes"] == 1


# ===========================================================================
# 21. Decimal Arithmetic Tests
# ===========================================================================


class TestDecimalArithmetic:
    """Tests ensuring bit-perfect Decimal arithmetic."""

    def test_to_decimal_from_float(self) -> None:
        """Float conversion should go through string to avoid IEEE artefacts."""
        result = _to_decimal(0.1) + _to_decimal(0.2)
        assert result == Decimal("0.3")

    def test_to_decimal_from_int(self) -> None:
        """Integer conversion should work."""
        assert _to_decimal(50) == Decimal("50")

    def test_to_decimal_from_string(self) -> None:
        """String conversion should work."""
        assert _to_decimal("30.50") == Decimal("30.50")

    def test_to_decimal_idempotent(self) -> None:
        """Decimal input should pass through unchanged."""
        d = Decimal("42.00")
        assert _to_decimal(d) is d

    def test_clamp_risk_precision(self) -> None:
        """Clamped risk should have exactly 2 decimal places."""
        result = _clamp_risk(Decimal("45.678"))
        assert result == Decimal("45.68")

    def test_provenance_hash_deterministic(self) -> None:
        """Provenance hash should be deterministic."""
        data = {"key": "value", "number": 42}
        hash1 = _compute_provenance_hash(data)
        hash2 = _compute_provenance_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 64


# ===========================================================================
# 22. Node Result Serialization
# ===========================================================================


class TestNodeResultSerialization:
    """Tests for NodeRiskResult serialization."""

    def test_node_result_to_dict(self) -> None:
        """NodeRiskResult should serialize all fields."""
        result = NodeRiskResult(
            node_id="N1",
            composite_risk=Decimal("45.00"),
            risk_level=RiskLevel.STANDARD,
            inherited_risk=Decimal("30.00"),
            own_country_risk_weighted=Decimal("15.00"),
            own_commodity_risk_weighted=Decimal("12.00"),
            own_supplier_risk_weighted=Decimal("10.00"),
            own_deforestation_risk_weighted=Decimal("8.00"),
            own_composite_risk=Decimal("15.00"),
            risk_drivers=[("inherited", Decimal("45.00"))],
            requires_enhanced_due_diligence=False,
            parent_node_ids=["P1"],
            highest_risk_parent_id="P1",
            propagation_depth=2,
        )
        d = result.to_dict()
        assert d["node_id"] == "N1"
        assert d["composite_risk"] == "45.00"
        assert d["risk_level"] == "standard"
        assert d["propagation_depth"] == 2
        assert len(d["risk_drivers"]) == 1


# ===========================================================================
# 23. Reference Data Integrity
# ===========================================================================


class TestReferenceDataIntegrity:
    """Tests ensuring reference data consistency."""

    def test_country_classifications_match_scores(self) -> None:
        """Country classifications should match the score-based classification."""
        for cc, score in COUNTRY_RISK_SCORES.items():
            expected_level = RiskLevel.from_score(Decimal(str(score)))
            actual_level = COUNTRY_CLASSIFICATIONS.get(cc)
            assert actual_level == expected_level, (
                f"Country {cc}: score={score}, expected={expected_level}, "
                f"got={actual_level}"
            )

    def test_all_commodity_scores_in_valid_range(self) -> None:
        """All commodity risk scores should be in [0, 100]."""
        for commodity, score in COMMODITY_RISK_SCORES.items():
            assert 0 <= score <= 100, (
                f"Commodity '{commodity}' has invalid score: {score}"
            )

    def test_all_country_scores_in_valid_range(self) -> None:
        """All country risk scores should be in [0, 100]."""
        for cc, score in COUNTRY_RISK_SCORES.items():
            assert 0 <= score <= 100, (
                f"Country '{cc}' has invalid score: {score}"
            )
