"""
Tests for Causal Analysis Module

Comprehensive tests for CausalAnalysisService including:
- DAG construction and validation
- Root cause analysis
- Counterfactual reasoning
- Intervention recommendations
- Path tracing and effect computation

Author: GreenLang AI Team
"""

import pytest
import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from explainability.explanation_schemas import (
    CausalNode,
    CausalEdge,
    CausalAnalysisResult,
    CounterfactualExplanation,
    RootCauseAnalysis,
)
from explainability.causal_analysis import (
    NodeType,
    EdgeType,
    DeviationType,
    CausalGraphConfig,
    CausalGraph,
    CausalAnalysisService,
    InterventionRecommendation,
    build_thermal_system_dag,
    format_causal_analysis_report,
)


# Test fixtures

@pytest.fixture
def simple_graph():
    """Create a simple causal graph: A -> B -> C."""
    graph = CausalGraph()
    graph.add_node("A", "Variable A", value=1.0)
    graph.add_node("B", "Variable B", value=2.0)
    graph.add_node("C", "Variable C", value=3.0)
    graph.add_edge("A", "B", effect_size=0.5)
    graph.add_edge("B", "C", effect_size=0.8)
    return graph


@pytest.fixture
def complex_graph():
    """Create a more complex causal graph with multiple paths."""
    graph = CausalGraph()
    graph.add_node("fuel", "Fuel Flow", value=100.0, unit="kg/s")
    graph.add_node("air", "Air Flow", value=1200.0, unit="kg/s")
    graph.add_node("temp", "Combustion Temp", value=1500.0, unit="K")
    graph.add_node("efficiency", "Efficiency", value=0.85)
    graph.add_node("emissions", "Emissions", value=500.0, unit="kg/h")

    graph.add_edge("fuel", "temp", effect_size=0.3)
    graph.add_edge("air", "temp", effect_size=0.2)
    graph.add_edge("fuel", "emissions", effect_size=0.9)
    graph.add_edge("temp", "efficiency", effect_size=0.5)
    graph.add_edge("temp", "emissions", effect_size=0.1)

    return graph


@pytest.fixture
def causal_service():
    """Create a causal analysis service."""
    return CausalAnalysisService(agent_id="GL-TEST")


class TestCausalGraph:
    """Tests for CausalGraph class."""

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = CausalGraph()
        node = graph.add_node("test", "Test Node", value=1.0, unit="kg")

        assert node.node_id == "test"
        assert node.name == "Test Node"
        assert node.value == 1.0
        assert node.unit == "kg"

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = CausalGraph()
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")

        edge = graph.add_edge("A", "B", effect_size=0.5, mechanism="Direct cause")

        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.effect_size == 0.5
        assert edge.mechanism == "Direct cause"

    def test_add_edge_missing_source(self):
        """Test adding edge with missing source node."""
        graph = CausalGraph()
        graph.add_node("B", "Node B")

        with pytest.raises(ValueError, match="Source node"):
            graph.add_edge("A", "B")

    def test_add_edge_missing_target(self):
        """Test adding edge with missing target node."""
        graph = CausalGraph()
        graph.add_node("A", "Node A")

        with pytest.raises(ValueError, match="Target node"):
            graph.add_edge("A", "B")

    def test_add_edge_creates_cycle(self):
        """Test that adding edge that creates cycle is rejected."""
        graph = CausalGraph()
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")
        graph.add_node("C", "Node C")

        graph.add_edge("A", "B")
        graph.add_edge("B", "C")

        with pytest.raises(ValueError, match="cycle"):
            graph.add_edge("C", "A")

    def test_get_node(self, simple_graph):
        """Test getting node by ID."""
        node = simple_graph.get_node("A")

        assert node is not None
        assert node.node_id == "A"

    def test_get_node_nonexistent(self, simple_graph):
        """Test getting nonexistent node returns None."""
        node = simple_graph.get_node("X")

        assert node is None

    def test_get_nodes(self, simple_graph):
        """Test getting all nodes."""
        nodes = simple_graph.get_nodes()

        assert len(nodes) == 3

    def test_get_edges(self, simple_graph):
        """Test getting all edges."""
        edges = simple_graph.get_edges()

        assert len(edges) == 2

    def test_get_parents(self, simple_graph):
        """Test getting parent nodes."""
        parents = simple_graph.get_parents("B")

        assert len(parents) == 1
        assert "A" in parents

    def test_get_children(self, simple_graph):
        """Test getting child nodes."""
        children = simple_graph.get_children("A")

        assert len(children) == 1
        assert "B" in children

    def test_get_ancestors(self, simple_graph):
        """Test getting all ancestors."""
        ancestors = simple_graph.get_ancestors("C")

        assert len(ancestors) == 2
        assert "A" in ancestors
        assert "B" in ancestors

    def test_get_descendants(self, simple_graph):
        """Test getting all descendants."""
        descendants = simple_graph.get_descendants("A")

        assert len(descendants) == 2
        assert "B" in descendants
        assert "C" in descendants

    def test_find_paths(self, simple_graph):
        """Test finding paths between nodes."""
        paths = simple_graph.find_paths("A", "C")

        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]

    def test_find_paths_no_path(self, simple_graph):
        """Test finding paths when none exist."""
        paths = simple_graph.find_paths("C", "A")

        assert len(paths) == 0

    def test_is_valid_dag(self, simple_graph):
        """Test DAG validity check."""
        assert simple_graph.is_valid_dag()

    def test_topological_sort(self, simple_graph):
        """Test topological sort."""
        sorted_nodes = simple_graph.topological_sort()

        assert len(sorted_nodes) == 3
        assert sorted_nodes.index("A") < sorted_nodes.index("B")
        assert sorted_nodes.index("B") < sorted_nodes.index("C")

    def test_get_root_nodes(self, simple_graph):
        """Test getting root nodes."""
        roots = simple_graph.get_root_nodes()

        assert len(roots) == 1
        assert "A" in roots

    def test_get_leaf_nodes(self, simple_graph):
        """Test getting leaf nodes."""
        leaves = simple_graph.get_leaf_nodes()

        assert len(leaves) == 1
        assert "C" in leaves

    def test_compute_total_effect(self, simple_graph):
        """Test computing total causal effect."""
        effect = simple_graph.compute_total_effect("A", "C")

        # A -> B (0.5) -> C (0.8) = 0.4
        assert abs(effect - 0.4) < 0.01

    def test_to_dict(self, simple_graph):
        """Test graph serialization."""
        result = simple_graph.to_dict()

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 2


class TestCausalAnalysisService:
    """Tests for CausalAnalysisService."""

    def test_initialization(self):
        """Test service initialization."""
        service = CausalAnalysisService(
            agent_id="GL-TEST",
            agent_version="1.0.0"
        )

        assert service.agent_id == "GL-TEST"
        assert service.graph is not None

    def test_add_variable(self, causal_service):
        """Test adding variables."""
        node = causal_service.add_variable(
            "temperature",
            name="Temperature",
            value=350.0,
            unit="K"
        )

        assert node.node_id == "temperature"
        assert node.value == 350.0

    def test_add_causal_relationship(self, causal_service):
        """Test adding causal relationships."""
        causal_service.add_variable("A", value=1.0)
        causal_service.add_variable("B", value=2.0)

        edge = causal_service.add_causal_relationship(
            "A", "B",
            effect_size=0.5,
            mechanism="Direct heating"
        )

        assert edge.source == "A"
        assert edge.target == "B"

    def test_set_baseline(self, causal_service):
        """Test setting baseline statistics."""
        causal_service.add_variable("temp", value=350.0)
        causal_service.set_baseline("temp", mean=345.0, std=10.0)

        assert "temp" in causal_service._baselines
        assert causal_service._baselines["temp"]["mean"] == 345.0

    def test_analyze(self, causal_service):
        """Test comprehensive analysis."""
        # Build a simple system
        causal_service.add_variable("input", value=100.0)
        causal_service.add_variable("intermediate", value=50.0)
        causal_service.add_variable("output", value=25.0)

        causal_service.add_causal_relationship("input", "intermediate", effect_size=0.5)
        causal_service.add_causal_relationship("intermediate", "output", effect_size=0.5)

        result = causal_service.analyze(outcome_variable="output")

        assert isinstance(result, CausalAnalysisResult)
        assert len(result.nodes) == 3
        assert len(result.edges) == 2
        assert result.provenance_hash is not None

    def test_identify_root_causes(self, causal_service):
        """Test root cause identification."""
        # Build system
        causal_service.add_variable("fuel", value=100.0)
        causal_service.add_variable("air", value=1200.0)
        causal_service.add_variable("temp", value=1500.0)
        causal_service.add_variable("efficiency", value=0.85)

        causal_service.add_causal_relationship("fuel", "temp", effect_size=0.3)
        causal_service.add_causal_relationship("air", "temp", effect_size=0.2)
        causal_service.add_causal_relationship("temp", "efficiency", effect_size=0.5)

        causal_service.set_baseline("efficiency", mean=0.90, std=0.02)

        rca = causal_service.identify_root_causes(
            outcome_variable="efficiency",
            deviation=-0.05
        )

        assert isinstance(rca, RootCauseAnalysis)
        assert rca.outcome_variable == "efficiency"
        assert len(rca.root_causes) > 0
        assert len(rca.recommendations) > 0

    def test_generate_counterfactuals(self, causal_service):
        """Test counterfactual generation."""
        causal_service.add_variable("input", value=100.0)
        causal_service.add_variable("output", value=50.0)

        causal_service.add_causal_relationship("input", "output", effect_size=0.5)
        causal_service.set_baseline("output", mean=55.0, std=5.0)

        counterfactuals = causal_service.generate_counterfactuals(
            outcome_variable="output",
            target_outcome=55.0
        )

        assert isinstance(counterfactuals, list)
        if len(counterfactuals) > 0:
            cf = counterfactuals[0]
            assert isinstance(cf, CounterfactualExplanation)
            assert cf.original_outcome == 50.0
            assert cf.counterfactual_outcome == 55.0

    def test_recommend_interventions(self, causal_service):
        """Test intervention recommendations."""
        causal_service.add_variable("input", value=100.0)
        causal_service.add_variable("output", value=50.0)

        causal_service.add_causal_relationship("input", "output", effect_size=0.5)

        recommendations = causal_service.recommend_interventions(
            outcome_variable="output",
            optimization_goal="maximize"
        )

        assert isinstance(recommendations, list)
        if len(recommendations) > 0:
            rec = recommendations[0]
            assert isinstance(rec, InterventionRecommendation)
            assert rec.target_variable == "input"

    def test_trace_causal_path(self, causal_service):
        """Test causal path tracing."""
        causal_service.add_variable("A", value=1.0)
        causal_service.add_variable("B", value=2.0)
        causal_service.add_variable("C", value=3.0)

        causal_service.add_causal_relationship("A", "B", effect_size=0.5, mechanism="Step 1")
        causal_service.add_causal_relationship("B", "C", effect_size=0.8, mechanism="Step 2")

        trace = causal_service.trace_causal_path("A", "C")

        assert "source" in trace
        assert "target" in trace
        assert "num_paths" in trace
        assert trace["num_paths"] == 1
        assert "total_effect" in trace

    def test_simulate_intervention(self, causal_service):
        """Test intervention simulation."""
        causal_service.add_variable("input", value=100.0)
        causal_service.add_variable("output", value=50.0)

        causal_service.add_causal_relationship("input", "output", effect_size=0.5)

        new_values = causal_service.simulate_intervention("input", 110.0)

        assert "input" in new_values
        assert new_values["input"] == 110.0
        assert "output" in new_values
        # Change of 10 * 0.5 = 5
        assert new_values["output"] == 55.0


class TestCausalGraphConfig:
    """Tests for CausalGraphConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = CausalGraphConfig()

        assert config.max_path_length == 10
        assert config.min_effect_threshold == 0.01
        assert config.confidence_threshold == 0.7
        assert config.enable_caching is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = CausalGraphConfig(
            max_path_length=5,
            min_effect_threshold=0.05,
            enable_caching=False
        )

        assert config.max_path_length == 5
        assert config.min_effect_threshold == 0.05
        assert config.enable_caching is False


class TestInterventionRecommendation:
    """Tests for InterventionRecommendation."""

    def test_to_dict(self):
        """Test serialization."""
        rec = InterventionRecommendation(
            intervention_id="int123",
            target_variable="temperature",
            current_value=350.0,
            recommended_value=360.0,
            expected_effect=0.05,
            confidence=0.9,
            feasibility_score=0.85,
            side_effects=[{"variable": "pressure", "expected_change": 1000}],
            mechanism="Direct heating",
            priority=1
        )

        result = rec.to_dict()

        assert result["intervention_id"] == "int123"
        assert result["target_variable"] == "temperature"
        assert result["expected_effect"] == 0.05
        assert result["priority"] == 1


class TestCausalSchemas:
    """Tests for causal analysis schemas."""

    def test_causal_node_creation(self):
        """Test CausalNode creation."""
        node = CausalNode(
            node_id="temp",
            name="Temperature",
            node_type="variable",
            value=350.0,
            unit="K"
        )

        assert node.node_id == "temp"
        assert node.value == 350.0

    def test_causal_node_to_dict(self):
        """Test CausalNode serialization."""
        node = CausalNode(
            node_id="temp",
            name="Temperature",
            node_type="variable"
        )

        result = node.to_dict()

        assert "node_id" in result
        assert "name" in result

    def test_causal_edge_creation(self):
        """Test CausalEdge creation."""
        edge = CausalEdge(
            source="A",
            target="B",
            effect_size=0.5,
            confidence=0.95,
            mechanism="Direct cause"
        )

        assert edge.source == "A"
        assert edge.effect_size == 0.5

    def test_causal_edge_to_dict(self):
        """Test CausalEdge serialization."""
        edge = CausalEdge("A", "B", 0.5)

        result = edge.to_dict()

        assert "source" in result
        assert "target" in result
        assert "effect_size" in result

    def test_counterfactual_explanation_creation(self):
        """Test CounterfactualExplanation creation."""
        cf = CounterfactualExplanation(
            counterfactual_id="cf123",
            original_outcome=0.8,
            counterfactual_outcome=0.9,
            interventions={"temp": 360.0},
            effect_size=0.1,
            feasibility_score=0.85
        )

        assert cf.counterfactual_id == "cf123"
        assert cf.effect_size == 0.1

    def test_root_cause_analysis_creation(self):
        """Test RootCauseAnalysis creation."""
        rca = RootCauseAnalysis(
            analysis_id="rca123",
            outcome_variable="efficiency",
            outcome_deviation=-0.05,
            root_causes=[{"variable": "fuel", "contribution": 0.8}],
            causal_paths=[["fuel", "temp", "efficiency"]],
            recommendations=["Reduce fuel flow"]
        )

        assert rca.analysis_id == "rca123"
        assert len(rca.root_causes) == 1
        assert rca.provenance_hash is not None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_build_thermal_system_dag(self):
        """Test building sample thermal system DAG."""
        graph = build_thermal_system_dag()

        assert len(graph.get_nodes()) > 0
        assert len(graph.get_edges()) > 0
        assert graph.is_valid_dag()

        # Check expected nodes exist
        assert graph.get_node("fuel_flow") is not None
        assert graph.get_node("efficiency") is not None

    def test_format_causal_analysis_report(self, causal_service):
        """Test report formatting."""
        causal_service.add_variable("input", value=100.0)
        causal_service.add_variable("output", value=50.0)
        causal_service.add_causal_relationship("input", "output", effect_size=0.5)

        result = causal_service.analyze(outcome_variable="output")
        report = format_causal_analysis_report(result)

        assert "CAUSAL ANALYSIS REPORT" in report
        assert "Analysis ID" in report
        assert "CAUSAL GRAPH SUMMARY" in report


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_types_defined(self):
        """Test all expected node types exist."""
        expected = ["VARIABLE", "INTERVENTION", "OUTCOME", "CONFOUNDER", "MEDIATOR", "INSTRUMENT"]

        for type_name in expected:
            assert hasattr(NodeType, type_name)


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_types_defined(self):
        """Test all expected edge types exist."""
        expected = ["DIRECT", "INDIRECT", "CONFOUNDED", "BIDIRECTIONAL"]

        for type_name in expected:
            assert hasattr(EdgeType, type_name)


class TestDeviationType:
    """Tests for DeviationType enum."""

    def test_all_types_defined(self):
        """Test all expected deviation types exist."""
        expected = ["HIGH", "LOW", "WITHIN_NORMAL", "CRITICAL"]

        for type_name in expected:
            assert hasattr(DeviationType, type_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
