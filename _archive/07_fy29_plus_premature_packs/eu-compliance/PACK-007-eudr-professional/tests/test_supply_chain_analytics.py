"""
Unit tests for PACK-007 EUDR Professional Pack - Supply Chain Analytics Engine

Tests supply chain mapping, critical node identification, concentration risk,
origin tracing, mass balance, and network analysis.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from decimal import Decimal
from typing import List, Dict, Any


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import supply chain analytics module
supply_chain_mod = _import_from_path(
    "pack_007_supply_chain_analytics",
    _PACK_007_DIR / "engines" / "supply_chain_analytics.py"
)

pytestmark = pytest.mark.skipif(
    supply_chain_mod is None,
    reason="PACK-007 supply_chain_analytics module not available"
)


@pytest.fixture
def supply_chain_engine():
    """Create supply chain analytics engine instance."""
    if supply_chain_mod is None:
        pytest.skip("supply_chain_analytics module not available")
    return supply_chain_mod.SupplyChainAnalyticsEngine()


@pytest.fixture
def sample_tier1_suppliers():
    """Sample tier 1 suppliers."""
    return [
        {"supplier_id": "s1", "name": "Supplier A", "country": "BR", "product": "coffee"},
        {"supplier_id": "s2", "name": "Supplier B", "country": "CO", "product": "coffee"},
        {"supplier_id": "s3", "name": "Supplier C", "country": "VN", "product": "coffee"},
    ]


@pytest.fixture
def sample_multi_tier_network():
    """Sample multi-tier supply chain network."""
    return {
        "tier_1": [
            {"supplier_id": "t1_s1", "name": "Tier1 Supplier A"},
            {"supplier_id": "t1_s2", "name": "Tier1 Supplier B"},
        ],
        "tier_2": [
            {"supplier_id": "t2_s1", "name": "Tier2 Supplier A", "upstream": ["t1_s1"]},
            {"supplier_id": "t2_s2", "name": "Tier2 Supplier B", "upstream": ["t1_s2"]},
            {"supplier_id": "t2_s3", "name": "Tier2 Supplier C", "upstream": ["t1_s1", "t1_s2"]},
        ],
        "tier_3": [
            {"supplier_id": "t3_s1", "name": "Tier3 Supplier A", "upstream": ["t2_s1", "t2_s3"]},
        ]
    }


class TestSupplyChainMapping:
    """Test supply chain mapping features."""

    def test_map_supply_chain_tier1(self, supply_chain_engine, sample_tier1_suppliers):
        """Test mapping tier 1 supply chain."""
        result = supply_chain_engine.map_supply_chain(
            suppliers=sample_tier1_suppliers,
            depth=1
        )

        assert result is not None
        assert "tier_1" in result or "tiers" in result
        assert len(result.get("tier_1", [])) >= 3

    def test_map_supply_chain_tier5(self, supply_chain_engine, sample_multi_tier_network):
        """Test mapping deep supply chain (5 tiers)."""
        result = supply_chain_engine.map_supply_chain(
            network=sample_multi_tier_network,
            depth=5
        )

        assert result is not None
        assert "depth" in result or "max_tier" in result
        # Should map as deep as data allows

    def test_critical_nodes_identification(self, supply_chain_engine, sample_multi_tier_network):
        """Test identifying critical nodes in supply chain."""
        critical_nodes = supply_chain_engine.identify_critical_nodes(
            network=sample_multi_tier_network
        )

        assert critical_nodes is not None
        assert isinstance(critical_nodes, list)
        # Critical nodes are those with high degree or betweenness centrality
        if len(critical_nodes) > 0:
            assert "supplier_id" in critical_nodes[0]
            assert "criticality_score" in critical_nodes[0] or "score" in critical_nodes[0]

    def test_concentration_risk(self, supply_chain_engine, sample_tier1_suppliers):
        """Test concentration risk analysis."""
        risk = supply_chain_engine.analyze_concentration_risk(
            suppliers=sample_tier1_suppliers
        )

        assert risk is not None
        assert "concentration_score" in risk or "risk_score" in risk
        assert "geographic_concentration" in risk or "geo_concentration" in risk
        assert "supplier_concentration" in risk or "supplier_count" in risk

    def test_origin_tracing(self, supply_chain_engine):
        """Test origin tracing from product to source."""
        trace_result = supply_chain_engine.trace_origin(
            product_id="product_123",
            batch_id="batch_456"
        )

        assert trace_result is not None
        assert "origin_chain" in trace_result or "trace" in trace_result
        assert "source_location" in trace_result or "origin" in trace_result

    def test_mass_balance_tracking(self, supply_chain_engine):
        """Test mass balance tracking through supply chain."""
        # Test mass balance for a batch
        balance = supply_chain_engine.track_mass_balance(
            input_quantity=1000.0,  # kg
            output_quantity=950.0,  # kg
            product="coffee"
        )

        assert balance is not None
        assert "balanced" in balance or "is_balanced" in balance
        assert "loss_percentage" in balance or "variance" in balance

        # Calculate expected loss
        expected_loss = ((1000.0 - 950.0) / 1000.0) * 100
        if "loss_percentage" in balance:
            assert abs(float(balance["loss_percentage"]) - expected_loss) < 0.1


class TestSupplyChainRiskAnalysis:
    """Test supply chain risk analysis features."""

    def test_diversification_scoring(self, supply_chain_engine, sample_tier1_suppliers):
        """Test supply chain diversification scoring."""
        score = supply_chain_engine.calculate_diversification_score(
            suppliers=sample_tier1_suppliers
        )

        assert score is not None
        assert isinstance(score, (int, float, Decimal))
        assert 0 <= float(score) <= 100

    def test_risk_propagation(self, supply_chain_engine, sample_multi_tier_network):
        """Test risk propagation through supply chain tiers."""
        # Introduce risk at tier 3
        tier3_risk = 80.0  # High risk

        propagated_risk = supply_chain_engine.propagate_risk(
            network=sample_multi_tier_network,
            source_tier=3,
            source_risk=tier3_risk
        )

        assert propagated_risk is not None
        assert "tier_1_risk" in propagated_risk or "propagated_risks" in propagated_risk
        # Risk should propagate to upstream tiers

    def test_alternative_suppliers(self, supply_chain_engine):
        """Test identifying alternative suppliers."""
        alternatives = supply_chain_engine.find_alternative_suppliers(
            current_supplier_id="s1",
            product="coffee",
            region="South America"
        )

        assert alternatives is not None
        assert isinstance(alternatives, list)
        # Should return list of alternative suppliers
        if len(alternatives) > 0:
            assert "supplier_id" in alternatives[0]
            assert "similarity_score" in alternatives[0] or "score" in alternatives[0]

    def test_scenario_planning(self, supply_chain_engine, sample_tier1_suppliers):
        """Test scenario planning for supply chain disruptions."""
        # Test disruption scenario
        scenario = supply_chain_engine.run_scenario(
            scenario_type="supplier_loss",
            affected_supplier_id="s1",
            suppliers=sample_tier1_suppliers
        )

        assert scenario is not None
        assert "impact_score" in scenario or "impact" in scenario
        assert "mitigation_options" in scenario or "alternatives" in scenario

    def test_network_graph_generation(self, supply_chain_engine, sample_multi_tier_network):
        """Test generating network graph data."""
        graph = supply_chain_engine.generate_network_graph(
            network=sample_multi_tier_network
        )

        assert graph is not None
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0


class TestSupplyChainMetrics:
    """Test supply chain metrics calculation."""

    def test_supply_chain_length(self, supply_chain_engine, sample_multi_tier_network):
        """Test calculating supply chain length (number of tiers)."""
        length = supply_chain_engine.calculate_chain_length(
            network=sample_multi_tier_network
        )

        assert length is not None
        assert isinstance(length, int)
        assert length >= 3  # Sample network has 3 tiers

    def test_node_degree_distribution(self, supply_chain_engine, sample_multi_tier_network):
        """Test node degree distribution analysis."""
        distribution = supply_chain_engine.analyze_node_degree_distribution(
            network=sample_multi_tier_network
        )

        assert distribution is not None
        assert "mean_degree" in distribution or "average_degree" in distribution
        assert "max_degree" in distribution or "maximum_degree" in distribution

    def test_centrality_metrics(self, supply_chain_engine, sample_multi_tier_network):
        """Test centrality metrics (betweenness, closeness, etc.)."""
        centrality = supply_chain_engine.calculate_centrality_metrics(
            network=sample_multi_tier_network
        )

        assert centrality is not None
        # Should have centrality scores for each node
        if isinstance(centrality, dict):
            assert len(centrality) > 0

    def test_clustering_coefficient(self, supply_chain_engine, sample_multi_tier_network):
        """Test clustering coefficient calculation."""
        coefficient = supply_chain_engine.calculate_clustering_coefficient(
            network=sample_multi_tier_network
        )

        assert coefficient is not None
        assert isinstance(coefficient, (int, float, Decimal))
        assert 0 <= float(coefficient) <= 1


class TestSupplyChainOptimization:
    """Test supply chain optimization features."""

    def test_bottleneck_identification(self, supply_chain_engine, sample_multi_tier_network):
        """Test identifying bottlenecks in supply chain."""
        bottlenecks = supply_chain_engine.identify_bottlenecks(
            network=sample_multi_tier_network
        )

        assert bottlenecks is not None
        assert isinstance(bottlenecks, list)
        # Bottlenecks are nodes with high criticality

    def test_redundancy_analysis(self, supply_chain_engine, sample_multi_tier_network):
        """Test analyzing redundancy in supply chain."""
        redundancy = supply_chain_engine.analyze_redundancy(
            network=sample_multi_tier_network
        )

        assert redundancy is not None
        assert "redundancy_score" in redundancy or "score" in redundancy
        # Higher redundancy = lower risk

    def test_optimization_recommendations(self, supply_chain_engine, sample_tier1_suppliers):
        """Test generating optimization recommendations."""
        recommendations = supply_chain_engine.generate_optimization_recommendations(
            suppliers=sample_tier1_suppliers
        )

        assert recommendations is not None
        assert isinstance(recommendations, list)
        # Should provide actionable recommendations
        if len(recommendations) > 0:
            assert "recommendation" in recommendations[0] or "action" in recommendations[0]
            assert "priority" in recommendations[0] or "importance" in recommendations[0]


class TestSupplyChainReporting:
    """Test supply chain reporting features."""

    def test_generate_supply_chain_map_report(self, supply_chain_engine, sample_multi_tier_network):
        """Test generating supply chain map report."""
        report = supply_chain_engine.generate_map_report(
            network=sample_multi_tier_network
        )

        assert report is not None
        assert "total_tiers" in report or "tiers" in report
        assert "total_suppliers" in report or "supplier_count" in report
        assert "critical_nodes" in report or "critical_suppliers" in report

    def test_concentration_risk_report(self, supply_chain_engine, sample_tier1_suppliers):
        """Test generating concentration risk report."""
        report = supply_chain_engine.generate_concentration_report(
            suppliers=sample_tier1_suppliers
        )

        assert report is not None
        assert "geographic_concentration" in report
        assert "supplier_concentration" in report
        assert "risk_level" in report or "overall_risk" in report

    def test_diversification_report(self, supply_chain_engine, sample_tier1_suppliers):
        """Test generating diversification report."""
        report = supply_chain_engine.generate_diversification_report(
            suppliers=sample_tier1_suppliers
        )

        assert report is not None
        assert "diversification_score" in report or "score" in report
        assert "recommendations" in report or "actions" in report

    def test_mass_balance_report(self, supply_chain_engine):
        """Test generating mass balance report."""
        transactions = [
            {"input": 1000, "output": 950, "product": "coffee", "batch": "b1"},
            {"input": 500, "output": 480, "product": "coffee", "batch": "b2"},
        ]

        report = supply_chain_engine.generate_mass_balance_report(transactions)

        assert report is not None
        assert "total_input" in report
        assert "total_output" in report
        assert "overall_loss_percentage" in report or "loss_rate" in report


class TestSupplyChainVisualization:
    """Test supply chain visualization data generation."""

    def test_tier_visualization_data(self, supply_chain_engine, sample_multi_tier_network):
        """Test generating tier visualization data."""
        viz_data = supply_chain_engine.generate_tier_visualization(
            network=sample_multi_tier_network
        )

        assert viz_data is not None
        assert "tiers" in viz_data
        assert len(viz_data["tiers"]) >= 3

    def test_flow_diagram_data(self, supply_chain_engine, sample_multi_tier_network):
        """Test generating flow diagram data."""
        flow_data = supply_chain_engine.generate_flow_diagram(
            network=sample_multi_tier_network
        )

        assert flow_data is not None
        assert "nodes" in flow_data
        assert "flows" in flow_data or "edges" in flow_data

    def test_heatmap_data(self, supply_chain_engine, sample_tier1_suppliers):
        """Test generating geographic heatmap data."""
        heatmap = supply_chain_engine.generate_geographic_heatmap(
            suppliers=sample_tier1_suppliers
        )

        assert heatmap is not None
        assert "regions" in heatmap or "locations" in heatmap
        # Should include concentration data by region
