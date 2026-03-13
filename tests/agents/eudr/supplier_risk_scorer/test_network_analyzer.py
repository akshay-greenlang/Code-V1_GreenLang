# -*- coding: utf-8 -*-
"""
Unit tests for NetworkAnalyzer - AGENT-EUDR-017 Engine 6

Tests supplier-to-supplier relationship mapping and risk propagation with
supply chain depth tracking, circular dependency detection, shared supplier
detection, and network centrality analysis.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from decimal import Decimal
import pytest

from greenlang.agents.eudr.supplier_risk_scorer.network_analyzer import (
    NetworkAnalyzer,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    SupplierType,
    RiskLevel,
)


class TestNetworkAnalyzerInit:
    """Tests for NetworkAnalyzer initialization."""

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        analyzer = NetworkAnalyzer()
        assert analyzer._networks == {}


class TestAnalyzeNetwork:
    """Tests for analyze_network method."""

    @pytest.mark.unit
    def test_analyze_network_returns_result(
        self, network_analyzer, sample_network
    ):
        result = network_analyzer.analyze_network(
            supplier_id=sample_network["supplier_id"],
            sub_suppliers=sample_network["sub_suppliers"],
        )
        assert result is not None
        assert "network_id" in result
        assert result["supplier_id"] == sample_network["supplier_id"]


class TestMapRelationships:
    """Tests for relationship mapping."""

    @pytest.mark.unit
    def test_map_relationships_creates_graph(self, network_analyzer):
        relationships = [
            {"from": "SUPP-A", "to": "SUPP-B", "type": "supplies"},
            {"from": "SUPP-B", "to": "SUPP-C", "type": "supplies"},
        ]
        graph = network_analyzer.map_relationships(relationships)
        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2


class TestDetectCycles:
    """Tests for circular dependency detection."""

    @pytest.mark.unit
    def test_detect_cycles_finds_circular(self, network_analyzer):
        # A -> B -> C -> A (circular)
        relationships = [
            {"from": "SUPP-A", "to": "SUPP-B"},
            {"from": "SUPP-B", "to": "SUPP-C"},
            {"from": "SUPP-C", "to": "SUPP-A"},
        ]
        cycles = network_analyzer.detect_cycles(relationships)
        assert len(cycles) > 0

    @pytest.mark.unit
    def test_detect_cycles_no_circular(self, network_analyzer):
        # A -> B -> C (linear, no cycle)
        relationships = [
            {"from": "SUPP-A", "to": "SUPP-B"},
            {"from": "SUPP-B", "to": "SUPP-C"},
        ]
        cycles = network_analyzer.detect_cycles(relationships)
        assert len(cycles) == 0


class TestRiskPropagation:
    """Tests for risk propagation modeling."""

    @pytest.mark.unit
    def test_propagate_risk_with_decay(
        self, network_analyzer, mock_config
    ):
        sub_supplier_risk = Decimal("80.0")
        tier = 2
        propagated = network_analyzer.propagate_risk(
            sub_supplier_risk,
            tier,
            decay_factor=mock_config.risk_propagation_decay,
        )
        # Risk should be reduced by decay factor
        assert propagated < sub_supplier_risk


class TestSharedSuppliers:
    """Tests for shared supplier detection."""

    @pytest.mark.unit
    def test_detect_shared_suppliers(self, network_analyzer):
        networks = {
            "IMPORTER-A": [
                {"supplier_id": "SUPP-1"},
                {"supplier_id": "SUPP-2"},
            ],
            "IMPORTER-B": [
                {"supplier_id": "SUPP-2"},  # Shared
                {"supplier_id": "SUPP-3"},
            ],
        }
        shared = network_analyzer.detect_shared_suppliers(networks)
        assert "SUPP-2" in [s["supplier_id"] for s in shared]


class TestCentrality:
    """Tests for network centrality analysis."""

    @pytest.mark.unit
    def test_calculate_centrality(self, network_analyzer):
        # Hub node with many connections
        graph = {
            "nodes": ["SUPP-HUB", "SUPP-A", "SUPP-B", "SUPP-C"],
            "edges": [
                {"from": "SUPP-HUB", "to": "SUPP-A"},
                {"from": "SUPP-HUB", "to": "SUPP-B"},
                {"from": "SUPP-HUB", "to": "SUPP-C"},
            ],
        }
        centrality = network_analyzer.calculate_centrality(graph)
        # Hub should have highest centrality
        assert centrality["SUPP-HUB"] > centrality.get("SUPP-A", 0)


class TestClustering:
    """Tests for clustering coefficient."""

    @pytest.mark.unit
    def test_calculate_clustering_coefficient(self, network_analyzer):
        graph = {
            "nodes": ["A", "B", "C", "D"],
            "edges": [
                {"from": "A", "to": "B"},
                {"from": "B", "to": "C"},
                {"from": "C", "to": "A"},  # Forms triangle
            ],
        }
        clustering = network_analyzer.calculate_clustering_coefficient(graph)
        assert Decimal("0.0") <= clustering <= Decimal("1.0")


class TestRoutingAnalysis:
    """Tests for routing analysis."""

    @pytest.mark.unit
    def test_analyze_routing_paths(self, network_analyzer):
        result = network_analyzer.analyze_routing(
            origin="BR",
            destination="NL",
            intermediaries=["PA", "DE"],
        )
        assert "route" in result
        assert len(result["route"]) >= 2


class TestIntermediaryRisk:
    """Tests for intermediary risk amplification."""

    @pytest.mark.unit
    def test_calculate_intermediary_risk(self, network_analyzer):
        intermediaries = [
            {"country": "PA", "type": SupplierType.BROKER},  # High-risk country
            {"country": "CH", "type": SupplierType.TRADER},
        ]
        risk = network_analyzer.calculate_intermediary_risk(intermediaries)
        assert "amplification_factor" in risk
        assert risk["amplification_factor"] >= Decimal("1.0")


class TestUltimateSource:
    """Tests for ultimate source tracing."""

    @pytest.mark.unit
    def test_trace_ultimate_source(self, network_analyzer):
        chain = [
            {"supplier_id": "IMPORTER", "tier": 1, "type": SupplierType.IMPORTER},
            {"supplier_id": "TRADER", "tier": 2, "type": SupplierType.TRADER},
            {"supplier_id": "PRODUCER", "tier": 3, "type": SupplierType.PRODUCER},
        ]
        ultimate = network_analyzer.trace_ultimate_source(chain)
        assert ultimate["supplier_id"] == "PRODUCER"
        assert ultimate["type"] == SupplierType.PRODUCER


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_analysis_includes_provenance_hash(
        self, network_analyzer, sample_network
    ):
        result = network_analyzer.analyze_network(
            supplier_id=sample_network["supplier_id"],
            sub_suppliers=sample_network["sub_suppliers"],
        )
        assert "provenance_hash" in result


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_supplier_id_raises_error(self, network_analyzer):
        with pytest.raises(ValueError):
            network_analyzer.analyze_network(
                supplier_id="",
                sub_suppliers=[],
            )
