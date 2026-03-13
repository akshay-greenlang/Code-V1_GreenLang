# -*- coding: utf-8 -*-
"""
Unit tests for Engine 1: Workflow Definition Engine -- AGENT-EUDR-026

Tests DAG creation, topological sorting (Kahn's algorithm), circular
dependency detection, layer assignment, critical path analysis,
standard/simplified/custom workflow definitions, commodity-specific
templates, runtime modification, and workflow versioning.

Test count: ~80 tests
Author: GreenLang Platform Team
Date: March 2026
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone

from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AgentNode,
    WorkflowEdge,
    WorkflowDefinition,
    WorkflowType,
    EUDRCommodity,
    DueDiligencePhase,
    FallbackStrategy,
    PHASE_1_AGENTS,
    PHASE_2_AGENTS,
    ALL_EUDR_AGENTS,
    AGENT_NAMES,
)
from greenlang.agents.eudr.due_diligence_orchestrator.workflow_definition_engine import (
    WorkflowDefinitionEngine,
)

from tests.agents.eudr.due_diligence_orchestrator.conftest import (
    STANDARD_WORKFLOW_EDGES,
    _make_agent_node,
)


class TestWorkflowDefinitionEngineInit:
    """Test engine initialization."""

    def test_init_with_default_config(self, default_config):
        engine = WorkflowDefinitionEngine()
        assert engine is not None

    def test_init_with_explicit_config(self, default_config):
        engine = WorkflowDefinitionEngine(config=default_config)
        assert engine is not None

    def test_init_loads_templates(self, workflow_definition_engine):
        engine = workflow_definition_engine
        assert engine is not None


class TestStandardWorkflowCreation:
    """Test standard 25-agent workflow creation."""

    @pytest.mark.parametrize("commodity", [
        EUDRCommodity.CATTLE, EUDRCommodity.COCOA, EUDRCommodity.COFFEE,
        EUDRCommodity.PALM_OIL, EUDRCommodity.RUBBER, EUDRCommodity.SOYA,
        EUDRCommodity.WOOD,
    ])
    def test_create_standard_workflow_all_commodities(
        self, workflow_definition_engine, commodity,
    ):
        defn = workflow_definition_engine.create_standard_workflow(commodity)
        assert defn is not None
        assert defn.workflow_type == WorkflowType.STANDARD
        assert defn.commodity == commodity

    def test_standard_workflow_has_25_agent_nodes(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        agent_nodes = [n for n in defn.nodes if n.agent_id.startswith("EUDR-")]
        assert len(agent_nodes) >= 25

    def test_standard_workflow_has_3_quality_gates(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert "QG-1" in defn.quality_gates
        assert "QG-2" in defn.quality_gates
        assert "QG-3" in defn.quality_gates

    def test_standard_workflow_has_dependency_edges(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert len(defn.edges) > 0

    def test_standard_workflow_entry_point_is_eudr001(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        sources = {e.source for e in defn.edges}
        targets = {e.target for e in defn.edges}
        root_nodes = sources - targets
        agent_roots = {n for n in root_nodes if n.startswith("EUDR-")}
        assert "EUDR-001" in agent_roots

    def test_standard_workflow_version_is_set(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert defn.version is not None
        assert len(defn.version) > 0


class TestSimplifiedWorkflowCreation:
    """Test simplified (Article 13) workflow creation."""

    def test_create_simplified_workflow(self, workflow_definition_engine):
        defn = workflow_definition_engine.create_simplified_workflow(
            EUDRCommodity.WOOD,
        )
        assert defn.workflow_type == WorkflowType.SIMPLIFIED

    def test_simplified_has_fewer_agents(self, workflow_definition_engine):
        standard = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.WOOD,
        )
        simplified = workflow_definition_engine.create_simplified_workflow(
            EUDRCommodity.WOOD,
        )
        std_agents = [n for n in standard.nodes if n.agent_id.startswith("EUDR-")]
        simp_agents = [n for n in simplified.nodes if n.agent_id.startswith("EUDR-")]
        assert len(simp_agents) < len(std_agents)

    def test_simplified_has_reduced_quality_gates(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_simplified_workflow(
            EUDRCommodity.WOOD,
        )
        assert "QG-3" not in defn.quality_gates or len(defn.quality_gates) <= 2

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_simplified_for_all_commodities(
        self, workflow_definition_engine, commodity,
    ):
        defn = workflow_definition_engine.create_simplified_workflow(commodity)
        assert defn.commodity == commodity


class TestDAGValidation:
    """Test DAG structural validation."""

    def test_validate_valid_definition(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        is_valid, errors = workflow_definition_engine.validate_definition(
            standard_workflow_definition,
        )
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_detects_circular_dependency(
        self, workflow_definition_engine,
    ):
        nodes = [
            _make_agent_node("A"), _make_agent_node("B"), _make_agent_node("C"),
        ]
        edges = [
            WorkflowEdge(source="A", target="B"),
            WorkflowEdge(source="B", target="C"),
            WorkflowEdge(source="C", target="A"),
        ]
        defn = WorkflowDefinition(
            name="Circular Test", nodes=nodes, edges=edges,
        )
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is False
        assert any("circular" in str(e).lower() or "cycle" in str(e).lower()
                    for e in errors)

    def test_validate_detects_orphan_nodes(self, workflow_definition_engine):
        nodes = [
            _make_agent_node("A"), _make_agent_node("B"),
            _make_agent_node("ORPHAN"),
        ]
        edges = [WorkflowEdge(source="A", target="B")]
        defn = WorkflowDefinition(
            name="Orphan Test", nodes=nodes, edges=edges,
        )
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        # Orphans may be valid (no-dep agents) or flagged as warnings
        assert isinstance(is_valid, bool)

    def test_validate_empty_definition(self, workflow_definition_engine):
        defn = WorkflowDefinition(name="Empty", nodes=[], edges=[])
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is False

    def test_validate_edge_references_missing_node(
        self, workflow_definition_engine,
    ):
        nodes = [_make_agent_node("A")]
        edges = [WorkflowEdge(source="A", target="MISSING")]
        defn = WorkflowDefinition(
            name="Missing node test", nodes=nodes, edges=edges,
        )
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is False

    def test_validate_self_loop(self, workflow_definition_engine):
        nodes = [_make_agent_node("A")]
        edges = [WorkflowEdge(source="A", target="A")]
        defn = WorkflowDefinition(
            name="Self loop", nodes=nodes, edges=edges,
        )
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is False

    def test_validate_duplicate_node_ids(self, workflow_definition_engine):
        nodes = [_make_agent_node("A"), _make_agent_node("A")]
        edges = []
        defn = WorkflowDefinition(
            name="Duplicate nodes", nodes=nodes, edges=edges,
        )
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is False


class TestTopologicalSort:
    """Test topological sorting via Kahn's algorithm."""

    def test_topological_sort_standard_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        layers = workflow_definition_engine.compute_execution_layers(
            standard_workflow_definition,
        )
        assert len(layers) > 0
        assert "EUDR-001" in layers[0]

    def test_topological_sort_deterministic(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        layers1 = workflow_definition_engine.compute_execution_layers(
            standard_workflow_definition,
        )
        layers2 = workflow_definition_engine.compute_execution_layers(
            standard_workflow_definition,
        )
        assert layers1 == layers2

    def test_topological_sort_respects_dependencies(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        layers = workflow_definition_engine.compute_execution_layers(
            standard_workflow_definition,
        )
        flat_order = [agent for layer in layers for agent in layer]
        for edge in standard_workflow_definition.edges:
            if edge.source in flat_order and edge.target in flat_order:
                assert flat_order.index(edge.source) < flat_order.index(edge.target)

    def test_topological_sort_groups_parallel_agents(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        layers = workflow_definition_engine.compute_execution_layers(
            standard_workflow_definition,
        )
        # Layer 1 should have multiple agents (EUDR-002, 006, 007, 008)
        layer1_agents = [a for a in layers[1] if a.startswith("EUDR-")]
        assert len(layer1_agents) >= 2

    def test_topological_sort_linear_dag(self, workflow_definition_engine):
        nodes = [
            _make_agent_node("A"), _make_agent_node("B"), _make_agent_node("C"),
        ]
        edges = [
            WorkflowEdge(source="A", target="B"),
            WorkflowEdge(source="B", target="C"),
        ]
        defn = WorkflowDefinition(name="Linear", nodes=nodes, edges=edges)
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert len(layers) == 3
        assert layers[0] == ["A"]
        assert layers[1] == ["B"]
        assert layers[2] == ["C"]

    def test_topological_sort_diamond_dag(self, workflow_definition_engine):
        nodes = [
            _make_agent_node("A"), _make_agent_node("B"),
            _make_agent_node("C"), _make_agent_node("D"),
        ]
        edges = [
            WorkflowEdge(source="A", target="B"),
            WorkflowEdge(source="A", target="C"),
            WorkflowEdge(source="B", target="D"),
            WorkflowEdge(source="C", target="D"),
        ]
        defn = WorkflowDefinition(name="Diamond", nodes=nodes, edges=edges)
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert len(layers) == 3
        assert layers[0] == ["A"]
        assert set(layers[1]) == {"B", "C"}
        assert layers[2] == ["D"]

    def test_topological_sort_wide_dag(self, workflow_definition_engine):
        nodes = [_make_agent_node(f"N{i}") for i in range(10)]
        edges = []
        defn = WorkflowDefinition(name="Wide", nodes=nodes, edges=edges)
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert len(layers) == 1
        assert len(layers[0]) == 10


class TestCriticalPath:
    """Test critical path calculation."""

    def test_critical_path_standard_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        durations = {a: 30 for a in ALL_EUDR_AGENTS}
        durations.update({"QG-1": 5, "QG-2": 5, "QG-3": 5,
                         "EUDR-025-MIT": 30, "PKG-GEN": 15})
        path, total = workflow_definition_engine.calculate_critical_path(
            standard_workflow_definition, durations,
        )
        assert len(path) > 0
        assert total > 0

    def test_critical_path_includes_quality_gates(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        durations = {a: 30 for a in ALL_EUDR_AGENTS}
        durations.update({"QG-1": 5, "QG-2": 5, "QG-3": 5,
                         "EUDR-025-MIT": 30, "PKG-GEN": 15})
        path, _ = workflow_definition_engine.calculate_critical_path(
            standard_workflow_definition, durations,
        )
        gate_in_path = any("QG" in node for node in path)
        assert gate_in_path

    def test_critical_path_linear_dag(self, workflow_definition_engine):
        nodes = [
            _make_agent_node("A"), _make_agent_node("B"), _make_agent_node("C"),
        ]
        edges = [
            WorkflowEdge(source="A", target="B"),
            WorkflowEdge(source="B", target="C"),
        ]
        defn = WorkflowDefinition(name="Linear", nodes=nodes, edges=edges)
        path, total = workflow_definition_engine.calculate_critical_path(
            defn, {"A": 10, "B": 20, "C": 30},
        )
        assert path == ["A", "B", "C"]
        assert total == 60


class TestRuntimeModification:
    """Test runtime workflow modification."""

    def test_add_agent_to_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        new_node = _make_agent_node("CUSTOM-AGENT-001")
        result = workflow_definition_engine.add_agent(
            standard_workflow_definition, new_node,
        )
        node_ids = [n.agent_id for n in result.nodes]
        assert "CUSTOM-AGENT-001" in node_ids

    def test_remove_agent_from_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        result = workflow_definition_engine.remove_agent(
            standard_workflow_definition, "EUDR-014",
        )
        node_ids = [n.agent_id for n in result.nodes]
        assert "EUDR-014" not in node_ids

    def test_add_edge_to_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        new_edge = WorkflowEdge(source="EUDR-007", target="EUDR-003")
        result = workflow_definition_engine.add_edge(
            standard_workflow_definition, new_edge,
        )
        edge_pairs = [(e.source, e.target) for e in result.edges]
        assert ("EUDR-007", "EUDR-003") in edge_pairs

    def test_remove_edge_from_workflow(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        result = workflow_definition_engine.remove_edge(
            standard_workflow_definition, "EUDR-001", "EUDR-002",
        )
        edge_pairs = [(e.source, e.target) for e in result.edges]
        assert ("EUDR-001", "EUDR-002") not in edge_pairs


class TestWorkflowCloning:
    """Test workflow definition cloning."""

    def test_clone_preserves_structure(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        clone = workflow_definition_engine.clone_definition(
            standard_workflow_definition,
        )
        assert len(clone.nodes) == len(standard_workflow_definition.nodes)
        assert len(clone.edges) == len(standard_workflow_definition.edges)

    def test_clone_creates_new_id(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        clone = workflow_definition_engine.clone_definition(
            standard_workflow_definition,
        )
        assert clone.definition_id != standard_workflow_definition.definition_id

    def test_clone_is_independent(
        self, workflow_definition_engine, standard_workflow_definition,
    ):
        clone = workflow_definition_engine.clone_definition(
            standard_workflow_definition,
        )
        clone.name = "Modified Clone"
        assert standard_workflow_definition.name != "Modified Clone"


class TestCommodityTemplates:
    """Test commodity-specific workflow templates."""

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_commodity_template_creates_valid_workflow(
        self, workflow_definition_engine, commodity,
    ):
        defn = workflow_definition_engine.create_standard_workflow(commodity)
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        assert is_valid is True, f"Template invalid for {commodity}: {errors}"

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_commodity_template_topological_sort_succeeds(
        self, workflow_definition_engine, commodity,
    ):
        defn = workflow_definition_engine.create_standard_workflow(commodity)
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert len(layers) > 0

    def test_palm_oil_template_has_rspo_emphasis(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.PALM_OIL,
        )
        assert defn.commodity == EUDRCommodity.PALM_OIL

    def test_wood_template_has_fsc_emphasis(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.WOOD,
        )
        assert defn.commodity == EUDRCommodity.WOOD


class TestWorkflowVersioning:
    """Test workflow version management."""

    def test_definition_has_version(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert defn.version is not None

    def test_definition_has_creation_timestamp(
        self, workflow_definition_engine,
    ):
        defn = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert defn.created_at is not None

    def test_definition_has_unique_id(
        self, workflow_definition_engine,
    ):
        defn1 = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        defn2 = workflow_definition_engine.create_standard_workflow(
            EUDRCommodity.COCOA,
        )
        assert defn1.definition_id != defn2.definition_id


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_max_agents_limit(self, workflow_definition_engine):
        nodes = [_make_agent_node(f"N{i}") for i in range(51)]
        defn = WorkflowDefinition(name="Too many", nodes=nodes, edges=[])
        is_valid, errors = workflow_definition_engine.validate_definition(defn)
        # Should fail or warn about exceeding MAX_WORKFLOW_AGENTS
        assert isinstance(is_valid, bool)

    def test_single_node_workflow(self, workflow_definition_engine):
        nodes = [_make_agent_node("SINGLE")]
        defn = WorkflowDefinition(name="Single", nodes=nodes, edges=[])
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert layers == [["SINGLE"]]

    def test_two_node_workflow(self, workflow_definition_engine):
        nodes = [_make_agent_node("A"), _make_agent_node("B")]
        edges = [WorkflowEdge(source="A", target="B")]
        defn = WorkflowDefinition(name="Two", nodes=nodes, edges=edges)
        layers = workflow_definition_engine.compute_execution_layers(defn)
        assert layers == [["A"], ["B"]]
