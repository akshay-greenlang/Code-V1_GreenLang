"""
Tests for Enhanced Provenance Tracker Module

Comprehensive test coverage for:
- ProvenanceNode creation and hashing
- Hash chain integrity
- Lineage graph building
- Merkle tree verification
- Model version tracking

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone
from uuid import UUID

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from audit.provenance_enhanced import (
    EnhancedProvenanceTracker,
    ProvenanceNode,
    ProvenanceEdge,
    ProvenanceNodeType,
    LineageGraph,
    ModelVersionRecord,
    HashAlgorithm,
)


class TestModelVersionRecord:
    """Tests for ModelVersionRecord model."""

    def test_create_model_version_record(self):
        """Test creating a model version record."""
        record = ModelVersionRecord(
            model_id="mdl-001",
            model_name="demand_forecast",
            model_version="2.1.0",
            model_hash="abc123def456",
            framework="sklearn",
            training_timestamp=datetime(2024, 1, 15, tzinfo=timezone.utc),
            validation_metrics={"mae": 0.05, "rmse": 0.08},
        )

        assert record.model_id == "mdl-001"
        assert record.framework == "sklearn"
        assert record.validation_metrics["mae"] == 0.05

    def test_model_version_immutable(self):
        """Test that ModelVersionRecord is immutable."""
        record = ModelVersionRecord(
            model_id="mdl-001",
            model_name="test",
            model_version="1.0.0",
            model_hash="hash123",
            framework="pytorch",
        )

        with pytest.raises(TypeError):
            record.model_name = "new_name"


class TestProvenanceNode:
    """Tests for ProvenanceNode model."""

    def test_create_provenance_node(self):
        """Test creating a provenance node."""
        node = ProvenanceNode(
            correlation_id="corr-12345",
            node_type=ProvenanceNodeType.INPUT,
            data_hash="sha256hash",
            data_size_bytes=1024,
            source_system="OPC-UA",
            source_type="sensor",
        )

        assert isinstance(node.node_id, UUID)
        assert node.node_type == ProvenanceNodeType.INPUT
        assert node.data_size_bytes == 1024

    def test_node_hash_calculation(self):
        """Test node hash is deterministic."""
        node = ProvenanceNode(
            correlation_id="corr-12345",
            node_type=ProvenanceNodeType.INPUT,
            data_hash="sha256hash",
            data_size_bytes=1024,
            source_system="OPC-UA",
            source_type="sensor",
        )

        hash1 = node.node_hash
        hash2 = node.node_hash

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_node_immutable(self):
        """Test that ProvenanceNode is immutable."""
        node = ProvenanceNode(
            correlation_id="corr-12345",
            node_type=ProvenanceNodeType.INPUT,
            data_hash="sha256hash",
            data_size_bytes=1024,
            source_system="OPC-UA",
            source_type="sensor",
        )

        with pytest.raises(TypeError):
            node.data_hash = "new_hash"


class TestEnhancedProvenanceTracker:
    """Tests for EnhancedProvenanceTracker."""

    @pytest.fixture
    def tracker(self):
        """Create fresh tracker for each test."""
        return EnhancedProvenanceTracker()

    def test_tracker_initialization(self, tracker):
        """Test tracker initializes correctly."""
        assert tracker.hash_algorithm == HashAlgorithm.SHA256
        assert len(tracker._nodes) == 0
        assert len(tracker._chain) == 0

    def test_create_input_node(self, tracker):
        """Test creating an input provenance node."""
        data = {"temperature": 450.5, "pressure": 100.2}

        node = tracker.create_input_node(
            data=data,
            correlation_id="corr-12345",
            source_system="OPC-UA",
            source_type="sensor",
            schema_version="1.0.0",
        )

        assert node.node_type == ProvenanceNodeType.INPUT
        assert node.correlation_id == "corr-12345"
        assert node.source_system == "OPC-UA"
        assert node.schema_version == "1.0.0"

        # Verify node is stored
        assert str(node.node_id) in tracker._nodes

    def test_create_transformation_node(self, tracker):
        """Test creating a transformation node."""
        # Create input first
        input_node = tracker.create_input_node(
            data={"raw_temp": 450},
            correlation_id="corr-12345",
            source_system="OPC-UA",
        )

        # Create transformation
        transformed_data = {"temp_celsius": 232.2}
        trans_node = tracker.create_transformation_node(
            data=transformed_data,
            correlation_id="corr-12345",
            operation_name="fahrenheit_to_celsius",
            operation_version="1.0.0",
            parent_node_ids=[str(input_node.node_id)],
        )

        assert trans_node.node_type == ProvenanceNodeType.TRANSFORMATION
        assert trans_node.operation_name == "fahrenheit_to_celsius"
        assert str(input_node.node_id) in trans_node.parent_node_ids

    def test_create_inference_node(self, tracker):
        """Test creating a model inference node."""
        model_version = ModelVersionRecord(
            model_id="mdl-001",
            model_name="demand_forecast",
            model_version="2.1.0",
            model_hash="abc123",
            framework="sklearn",
        )

        input_node = tracker.create_input_node(
            data={"features": [1, 2, 3]},
            correlation_id="corr-12345",
            source_system="feature_store",
        )

        inference_node = tracker.create_inference_node(
            data={"prediction": 1500.0},
            correlation_id="corr-12345",
            model_version=model_version,
            parent_node_ids=[str(input_node.node_id)],
        )

        assert inference_node.node_type == ProvenanceNodeType.MODEL_INFERENCE
        assert inference_node.model_version_record.model_name == "demand_forecast"

    def test_create_calculation_node(self, tracker):
        """Test creating a calculation node (zero-hallucination)."""
        input_node = tracker.create_input_node(
            data={"activity_data": 100, "emission_factor": 0.5},
            correlation_id="corr-12345",
            source_system="erp",
        )

        calc_node = tracker.create_calculation_node(
            data={"emissions": 50.0},
            correlation_id="corr-12345",
            calculation_name="emissions_calculation",
            calculation_version="1.0.0",
            parent_node_ids=[str(input_node.node_id)],
            formula_hash="formula_sha256",
        )

        assert calc_node.node_type == ProvenanceNodeType.CALCULATION
        assert calc_node.metadata.get("formula_hash") == "formula_sha256"

    def test_hash_chain_linking(self, tracker):
        """Test that nodes are linked in hash chain."""
        node1 = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        node2 = tracker.create_input_node(
            data={"temp": 101},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        # Second node should reference first node's hash
        assert node2.previous_node_hash == node1.node_hash

    def test_verify_chain_valid(self, tracker):
        """Test chain verification passes for valid chain."""
        # Create several nodes
        for i in range(5):
            tracker.create_input_node(
                data={"value": i},
                correlation_id="corr-12345",
                source_system="sensor",
            )

        is_valid, error = tracker.verify_chain()
        assert is_valid is True
        assert error is None

    def test_link_nodes(self, tracker):
        """Test linking nodes with edges."""
        node1 = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        node2 = tracker.create_output_node(
            data={"recommendation": "increase"},
            correlation_id="corr-12345",
            parent_node_ids=[],
        )

        edge = tracker.link_nodes(
            str(node1.node_id),
            str(node2.node_id),
            edge_type="influences",
        )

        assert edge.source_node_id == str(node1.node_id)
        assert edge.target_node_id == str(node2.node_id)

    def test_link_nodes_invalid_source(self, tracker):
        """Test linking with invalid source raises error."""
        node = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        with pytest.raises(ValueError, match="Source node not found"):
            tracker.link_nodes("invalid-id", str(node.node_id))

    def test_get_node(self, tracker):
        """Test retrieving node by ID."""
        node = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        retrieved = tracker.get_node(str(node.node_id))
        assert retrieved.node_id == node.node_id

    def test_get_node_not_found(self, tracker):
        """Test getting non-existent node returns None."""
        result = tracker.get_node("non-existent-id")
        assert result is None

    def test_get_nodes_by_correlation(self, tracker):
        """Test retrieving nodes by correlation ID."""
        # Create nodes with same correlation
        for i in range(3):
            tracker.create_input_node(
                data={"value": i},
                correlation_id="corr-12345",
                source_system="sensor",
            )

        # Create node with different correlation
        tracker.create_input_node(
            data={"value": 999},
            correlation_id="corr-other",
            source_system="sensor",
        )

        nodes = tracker.get_nodes_by_correlation("corr-12345")
        assert len(nodes) == 3

    def test_get_parent_nodes(self, tracker):
        """Test getting parent nodes."""
        input1 = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        input2 = tracker.create_input_node(
            data={"pressure": 50},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        output = tracker.create_output_node(
            data={"result": "optimal"},
            correlation_id="corr-12345",
            parent_node_ids=[str(input1.node_id), str(input2.node_id)],
        )

        parents = tracker.get_parent_nodes(str(output.node_id))
        assert len(parents) == 2

    def test_get_child_nodes(self, tracker):
        """Test getting child nodes."""
        input_node = tracker.create_input_node(
            data={"temp": 100},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        # Create multiple children
        for i in range(3):
            tracker.create_transformation_node(
                data={"transformed": i},
                correlation_id="corr-12345",
                operation_name=f"transform_{i}",
                operation_version="1.0.0",
                parent_node_ids=[str(input_node.node_id)],
            )

        children = tracker.get_child_nodes(str(input_node.node_id))
        assert len(children) == 3


class TestLineageGraph:
    """Tests for lineage graph building."""

    @pytest.fixture
    def tracker_with_graph(self):
        """Create tracker with sample lineage graph."""
        tracker = EnhancedProvenanceTracker()

        # Create input nodes
        input1 = tracker.create_input_node(
            data={"sensor_a": 100},
            correlation_id="corr-12345",
            source_system="OPC-UA",
        )

        input2 = tracker.create_input_node(
            data={"sensor_b": 200},
            correlation_id="corr-12345",
            source_system="OPC-UA",
        )

        # Create transformation
        transform = tracker.create_transformation_node(
            data={"combined": 300},
            correlation_id="corr-12345",
            operation_name="aggregate",
            operation_version="1.0.0",
            parent_node_ids=[str(input1.node_id), str(input2.node_id)],
        )

        # Create output
        output = tracker.create_output_node(
            data={"recommendation": "increase"},
            correlation_id="corr-12345",
            parent_node_ids=[str(transform.node_id)],
        )

        return tracker, output

    def test_build_lineage_graph(self, tracker_with_graph):
        """Test building lineage graph."""
        tracker, output = tracker_with_graph

        graph = tracker.build_lineage_graph(
            correlation_id="corr-12345",
            root_node_id=str(output.node_id),
        )

        assert isinstance(graph, LineageGraph)
        assert len(graph.nodes) == 4  # 2 inputs + 1 transform + 1 output
        assert len(graph.input_node_ids) == 2
        assert len(graph.output_node_ids) == 1

    def test_lineage_graph_merkle_root(self, tracker_with_graph):
        """Test Merkle root is calculated."""
        tracker, output = tracker_with_graph

        graph = tracker.build_lineage_graph(
            correlation_id="corr-12345",
            root_node_id=str(output.node_id),
        )

        assert graph.merkle_root is not None
        assert len(graph.merkle_root) == 64  # SHA-256

    def test_verify_lineage(self, tracker_with_graph):
        """Test lineage verification."""
        tracker, output = tracker_with_graph

        graph = tracker.build_lineage_graph(
            correlation_id="corr-12345",
            root_node_id=str(output.node_id),
        )

        is_valid, error = tracker.verify_lineage(graph)
        assert is_valid is True
        assert error is None

    def test_lineage_graph_hash(self, tracker_with_graph):
        """Test graph hash is deterministic."""
        tracker, output = tracker_with_graph

        graph = tracker.build_lineage_graph(
            correlation_id="corr-12345",
            root_node_id=str(output.node_id),
        )

        hash1 = graph.graph_hash
        hash2 = graph.graph_hash

        assert hash1 == hash2


class TestModelVersionTracking:
    """Tests for model version tracking."""

    @pytest.fixture
    def tracker(self):
        return EnhancedProvenanceTracker()

    def test_register_model_version(self, tracker):
        """Test registering a model version."""
        model = ModelVersionRecord(
            model_id="mdl-001",
            model_name="demand_forecast",
            model_version="2.1.0",
            model_hash="abc123",
            framework="sklearn",
        )

        tracker.register_model_version(model)

        retrieved = tracker.get_model_version("mdl-001")
        assert retrieved.model_name == "demand_forecast"

    def test_get_all_model_versions(self, tracker):
        """Test getting all model versions."""
        models = [
            ModelVersionRecord(
                model_id=f"mdl-{i}",
                model_name=f"model_{i}",
                model_version="1.0.0",
                model_hash=f"hash_{i}",
                framework="sklearn",
            )
            for i in range(3)
        ]

        for model in models:
            tracker.register_model_version(model)

        all_models = tracker.get_all_model_versions()
        assert len(all_models) == 3


class TestHashAlgorithms:
    """Tests for different hash algorithms."""

    def test_sha256_algorithm(self):
        """Test SHA-256 hashing."""
        tracker = EnhancedProvenanceTracker(hash_algorithm=HashAlgorithm.SHA256)

        hash_result = tracker._compute_hash({"test": "data"})
        assert len(hash_result) == 64

    def test_sha384_algorithm(self):
        """Test SHA-384 hashing."""
        tracker = EnhancedProvenanceTracker(hash_algorithm=HashAlgorithm.SHA384)

        hash_result = tracker._compute_hash({"test": "data"})
        assert len(hash_result) == 96

    def test_sha512_algorithm(self):
        """Test SHA-512 hashing."""
        tracker = EnhancedProvenanceTracker(hash_algorithm=HashAlgorithm.SHA512)

        hash_result = tracker._compute_hash({"test": "data"})
        assert len(hash_result) == 128


class TestTrackerStatistics:
    """Tests for tracker statistics."""

    def test_get_statistics(self):
        """Test getting tracker statistics."""
        tracker = EnhancedProvenanceTracker()

        # Create various nodes
        for i in range(3):
            tracker.create_input_node(
                data={"value": i},
                correlation_id="corr-12345",
                source_system="sensor",
            )

        tracker.create_output_node(
            data={"result": "ok"},
            correlation_id="corr-12345",
            parent_node_ids=[],
        )

        stats = tracker.get_statistics()

        assert stats["total_nodes"] == 4
        assert stats["nodes_by_type"]["INPUT"] == 3
        assert stats["nodes_by_type"]["OUTPUT"] == 1

    def test_export_chain(self):
        """Test exporting hash chain."""
        tracker = EnhancedProvenanceTracker()

        for i in range(3):
            tracker.create_input_node(
                data={"value": i},
                correlation_id="corr-12345",
                source_system="sensor",
            )

        chain = tracker.export_chain()

        assert len(chain) == 3
        assert "node_id" in chain[0]
        assert "node_hash" in chain[0]
        assert "previous_hash" in chain[0]

    def test_clear_tracker(self):
        """Test clearing tracker."""
        tracker = EnhancedProvenanceTracker()

        tracker.create_input_node(
            data={"value": 1},
            correlation_id="corr-12345",
            source_system="sensor",
        )

        tracker.clear()

        assert len(tracker._nodes) == 0
        assert len(tracker._chain) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
