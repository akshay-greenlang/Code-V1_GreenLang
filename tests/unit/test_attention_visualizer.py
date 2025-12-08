# -*- coding: utf-8 -*-
"""
Unit Tests for Attention Visualizer Module
============================================

Tests for transformer attention visualization and analysis functionality.

Test Coverage:
  - AttentionWeights validation and shape handling
  - AttentionSummary generation and statistics
  - Feature importance extraction
  - Visualization generation (heatmap, etc.)
  - Export formats (HTML, JSON, CSV, PNG)
  - Model type detection
  - Error handling and edge cases
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from greenlang.ml.explainability.attention_visualizer import (
    AttentionSummary,
    AttentionVisualizer,
    AttentionWeights,
    ExportFormat,
    VisualizationType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_transformer_model():
    """Create a mock transformer model."""
    model = MagicMock()
    model.__class__.__name__ = "BertModel"
    return model


@pytest.fixture
def sample_attention_weights():
    """Create sample attention weights for testing."""
    # Shape: (num_layers=2, num_heads=4, seq_len=8, seq_len=8)
    np.random.seed(42)
    # Create properly normalized attention weights (sum to 1.0 along last dim)
    weights = np.zeros((2, 4, 8, 8), dtype=np.float32)
    for i in range(2):
        for j in range(4):
            for k in range(8):
                weights[i, j, k, :] = np.random.dirichlet(np.ones(8))
    return weights


@pytest.fixture
def sample_feature_names():
    """Create sample feature names for sensors."""
    return [
        "temperature",
        "pressure",
        "flow_rate",
        "oxygen_level",
        "fuel_flow",
        "stack_temp",
        "efficiency",
        "emissions"
    ]


@pytest.fixture
def attention_weights_object(sample_attention_weights, sample_feature_names):
    """Create AttentionWeights object for testing."""
    return AttentionWeights(
        weights=sample_attention_weights,
        feature_names=sample_feature_names,
        model_name="test_model",
        input_shape=(1, 8, 768)
    )


# =============================================================================
# TESTS: AttentionWeights CLASS
# =============================================================================


class TestAttentionWeights:
    """Tests for AttentionWeights data model."""

    def test_init_valid_weights(self, sample_attention_weights, sample_feature_names):
        """Test initialization with valid weights."""
        attn = AttentionWeights(
            weights=sample_attention_weights,
            feature_names=sample_feature_names
        )
        assert attn.weights.shape == (2, 4, 8, 8)
        assert attn.feature_names == sample_feature_names
        assert attn.model_name == "unknown"

    def test_get_shape(self, attention_weights_object):
        """Test get_shape method."""
        shape = attention_weights_object.get_shape()
        assert shape == (2, 4, 8, 8)
        assert len(shape) == 4

    def test_validate_valid_weights(self, attention_weights_object):
        """Test validation of properly normalized weights."""
        assert attention_weights_object.validate() is True

    def test_validate_invalid_weights_empty(self):
        """Test validation rejects empty weights."""
        attn = AttentionWeights(weights=np.array([]))
        assert attn.validate() is False

    def test_validate_invalid_shape(self):
        """Test validation rejects wrong shape."""
        # Wrong shape: only 3D instead of 4D
        weights = np.random.rand(2, 4, 8)
        attn = AttentionWeights(weights=weights)
        assert attn.validate() is False

    def test_validate_unnormalized_weights(self):
        """Test validation of unnormalized weights."""
        weights = np.random.rand(2, 4, 8, 8)  # Not normalized to 1.0
        attn = AttentionWeights(weights=weights)
        assert attn.validate() is False

    def test_timestamp_default(self, attention_weights_object):
        """Test timestamp is set by default."""
        assert attention_weights_object.timestamp is not None

    def test_input_shape_metadata(self, attention_weights_object):
        """Test input shape metadata."""
        assert attention_weights_object.input_shape == (1, 8, 768)


# =============================================================================
# TESTS: AttentionSummary CLASS
# =============================================================================


class TestAttentionSummary:
    """Tests for AttentionSummary data model."""

    def test_init_empty(self):
        """Test initialization with empty data."""
        summary = AttentionSummary()
        assert summary.aggregated_weights.size == 0
        assert summary.feature_importance.size == 0
        assert len(summary.top_attended_features) == 0
        assert summary.processing_time_ms == 0.0

    def test_calculate_provenance(self):
        """Test provenance hash calculation."""
        summary = AttentionSummary(
            aggregated_weights=np.random.rand(2, 8, 8),
            feature_importance=np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.1, 0.1, 0.05])
        )
        hash1 = summary.calculate_provenance()
        assert len(hash1) == 64  # SHA-256 hex digest
        assert hash1.isalnum()

        # Different data should produce different hash
        summary2 = AttentionSummary(
            aggregated_weights=np.random.rand(2, 8, 8),
            feature_importance=np.array([0.2, 0.1, 0.2, 0.2, 0.1, 0.05, 0.1, 0.05])
        )
        hash2 = summary2.calculate_provenance()
        assert hash1 != hash2

    def test_layer_importance_storage(self):
        """Test layer importance storage."""
        layer_imp = np.array([0.5, 0.5])
        summary = AttentionSummary(layer_importance=layer_imp)
        np.testing.assert_array_equal(summary.layer_importance, layer_imp)

    def test_head_importance_dict(self):
        """Test head importance per-layer dictionary."""
        head_imp = {
            0: np.array([0.25, 0.25, 0.25, 0.25]),
            1: np.array([0.2, 0.3, 0.25, 0.25])
        }
        summary = AttentionSummary(head_importance=head_imp)
        assert len(summary.head_importance) == 2
        np.testing.assert_array_equal(summary.head_importance[0], head_imp[0])


# =============================================================================
# TESTS: AttentionVisualizer CLASS
# =============================================================================


class TestAttentionVisualizer:
    """Tests for main AttentionVisualizer class."""

    def test_init(self, mock_transformer_model, sample_feature_names):
        """Test visualizer initialization."""
        viz = AttentionVisualizer(mock_transformer_model, feature_names=sample_feature_names)
        assert viz.model == mock_transformer_model
        assert viz.feature_names == sample_feature_names
        assert viz.enable_caching is True

    def test_model_type_detection_huggingface(self):
        """Test detection of HuggingFace models."""
        model = MagicMock()
        model.__class__.__module__ = "transformers"
        viz = AttentionVisualizer(model)
        # Type detection happens in __init__
        assert hasattr(viz, '_model_type')

    def test_model_type_detection_pytorch(self):
        """Test detection of PyTorch transformer models."""
        model = MagicMock()
        model.__class__.__name__ = "TransformerModel"
        model.encoder = MagicMock()
        viz = AttentionVisualizer(model)
        assert viz._model_type == "pytorch_transformer"

    def test_cache_initialization(self, mock_transformer_model):
        """Test cache initialization."""
        viz = AttentionVisualizer(mock_transformer_model, enable_caching=True)
        assert viz._attention_cache == {}
        assert viz._visualization_cache == {}

    def test_numpy_to_torch_conversion(self, mock_transformer_model):
        """Test numpy to torch conversion (with mock)."""
        viz = AttentionVisualizer(mock_transformer_model)
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Mock torch import
        with patch('greenlang.ml.explainability.attention_visualizer.torch') as mock_torch:
            mock_tensor = MagicMock()
            mock_torch.from_numpy.return_value.float.return_value = mock_tensor

            result = viz._numpy_to_torch(data)
            assert result == mock_tensor


# =============================================================================
# TESTS: get_attention_summary METHOD
# =============================================================================


class TestGetAttentionSummary:
    """Tests for attention summary generation."""

    def test_summary_generation_basic(self, attention_weights_object, sample_feature_names):
        """Test basic summary generation."""
        viz = AttentionVisualizer(MagicMock(), feature_names=sample_feature_names)
        summary = viz.get_attention_summary(attention_weights_object, top_k=3)

        assert isinstance(summary, AttentionSummary)
        assert summary.aggregated_weights.shape == (2, 8, 8)
        assert len(summary.feature_importance) == 8
        assert len(summary.top_attended_features) == 3
        assert len(summary.top_attending_features) == 3

    def test_summary_feature_importance(self, attention_weights_object):
        """Test feature importance calculation."""
        viz = AttentionVisualizer(MagicMock())
        summary = viz.get_attention_summary(attention_weights_object)

        # Feature importance should be non-negative and between 0 and 1
        assert np.all(summary.feature_importance >= 0)
        assert np.all(summary.feature_importance <= 1)

    def test_summary_top_features(self, attention_weights_object):
        """Test top features extraction."""
        viz = AttentionVisualizer(MagicMock())
        summary = viz.get_attention_summary(attention_weights_object, top_k=5)

        # Check top attended features are sorted
        importances = [imp for _, imp in summary.top_attended_features]
        assert importances == sorted(importances, reverse=True)

        # All importances should be in [0, 1]
        for _, imp in summary.top_attended_features:
            assert 0 <= imp <= 1

    def test_summary_invalid_weights_raises(self, mock_transformer_model):
        """Test that invalid weights raise error."""
        viz = AttentionVisualizer(mock_transformer_model)
        invalid_weights = AttentionWeights(weights=np.array([]))

        with pytest.raises(ValueError):
            viz.get_attention_summary(invalid_weights)

    def test_summary_layer_importance(self, attention_weights_object):
        """Test layer importance calculation."""
        viz = AttentionVisualizer(MagicMock())
        summary = viz.get_attention_summary(attention_weights_object)

        assert len(summary.layer_importance) == 2  # 2 layers
        assert np.all(summary.layer_importance >= 0)

    def test_summary_provenance_hash(self, attention_weights_object):
        """Test provenance hash generation."""
        viz = AttentionVisualizer(MagicMock())
        summary = viz.get_attention_summary(attention_weights_object)

        assert len(summary.provenance_hash) == 64
        assert summary.provenance_hash.isalnum()

    def test_summary_processing_time(self, attention_weights_object):
        """Test processing time tracking."""
        viz = AttentionVisualizer(MagicMock())
        summary = viz.get_attention_summary(attention_weights_object)

        assert summary.processing_time_ms >= 0  # May be 0 on very fast systems
        assert summary.processing_time_ms < 10000  # Should be < 10 seconds


# =============================================================================
# TESTS: highlight_important_features METHOD
# =============================================================================


class TestHighlightImportantFeatures:
    """Tests for feature importance highlighting."""

    def test_highlight_with_threshold(self, attention_weights_object):
        """Test highlighting with attention threshold."""
        viz = AttentionVisualizer(MagicMock())
        result = viz.highlight_important_features(
            attention_weights_object,
            threshold=0.1
        )

        assert "important_features" in result
        assert "feature_importances" in result
        assert "threshold" in result
        assert result["threshold"] == 0.1

    def test_highlight_with_top_k(self, attention_weights_object):
        """Test highlighting with top-K selection."""
        viz = AttentionVisualizer(MagicMock())
        result = viz.highlight_important_features(
            attention_weights_object,
            top_k=3
        )

        assert len(result["important_features"]) == 3
        for feature in result["important_features"]:
            assert "name" in feature
            assert "importance" in feature
            assert "rank" in feature

    def test_highlight_feature_names(self, attention_weights_object, sample_feature_names):
        """Test that feature names are correctly included."""
        viz = AttentionVisualizer(MagicMock(), feature_names=sample_feature_names)
        result = viz.highlight_important_features(attention_weights_object, top_k=5)

        for feature in result["important_features"]:
            assert feature["name"] in sample_feature_names

    def test_highlight_invalid_weights_raises(self, mock_transformer_model):
        """Test that invalid weights raise error."""
        viz = AttentionVisualizer(mock_transformer_model)
        invalid_weights = AttentionWeights(weights=np.array([]))

        with pytest.raises(ValueError):
            viz.highlight_important_features(invalid_weights)


# =============================================================================
# TESTS: EXPORT FUNCTIONALITY
# =============================================================================


class TestExportFunctionality:
    """Tests for visualization export."""

    @pytest.fixture
    def sample_summary(self, attention_weights_object):
        """Generate a sample summary for export tests."""
        viz = AttentionVisualizer(MagicMock())
        return viz.get_attention_summary(attention_weights_object)

    def test_export_json(self, sample_summary):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "attention.json"
            result = AttentionVisualizer(MagicMock()).export_visualization(
                sample_summary,
                format=ExportFormat.JSON,
                output_path=output_path
            )

            assert result.exists()
            with open(result) as f:
                data = json.load(f)

            assert "provenance_hash" in data
            assert "processing_time_ms" in data
            assert "top_attended_features" in data

    def test_export_csv(self, sample_summary):
        """Test CSV export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "attention.csv"
            result = AttentionVisualizer(MagicMock()).export_visualization(
                sample_summary,
                format=ExportFormat.CSV,
                output_path=output_path
            )

            assert result.exists()
            content = result.read_text()
            assert "Feature" in content
            assert "Attention Score" in content

    def test_export_html(self, sample_summary):
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "attention.html"
            result = AttentionVisualizer(MagicMock()).export_visualization(
                sample_summary,
                format=ExportFormat.HTML,
                output_path=output_path
            )

            assert result.exists()
            content = result.read_text()
            assert "<!DOCTYPE html>" in content
            assert sample_summary.provenance_hash in content

    def test_export_default_paths(self, sample_summary):
        """Test export with default output paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = AttentionVisualizer(MagicMock()).export_visualization(
                    sample_summary,
                    format=ExportFormat.JSON
                )
                assert result.exists()
                assert result.name == "attention_summary.json"
            finally:
                os.chdir(old_cwd)

    def test_export_unsupported_format_raises(self, sample_summary, mock_transformer_model):
        """Test that unsupported format raises error."""
        viz = AttentionVisualizer(mock_transformer_model)
        # This will fail because we're passing an invalid enum
        with pytest.raises((ValueError, AttributeError)):
            viz.export_visualization(sample_summary, format="invalid_format")


# =============================================================================
# TESTS: VISUALIZATION GENERATION
# =============================================================================


class TestVisualizationGeneration:
    """Tests for visualization generation."""

    def test_visualize_attention_heatmap(self, attention_weights_object):
        """Test heatmap visualization generation."""
        pytest.importorskip("matplotlib")
        viz = AttentionVisualizer(MagicMock())
        result = viz.visualize_attention(
            attention_weights_object,
            layer=0,
            head=0,
            viz_type=VisualizationType.HEATMAP
        )

        assert "figure" in result
        assert "matrix" in result
        assert "layer" in result
        assert "head" in result
        assert result["layer"] == 0
        assert result["head"] == 0
        assert result["type"] == "heatmap"

    def test_visualize_invalid_layer_raises(self, attention_weights_object):
        """Test that invalid layer index raises error."""
        pytest.importorskip("matplotlib")
        viz = AttentionVisualizer(MagicMock())

        with pytest.raises(ValueError):
            viz.visualize_attention(attention_weights_object, layer=999, head=0)

    def test_visualize_invalid_head_raises(self, attention_weights_object):
        """Test that invalid head index raises error."""
        pytest.importorskip("matplotlib")
        viz = AttentionVisualizer(MagicMock())

        with pytest.raises(ValueError):
            viz.visualize_attention(attention_weights_object, layer=0, head=999)

    def test_visualize_matplotlib_missing_raises(self, attention_weights_object):
        """Test that missing matplotlib is reported."""
        with patch('greenlang.ml.explainability.attention_visualizer.MATPLOTLIB_AVAILABLE', False):
            viz = AttentionVisualizer(MagicMock())
            with pytest.raises(ImportError, match="matplotlib"):
                viz.visualize_attention(attention_weights_object)


# =============================================================================
# TESTS: ENUMS AND CONSTANTS
# =============================================================================


class TestEnumsAndConstants:
    """Tests for enums and constants."""

    def test_visualization_type_enum(self):
        """Test VisualizationType enum values."""
        assert VisualizationType.HEATMAP.value == "heatmap"
        assert VisualizationType.FLOW.value == "flow"
        assert VisualizationType.NETWORK.value == "network"
        assert VisualizationType.TIMESERIES.value == "timeseries"
        assert VisualizationType.DISTRIBUTION.value == "distribution"

    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.PNG.value == "png"
        assert ExportFormat.SVG.value == "svg"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"


# =============================================================================
# TESTS: INTEGRATION SCENARIOS
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""

    def test_full_workflow(self, attention_weights_object, sample_feature_names):
        """Test complete workflow: extraction -> summary -> export."""
        # Create visualizer
        viz = AttentionVisualizer(
            MagicMock(),
            feature_names=sample_feature_names
        )

        # Generate summary
        summary = viz.get_attention_summary(attention_weights_object, top_k=5)
        assert summary is not None

        # Highlight features
        highlights = viz.highlight_important_features(attention_weights_object, top_k=3)
        assert len(highlights["important_features"]) == 3

        # Export as JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "full_workflow.json"
            result_path = viz.export_visualization(
                summary,
                format=ExportFormat.JSON,
                output_path=output_path
            )
            assert result_path.exists()

    def test_process_heat_specific_workflow(self):
        """Test workflow with process heat sensor names."""
        sensor_names = [
            "boiler_temp",
            "pressure_in",
            "pressure_out",
            "steam_flow",
            "fuel_flow",
            "oxygen",
            "efficiency",
            "co2_emissions"
        ]

        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(8), size=(3, 8, 8, 8))

        attention_weights = AttentionWeights(
            weights=weights.astype(np.float32),
            feature_names=sensor_names,
            model_name="process_heat_transformer"
        )

        viz = AttentionVisualizer(MagicMock(), feature_names=sensor_names)
        summary = viz.get_attention_summary(attention_weights, top_k=5)

        # Verify sensor-specific analysis
        assert all(name in sensor_names for name, _ in summary.top_attended_features)

        # Check feature importance for key sensors
        feature_importance = dict(zip(sensor_names, summary.feature_importance))
        assert "boiler_temp" in feature_importance
        assert "efficiency" in feature_importance


# =============================================================================
# TESTS: ERROR HANDLING AND EDGE CASES
# =============================================================================


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge cases."""

    def test_empty_attention_weights(self, mock_transformer_model):
        """Test handling of empty attention weights."""
        viz = AttentionVisualizer(mock_transformer_model)
        invalid_weights = AttentionWeights(weights=np.array([]))

        with pytest.raises(ValueError):
            viz.get_attention_summary(invalid_weights)

    def test_single_feature(self, mock_transformer_model):
        """Test handling of single feature."""
        weights = np.ones((1, 1, 1, 1), dtype=np.float32)
        attn = AttentionWeights(weights=weights, feature_names=["single"])

        viz = AttentionVisualizer(mock_transformer_model, feature_names=["single"])
        summary = viz.get_attention_summary(attn, top_k=1)

        assert len(summary.feature_importance) == 1
        assert summary.feature_importance[0] == 1.0

    def test_many_layers_and_heads(self, mock_transformer_model):
        """Test handling of many layers and heads."""
        np.random.seed(42)
        # 12 layers, 16 heads, 512 sequence length (realistic transformer)
        weights = np.random.dirichlet(np.ones(512), size=(12, 16, 512, 512))
        attn = AttentionWeights(weights=weights.astype(np.float32))

        viz = AttentionVisualizer(mock_transformer_model)
        summary = viz.get_attention_summary(attn, top_k=10)

        assert summary.layer_importance.shape == (12,)
        assert len(summary.head_importance) == 12
        assert len(summary.top_attended_features) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
