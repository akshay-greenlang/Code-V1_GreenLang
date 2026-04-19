# -*- coding: utf-8 -*-
"""
Unit tests for TrapStateClassifier.

Tests multimodal classification, late fusion, and deterministic behavior.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trap_state_classifier import (
    TrapStateClassifier,
    ClassificationConfig,
    SensorInput,
    TrapCondition,
    ConfidenceLevel,
)


class TestTrapStateClassifier:
    """Tests for TrapStateClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create default classifier."""
        return TrapStateClassifier()

    @pytest.fixture
    def healthy_input(self):
        """Create input for healthy trap."""
        return SensorInput(
            trap_id="ST-001",
            acoustic_amplitude_db=65.0,
            acoustic_frequency_khz=38.0,
            inlet_temp_c=185.0,
            outlet_temp_c=95.0,
            pressure_bar_g=10.0,
        )

    @pytest.fixture
    def failed_input(self):
        """Create input for failed trap."""
        return SensorInput(
            trap_id="ST-002",
            acoustic_amplitude_db=95.0,
            acoustic_frequency_khz=25.0,
            inlet_temp_c=185.0,
            outlet_temp_c=180.0,  # High outlet = blow-through
            pressure_bar_g=10.0,
        )

    @pytest.fixture
    def leaking_input(self):
        """Create input for leaking trap."""
        return SensorInput(
            trap_id="ST-003",
            acoustic_amplitude_db=78.0,
            acoustic_frequency_khz=32.0,
            inlet_temp_c=185.0,
            outlet_temp_c=150.0,
            pressure_bar_g=10.0,
        )

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert classifier.config is not None

    def test_classify_healthy_trap(self, classifier, healthy_input):
        """Test classification of healthy trap."""
        result = classifier.classify(healthy_input)

        assert result is not None
        assert result.trap_id == "ST-001"
        assert result.condition == TrapCondition.HEALTHY
        assert result.confidence >= 0.5
        assert result.provenance_hash is not None

    def test_classify_failed_trap(self, classifier, failed_input):
        """Test classification of failed trap."""
        result = classifier.classify(failed_input)

        assert result is not None
        assert result.trap_id == "ST-002"
        assert result.condition == TrapCondition.FAILED
        assert result.confidence >= 0.5

    def test_classify_leaking_trap(self, classifier, leaking_input):
        """Test classification of leaking trap."""
        result = classifier.classify(leaking_input)

        assert result is not None
        assert result.trap_id == "ST-003"
        assert result.condition in (TrapCondition.LEAKING, TrapCondition.DEGRADED)

    def test_deterministic_classification(self, classifier, healthy_input):
        """Test that same input produces same output."""
        result1 = classifier.classify(healthy_input)
        result2 = classifier.classify(healthy_input)

        assert result1.condition == result2.condition
        assert result1.confidence == result2.confidence
        assert result1.provenance_hash == result2.provenance_hash

    def test_classification_with_missing_acoustic(self, classifier):
        """Test classification with missing acoustic data."""
        input_data = SensorInput(
            trap_id="ST-004",
            acoustic_amplitude_db=None,
            acoustic_frequency_khz=None,
            inlet_temp_c=185.0,
            outlet_temp_c=95.0,
            pressure_bar_g=10.0,
        )

        result = classifier.classify(input_data)
        assert result is not None
        assert result.confidence_level in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)

    def test_classification_with_missing_thermal(self, classifier):
        """Test classification with missing thermal data."""
        input_data = SensorInput(
            trap_id="ST-005",
            acoustic_amplitude_db=65.0,
            acoustic_frequency_khz=38.0,
            inlet_temp_c=None,
            outlet_temp_c=None,
            pressure_bar_g=10.0,
        )

        result = classifier.classify(input_data)
        assert result is not None

    def test_feature_importance_calculated(self, classifier, healthy_input):
        """Test that feature importance is calculated."""
        result = classifier.classify(healthy_input)

        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0

    def test_uncertainty_bounds(self, classifier, healthy_input):
        """Test that uncertainty bounds are provided."""
        result = classifier.classify(healthy_input)

        assert result.confidence_lower is not None
        assert result.confidence_upper is not None
        assert result.confidence_lower <= result.confidence
        assert result.confidence <= result.confidence_upper

    def test_provenance_hash_format(self, classifier, healthy_input):
        """Test provenance hash has correct format."""
        result = classifier.classify(healthy_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16  # 16 hex chars
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_batch_classification(self, classifier, healthy_input, failed_input):
        """Test batch classification of multiple traps."""
        inputs = [healthy_input, failed_input]
        results = [classifier.classify(inp) for inp in inputs]

        assert len(results) == 2
        assert results[0].condition == TrapCondition.HEALTHY
        assert results[1].condition == TrapCondition.FAILED

    def test_extreme_values(self, classifier):
        """Test classification with extreme values."""
        extreme_input = SensorInput(
            trap_id="ST-EXTREME",
            acoustic_amplitude_db=120.0,  # Very high
            acoustic_frequency_khz=100.0,
            inlet_temp_c=300.0,  # Very high
            outlet_temp_c=290.0,
            pressure_bar_g=50.0,
        )

        result = classifier.classify(extreme_input)
        assert result is not None
        assert result.condition == TrapCondition.FAILED

    def test_custom_config(self):
        """Test classifier with custom configuration."""
        config = ClassificationConfig(
            acoustic_weight=0.6,
            thermal_weight=0.4,
            context_weight=0.0,
            confidence_threshold=0.8,
        )
        classifier = TrapStateClassifier(config)

        assert classifier.config.acoustic_weight == 0.6
        assert classifier.config.thermal_weight == 0.4


class TestClassificationConfig:
    """Tests for ClassificationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ClassificationConfig()

        assert config.acoustic_weight >= 0
        assert config.thermal_weight >= 0
        assert config.confidence_threshold >= 0
        assert config.confidence_threshold <= 1

    def test_config_weights_sum(self):
        """Test that weights sum appropriately."""
        config = ClassificationConfig()
        total = config.acoustic_weight + config.thermal_weight + config.context_weight

        assert 0.9 <= total <= 1.1  # Allow small tolerance
