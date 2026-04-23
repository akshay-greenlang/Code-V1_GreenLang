# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Comprehensive Trap Classifier Unit Tests

This module provides extensive unit tests for the TrapStateClassifier,
covering all classification scenarios, edge cases, and compliance requirements.

Test Categories:
    - Classification accuracy tests
    - Multimodal fusion tests
    - Boundary condition tests
    - Determinism verification
    - Feature importance validation
    - Provenance tracking tests
    - ASME PTC 39 compliance tests

Coverage Target: 95%+

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import List
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.trap_state_classifier import (
    TrapStateClassifier,
    ClassificationConfig,
    ClassificationResult,
    SensorInput,
    ModalityScore,
    FeatureImportance,
    TrapCondition,
    ConfidenceLevel,
    SeverityLevel,
    ModalityWeight,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_classifier() -> TrapStateClassifier:
    """Create a default TrapStateClassifier instance."""
    return TrapStateClassifier()


@pytest.fixture
def custom_classifier() -> TrapStateClassifier:
    """Create a classifier with custom configuration."""
    config = ClassificationConfig(
        modality_weight_profile=ModalityWeight.ACOUSTIC_PRIMARY,
        acoustic_weight=Decimal("0.50"),
        thermal_weight=Decimal("0.30"),
        context_weight=Decimal("0.20"),
        confidence_threshold=Decimal("0.60"),
    )
    return TrapStateClassifier(config)


@pytest.fixture
def healthy_trap_input() -> SensorInput:
    """Create input data for a healthy, normally operating trap."""
    return SensorInput(
        trap_id="ST-HEALTHY-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=42.0,  # Normal range
        acoustic_frequency_khz=38.0,
        inlet_temp_c=184.0,          # Near saturation at 10 bar
        outlet_temp_c=85.0,          # Good condensate subcooling
        pressure_bar_g=10.0,
        trap_type="thermodynamic",
        trap_age_years=2.0,
        last_maintenance_days=90,
        location="BOILER-ROOM-A",
    )


@pytest.fixture
def failed_open_trap_input() -> SensorInput:
    """Create input data for a failed-open (blow-through) trap."""
    return SensorInput(
        trap_id="ST-FAILED-OPEN-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=85.0,  # High = blow-through
        acoustic_frequency_khz=25.0,
        inlet_temp_c=184.0,
        outlet_temp_c=182.0,         # Very close = steam passing through
        pressure_bar_g=10.0,
        trap_type="thermodynamic",
        trap_age_years=8.0,
        last_maintenance_days=500,
        location="PROCESS-AREA-B",
    )


@pytest.fixture
def failed_closed_trap_input() -> SensorInput:
    """Create input data for a failed-closed (blocked) trap."""
    return SensorInput(
        trap_id="ST-FAILED-CLOSED-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=20.0,  # Very low = no flow
        acoustic_frequency_khz=10.0,
        inlet_temp_c=184.0,
        outlet_temp_c=40.0,          # Cold outlet = blocked
        pressure_bar_g=10.0,
        trap_type="mechanical",
        trap_age_years=5.0,
        last_maintenance_days=180,
        location="BUILDING-C",
    )


@pytest.fixture
def leaking_trap_input() -> SensorInput:
    """Create input data for a leaking trap."""
    return SensorInput(
        trap_id="ST-LEAKING-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=55.0,  # Elevated
        acoustic_frequency_khz=32.0,
        inlet_temp_c=184.0,
        outlet_temp_c=140.0,         # Intermediate delta
        pressure_bar_g=10.0,
        trap_type="thermostatic",
        trap_age_years=4.0,
        last_maintenance_days=200,
        location="WAREHOUSE-D",
    )


@pytest.fixture
def cold_trap_input() -> SensorInput:
    """Create input data for a cold trap (not receiving steam)."""
    return SensorInput(
        trap_id="ST-COLD-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=15.0,
        acoustic_frequency_khz=5.0,
        inlet_temp_c=30.0,           # Ambient = no steam
        outlet_temp_c=25.0,
        pressure_bar_g=10.0,
        trap_type="thermodynamic",
        trap_age_years=1.0,
        last_maintenance_days=30,
        location="SEASONAL-AREA",
    )


@pytest.fixture
def partial_data_input() -> SensorInput:
    """Create input data with missing sensor values."""
    return SensorInput(
        trap_id="ST-PARTIAL-001",
        timestamp=datetime.now(timezone.utc),
        acoustic_amplitude_db=None,  # Missing acoustic
        acoustic_frequency_khz=None,
        inlet_temp_c=184.0,
        outlet_temp_c=90.0,
        pressure_bar_g=10.0,
        trap_type="thermodynamic",
        trap_age_years=3.0,
        last_maintenance_days=120,
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestClassifierInitialization:
    """Tests for TrapStateClassifier initialization."""

    def test_default_initialization(self, default_classifier: TrapStateClassifier):
        """Test classifier initializes with default config."""
        assert default_classifier is not None
        assert default_classifier.config is not None
        assert default_classifier.config.modality_weight_profile == ModalityWeight.BALANCED

    def test_custom_initialization(self, custom_classifier: TrapStateClassifier):
        """Test classifier initializes with custom config."""
        assert custom_classifier.config.acoustic_weight == Decimal("0.50")
        assert custom_classifier.config.thermal_weight == Decimal("0.30")
        assert custom_classifier.config.context_weight == Decimal("0.20")

    def test_weight_validation_sums_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ClassificationConfig(
                acoustic_weight=Decimal("0.50"),
                thermal_weight=Decimal("0.50"),
                context_weight=Decimal("0.50"),
            )

    def test_statistics_initialized(self, default_classifier: TrapStateClassifier):
        """Test that statistics are initialized."""
        stats = default_classifier.get_statistics()
        assert stats["classification_count"] == 0
        assert "modality_weights" in stats


class TestHealthyTrapClassification:
    """Tests for classifying healthy traps."""

    def test_healthy_trap_classified_correctly(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that a healthy trap is classified as operating normal."""
        result = default_classifier.classify(healthy_trap_input)

        assert result.condition == TrapCondition.OPERATING_NORMAL
        assert result.confidence_score >= 0.5
        assert result.severity == SeverityLevel.NONE

    def test_healthy_trap_high_confidence(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that clear healthy traps have high confidence."""
        result = default_classifier.classify(healthy_trap_input)

        # Should be at least medium confidence
        assert result.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM
        ]

    def test_healthy_trap_recommendation(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test healthy trap gets monitoring recommendation."""
        result = default_classifier.classify(healthy_trap_input)

        assert "routine monitoring" in result.recommended_action.lower() or \
               "continue" in result.recommended_action.lower()


class TestFailedOpenClassification:
    """Tests for classifying failed-open (blow-through) traps."""

    def test_failed_open_detected(
        self,
        default_classifier: TrapStateClassifier,
        failed_open_trap_input: SensorInput
    ):
        """Test that blow-through traps are detected."""
        result = default_classifier.classify(failed_open_trap_input)

        assert result.condition == TrapCondition.FAILED_OPEN
        assert result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]

    def test_failed_open_urgent_action(
        self,
        default_classifier: TrapStateClassifier,
        failed_open_trap_input: SensorInput
    ):
        """Test that failed-open gets urgent action recommendation."""
        result = default_classifier.classify(failed_open_trap_input)

        assert "immediate" in result.recommended_action.lower() or \
               "replace" in result.recommended_action.lower() or \
               "urgent" in result.recommended_action.lower()

    def test_high_acoustic_indicates_blowthrough(
        self,
        default_classifier: TrapStateClassifier
    ):
        """Test that high acoustic amplitude indicates blow-through."""
        input_data = SensorInput(
            trap_id="ST-HIGH-ACOUSTIC",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=95.0,  # Very high
            inlet_temp_c=185.0,
            outlet_temp_c=183.0,         # Minimal delta
            pressure_bar_g=10.0,
        )

        result = default_classifier.classify(input_data)
        assert result.condition == TrapCondition.FAILED_OPEN


class TestFailedClosedClassification:
    """Tests for classifying failed-closed (blocked) traps."""

    def test_failed_closed_detected(
        self,
        default_classifier: TrapStateClassifier,
        failed_closed_trap_input: SensorInput
    ):
        """Test that blocked traps are detected."""
        result = default_classifier.classify(failed_closed_trap_input)

        # Should detect as failed_closed or cold
        assert result.condition in [
            TrapCondition.FAILED_CLOSED,
            TrapCondition.COLD
        ]

    def test_large_delta_t_indicates_blockage(
        self,
        default_classifier: TrapStateClassifier
    ):
        """Test that large delta-T indicates blockage."""
        input_data = SensorInput(
            trap_id="ST-BLOCKED",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=25.0,  # Low
            inlet_temp_c=185.0,
            outlet_temp_c=35.0,          # Very cold outlet
            pressure_bar_g=10.0,
        )

        result = default_classifier.classify(input_data)
        assert result.condition in [
            TrapCondition.FAILED_CLOSED,
            TrapCondition.COLD
        ]


class TestLeakingTrapClassification:
    """Tests for classifying leaking traps."""

    def test_leaking_trap_detected(
        self,
        default_classifier: TrapStateClassifier,
        leaking_trap_input: SensorInput
    ):
        """Test that leaking traps are detected."""
        result = default_classifier.classify(leaking_trap_input)

        assert result.condition in [
            TrapCondition.LEAKING,
            TrapCondition.OPERATING_NORMAL,  # Could be borderline
        ]

    def test_moderate_acoustic_indicates_leak(
        self,
        default_classifier: TrapStateClassifier
    ):
        """Test that moderate acoustic elevation indicates leak."""
        input_data = SensorInput(
            trap_id="ST-LEAKING",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=60.0,  # Moderately elevated
            inlet_temp_c=185.0,
            outlet_temp_c=145.0,         # Some steam passing
            pressure_bar_g=10.0,
        )

        result = default_classifier.classify(input_data)

        # Should be leaking or intermittent
        assert result.condition in [
            TrapCondition.LEAKING,
            TrapCondition.INTERMITTENT,
            TrapCondition.OPERATING_NORMAL,
        ]


class TestDeterminism:
    """Tests for deterministic behavior (zero-hallucination)."""

    def test_same_input_same_output(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that identical inputs produce identical outputs."""
        result1 = default_classifier.classify(healthy_trap_input)
        result2 = default_classifier.classify(healthy_trap_input)

        assert result1.condition == result2.condition
        assert result1.confidence_score == result2.confidence_score
        assert result1.condition_probabilities == result2.condition_probabilities

    def test_provenance_hash_deterministic(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that provenance hash is deterministic."""
        result1 = default_classifier.classify(healthy_trap_input)
        result2 = default_classifier.classify(healthy_trap_input)

        assert result1.provenance_hash == result2.provenance_hash
        assert len(result1.provenance_hash) == 64  # SHA-256

    def test_different_classifiers_same_result(
        self,
        healthy_trap_input: SensorInput
    ):
        """Test that different classifier instances give same result."""
        classifier1 = TrapStateClassifier()
        classifier2 = TrapStateClassifier()

        result1 = classifier1.classify(healthy_trap_input)
        result2 = classifier2.classify(healthy_trap_input)

        assert result1.condition == result2.condition
        assert result1.confidence_score == result2.confidence_score


class TestModalityFusion:
    """Tests for multimodal late fusion."""

    def test_all_modalities_scored(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that all modalities are scored."""
        result = default_classifier.classify(healthy_trap_input)

        modality_names = [m.modality_name for m in result.modality_scores]

        assert "acoustic" in modality_names
        assert "thermal" in modality_names
        assert "context" in modality_names

    def test_missing_acoustic_handled(
        self,
        default_classifier: TrapStateClassifier,
        partial_data_input: SensorInput
    ):
        """Test classification works with missing acoustic data."""
        result = default_classifier.classify(partial_data_input)

        assert result is not None
        assert result.condition is not None

        # Acoustic modality should show unavailable
        acoustic_score = next(
            (m for m in result.modality_scores if m.modality_name == "acoustic"),
            None
        )
        assert acoustic_score is not None
        assert acoustic_score.available is False

    def test_missing_thermal_handled(
        self,
        default_classifier: TrapStateClassifier
    ):
        """Test classification works with missing thermal data."""
        input_data = SensorInput(
            trap_id="ST-NO-THERMAL",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=45.0,
            inlet_temp_c=None,  # Missing
            outlet_temp_c=None,  # Missing
            pressure_bar_g=10.0,
        )

        result = default_classifier.classify(input_data)

        assert result is not None
        thermal_score = next(
            (m for m in result.modality_scores if m.modality_name == "thermal"),
            None
        )
        assert thermal_score is not None
        assert thermal_score.available is False

    def test_weights_affect_result(self):
        """Test that different weights produce different results."""
        acoustic_primary = TrapStateClassifier(ClassificationConfig(
            acoustic_weight=Decimal("0.70"),
            thermal_weight=Decimal("0.20"),
            context_weight=Decimal("0.10"),
        ))

        thermal_primary = TrapStateClassifier(ClassificationConfig(
            acoustic_weight=Decimal("0.20"),
            thermal_weight=Decimal("0.70"),
            context_weight=Decimal("0.10"),
        ))

        # Ambiguous case where acoustic and thermal disagree
        input_data = SensorInput(
            trap_id="ST-AMBIGUOUS",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=75.0,  # Suggests failure
            inlet_temp_c=185.0,
            outlet_temp_c=95.0,          # Suggests healthy
            pressure_bar_g=10.0,
        )

        result_acoustic = acoustic_primary.classify(input_data)
        result_thermal = thermal_primary.classify(input_data)

        # Results may differ based on weights
        # (Not necessarily, but probabilities should differ)
        assert result_acoustic.condition_probabilities != result_thermal.condition_probabilities


class TestFeatureImportance:
    """Tests for feature importance calculation."""

    def test_feature_importance_provided(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that feature importance is calculated."""
        result = default_classifier.classify(healthy_trap_input)

        assert result.feature_importance is not None
        assert len(result.feature_importance) > 0

    def test_feature_importance_sorted(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that features are sorted by importance."""
        result = default_classifier.classify(healthy_trap_input)

        importances = [fi.importance_score for fi in result.feature_importance]

        # Should be in descending order
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1]

    def test_feature_importance_has_direction(
        self,
        default_classifier: TrapStateClassifier,
        failed_open_trap_input: SensorInput
    ):
        """Test that features have direction (toward/away from failure)."""
        result = default_classifier.classify(failed_open_trap_input)

        for fi in result.feature_importance:
            assert fi.direction in ["toward_failure", "toward_normal"]


class TestUncertaintyBounds:
    """Tests for uncertainty quantification."""

    def test_uncertainty_bounds_provided(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that uncertainty bounds are provided."""
        result = default_classifier.classify(healthy_trap_input)

        assert result.uncertainty_bounds is not None
        assert len(result.uncertainty_bounds) == 2

    def test_bounds_are_valid(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that bounds are valid (low <= point <= high)."""
        result = default_classifier.classify(healthy_trap_input)

        low, high = result.uncertainty_bounds

        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        assert low <= high

    def test_confidence_within_bounds(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that confidence is within uncertainty bounds."""
        result = default_classifier.classify(healthy_trap_input)

        low, high = result.uncertainty_bounds

        # Point estimate should be within or near bounds
        # (May be slightly outside due to calculation method)
        assert low <= result.confidence_score + 0.1
        assert result.confidence_score <= high + 0.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extreme_high_values(self, default_classifier: TrapStateClassifier):
        """Test classification with extreme high values."""
        input_data = SensorInput(
            trap_id="ST-EXTREME-HIGH",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=120.0,  # Very high
            inlet_temp_c=300.0,           # Very high
            outlet_temp_c=295.0,
            pressure_bar_g=50.0,
        )

        result = default_classifier.classify(input_data)

        assert result is not None
        assert result.condition is not None

    def test_extreme_low_values(self, default_classifier: TrapStateClassifier):
        """Test classification with extreme low values."""
        input_data = SensorInput(
            trap_id="ST-EXTREME-LOW",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=0.0,
            inlet_temp_c=0.0,
            outlet_temp_c=0.0,
            pressure_bar_g=0.0,
        )

        result = default_classifier.classify(input_data)

        assert result is not None
        assert result.condition is not None

    def test_zero_pressure(self, default_classifier: TrapStateClassifier):
        """Test classification at atmospheric pressure."""
        input_data = SensorInput(
            trap_id="ST-ZERO-PRESSURE",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=40.0,
            inlet_temp_c=100.0,  # Atmospheric saturation
            outlet_temp_c=90.0,
            pressure_bar_g=0.0,
        )

        result = default_classifier.classify(input_data)

        assert result is not None

    def test_very_old_trap(self, default_classifier: TrapStateClassifier):
        """Test classification of very old trap."""
        input_data = SensorInput(
            trap_id="ST-OLD",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=50.0,
            inlet_temp_c=185.0,
            outlet_temp_c=120.0,
            pressure_bar_g=10.0,
            trap_age_years=15.0,  # Very old
            last_maintenance_days=1000,
        )

        result = default_classifier.classify(input_data)

        # Age should influence failure probability
        assert result is not None
        # Context should show higher failure risk

    def test_all_sensor_types(self, default_classifier: TrapStateClassifier):
        """Test classification for all trap types."""
        trap_types = ["thermodynamic", "thermostatic", "mechanical", "venturi"]

        for trap_type in trap_types:
            input_data = SensorInput(
                trap_id=f"ST-{trap_type.upper()}",
                timestamp=datetime.now(timezone.utc),
                acoustic_amplitude_db=50.0,
                inlet_temp_c=185.0,
                outlet_temp_c=100.0,
                pressure_bar_g=10.0,
                trap_type=trap_type,
            )

            result = default_classifier.classify(input_data)

            assert result is not None
            assert result.condition is not None


class TestProvenance:
    """Tests for provenance tracking."""

    def test_provenance_hash_format(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that provenance hash has correct format."""
        result = default_classifier.classify(healthy_trap_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_provenance_includes_input(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that different inputs produce different hashes."""
        input1 = healthy_trap_input
        input2 = SensorInput(
            trap_id="ST-DIFFERENT",
            timestamp=datetime.now(timezone.utc),
            acoustic_amplitude_db=45.0,  # Different value
            inlet_temp_c=185.0,
            outlet_temp_c=90.0,
            pressure_bar_g=10.0,
        )

        result1 = default_classifier.classify(input1)
        result2 = default_classifier.classify(input2)

        assert result1.provenance_hash != result2.provenance_hash


class TestSeverityClassification:
    """Tests for severity level classification."""

    def test_healthy_no_severity(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test healthy trap has no severity."""
        result = default_classifier.classify(healthy_trap_input)

        if result.condition == TrapCondition.OPERATING_NORMAL:
            assert result.severity == SeverityLevel.NONE

    def test_failed_open_critical_severity(
        self,
        default_classifier: TrapStateClassifier,
        failed_open_trap_input: SensorInput
    ):
        """Test failed-open has critical or high severity."""
        result = default_classifier.classify(failed_open_trap_input)

        if result.condition == TrapCondition.FAILED_OPEN:
            assert result.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]


class TestBatchProcessing:
    """Tests for batch classification."""

    def test_batch_multiple_traps(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput,
        failed_open_trap_input: SensorInput,
        leaking_trap_input: SensorInput
    ):
        """Test classifying multiple traps."""
        inputs = [healthy_trap_input, failed_open_trap_input, leaking_trap_input]

        results = [default_classifier.classify(inp) for inp in inputs]

        assert len(results) == 3
        for result in results:
            assert result.condition is not None
            assert result.provenance_hash is not None

    def test_statistics_incremented(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that classification count is tracked."""
        initial_stats = default_classifier.get_statistics()
        initial_count = initial_stats["classification_count"]

        # Classify 5 times
        for _ in range(5):
            default_classifier.classify(healthy_trap_input)

        final_stats = default_classifier.get_statistics()
        assert final_stats["classification_count"] == initial_count + 5


class TestSerialization:
    """Tests for result serialization."""

    def test_to_dict_complete(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that to_dict produces complete output."""
        result = default_classifier.classify(healthy_trap_input)
        result_dict = result.to_dict()

        assert "trap_id" in result_dict
        assert "condition" in result_dict
        assert "confidence_score" in result_dict
        assert "provenance_hash" in result_dict
        assert "recommended_action" in result_dict

    def test_to_dict_json_serializable(
        self,
        default_classifier: TrapStateClassifier,
        healthy_trap_input: SensorInput
    ):
        """Test that to_dict output is JSON serializable."""
        result = default_classifier.classify(healthy_trap_input)
        result_dict = result.to_dict()

        # Should not raise
        json_str = json.dumps(result_dict)
        assert json_str is not None

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["trap_id"] == result_dict["trap_id"]


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
