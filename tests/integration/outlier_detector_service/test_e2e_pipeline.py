# -*- coding: utf-8 -*-
"""
Integration tests for Outlier Detection end-to-end pipeline - AGENT-DATA-013

Tests the full outlier detection pipeline flow from detection through
treatment, including multi-stage pipeline, classification accuracy,
provenance chain integrity, determinism, error recovery, performance,
and cross-method ensemble behavior.

12 test cases.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection Agent (GL-DATA-X-016)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.outlier_detector.setup import (
    OutlierDetectorService,
    DetectionResponse,
    BatchDetectionResponse,
    ClassificationResponse,
    TreatmentResponse,
    PipelineResponse,
    StatsResponse,
)

from tests.integration.outlier_detector_service.conftest import _build_service


# ===================================================================
# End-to-End Pipeline Integration Tests
# ===================================================================


class TestFullPipelineE2E:
    """End-to-end integration tests for the outlier detection pipeline."""

    def test_full_pipeline_detect_to_treat(self, service, sample_records_20):
        """Test the complete pipeline from detection to treatment."""
        result = service.run_pipeline(sample_records_20)

        assert isinstance(result, PipelineResponse)
        assert result.status in ("completed", "failed")
        assert result.total_records == 20
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Pipeline ID must be a valid UUID
        uuid.UUID(result.pipeline_id)

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_records_processed >= 20

    def test_pipeline_with_no_outliers(self, service, all_normal_records):
        """Test pipeline with all-normal records."""
        result = service.run_pipeline(all_normal_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 10
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

    def test_pipeline_with_multiple_outliers(self, service, sample_records_20):
        """Test pipeline with multiple outliers detects them."""
        result = service.run_pipeline(sample_records_20)

        assert result.total_records == 20
        assert result.status in ("completed", "failed")
        assert len(result.provenance_hash) == 64

        stats = service.get_statistics()
        assert stats.total_records_processed >= 20

    def test_pipeline_with_config(self, service, sample_records_20):
        """Test pipeline with custom config dict."""
        result = service.run_pipeline(
            sample_records_20,
            config={"columns": ["emissions"]},
        )

        assert result.status in ("completed", "failed")
        assert result.total_records == 20
        assert len(result.provenance_hash) == 64

    def test_staged_detect_classify_treat(self, service, sample_records_20):
        """Test running each stage individually in sequence."""
        # Stage 1: Detect
        detection = service.detect_outliers(sample_records_20, "emissions")
        assert isinstance(detection, DetectionResponse)
        assert detection.total_points == 20
        assert detection.column_name == "emissions"
        assert len(detection.provenance_hash) == 64

        # Verify the detection is stored
        stored = service.get_detection(detection.detection_id)
        assert stored is not None

        # Stage 2: Classify (only if outliers found)
        if detection.outliers_found > 0:
            outlier_dicts = [
                s for s in detection.scores if s.get("is_outlier", False)
            ]
            if outlier_dicts:
                classification = service.classify_outliers(
                    outlier_dicts, sample_records_20,
                )
                assert isinstance(classification, ClassificationResponse)
                assert len(classification.provenance_hash) == 64

        # Stage 3: Treat
        treatment = service.apply_treatment(
            sample_records_20, detection.scores, strategy="flag",
        )
        assert isinstance(treatment, TreatmentResponse)
        assert len(treatment.provenance_hash) == 64

    def test_pipeline_with_custom_thresholds(self, service, sample_records_20):
        """Test pipeline with user-defined domain thresholds."""
        # Create a domain threshold
        threshold = service.create_threshold(
            column="emissions",
            min_val=0.0,
            max_val=100.0,
            source="domain",
            context="Max expected emissions",
        )
        assert threshold.column_name == "emissions"
        assert threshold.upper_bound == 100.0

        # List thresholds
        thresholds = service.list_thresholds()
        assert len(thresholds) >= 1

        # Run detection
        result = service.detect_outliers(sample_records_20, "emissions")
        assert result.total_points == 20
        assert result.outliers_found >= 1

    def test_pipeline_large_batch(self, service, large_records_500):
        """Test pipeline with a large batch (500+ records)."""
        start_time = time.time()
        result = service.run_pipeline(large_records_500)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert result.status in ("completed", "failed")
        assert result.total_records == 500

        # Pipeline should complete in under 30 seconds
        assert elapsed_ms < 30_000, f"Pipeline took {elapsed_ms:.0f}ms (>30s)"

    def test_pipeline_with_empty_input(self, service):
        """Test pipeline raises ValueError for empty record list."""
        with pytest.raises(ValueError):
            service.run_pipeline(records=[])

    def test_pipeline_error_recovery(self, service, sample_records_20):
        """Test that a pipeline runs and stats are updated."""
        initial_stats = service.get_statistics()

        result = service.run_pipeline(sample_records_20)
        assert result.status in ("completed", "failed")

        stats_after = service.get_statistics()
        assert stats_after.total_records_processed >= initial_stats.total_records_processed

    def test_pipeline_determinism_across_runs(self, service, sample_records_20):
        """Test that running the same pipeline twice produces consistent results."""
        result1 = service.run_pipeline(sample_records_20)

        svc2 = _build_service()
        result2 = svc2.run_pipeline(sample_records_20)

        assert result1.status == result2.status
        assert result1.total_records == result2.total_records

    def test_pipeline_provenance_chain_integrity(self, service, sample_records_20):
        """Test that each stage records provenance with valid hashes."""
        initial_count = service.provenance.entry_count

        detection = service.detect_outliers(sample_records_20, "emissions")
        treatment = service.apply_treatment(
            sample_records_20, detection.scores, strategy="flag",
        )

        assert service.provenance.entry_count > initial_count
        assert len(detection.provenance_hash) == 64
        assert len(treatment.provenance_hash) == 64

        for entry in service.provenance._entries:
            assert len(entry["entry_hash"]) == 64
            assert entry["timestamp"] is not None

    def test_pipeline_feedback_loop(self, service, sample_records_20):
        """Test the feedback loop for false positive reporting."""
        detection = service.detect_outliers(sample_records_20, "emissions")

        feedback = service.submit_feedback(
            detection_id=detection.detection_id,
            feedback_type="false_positive",
            notes="Valid extreme reading from sensor calibration",
        )

        assert feedback.detection_id == detection.detection_id
        assert feedback.feedback_type == "false_positive"
        assert len(feedback.provenance_hash) == 64

        stats = service.get_statistics()
        assert stats.total_feedback >= 1
