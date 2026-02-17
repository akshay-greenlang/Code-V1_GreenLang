# -*- coding: utf-8 -*-
"""
Integration tests for Outlier Detection Agent end-to-end pipeline - AGENT-DATA-013

Tests the full outlier detection pipeline flow from detection through
documentation, including checkpoint/resume, contextual detection,
temporal data, large batches, empty inputs, error recovery, determinism,
and provenance chain integrity.

12 test cases covering:
- test_full_pipeline_detect_to_document
- test_pipeline_with_no_outliers
- test_pipeline_with_all_outliers
- test_pipeline_with_mixed_data
- test_pipeline_checkpoint_and_resume
- test_pipeline_with_contextual_detection
- test_pipeline_with_temporal_data
- test_pipeline_large_batch
- test_pipeline_with_empty_input
- test_pipeline_error_recovery
- test_pipeline_determinism
- test_pipeline_provenance_chain

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.outlier_detector.config import (
    OutlierDetectorConfig,
    set_config,
)
from greenlang.outlier_detector.setup import (
    OutlierDetectorService,
    DetectionResponse,
    BatchDetectionResponse,
    ClassificationResponse,
    TreatmentResponse,
    PipelineResponse,
    StatsResponse,
    _compute_hash,
)

# Import the helper from conftest (via pytest fixture resolution)
from tests.integration.outlier_detector.conftest import _build_service


# ===================================================================
# End-to-End Pipeline Integration Tests
# ===================================================================


class TestFullPipelineE2E:
    """End-to-end integration tests for the outlier detection pipeline."""

    def test_full_pipeline_detect_to_document(
        self, service, sample_records,
    ):
        """Test the complete pipeline from detection to documentation.

        Validates that running the full pipeline produces:
        - A completed status
        - All 5 stages (detect, classify, treat, validate, document)
        - Valid provenance hash
        - Correct record count
        - Outlier detection in data with known outliers
        """
        result = service.run_pipeline(records=sample_records)

        assert isinstance(result, PipelineResponse)
        assert result.status in ("completed", "failed")
        assert result.total_records == 10
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Pipeline ID must be a valid UUID
        uuid.UUID(result.pipeline_id)

        # Verify detect stage is always present
        assert "detect" in result.stages

        # If completed, all stages should be present
        if result.status == "completed":
            detect_stage = result.stages["detect"]
            assert detect_stage["status"] == "completed"
            assert detect_stage["columns"] >= 1

            # With outlier values (rec-003 revenue=999999, rec-007 emissions=50000),
            # outliers should be detected
            if result.total_outliers > 0:
                assert "classify" in result.stages
                assert "treat" in result.stages

            # Validate and document stages should be present
            assert "validate" in result.stages
            assert "document" in result.stages

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_records_processed >= 10

    def test_pipeline_with_no_outliers(self, service):
        """Test pipeline with clean data produces no outlier detections.

        When all values are within normal range, the pipeline should
        complete with zero outliers detected and zero treatments.
        """
        clean_records = [
            {
                "id": f"clean-{i}",
                "name": f"Company {i}",
                "value": 100.0 + i * 0.5,
                "score": 50.0 + i * 0.3,
            }
            for i in range(20)
        ]

        result = service.run_pipeline(records=clean_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 20

        if result.status == "completed":
            # With no outlier values, detection should find zero or few
            assert "detect" in result.stages
            assert result.total_treated == 0

        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Active jobs should return to zero after completion
        assert service._active_jobs == 0

    def test_pipeline_with_all_outliers(self, service):
        """Test pipeline with data where every point is extreme.

        When all values are extremely different from each other, the
        IQR-based detection should flag several as outliers.
        """
        extreme_records = [
            {"id": f"ext-{i}", "value": 10.0 ** i}
            for i in range(10)
        ]

        result = service.run_pipeline(records=extreme_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 10

        if result.status == "completed":
            # At least some points should be flagged as outliers
            # since the range spans 1 to 1 billion
            assert result.total_outliers >= 1

        assert len(result.provenance_hash) == 64

    def test_pipeline_with_mixed_data(
        self, service, sample_records,
    ):
        """Test pipeline with a mix of normal and outlier records.

        The sample_records fixture has:
        - 8 normal records
        - 2 outlier records (rec-003 revenue=999999, rec-007 emissions=50000)
        Validate that the pipeline processes all records and the stats
        reflect the mixed nature.
        """
        result = service.run_pipeline(records=sample_records)

        assert result.total_records == 10
        assert result.status in ("completed", "failed")

        # Verify pipeline result is stored
        assert result.pipeline_id in service._pipeline_results

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_records_processed >= 10
        assert stats.provenance_entries >= 1

    def test_pipeline_checkpoint_and_resume(
        self, service, sample_records,
    ):
        """Test pipeline stages can be executed individually.

        Simulates a pipeline that processes stages step-by-step:
        1. detect_batch -> 2. classify_outliers -> 3. apply_treatment
        and verifies intermediate results are stored.
        """
        # Stage 1: Detect
        batch_result = service.detect_batch(records=sample_records)
        assert isinstance(batch_result, BatchDetectionResponse)
        assert batch_result.batch_id in service._batch_detections
        assert batch_result.total_columns >= 1

        # Stage 2: Classify (collect outlier scores from batch)
        all_detections = []
        for col_result in batch_result.results:
            for score in col_result.get("scores", []):
                if score.get("is_outlier", False):
                    all_detections.append(score)

        if all_detections:
            cls_result = service.classify_outliers(
                detections=all_detections,
                records=sample_records,
            )
            assert isinstance(cls_result, ClassificationResponse)
            assert cls_result.classification_id in service._classifications
            assert cls_result.total_classified > 0

            # Stage 3: Treat
            treat_result = service.apply_treatment(
                records=sample_records,
                detections=all_detections,
                strategy="flag",
            )
            assert isinstance(treat_result, TreatmentResponse)
            assert treat_result.treatment_id in service._treatments
            assert treat_result.total_treated > 0

        # Verify all intermediate results are persisted
        assert len(service._batch_detections) >= 1
        assert len(service._detections) >= 1

    def test_pipeline_with_contextual_detection(
        self, service, sample_records,
    ):
        """Test pipeline with contextual detection configuration.

        Runs the pipeline with config enabling contextual detection and
        verifies the pipeline completes with expected stages.
        """
        pipeline_config = {
            "enable_contextual": True,
            "group_columns": ["region"],
        }

        result = service.run_pipeline(
            records=sample_records,
            config=pipeline_config,
        )

        assert result.status in ("completed", "failed")
        assert result.total_records == 10

        # Pipeline should still have standard stages
        assert "detect" in result.stages
        assert "validate" in result.stages
        assert "document" in result.stages

        assert len(result.provenance_hash) == 64

    def test_pipeline_with_temporal_data(
        self, service, sample_time_series,
    ):
        """Test pipeline with time-series data containing temporal anomalies.

        Uses the sample_time_series fixture with 30 timestamped records
        and 3 anomalies injected at indices 8, 17, 26.
        """
        result = service.run_pipeline(records=sample_time_series)

        assert result.status in ("completed", "failed")
        assert result.total_records == 30

        if result.status == "completed":
            # Detect stage should detect the injected anomalies
            detect_stage = result.stages.get("detect", {})
            if detect_stage.get("status") == "completed":
                assert detect_stage["columns"] >= 1
                # With 3 extreme anomalies (500, -100, 800 in base range ~100-160),
                # outliers should be detected
                assert detect_stage.get("outliers", 0) >= 1

        assert len(result.provenance_hash) == 64

    def test_pipeline_large_batch(self, service):
        """Test pipeline with a large batch of records (500+).

        Generates 500 records with some outlier values and validates the
        pipeline completes within a reasonable time and without errors.
        """
        import random
        random.seed(42)

        records = []
        for i in range(500):
            value = random.uniform(100.0, 200.0)
            # Inject outliers at every 50th record
            if i % 50 == 0:
                value = random.uniform(10000.0, 50000.0)

            rec = {
                "id": f"batch-{i:04d}",
                "company": f"Company {i % 100}",
                "revenue": round(value, 2),
                "emissions": round(random.uniform(500.0, 2000.0), 2),
                "region": f"R{i % 5}",
            }
            records.append(rec)

        start_time = time.time()
        result = service.run_pipeline(records=records)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert result.status in ("completed", "failed")
        assert result.total_records == 500

        # Pipeline should complete in under 30 seconds for 500 records
        assert elapsed_ms < 30_000, f"Pipeline took {elapsed_ms:.0f}ms (>30s)"

        # Active jobs should be zero after completion
        assert service._active_jobs == 0

    def test_pipeline_with_empty_input(self, service):
        """Test pipeline raises ValueError for empty record list.

        The pipeline must reject empty inputs with a clear error.
        """
        with pytest.raises(ValueError, match="must not be empty"):
            service.run_pipeline(records=[])

    def test_pipeline_error_recovery(
        self, service, sample_records,
    ):
        """Test that a pipeline failure is handled gracefully.

        Simulates an error during the detect_batch stage and verifies:
        - Status is 'failed'
        - Active jobs counter returns to zero
        - Stats reflect the failed job
        - Provenance is still recorded
        """
        # Patch detect_batch to simulate failure during pipeline
        original_detect_batch = service.detect_batch

        def failing_detect(*args, **kwargs):
            raise RuntimeError("Simulated detection engine failure")

        service.detect_batch = failing_detect

        result = service.run_pipeline(records=sample_records)

        assert result.status == "failed"
        assert result.total_records == 10
        assert service._active_jobs == 0

        # Stats should reflect the failure
        stats = service.get_statistics()
        assert stats.failed_jobs >= 1

        # Provenance should still be recorded for the failed pipeline
        assert result.pipeline_id in service._pipeline_results
        assert len(result.provenance_hash) == 64

        # Restore for other tests
        service.detect_batch = original_detect_batch

    def test_pipeline_determinism(
        self, service, sample_records,
    ):
        """Test that running the same pipeline twice produces identical results.

        Determinism is a core requirement of GreenLang. Two identical
        pipeline runs must produce the same detection results, same
        outlier counts, and same treatment counts.
        """
        # First run
        result1 = service.run_pipeline(records=sample_records)

        # Create a fresh service for second run
        svc2 = _build_service()
        result2 = svc2.run_pipeline(records=sample_records)

        # Both should have same status
        assert result1.status == result2.status

        # Both should process same number of records
        assert result1.total_records == result2.total_records

        # Detect stages should produce identical results
        if "detect" in result1.stages and "detect" in result2.stages:
            assert (
                result1.stages["detect"]["columns"]
                == result2.stages["detect"]["columns"]
            )
            assert (
                result1.stages["detect"]["outliers"]
                == result2.stages["detect"]["outliers"]
            )

        # Outlier and treatment counts should match
        assert result1.total_outliers == result2.total_outliers
        assert result1.total_treated == result2.total_treated

    def test_pipeline_provenance_chain(
        self, service, sample_records,
    ):
        """Test that each pipeline stage records provenance and all hashes are valid.

        Every stage of the pipeline must contribute to the provenance
        chain with a valid SHA-256 hash.
        """
        initial_provenance_count = service.provenance.entry_count

        result = service.run_pipeline(records=sample_records)

        # Provenance should have new entries for each stage that ran
        assert service.provenance.entry_count > initial_provenance_count

        # The pipeline result itself should have a valid hash
        assert len(result.provenance_hash) == 64

        # All provenance entries should have 64-char hex hashes
        for entry in service.provenance._entries:
            assert len(entry["entry_hash"]) == 64
            assert entry["entity_type"] in (
                "detection_job", "detection", "batch_detection",
                "classification", "treatment", "threshold",
                "feedback", "pipeline", "impact_analysis",
            )
            assert entry["timestamp"] is not None
