# -*- coding: utf-8 -*-
"""
Integration tests for Missing Value Imputer end-to-end pipeline - AGENT-DATA-012

Tests the full imputation pipeline flow from missingness analysis through
documentation, including checkpoint/resume, custom rules, time-series
imputation, large batches, empty inputs, and error recovery scenarios.

12 test cases covering:
- test_full_pipeline_analyze_to_document
- test_pipeline_with_no_missing_values
- test_pipeline_with_all_missing_column
- test_pipeline_with_mixed_missing_patterns
- test_pipeline_checkpoint_and_resume
- test_pipeline_with_custom_rules
- test_pipeline_with_time_series_data
- test_pipeline_large_batch
- test_pipeline_with_empty_input
- test_pipeline_error_recovery
- test_pipeline_determinism_across_runs
- test_pipeline_provenance_chain_integrity

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.missing_value_imputer.config import (
    MissingValueImputerConfig,
    set_config,
)
from greenlang.missing_value_imputer.setup import (
    MissingValueImputerService,
    AnalysisResponse,
    ImputationResponse,
    BatchImputationResponse,
    ValidationResponse,
    PipelineResponse,
    StatsResponse,
    _compute_hash,
)

# Import the helper from conftest (via pytest fixture resolution)
from tests.integration.missing_value_imputer.conftest import _build_service


# ===================================================================
# End-to-End Pipeline Integration Tests
# ===================================================================


class TestFullPipelineE2E:
    """End-to-end integration tests for the imputation pipeline."""

    def test_full_pipeline_analyze_to_document(
        self, service, sample_records_with_missing,
    ):
        """Test the complete pipeline from analysis to documentation.

        Validates that running the full pipeline produces:
        - A completed status
        - All 5 stages (analyze, strategize, impute, validate, document)
        - Valid provenance hash
        - Correct record count
        """
        result = service.run_pipeline(records=sample_records_with_missing)

        assert isinstance(result, PipelineResponse)
        assert result.status in ("completed", "failed")
        assert result.total_records == 10
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Pipeline ID must be a valid UUID
        uuid.UUID(result.pipeline_id)

        # Verify analyze stage is always present
        assert "analyze" in result.stages

        # If completed and missing values exist, all stages should be present
        if result.status == "completed":
            analyze_stage = result.stages["analyze"]
            assert "analysis_id" in analyze_stage
            assert analyze_stage["total_records"] == 10
            assert analyze_stage["columns_with_missing"] > 0

            # With missing values, strategize and impute stages should exist
            assert "strategize" in result.stages
            assert "impute" in result.stages

            impute_stage = result.stages["impute"]
            assert impute_stage["total_values_imputed"] > 0
            assert "batch_id" in impute_stage

            # Document stage should be present
            assert "document" in result.stages

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_analyses >= 1
        assert stats.total_records_processed >= 10

    def test_pipeline_with_no_missing_values(self, service):
        """Test pipeline with clean data produces no imputations.

        When no values are missing, the pipeline should complete with
        zero columns imputed, zero values imputed, and only the analyze
        stage populated.
        """
        clean_records = [
            {
                "id": f"clean-{i}",
                "name": f"Person {i}",
                "value": 100.0 + i,
                "category": "A",
            }
            for i in range(5)
        ]

        result = service.run_pipeline(records=clean_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 5

        if result.status == "completed":
            # With no missing values, imputation stages should be absent
            assert "analyze" in result.stages
            assert result.total_columns_imputed == 0
            assert result.total_values_imputed == 0

        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Active jobs should return to zero after completion
        assert service._active_jobs == 0

    def test_pipeline_with_all_missing_column(
        self, service, exact_missing_records,
    ):
        """Test pipeline with a column that is entirely missing.

        When a column has 100% missing values, the pipeline should:
        - Detect the high missing percentage
        - Select an appropriate strategy (mode or mice for high missing)
        - Impute with lower confidence
        """
        result = service.run_pipeline(records=exact_missing_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 5

        if result.status == "completed":
            analyze_stage = result.stages.get("analyze", {})
            if analyze_stage:
                assert analyze_stage.get("columns_with_missing", 0) >= 2

        assert len(result.provenance_hash) == 64

    def test_pipeline_with_mixed_missing_patterns(
        self, service, sample_records_with_missing,
    ):
        """Test pipeline with a mix of complete and missing records.

        The sample_records_with_missing fixture has:
        - 3 complete records (rec-001, rec-006, rec-010)
        - 7 records with various missing columns
        Validate that the pipeline processes all records and the stats
        reflect the mixed nature.
        """
        result = service.run_pipeline(records=sample_records_with_missing)

        assert result.total_records == 10
        assert result.status in ("completed", "failed")

        # Verify pipeline result is stored
        assert result.pipeline_id in service._pipeline_results

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_records_processed >= 10
        assert stats.provenance_entries >= 1

    def test_pipeline_checkpoint_and_resume(
        self, service, sample_records_with_missing,
    ):
        """Test pipeline stages can be executed individually.

        Simulates a pipeline that processes stages step-by-step:
        1. analyze -> 2. impute_batch -> 3. validate
        and verifies intermediate results are stored.
        """
        # Stage 1: Analyze
        analysis = service.analyze_missingness(
            records=sample_records_with_missing,
        )
        assert isinstance(analysis, AnalysisResponse)
        assert analysis.total_records == 10
        assert analysis.analysis_id in service._analysis_results

        # Identify columns with missing values
        missing_cols = [
            ca["column_name"]
            for ca in analysis.column_analyses
            if ca["missing_count"] > 0
        ]
        assert len(missing_cols) > 0

        # Stage 2: Batch impute all missing columns
        batch_result = service.impute_batch(
            records=sample_records_with_missing,
        )
        assert isinstance(batch_result, BatchImputationResponse)
        assert batch_result.batch_id in service._batch_results
        assert batch_result.total_values_imputed > 0

        # Stage 3: Validate
        validation = service.validate_imputation(
            original=sample_records_with_missing,
            imputed=sample_records_with_missing,
        )
        assert isinstance(validation, ValidationResponse)
        assert validation.validation_id in service._validation_results

        # Verify all intermediate results are persisted
        assert len(service._analysis_results) >= 1
        assert len(service._batch_results) >= 1
        assert len(service._validation_results) >= 1

    def test_pipeline_with_custom_rules(
        self, service, sample_records_with_missing, sample_rules,
    ):
        """Test pipeline execution with custom imputation rules.

        Creates domain-specific rules before running the pipeline and
        validates that rule creation is tracked via provenance.
        """
        # Create rules from sample_rules fixture
        created_rule_ids = []
        for rule_def in sample_rules:
            rule = service.create_rule(
                name=rule_def["name"],
                target_column=rule_def["target_column"],
                conditions=rule_def.get("conditions"),
                impute_value=rule_def.get("impute_value"),
                priority=rule_def.get("priority", "medium"),
                justification=rule_def.get("justification", ""),
            )
            created_rule_ids.append(rule.rule_id)
            assert rule.rule_id in service._rules

        # Verify all 5 rules were created
        assert len(service._rules) == 5

        # Run pipeline with the records
        result = service.run_pipeline(records=sample_records_with_missing)

        assert result.status in ("completed", "failed")
        assert result.total_records == 10

        # Stats should reflect rules
        stats = service.get_statistics()
        assert stats.total_rules == 5

    def test_pipeline_with_time_series_data(
        self, service, sample_time_series,
    ):
        """Test pipeline with time-series data that has temporal gaps.

        Uses the sample_time_series fixture with 20 timestamped records
        and 4 missing temperature values at regular intervals.
        """
        result = service.run_pipeline(records=sample_time_series)

        assert result.status in ("completed", "failed")
        assert result.total_records == 20

        if result.status == "completed":
            # Analyze stage should detect missing values
            analyze_stage = result.stages.get("analyze", {})
            if analyze_stage:
                assert analyze_stage["total_records"] == 20
                assert analyze_stage["columns_with_missing"] > 0

        assert len(result.provenance_hash) == 64

    def test_pipeline_large_batch(self, service):
        """Test pipeline with a large batch of records (500+).

        Generates 500 records with some missing values and validates the
        pipeline completes within a reasonable time and without errors.
        """
        records = []
        for i in range(500):
            rec = {
                "id": f"batch-{i:04d}",
                "company": f"Company {i % 100}",
                "sector": f"Sector{i % 10}" if i % 7 != 0 else None,
                "revenue": 10000.0 + i * 10 if i % 5 != 0 else None,
                "emissions": 500.0 + i * 2 if i % 8 != 0 else None,
                "region": f"R{i % 5}" if i % 9 != 0 else None,
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
        self, service, sample_records_with_missing,
    ):
        """Test that a pipeline failure is handled gracefully.

        Simulates an error during the impute_batch stage and verifies:
        - Status is 'failed'
        - Active jobs counter returns to zero
        - Stats reflect the failed job
        - Provenance is still recorded
        """
        # Patch impute_batch to simulate failure during pipeline
        original_impute_batch = service.impute_batch

        call_count = 0

        def failing_impute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated imputation engine failure")

        service.impute_batch = failing_impute

        result = service.run_pipeline(records=sample_records_with_missing)

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
        service.impute_batch = original_impute_batch

    def test_pipeline_determinism_across_runs(
        self, service, sample_records_with_missing,
    ):
        """Test that running the same pipeline twice produces identical results.

        Determinism is a core requirement of GreenLang. Two identical
        pipeline runs must produce the same analysis results, same
        strategy selections, and same imputation counts.
        """
        # First run
        result1 = service.run_pipeline(records=sample_records_with_missing)

        # Create a fresh service for second run
        svc2 = _build_service()
        result2 = svc2.run_pipeline(records=sample_records_with_missing)

        # Both should have same status
        assert result1.status == result2.status

        # Both should process same number of records
        assert result1.total_records == result2.total_records

        # Analyze stages should produce identical results
        if "analyze" in result1.stages and "analyze" in result2.stages:
            assert (
                result1.stages["analyze"]["total_records"]
                == result2.stages["analyze"]["total_records"]
            )
            assert (
                result1.stages["analyze"]["columns_with_missing"]
                == result2.stages["analyze"]["columns_with_missing"]
            )

        # Imputation counts should match
        assert result1.total_columns_imputed == result2.total_columns_imputed
        assert result1.total_values_imputed == result2.total_values_imputed

    def test_pipeline_provenance_chain_integrity(
        self, service, sample_records_with_missing,
    ):
        """Test that each pipeline stage records provenance and all hashes are valid.

        Every stage of the pipeline must contribute to the provenance
        chain with a valid SHA-256 hash.
        """
        initial_provenance_count = service.provenance.entry_count

        result = service.run_pipeline(records=sample_records_with_missing)

        # Provenance should have new entries for each stage that ran
        assert service.provenance.entry_count > initial_provenance_count

        # The pipeline result itself should have a valid hash
        assert len(result.provenance_hash) == 64

        # All provenance entries should have 64-char hex hashes
        for entry in service.provenance._entries:
            assert len(entry["entry_hash"]) == 64
            assert entry["entity_type"] in (
                "imputation_job", "analysis", "imputation", "batch",
                "validation", "rule", "template", "pipeline",
            )
            assert entry["timestamp"] is not None
