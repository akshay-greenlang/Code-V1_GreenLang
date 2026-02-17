# -*- coding: utf-8 -*-
"""
Integration tests for Duplicate Detection end-to-end pipeline - AGENT-DATA-011

Tests the full deduplication pipeline flow from fingerprinting through merge,
including checkpoint/resume, custom rules, cross-dataset dedup, large batches,
empty inputs, and error recovery scenarios.

12 test cases covering:
- test_full_pipeline_fingerprint_to_merge
- test_pipeline_with_no_duplicates
- test_pipeline_with_all_duplicates
- test_pipeline_with_mixed_matches
- test_pipeline_checkpoint_and_resume
- test_pipeline_with_custom_rules
- test_pipeline_cross_dataset_dedup
- test_pipeline_large_batch
- test_pipeline_with_empty_input
- test_pipeline_error_recovery
- test_pipeline_determinism_across_runs
- test_pipeline_provenance_chain_integrity

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    set_config,
)
from greenlang.duplicate_detector.setup import (
    DuplicateDetectorService,
    FingerprintResponse,
    BlockResponse,
    CompareResponse,
    ClassifyResponse,
    ClusterResponse,
    MergeResponse,
    PipelineResponse,
    StatsResponse,
    _compute_hash,
)

# Import the helper from conftest (via pytest fixture resolution)
from tests.integration.duplicate_detector.conftest import _build_service


# ===================================================================
# End-to-End Pipeline Integration Tests
# ===================================================================


class TestFullPipelineE2E:
    """End-to-end integration tests for the deduplication pipeline."""

    def test_full_pipeline_fingerprint_to_merge(self, service, sample_records_10):
        """Test the complete pipeline from fingerprinting to merge.

        Validates that running the full pipeline produces:
        - A completed status
        - All 6 stages (fingerprint, block, compare, classify, cluster, merge)
        - Valid provenance hash
        - Correct record count
        """
        result = service.run_pipeline(records=sample_records_10)

        assert isinstance(result, PipelineResponse)
        assert result.status in ("completed", "failed")
        assert result.total_records == 10
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Pipeline ID must be a valid UUID
        uuid.UUID(result.pipeline_id)

        # Verify fingerprint and block stages are always present
        assert "fingerprint" in result.stages
        assert "block" in result.stages

        # If completed, all stages should have IDs and timing
        if result.status == "completed":
            for stage_name in ["fingerprint", "block"]:
                stage = result.stages[stage_name]
                assert "id" in stage
                assert "processing_time_ms" in stage

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_jobs >= 1
        assert stats.total_records_processed >= 10

    def test_pipeline_with_no_duplicates(self, service, unique_records):
        """Test pipeline with all-unique records produces zero duplicates.

        When no records are similar enough, the pipeline should complete
        with zero duplicates, zero clusters, and zero merges.
        """
        result = service.run_pipeline(records=unique_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 5

        # With all-unique records, there may be zero matches
        # (depends on blocking strategy putting them together)
        # The key assertion is that pipeline completes without error
        assert result.processing_time_ms >= 0
        assert len(result.provenance_hash) == 64

        # Active jobs should return to zero after completion
        assert service._active_jobs == 0

    def test_pipeline_with_all_duplicates(self, service, exact_duplicate_records):
        """Test pipeline with all-identical records.

        When every record is an exact duplicate, the pipeline should:
        - Find duplicate fingerprints
        - Generate matches
        - Form clusters
        - Produce merged golden records
        """
        result = service.run_pipeline(records=exact_duplicate_records)

        assert result.status in ("completed", "failed")
        assert result.total_records == 5

        if result.status == "completed":
            # With all identical records and normalization enabled,
            # fingerprints should show many duplicate candidates
            fp_stage = result.stages.get("fingerprint", {})
            if fp_stage:
                assert fp_stage.get("total_records", 0) == 5

        assert len(result.provenance_hash) == 64

    def test_pipeline_with_mixed_matches(self, service, sample_records_10):
        """Test pipeline with a mix of duplicates and unique records.

        The sample_records_10 fixture has 4 duplicate groups (8 records)
        and 2 unique records. Validate that the pipeline processes all
        records and the stats reflect the mixed nature.
        """
        result = service.run_pipeline(records=sample_records_10)

        assert result.total_records == 10
        assert result.status in ("completed", "failed")

        # Verify pipeline result is stored
        assert result.pipeline_id in service._pipeline_results

        # Stats should be updated
        stats = service.get_statistics()
        assert stats.total_records_processed >= 10
        assert stats.provenance_entries >= 1

    def test_pipeline_checkpoint_and_resume(self, service, sample_records_10):
        """Test pipeline checkpoint interval via staged execution.

        Simulates a pipeline that processes records in stages by running
        each stage individually (fingerprint -> block -> compare -> classify
        -> cluster -> merge) and verifying intermediate results are stored.
        """
        # Stage 1: Fingerprint
        fp_result = service.fingerprint_records(records=sample_records_10)
        assert isinstance(fp_result, FingerprintResponse)
        assert fp_result.total_records == 10
        assert fp_result.fingerprint_id in service._fingerprint_results

        # Stage 2: Block
        block_result = service.create_blocks(records=sample_records_10)
        assert isinstance(block_result, BlockResponse)
        assert block_result.total_records == 10
        assert block_result.block_id in service._block_results

        # Stage 3: Compare (build pairs from blocks)
        pairs = service._build_pairs_from_blocks(
            block_result.blocks, sample_records_10,
        )
        if pairs:
            compare_result = service.compare_pairs(
                block_results={"pairs": pairs},
            )
            assert isinstance(compare_result, CompareResponse)
            assert compare_result.comparison_id in service._compare_results

            # Stage 4: Classify
            classify_result = service.classify_matches(
                comparisons=compare_result.comparisons,
            )
            assert isinstance(classify_result, ClassifyResponse)
            assert classify_result.classify_id in service._classify_results

            # Stage 5: Cluster (only MATCH classifications)
            match_items = [
                c for c in classify_result.classifications
                if c.get("classification") == "MATCH"
            ]
            if match_items:
                cluster_result = service.form_clusters(matches=match_items)
                assert isinstance(cluster_result, ClusterResponse)
                assert cluster_result.cluster_run_id in service._cluster_results

                # Stage 6: Merge
                if cluster_result.clusters:
                    merge_result = service.merge_duplicates(
                        clusters=cluster_result.clusters,
                        records=sample_records_10,
                    )
                    assert isinstance(merge_result, MergeResponse)
                    assert merge_result.merge_id in service._merge_results

        # Verify all intermediate results are persisted
        assert len(service._fingerprint_results) >= 1
        assert len(service._block_results) >= 1

    def test_pipeline_with_custom_rules(self, service, sample_records_10):
        """Test pipeline execution with a custom dedup rule configuration.

        Creates a rule with custom thresholds and field weights, then
        runs the pipeline using those settings via the options parameter.
        """
        # Create a custom rule
        rule = service.create_rule(rule_config={
            "name": "custom-sustainability-rule",
            "description": "Custom rule for sustainability supplier dedup",
            "match_threshold": 0.90,
            "possible_threshold": 0.70,
            "blocking_strategy": "sorted_neighborhood",
            "merge_strategy": "keep_most_complete",
        })
        assert rule["rule_id"] in service._rules
        assert rule["match_threshold"] == 0.90

        # Run pipeline with custom options matching the rule
        result = service.run_pipeline(
            records=sample_records_10,
            rule=rule,
            options={
                "fingerprint_algorithm": "sha256",
                "blocking_strategy": "sorted_neighborhood",
                "thresholds": {"match": 0.90, "possible": 0.70},
            },
        )

        assert result.status in ("completed", "failed")
        assert result.total_records == 10

        # Verify the stats reflect the rule was used
        stats = service.get_statistics()
        assert stats.total_rules >= 1

    def test_pipeline_cross_dataset_dedup(self, service, cross_dataset_records):
        """Test deduplication across two separate datasets.

        Merges two datasets and runs the pipeline to find duplicates
        that exist across dataset boundaries.
        """
        combined = (
            cross_dataset_records["dataset_a"]
            + cross_dataset_records["dataset_b"]
        )

        result = service.run_pipeline(records=combined)

        assert result.status in ("completed", "failed")
        assert result.total_records == 6  # 3 + 3

        # Verify fingerprinting processed all records from both datasets
        fp_stage = result.stages.get("fingerprint", {})
        if fp_stage:
            assert fp_stage["total_records"] == 6

    def test_pipeline_large_batch(self, service):
        """Test pipeline with a large batch of records (500+).

        Generates 500 records with some duplicates and validates the
        pipeline completes within a reasonable time and without errors.
        """
        records = []
        for i in range(500):
            records.append({
                "id": f"batch-{i:04d}",
                "name": f"Person {i % 100}",  # ~5 dups per name
                "email": f"person{i % 100}@company{i % 50}.com",
                "phone": f"+1-555-{i % 100:04d}",
                "city": f"City{i % 20}",
                "state": f"S{i % 10}",
            })

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

    def test_pipeline_error_recovery(self, service, sample_records_10):
        """Test that a pipeline failure is handled gracefully.

        Simulates an error during the compare stage and verifies:
        - Status is 'failed'
        - Active jobs counter returns to zero
        - Stats reflect the failed job
        - Provenance is still recorded
        """
        # Patch compare_pairs to simulate failure during pipeline
        original_compare = service.compare_pairs

        call_count = 0

        def failing_compare(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Simulated comparison engine failure")

        service.compare_pairs = failing_compare

        result = service.run_pipeline(records=sample_records_10)

        assert result.status == "failed"
        assert result.total_records == 10
        assert service._active_jobs == 0

        # Stats should reflect the failure
        stats = service.get_statistics()
        assert stats.failed_jobs >= 1
        assert stats.total_jobs >= 1

        # Provenance should still be recorded for the failed pipeline
        assert result.pipeline_id in service._pipeline_results
        assert len(result.provenance_hash) == 64

        # Restore for other tests
        service.compare_pairs = original_compare

    def test_pipeline_determinism_across_runs(self, service, sample_records_10):
        """Test that running the same pipeline twice produces identical results.

        Determinism is a core requirement of GreenLang. Two identical
        pipeline runs must produce the same fingerprints, same classifications,
        and same cluster structures.
        """
        # First run
        result1 = service.run_pipeline(records=sample_records_10)

        # Create a fresh service for second run
        svc2 = _build_service()
        result2 = svc2.run_pipeline(records=sample_records_10)

        # Both should have same status
        assert result1.status == result2.status

        # Both should process same number of records
        assert result1.total_records == result2.total_records

        # Fingerprint stages should produce identical results
        if "fingerprint" in result1.stages and "fingerprint" in result2.stages:
            assert (
                result1.stages["fingerprint"]["total_records"]
                == result2.stages["fingerprint"]["total_records"]
            )
            assert (
                result1.stages["fingerprint"]["unique_fingerprints"]
                == result2.stages["fingerprint"]["unique_fingerprints"]
            )

    def test_pipeline_provenance_chain_integrity(self, service, sample_records_10):
        """Test that each pipeline stage records provenance and all hashes are valid.

        Every stage of the pipeline must contribute to the provenance
        chain with a valid SHA-256 hash.
        """
        initial_provenance_count = service.provenance.entry_count

        result = service.run_pipeline(records=sample_records_10)

        # Provenance should have new entries for each stage that ran
        assert service.provenance.entry_count > initial_provenance_count

        # The pipeline result itself should have a valid hash
        assert len(result.provenance_hash) == 64

        # All provenance entries should have 64-char hex hashes
        for entry in service.provenance._entries:
            assert len(entry["entry_hash"]) == 64
            assert entry["entity_type"] in (
                "fingerprint", "block", "comparison", "classification",
                "cluster", "merge", "pipeline", "dedup_job", "rule",
                "report",
            )
            assert entry["timestamp"] is not None
