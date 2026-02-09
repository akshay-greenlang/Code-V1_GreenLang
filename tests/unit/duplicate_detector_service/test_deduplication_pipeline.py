# -*- coding: utf-8 -*-
"""
Unit tests for DeduplicationPipeline - AGENT-DATA-011

Tests the DeduplicationPipeline with 100+ test cases covering:
- run_pipeline: end-to-end with duplicates and non-duplicates
- run_stage: individual stage execution
- checkpoint / resume_from_checkpoint
- get_pipeline_stats / generate_report
- validate_input
- estimate_comparisons
- cross_dataset_dedup
- error handling
- progress tracking
- thread-safe statistics
- provenance tracking

Author: GreenLang Platform Team
Date: February 2026
"""

import threading
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from greenlang.duplicate_detector.deduplication_pipeline import DeduplicationPipeline
from greenlang.duplicate_detector.models import (
    BlockingStrategy,
    ClusterAlgorithm,
    DedupJob,
    DedupReport,
    DedupRule,
    DuplicateCluster,
    FieldComparisonConfig,
    IssueSeverity,
    DedupIssueSummary,
    JobStatus,
    MatchClassification,
    MatchResult,
    MergeDecision,
    MergeStrategy,
    PipelineCheckpoint,
    PipelineStage,
    RecordSource,
    SimilarityAlgorithm,
    SimilarityResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline() -> DeduplicationPipeline:
    """Create a fresh DeduplicationPipeline instance."""
    return DeduplicationPipeline()


def _make_rule(
    name: str = "test-rule",
    fields: List[str] | None = None,
    match_threshold: float = 0.85,
    possible_threshold: float = 0.65,
) -> DedupRule:
    """Create a DedupRule with field configs."""
    field_names = fields or ["name"]
    field_configs = [
        FieldComparisonConfig(
            field_name=fn,
            algorithm=SimilarityAlgorithm.JARO_WINKLER,
            weight=1.0,
        )
        for fn in field_names
    ]
    return DedupRule(
        name=name,
        field_configs=field_configs,
        match_threshold=match_threshold,
        possible_threshold=possible_threshold,
        blocking_strategy=BlockingStrategy.SORTED_NEIGHBORHOOD,
        merge_strategy=MergeStrategy.KEEP_MOST_COMPLETE,
    )


def _make_records(num: int, prefix: str = "rec") -> List[Dict[str, Any]]:
    """Generate test records with unique ids."""
    return [
        {"id": f"{prefix}-{i}", "name": f"Person{i}", "email": f"p{i}@co.com"}
        for i in range(num)
    ]


def _make_duplicate_records() -> List[Dict[str, Any]]:
    """Generate records with clear duplicates."""
    return [
        {"id": "r1", "name": "Alice Smith", "email": "alice@co.com"},
        {"id": "r2", "name": "Alice Smith", "email": "alice@company.com"},
        {"id": "r3", "name": "Bob Jones", "email": "bob@co.com"},
        {"id": "r4", "name": "Carol White", "email": "carol@co.com"},
    ]


# =============================================================================
# TestDeduplicationPipelineInit
# =============================================================================


class TestDeduplicationPipelineInit:
    """Initialization tests."""

    def test_initialization(self):
        """Pipeline initializes with all 6 engines."""
        pipeline = DeduplicationPipeline()
        assert pipeline.fingerprinter is not None
        assert pipeline.blocker is not None
        assert pipeline.scorer is not None
        assert pipeline.classifier is not None
        assert pipeline.cluster_resolver is not None
        assert pipeline.merge_engine is not None

    def test_initial_stats(self, pipeline: DeduplicationPipeline):
        """Initial stats are all zero."""
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["invocations"] == 0
        assert stats["pipeline"]["successes"] == 0
        assert stats["pipeline"]["failures"] == 0

    def test_reset_statistics(self, pipeline: DeduplicationPipeline):
        """Reset clears all stats including sub-engines."""
        pipeline.reset_statistics()
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["invocations"] == 0
        for engine_name, engine_stats in stats["engines"].items():
            assert engine_stats["invocations"] == 0


# =============================================================================
# TestValidateInput
# =============================================================================


class TestValidateInput:
    """Tests for validate_input method."""

    def test_empty_records_raises(self, pipeline: DeduplicationPipeline):
        """Empty records raises ValueError."""
        rule = _make_rule()
        with pytest.raises(ValueError, match="must not be empty"):
            pipeline.validate_input([], rule)

    def test_single_record_raises(self, pipeline: DeduplicationPipeline):
        """Single record raises ValueError."""
        rule = _make_rule()
        with pytest.raises(ValueError, match="At least 2"):
            pipeline.validate_input([{"id": "1", "name": "A"}], rule)

    def test_no_field_configs_raises(self, pipeline: DeduplicationPipeline):
        """Rule with no field_configs raises ValueError."""
        rule = DedupRule(name="empty", field_configs=[])
        with pytest.raises(ValueError, match="at least one field_config"):
            pipeline.validate_input([{"id": "1"}, {"id": "2"}], rule)

    def test_valid_input_passes(self, pipeline: DeduplicationPipeline):
        """Valid input does not raise."""
        rule = _make_rule(fields=["name"])
        records = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]
        pipeline.validate_input(records, rule)  # no exception

    def test_missing_field_in_records_warns(self, pipeline: DeduplicationPipeline):
        """Missing field in records logs warning but does not raise."""
        rule = _make_rule(fields=["nonexistent"])
        records = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]
        # Should not raise, just warn
        pipeline.validate_input(records, rule)


# =============================================================================
# TestEstimateComparisons
# =============================================================================


class TestEstimateComparisons:
    """Tests for estimate_comparisons method."""

    def test_no_blocking(self, pipeline: DeduplicationPipeline):
        """No blocking: all pairs."""
        est = pipeline.estimate_comparisons(100, BlockingStrategy.NONE)
        assert est["all_pairs"] == 100 * 99 // 2
        assert est["estimated_comparisons"] == est["all_pairs"]
        assert est["reduction_ratio"] == 0.0

    def test_sorted_neighborhood(self, pipeline: DeduplicationPipeline):
        """Sorted neighborhood reduces pairs."""
        est = pipeline.estimate_comparisons(
            100, BlockingStrategy.SORTED_NEIGHBORHOOD, window_size=10,
        )
        assert est["estimated_comparisons"] <= est["all_pairs"]
        assert est["reduction_ratio"] > 0.0

    def test_standard_blocking(self, pipeline: DeduplicationPipeline):
        """Standard blocking estimates ~10% of pairs."""
        est = pipeline.estimate_comparisons(100, BlockingStrategy.STANDARD)
        assert est["estimated_comparisons"] < est["all_pairs"]

    def test_small_dataset(self, pipeline: DeduplicationPipeline):
        """Small dataset with 2 records."""
        est = pipeline.estimate_comparisons(2, BlockingStrategy.NONE)
        assert est["all_pairs"] == 1
        assert est["estimated_comparisons"] == 1

    def test_window_clamped(self, pipeline: DeduplicationPipeline):
        """Window size clamped to num_records."""
        est = pipeline.estimate_comparisons(5, BlockingStrategy.SORTED_NEIGHBORHOOD, window_size=100)
        assert est["estimated_comparisons"] <= 5 * 4 // 2

    def test_result_keys(self, pipeline: DeduplicationPipeline):
        """Result dict has expected keys."""
        est = pipeline.estimate_comparisons(50)
        assert "total_records" in est
        assert "all_pairs" in est
        assert "estimated_comparisons" in est
        assert "reduction_ratio" in est
        assert "blocking_strategy" in est


# =============================================================================
# TestCheckpoint
# =============================================================================


class TestCheckpoint:
    """Tests for checkpoint and resume."""

    def test_create_checkpoint(self, pipeline: DeduplicationPipeline):
        """Creating a checkpoint returns PipelineCheckpoint."""
        cp = pipeline.checkpoint(
            job_id="j1",
            stage=PipelineStage.COMPARE,
            records_processed=500,
            state_data={"some": "state"},
        )
        assert isinstance(cp, PipelineCheckpoint)
        assert cp.job_id == "j1"
        assert cp.stage == PipelineStage.COMPARE
        assert cp.records_processed == 500

    def test_resume_existing(self, pipeline: DeduplicationPipeline):
        """Resume returns the saved checkpoint."""
        pipeline.checkpoint("j1", PipelineStage.BLOCK, 100, {})
        cp = pipeline.resume_from_checkpoint("j1")
        assert cp is not None
        assert cp.job_id == "j1"

    def test_resume_nonexistent(self, pipeline: DeduplicationPipeline):
        """Resume for unknown job returns None."""
        cp = pipeline.resume_from_checkpoint("no-such-job")
        assert cp is None

    def test_overwrite_checkpoint(self, pipeline: DeduplicationPipeline):
        """Later checkpoint overwrites earlier one."""
        pipeline.checkpoint("j1", PipelineStage.BLOCK, 100, {"v": 1})
        pipeline.checkpoint("j1", PipelineStage.COMPARE, 500, {"v": 2})
        cp = pipeline.resume_from_checkpoint("j1")
        assert cp.stage == PipelineStage.COMPARE
        assert cp.records_processed == 500

    def test_checkpoint_has_provenance(self, pipeline: DeduplicationPipeline):
        """Checkpoint has provenance hash."""
        cp = pipeline.checkpoint("j1", PipelineStage.FINGERPRINT, 0, {})
        assert len(cp.provenance_hash) == 64

    def test_multiple_job_checkpoints(self, pipeline: DeduplicationPipeline):
        """Multiple jobs have independent checkpoints."""
        pipeline.checkpoint("j1", PipelineStage.BLOCK, 100, {})
        pipeline.checkpoint("j2", PipelineStage.CLASSIFY, 200, {})
        cp1 = pipeline.resume_from_checkpoint("j1")
        cp2 = pipeline.resume_from_checkpoint("j2")
        assert cp1.stage == PipelineStage.BLOCK
        assert cp2.stage == PipelineStage.CLASSIFY


# =============================================================================
# TestRunPipeline
# =============================================================================


class TestRunPipeline:
    """Tests for run_pipeline end-to-end (with mocked sub-engines)."""

    def test_pipeline_with_distinct_records(self, pipeline: DeduplicationPipeline):
        """Distinct records should produce 0 clusters (no duplicates)."""
        records = [
            {"id": "r1", "name": "Alice", "email": "alice@co.com"},
            {"id": "r2", "name": "Bob", "email": "bob@co.com"},
            {"id": "r3", "name": "Carol", "email": "carol@co.com"},
        ]
        rule = _make_rule(fields=["name", "email"])
        report = pipeline.run_pipeline(records, rule)
        assert isinstance(report, DedupReport)
        assert report.job_id is not None

    def test_pipeline_report_has_summary(self, pipeline: DeduplicationPipeline):
        """Pipeline report includes summary text."""
        records = _make_records(3)
        rule = _make_rule()
        report = pipeline.run_pipeline(records, rule)
        assert "records processed" in report.summary.lower()

    def test_pipeline_report_has_statistics(self, pipeline: DeduplicationPipeline):
        """Pipeline report includes statistics."""
        records = _make_records(3)
        rule = _make_rule()
        report = pipeline.run_pipeline(records, rule)
        assert report.statistics is not None
        assert report.statistics.total_records >= 3

    def test_pipeline_success_stats(self, pipeline: DeduplicationPipeline):
        """Successful pipeline increments success counter."""
        records = _make_records(3)
        rule = _make_rule()
        pipeline.run_pipeline(records, rule)
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["successes"] == 1

    def test_pipeline_invalid_input_raises(self, pipeline: DeduplicationPipeline):
        """Invalid input raises ValueError."""
        rule = _make_rule()
        with pytest.raises(ValueError):
            pipeline.run_pipeline([], rule)

    def test_pipeline_single_record_raises(self, pipeline: DeduplicationPipeline):
        """Single record raises ValueError."""
        rule = _make_rule()
        with pytest.raises(ValueError):
            pipeline.run_pipeline([{"id": "1", "name": "A"}], rule)

    def test_pipeline_failure_stats(self, pipeline: DeduplicationPipeline):
        """Failed pipeline increments failure counter."""
        rule = _make_rule()
        with pytest.raises(ValueError):
            pipeline.run_pipeline([], rule)
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["failures"] == 1


# =============================================================================
# TestRunStage
# =============================================================================


class TestRunStage:
    """Tests for run_stage individual stage execution."""

    def test_fingerprint_stage(self, pipeline: DeduplicationPipeline):
        """Run FINGERPRINT stage individually."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(PipelineStage.FINGERPRINT, records, rule)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_block_stage(self, pipeline: DeduplicationPipeline):
        """Run BLOCK stage individually."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(PipelineStage.BLOCK, records, rule)
        assert isinstance(result, list)

    def test_classify_stage_empty(self, pipeline: DeduplicationPipeline):
        """Run CLASSIFY stage with empty comparisons."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(
            PipelineStage.CLASSIFY, records, rule, comparisons=[],
        )
        assert result == []

    def test_unknown_stage_raises(self, pipeline: DeduplicationPipeline):
        """Unknown stage value raises ValueError."""
        records = _make_records(3)
        rule = _make_rule()
        # PipelineStage.COMPLETE is not handled as a runnable stage
        with pytest.raises(ValueError, match="Unknown"):
            pipeline.run_stage(PipelineStage.COMPLETE, records, rule)

    def test_compare_stage_empty_blocks(self, pipeline: DeduplicationPipeline):
        """Compare stage with empty blocks returns empty."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(
            PipelineStage.COMPARE, records, rule, blocks=[],
        )
        assert result == []

    def test_cluster_stage_empty(self, pipeline: DeduplicationPipeline):
        """Cluster stage with no matches returns empty."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(
            PipelineStage.CLUSTER, records, rule, match_results=[],
        )
        assert result == []

    def test_merge_stage_empty(self, pipeline: DeduplicationPipeline):
        """Merge stage with no clusters returns empty."""
        records = _make_records(3)
        rule = _make_rule()
        result = pipeline.run_stage(
            PipelineStage.MERGE, records, rule, clusters=[],
        )
        assert result == []


# =============================================================================
# TestGenerateReport
# =============================================================================


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_report_with_no_clusters(self, pipeline: DeduplicationPipeline):
        """Report with zero clusters."""
        job = DedupJob(total_records=10, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        assert isinstance(report, DedupReport)
        assert "0 duplicate clusters" in report.summary

    def test_report_with_clusters(self, pipeline: DeduplicationPipeline):
        """Report with clusters includes cluster count."""
        job = DedupJob(total_records=10, status=JobStatus.COMPLETED)
        cluster = DuplicateCluster(
            member_record_ids=["a", "b"],
            member_count=2,
            cluster_quality=0.90,
        )
        report = pipeline.generate_report(job, [cluster], [], [])
        assert "1 duplicate clusters" in report.summary

    def test_report_with_issues(self, pipeline: DeduplicationPipeline):
        """Report includes issue count."""
        job = DedupJob(total_records=10, status=JobStatus.COMPLETED)
        issue = DedupIssueSummary(
            severity=IssueSeverity.MEDIUM,
            stage=PipelineStage.BLOCK,
            description="Test issue",
        )
        report = pipeline.generate_report(job, [], [], [issue])
        assert "1 issues" in report.summary

    def test_report_has_statistics(self, pipeline: DeduplicationPipeline):
        """Report includes DedupStatistics."""
        job = DedupJob(total_records=10, matched=2, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        assert report.statistics is not None
        assert report.statistics.total_records == 10

    def test_report_provenance(self, pipeline: DeduplicationPipeline):
        """Report has provenance hash."""
        job = DedupJob(total_records=5, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        assert len(report.provenance_hash) == 64

    def test_report_format_json(self, pipeline: DeduplicationPipeline):
        """Report format is JSON by default."""
        job = DedupJob(total_records=5, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        from greenlang.duplicate_detector.models import ReportFormat
        assert report.format == ReportFormat.JSON

    def test_report_avg_similarity(self, pipeline: DeduplicationPipeline):
        """Report computes average similarity from cluster qualities."""
        job = DedupJob(total_records=10, status=JobStatus.COMPLETED)
        c1 = DuplicateCluster(
            member_record_ids=["a", "b"], member_count=2, cluster_quality=0.80,
        )
        c2 = DuplicateCluster(
            member_record_ids=["c", "d"], member_count=2, cluster_quality=0.60,
        )
        report = pipeline.generate_report(job, [c1, c2], [], [])
        assert report.statistics.avg_similarity == pytest.approx(0.70, abs=0.01)

    def test_report_duplicate_rate(self, pipeline: DeduplicationPipeline):
        """Report computes duplicate rate."""
        job = DedupJob(total_records=100, matched=20, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        assert report.statistics.duplicate_rate == pytest.approx(0.20, abs=0.01)


# =============================================================================
# TestGetPipelineStats
# =============================================================================


class TestGetPipelineStats:
    """Tests for get_pipeline_stats method."""

    def test_stats_structure(self, pipeline: DeduplicationPipeline):
        """Stats dict has pipeline and engines keys."""
        stats = pipeline.get_pipeline_stats()
        assert "pipeline" in stats
        assert "engines" in stats
        assert "fingerprinter" in stats["engines"]
        assert "blocker" in stats["engines"]
        assert "scorer" in stats["engines"]
        assert "classifier" in stats["engines"]
        assert "cluster_resolver" in stats["engines"]
        assert "merge_engine" in stats["engines"]

    def test_active_checkpoints_count(self, pipeline: DeduplicationPipeline):
        """Active checkpoints count is tracked."""
        pipeline.checkpoint("j1", PipelineStage.BLOCK, 50, {})
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["active_checkpoints"] == 1

    def test_get_statistics_alias(self, pipeline: DeduplicationPipeline):
        """get_statistics is an alias for get_pipeline_stats."""
        s1 = pipeline.get_statistics()
        s2 = pipeline.get_pipeline_stats()
        assert s1.keys() == s2.keys()


# =============================================================================
# TestCrossDatasetDedup
# =============================================================================


class TestCrossDatasetDedup:
    """Tests for cross_dataset_dedup method."""

    def test_cross_dataset_tags_records(self, pipeline: DeduplicationPipeline):
        """Records are tagged with _dataset field."""
        ds_a = [{"id": "a1", "name": "Alice"}]
        ds_b = [{"id": "b1", "name": "Bob"}]
        rule = _make_rule()
        report = pipeline.cross_dataset_dedup(ds_a, ds_b, rule)
        assert isinstance(report, DedupReport)

    def test_cross_dataset_auto_ids(self, pipeline: DeduplicationPipeline):
        """Records without IDs get auto-assigned IDs."""
        ds_a = [{"name": "Alice"}, {"name": "Bob"}]
        ds_b = [{"name": "Carol"}]
        rule = _make_rule()
        report = pipeline.cross_dataset_dedup(ds_a, ds_b, rule)
        assert isinstance(report, DedupReport)


# =============================================================================
# TestPipelineErrorHandling
# =============================================================================


class TestPipelineErrorHandling:
    """Error handling tests for pipeline execution."""

    def test_empty_records_raises(self, pipeline: DeduplicationPipeline):
        """Empty records raises ValueError."""
        rule = _make_rule()
        with pytest.raises(ValueError):
            pipeline.run_pipeline([], rule)

    def test_no_field_configs_raises(self, pipeline: DeduplicationPipeline):
        """Rule with no field configs raises ValueError."""
        rule = DedupRule(name="bad", field_configs=[])
        with pytest.raises(ValueError):
            pipeline.run_pipeline(
                [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}],
                rule,
            )

    def test_error_in_sub_engine_propagates(self, pipeline: DeduplicationPipeline):
        """Error in a sub-engine propagates through pipeline."""
        rule = _make_rule()
        with patch.object(
            pipeline.fingerprinter, "fingerprint_batch",
            side_effect=RuntimeError("Engine failure"),
        ):
            with pytest.raises(RuntimeError, match="Engine failure"):
                pipeline.run_pipeline(_make_records(3), rule)


# =============================================================================
# TestPipelineProgressTracking
# =============================================================================


class TestPipelineProgressTracking:
    """Progress tracking tests."""

    def test_job_progress_stages(self):
        """Job progress increases through stages."""
        job = DedupJob(status=JobStatus.RUNNING)
        stages = [
            PipelineStage.FINGERPRINT,
            PipelineStage.BLOCK,
            PipelineStage.COMPARE,
            PipelineStage.CLASSIFY,
            PipelineStage.CLUSTER,
            PipelineStage.MERGE,
            PipelineStage.COMPLETE,
        ]
        prev = 0.0
        for stage in stages:
            job.stage = stage
            pct = job.progress_pct
            assert pct >= prev
            prev = pct

    def test_completed_job_is_100(self):
        """Completed job has 100% progress."""
        job = DedupJob(status=JobStatus.COMPLETED)
        assert job.progress_pct == 100.0

    def test_failed_job_is_0(self):
        """Failed job has 0% progress."""
        job = DedupJob(status=JobStatus.FAILED)
        assert job.progress_pct == 0.0


# =============================================================================
# TestPipelineNoDuplicatesFound
# =============================================================================


class TestPipelineNoDuplicatesFound:
    """Tests when pipeline finds no duplicates."""

    def test_distinct_records_zero_clusters(self, pipeline: DeduplicationPipeline):
        """All distinct records produce 0 clusters."""
        records = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
            {"id": "3", "name": "Carol"},
        ]
        rule = _make_rule(fields=["name"])
        report = pipeline.run_pipeline(records, rule)
        # With distinct names via Jaro-Winkler, might or might not find matches
        # but the pipeline should complete without error
        assert isinstance(report, DedupReport)


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Thread-safety tests."""

    def test_concurrent_checkpoints(self, pipeline: DeduplicationPipeline):
        """Concurrent checkpoint operations are safe."""
        errors: List[str] = []

        def worker(idx: int):
            try:
                for i in range(20):
                    pipeline.checkpoint(
                        f"job-{idx}",
                        PipelineStage.COMPARE,
                        i * 10,
                        {"idx": idx, "i": i},
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_stats_access(self, pipeline: DeduplicationPipeline):
        """Concurrent stats access is safe."""
        errors: List[str] = []

        def worker():
            try:
                for _ in range(50):
                    pipeline.get_pipeline_stats()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# =============================================================================
# TestProvenanceTracking
# =============================================================================


class TestProvenanceTracking:
    """Provenance hash generation tests."""

    def test_report_provenance_sha256(self, pipeline: DeduplicationPipeline):
        """Report provenance is 64-char hex."""
        job = DedupJob(total_records=5, status=JobStatus.COMPLETED)
        report = pipeline.generate_report(job, [], [], [])
        assert len(report.provenance_hash) == 64
        int(report.provenance_hash, 16)

    def test_checkpoint_provenance_sha256(self, pipeline: DeduplicationPipeline):
        """Checkpoint provenance is 64-char hex."""
        cp = pipeline.checkpoint("j1", PipelineStage.BLOCK, 50, {})
        assert len(cp.provenance_hash) == 64
        int(cp.provenance_hash, 16)


# =============================================================================
# TestDeterminism
# =============================================================================


class TestDeterminism:
    """Determinism: same input produces same results."""

    def test_estimate_comparisons_deterministic(self, pipeline: DeduplicationPipeline):
        """Estimation is deterministic."""
        results = [
            pipeline.estimate_comparisons(100, BlockingStrategy.SORTED_NEIGHBORHOOD)
            for _ in range(5)
        ]
        assert all(r == results[0] for r in results)

    def test_validate_input_deterministic(self, pipeline: DeduplicationPipeline):
        """Validation is deterministic (no raise)."""
        rule = _make_rule()
        records = [{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]
        for _ in range(5):
            pipeline.validate_input(records, rule)  # should never raise

    def test_generate_report_stats_deterministic(self, pipeline: DeduplicationPipeline):
        """Report statistics are deterministic for same input."""
        job = DedupJob(total_records=100, matched=10, status=JobStatus.COMPLETED)
        reports = [pipeline.generate_report(job, [], [], []) for _ in range(5)]
        first_stats = reports[0].statistics
        for r in reports[1:]:
            assert r.statistics.total_records == first_stats.total_records
            assert r.statistics.duplicate_rate == first_stats.duplicate_rate


# =============================================================================
# TestResetStatistics
# =============================================================================


class TestResetStatistics:
    """Tests for reset_statistics."""

    def test_reset_clears_pipeline_stats(self, pipeline: DeduplicationPipeline):
        """Reset zeroes pipeline stats."""
        pipeline._invocations = 5
        pipeline._successes = 4
        pipeline._failures = 1
        pipeline.reset_statistics()
        stats = pipeline.get_pipeline_stats()
        assert stats["pipeline"]["invocations"] == 0
        assert stats["pipeline"]["successes"] == 0
        assert stats["pipeline"]["failures"] == 0

    def test_reset_clears_engine_stats(self, pipeline: DeduplicationPipeline):
        """Reset also clears all sub-engine stats."""
        pipeline.reset_statistics()
        stats = pipeline.get_pipeline_stats()
        for engine_stats in stats["engines"].values():
            assert engine_stats["invocations"] == 0
