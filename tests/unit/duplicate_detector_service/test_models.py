# -*- coding: utf-8 -*-
"""
Unit tests for Duplicate Detection Data Models - AGENT-DATA-011

Tests all 13 enums, 15 SDK models, 7 request models, constants,
validators, serialization, edge cases, and model_config enforcement.
Target: 250+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.duplicate_detector.models import (
    # Constants
    BLOCKING_STRATEGIES,
    BLOCKING_STRATEGY_DEFAULTS,
    CONFLICT_RESOLUTIONS,
    DEFAULT_FIELD_WEIGHTS,
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_NGRAM_SIZE,
    DEFAULT_NON_MATCH_THRESHOLD,
    DEFAULT_POSSIBLE_THRESHOLD,
    FIELD_TYPES,
    FINGERPRINT_ALGORITHMS,
    ISSUE_SEVERITY_ORDER,
    MAX_BLOCK_COMPARISONS_WARN,
    MAX_RECORDS_PER_COMPARISON,
    MERGE_STRATEGIES,
    PIPELINE_STAGE_ORDER,
    REPORT_FORMAT_OPTIONS,
    SIMILARITY_ALGORITHM_DEFAULTS,
    SIMILARITY_ALGORITHMS,
    # Enums
    BlockingStrategy,
    ClusterAlgorithm,
    ConflictResolution,
    FieldType,
    FingerprintAlgorithm,
    IssueSeverity,
    JobStatus,
    MatchClassification,
    MergeStrategy,
    PipelineStage,
    RecordSource,
    ReportFormat,
    SimilarityAlgorithm,
    # SDK Models
    BlockResult,
    DedupIssueSummary,
    DedupJob,
    DedupJobSummary,
    DedupReport,
    DedupRule,
    DedupStatistics,
    DuplicateCluster,
    FieldComparisonConfig,
    MatchResult,
    MergeConflict,
    MergeDecision,
    PipelineCheckpoint,
    RecordFingerprint,
    SimilarityResult,
    # Request Models
    BlockRequest,
    ClassifyRequest,
    ClusterRequest,
    CompareRequest,
    FingerprintRequest,
    MergeRequest,
    PipelineRequest,
)


# =============================================================================
# 1. ENUM TESTS (13 enums)
# =============================================================================


class TestFingerprintAlgorithm:
    def test_values(self):
        assert FingerprintAlgorithm.SHA256.value == "sha256"
        assert FingerprintAlgorithm.SIMHASH.value == "simhash"
        assert FingerprintAlgorithm.MINHASH.value == "minhash"

    def test_member_count(self):
        assert len(FingerprintAlgorithm) == 3

    def test_membership(self):
        assert "sha256" in [e.value for e in FingerprintAlgorithm]

    def test_string_representation(self):
        assert str(FingerprintAlgorithm.SHA256) == "FingerprintAlgorithm.SHA256"

    def test_is_str_enum(self):
        assert isinstance(FingerprintAlgorithm.SHA256, str)
        assert FingerprintAlgorithm.SHA256 == "sha256"


class TestBlockingStrategy:
    def test_values(self):
        assert BlockingStrategy.SORTED_NEIGHBORHOOD.value == "sorted_neighborhood"
        assert BlockingStrategy.STANDARD.value == "standard"
        assert BlockingStrategy.CANOPY.value == "canopy"
        assert BlockingStrategy.NONE.value == "none"

    def test_member_count(self):
        assert len(BlockingStrategy) == 4

    def test_is_str_enum(self):
        assert isinstance(BlockingStrategy.NONE, str)


class TestSimilarityAlgorithm:
    def test_all_values(self):
        expected = {"exact", "levenshtein", "jaro_winkler", "soundex",
                    "ngram", "tfidf_cosine", "numeric", "date"}
        actual = {e.value for e in SimilarityAlgorithm}
        assert actual == expected

    def test_member_count(self):
        assert len(SimilarityAlgorithm) == 8

    def test_is_str_enum(self):
        assert isinstance(SimilarityAlgorithm.EXACT, str)
        assert SimilarityAlgorithm.EXACT == "exact"


class TestMatchClassification:
    def test_values(self):
        assert MatchClassification.MATCH.value == "match"
        assert MatchClassification.NON_MATCH.value == "non_match"
        assert MatchClassification.POSSIBLE.value == "possible"

    def test_member_count(self):
        assert len(MatchClassification) == 3


class TestClusterAlgorithm:
    def test_values(self):
        assert ClusterAlgorithm.UNION_FIND.value == "union_find"
        assert ClusterAlgorithm.CONNECTED_COMPONENTS.value == "connected_components"

    def test_member_count(self):
        assert len(ClusterAlgorithm) == 2


class TestMergeStrategy:
    def test_all_values(self):
        expected = {"keep_first", "keep_latest", "keep_most_complete",
                    "merge_fields", "golden_record", "custom"}
        actual = {e.value for e in MergeStrategy}
        assert actual == expected

    def test_member_count(self):
        assert len(MergeStrategy) == 6


class TestConflictResolution:
    def test_all_values(self):
        expected = {"first", "latest", "most_complete", "longest", "shortest"}
        actual = {e.value for e in ConflictResolution}
        assert actual == expected

    def test_member_count(self):
        assert len(ConflictResolution) == 5


class TestJobStatus:
    def test_all_values(self):
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        actual = {e.value for e in JobStatus}
        assert actual == expected

    def test_member_count(self):
        assert len(JobStatus) == 5


class TestPipelineStage:
    def test_all_values(self):
        expected = {"fingerprint", "block", "compare", "classify",
                    "cluster", "merge", "complete"}
        actual = {e.value for e in PipelineStage}
        assert actual == expected

    def test_member_count(self):
        assert len(PipelineStage) == 7


class TestRecordSource:
    def test_values(self):
        assert RecordSource.SINGLE_DATASET.value == "single_dataset"
        assert RecordSource.CROSS_DATASET.value == "cross_dataset"

    def test_member_count(self):
        assert len(RecordSource) == 2


class TestFieldType:
    def test_all_values(self):
        expected = {"string", "numeric", "date", "boolean", "categorical"}
        actual = {e.value for e in FieldType}
        assert actual == expected

    def test_member_count(self):
        assert len(FieldType) == 5


class TestIssueSeverity:
    def test_all_values(self):
        expected = {"critical", "high", "medium", "low", "info"}
        actual = {e.value for e in IssueSeverity}
        assert actual == expected

    def test_member_count(self):
        assert len(IssueSeverity) == 5


class TestReportFormat:
    def test_all_values(self):
        expected = {"json", "markdown", "html", "text", "csv"}
        actual = {e.value for e in ReportFormat}
        assert actual == expected

    def test_member_count(self):
        assert len(ReportFormat) == 5


# =============================================================================
# 2. CONSTANTS TESTS
# =============================================================================


class TestConstants:
    def test_default_field_weights_keys(self):
        expected_keys = {"name", "address", "email", "phone",
                         "company", "identifier", "date", "amount"}
        assert set(DEFAULT_FIELD_WEIGHTS.keys()) == expected_keys

    def test_default_field_weights_sum_to_one(self):
        total = sum(DEFAULT_FIELD_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_similarity_algorithm_defaults_keys(self):
        expected = {"exact", "levenshtein", "jaro_winkler", "soundex",
                    "ngram", "tfidf_cosine", "numeric", "date"}
        assert set(SIMILARITY_ALGORITHM_DEFAULTS.keys()) == expected

    def test_blocking_strategy_defaults_keys(self):
        expected = {"sorted_neighborhood", "standard", "canopy", "none"}
        assert set(BLOCKING_STRATEGY_DEFAULTS.keys()) == expected

    def test_max_records_per_comparison(self):
        assert MAX_RECORDS_PER_COMPARISON == 50_000

    def test_default_ngram_size(self):
        assert DEFAULT_NGRAM_SIZE == 3

    def test_default_match_threshold(self):
        assert DEFAULT_MATCH_THRESHOLD == pytest.approx(0.85)

    def test_default_possible_threshold(self):
        assert DEFAULT_POSSIBLE_THRESHOLD == pytest.approx(0.65)

    def test_default_non_match_threshold(self):
        assert DEFAULT_NON_MATCH_THRESHOLD == pytest.approx(0.40)

    def test_max_block_comparisons_warn(self):
        assert MAX_BLOCK_COMPARISONS_WARN == 100_000

    def test_fingerprint_algorithms_tuple(self):
        assert FINGERPRINT_ALGORITHMS == ("sha256", "simhash", "minhash")

    def test_blocking_strategies_tuple(self):
        assert BLOCKING_STRATEGIES == ("sorted_neighborhood", "standard", "canopy", "none")

    def test_similarity_algorithms_tuple(self):
        assert len(SIMILARITY_ALGORITHMS) == 8
        assert "jaro_winkler" in SIMILARITY_ALGORITHMS

    def test_merge_strategies_tuple(self):
        assert len(MERGE_STRATEGIES) == 6
        assert "golden_record" in MERGE_STRATEGIES

    def test_conflict_resolutions_tuple(self):
        assert len(CONFLICT_RESOLUTIONS) == 5
        assert "most_complete" in CONFLICT_RESOLUTIONS

    def test_pipeline_stage_order(self):
        assert PIPELINE_STAGE_ORDER == (
            "fingerprint", "block", "compare", "classify",
            "cluster", "merge", "complete",
        )

    def test_field_types_tuple(self):
        assert FIELD_TYPES == ("string", "numeric", "date", "boolean", "categorical")

    def test_issue_severity_order(self):
        assert ISSUE_SEVERITY_ORDER == ("info", "low", "medium", "high", "critical")

    def test_report_format_options(self):
        assert REPORT_FORMAT_OPTIONS == ("json", "markdown", "html", "text", "csv")


# =============================================================================
# 3. SDK MODEL TESTS (15 models)
# =============================================================================


# -- RecordFingerprint --

class TestRecordFingerprint:
    def test_create_minimal(self):
        fp = RecordFingerprint(record_id="r1")
        assert fp.record_id == "r1"
        assert fp.dataset_id == ""
        assert fp.field_set == []
        assert fp.fingerprint_hash == ""
        assert fp.algorithm == FingerprintAlgorithm.SHA256
        assert fp.normalized_fields is True
        assert isinstance(fp.created_at, datetime)
        assert fp.provenance_hash == ""

    def test_create_full(self):
        fp = RecordFingerprint(
            record_id="r42",
            dataset_id="ds-01",
            field_set=["name", "email"],
            fingerprint_hash="abc123",
            algorithm=FingerprintAlgorithm.MINHASH,
            normalized_fields=False,
            provenance_hash="xyz",
        )
        assert fp.record_id == "r42"
        assert fp.dataset_id == "ds-01"
        assert fp.field_set == ["name", "email"]
        assert fp.algorithm == FingerprintAlgorithm.MINHASH
        assert fp.normalized_fields is False

    def test_empty_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id must be non-empty"):
            RecordFingerprint(record_id="")

    def test_whitespace_record_id_raises(self):
        with pytest.raises(ValidationError, match="record_id must be non-empty"):
            RecordFingerprint(record_id="   ")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            RecordFingerprint(record_id="r1", unknown_field="bad")

    def test_serialization_roundtrip(self):
        fp = RecordFingerprint(record_id="r1", fingerprint_hash="h1")
        data = fp.model_dump()
        fp2 = RecordFingerprint(**data)
        assert fp2.record_id == fp.record_id
        assert fp2.fingerprint_hash == fp.fingerprint_hash

    def test_json_roundtrip(self):
        fp = RecordFingerprint(record_id="r1")
        json_str = fp.model_dump_json()
        fp2 = RecordFingerprint.model_validate_json(json_str)
        assert fp2.record_id == "r1"


# -- BlockResult --

class TestBlockResult:
    def test_create_minimal(self):
        br = BlockResult(block_key="abc")
        assert br.block_key == "abc"
        assert br.strategy == BlockingStrategy.SORTED_NEIGHBORHOOD
        assert br.record_ids == []
        assert br.record_count == 0

    def test_create_full(self):
        br = BlockResult(
            block_key="xyz",
            strategy=BlockingStrategy.CANOPY,
            record_ids=["r1", "r2"],
            record_count=2,
            provenance_hash="prov",
        )
        assert br.block_key == "xyz"
        assert br.strategy == BlockingStrategy.CANOPY
        assert len(br.record_ids) == 2

    def test_empty_block_key_raises(self):
        with pytest.raises(ValidationError, match="block_key must be non-empty"):
            BlockResult(block_key="")

    def test_whitespace_block_key_raises(self):
        with pytest.raises(ValidationError, match="block_key must be non-empty"):
            BlockResult(block_key="  ")

    def test_negative_record_count_raises(self):
        with pytest.raises(ValidationError):
            BlockResult(block_key="k", record_count=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            BlockResult(block_key="k", extra="bad")


# -- SimilarityResult --

class TestSimilarityResult:
    def test_create_minimal(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b")
        assert sr.record_a_id == "a"
        assert sr.record_b_id == "b"
        assert sr.overall_score == 0.0
        assert sr.field_scores == {}
        assert sr.algorithm_used == SimilarityAlgorithm.JARO_WINKLER
        assert sr.comparison_time_ms == 0.0

    def test_create_full(self):
        sr = SimilarityResult(
            record_a_id="r1",
            record_b_id="r2",
            field_scores={"name": 0.9, "email": 1.0},
            overall_score=0.95,
            algorithm_used=SimilarityAlgorithm.LEVENSHTEIN,
            comparison_time_ms=2.5,
            provenance_hash="ph",
        )
        assert sr.overall_score == pytest.approx(0.95)
        assert sr.algorithm_used == SimilarityAlgorithm.LEVENSHTEIN

    def test_empty_record_a_id_raises(self):
        with pytest.raises(ValidationError, match="record_a_id must be non-empty"):
            SimilarityResult(record_a_id="", record_b_id="b")

    def test_empty_record_b_id_raises(self):
        with pytest.raises(ValidationError, match="record_b_id must be non-empty"):
            SimilarityResult(record_a_id="a", record_b_id="")

    def test_overall_score_below_zero_raises(self):
        with pytest.raises(ValidationError):
            SimilarityResult(record_a_id="a", record_b_id="b", overall_score=-0.1)

    def test_overall_score_above_one_raises(self):
        with pytest.raises(ValidationError):
            SimilarityResult(record_a_id="a", record_b_id="b", overall_score=1.1)

    def test_comparison_time_ms_negative_raises(self):
        with pytest.raises(ValidationError):
            SimilarityResult(record_a_id="a", record_b_id="b", comparison_time_ms=-1.0)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            SimilarityResult(record_a_id="a", record_b_id="b", extra="x")


# -- FieldComparisonConfig --

class TestFieldComparisonConfig:
    def test_create_minimal(self):
        fcc = FieldComparisonConfig(field_name="name")
        assert fcc.field_name == "name"
        assert fcc.algorithm == SimilarityAlgorithm.JARO_WINKLER
        assert fcc.weight == 1.0
        assert fcc.field_type == FieldType.STRING
        assert fcc.case_sensitive is False
        assert fcc.strip_whitespace is True
        assert fcc.phonetic_encode is False

    def test_create_full(self):
        fcc = FieldComparisonConfig(
            field_name="phone",
            algorithm=SimilarityAlgorithm.SOUNDEX,
            weight=3.5,
            field_type=FieldType.STRING,
            case_sensitive=True,
            strip_whitespace=False,
            phonetic_encode=True,
        )
        assert fcc.field_name == "phone"
        assert fcc.algorithm == SimilarityAlgorithm.SOUNDEX
        assert fcc.weight == pytest.approx(3.5)
        assert fcc.phonetic_encode is True

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            FieldComparisonConfig(field_name="")

    def test_weight_zero_ok(self):
        fcc = FieldComparisonConfig(field_name="x", weight=0.0)
        assert fcc.weight == 0.0

    def test_weight_ten_ok(self):
        fcc = FieldComparisonConfig(field_name="x", weight=10.0)
        assert fcc.weight == 10.0

    def test_weight_below_zero_raises(self):
        with pytest.raises(ValidationError):
            FieldComparisonConfig(field_name="x", weight=-0.1)

    def test_weight_above_ten_raises(self):
        with pytest.raises(ValidationError):
            FieldComparisonConfig(field_name="x", weight=10.1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FieldComparisonConfig(field_name="x", extra="bad")

    def test_all_algorithms_accepted(self):
        for algo in SimilarityAlgorithm:
            fcc = FieldComparisonConfig(field_name="f", algorithm=algo)
            assert fcc.algorithm == algo

    def test_all_field_types_accepted(self):
        for ft in FieldType:
            fcc = FieldComparisonConfig(field_name="f", field_type=ft)
            assert fcc.field_type == ft


# -- MatchResult --

class TestMatchResult:
    def test_create_minimal(self):
        mr = MatchResult(
            record_a_id="a",
            record_b_id="b",
            classification=MatchClassification.MATCH,
        )
        assert mr.record_a_id == "a"
        assert mr.classification == MatchClassification.MATCH
        assert mr.confidence == 0.0
        assert mr.overall_score == 0.0

    def test_create_full(self):
        mr = MatchResult(
            record_a_id="r1",
            record_b_id="r2",
            classification=MatchClassification.POSSIBLE,
            confidence=0.72,
            field_scores={"name": 0.8},
            overall_score=0.72,
            decision_reason="Borderline",
            provenance_hash="abc",
        )
        assert mr.decision_reason == "Borderline"

    def test_empty_record_a_id_raises(self):
        with pytest.raises(ValidationError, match="record_a_id must be non-empty"):
            MatchResult(record_a_id="", record_b_id="b",
                        classification=MatchClassification.MATCH)

    def test_empty_record_b_id_raises(self):
        with pytest.raises(ValidationError, match="record_b_id must be non-empty"):
            MatchResult(record_a_id="a", record_b_id="",
                        classification=MatchClassification.MATCH)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            MatchResult(record_a_id="a", record_b_id="b",
                        classification=MatchClassification.MATCH,
                        confidence=1.5)

    def test_all_classifications(self):
        for cls in MatchClassification:
            mr = MatchResult(record_a_id="a", record_b_id="b", classification=cls)
            assert mr.classification == cls

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MatchResult(record_a_id="a", record_b_id="b",
                        classification=MatchClassification.MATCH, extra="x")


# -- MergeConflict --

class TestMergeConflict:
    def test_create_minimal(self):
        mc = MergeConflict(field_name="name")
        assert mc.field_name == "name"
        assert mc.values == {}
        assert mc.chosen_value is None
        assert mc.resolution_method == ConflictResolution.MOST_COMPLETE
        assert mc.source_record_id is None

    def test_create_full(self):
        mc = MergeConflict(
            field_name="email",
            values={"r1": "a@b.com", "r2": "c@d.com"},
            chosen_value="a@b.com",
            resolution_method=ConflictResolution.FIRST,
            source_record_id="r1",
        )
        assert mc.chosen_value == "a@b.com"
        assert mc.source_record_id == "r1"

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            MergeConflict(field_name="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MergeConflict(field_name="x", extra="bad")

    def test_all_resolutions(self):
        for res in ConflictResolution:
            mc = MergeConflict(field_name="f", resolution_method=res)
            assert mc.resolution_method == res


# -- DuplicateCluster --

class TestDuplicateCluster:
    def test_create_defaults(self):
        dc = DuplicateCluster()
        assert dc.cluster_id  # UUID should be generated
        assert dc.member_record_ids == []
        assert dc.representative_id is None
        assert dc.cluster_quality == 0.0
        assert dc.density == 0.0
        assert dc.diameter == 0.0
        assert dc.member_count == 0

    def test_create_full(self):
        dc = DuplicateCluster(
            cluster_id="c-1",
            member_record_ids=["r1", "r2"],
            representative_id="r1",
            cluster_quality=0.9,
            density=0.85,
            diameter=0.1,
            member_count=2,
            provenance_hash="ph",
        )
        assert dc.cluster_quality == pytest.approx(0.9)

    def test_empty_member_id_raises(self):
        with pytest.raises(ValidationError, match="must not contain empty strings"):
            DuplicateCluster(member_record_ids=["r1", ""])

    def test_whitespace_member_id_raises(self):
        with pytest.raises(ValidationError, match="must not contain empty strings"):
            DuplicateCluster(member_record_ids=["   "])

    def test_quality_above_one_raises(self):
        with pytest.raises(ValidationError):
            DuplicateCluster(cluster_quality=1.1)

    def test_density_below_zero_raises(self):
        with pytest.raises(ValidationError):
            DuplicateCluster(density=-0.1)

    def test_negative_member_count_raises(self):
        with pytest.raises(ValidationError):
            DuplicateCluster(member_count=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DuplicateCluster(extra="x")

    def test_cluster_id_default_is_uuid(self):
        dc = DuplicateCluster()
        # Should be a valid UUID
        uuid.UUID(dc.cluster_id)


# -- MergeDecision --

class TestMergeDecision:
    def test_create_minimal(self):
        md = MergeDecision(cluster_id="c-1")
        assert md.cluster_id == "c-1"
        assert md.strategy == MergeStrategy.KEEP_MOST_COMPLETE
        assert md.merged_record == {}
        assert md.conflicts == []
        assert md.source_records == []
        assert isinstance(md.decided_at, datetime)

    def test_create_full(self):
        conflict = MergeConflict(field_name="email",
                                  values={"r1": "a@b.com", "r2": "c@d.com"},
                                  chosen_value="a@b.com")
        md = MergeDecision(
            cluster_id="c-2",
            strategy=MergeStrategy.GOLDEN_RECORD,
            merged_record={"name": "Alice", "email": "a@b.com"},
            conflicts=[conflict],
            source_records=[{"id": "r1"}, {"id": "r2"}],
            provenance_hash="ph",
        )
        assert md.strategy == MergeStrategy.GOLDEN_RECORD
        assert len(md.conflicts) == 1

    def test_empty_cluster_id_raises(self):
        with pytest.raises(ValidationError, match="cluster_id must be non-empty"):
            MergeDecision(cluster_id="")

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            MergeDecision(cluster_id="c", extra="bad")


# -- DedupRule --

class TestDedupRule:
    def test_create_minimal(self):
        rule = DedupRule(name="test-rule")
        assert rule.name == "test-rule"
        assert rule.match_threshold == pytest.approx(0.85)
        assert rule.possible_threshold == pytest.approx(0.65)
        assert rule.blocking_strategy == BlockingStrategy.SORTED_NEIGHBORHOOD
        assert rule.merge_strategy == MergeStrategy.KEEP_MOST_COMPLETE
        assert rule.active is True

    def test_create_full(self, sample_field_configs):
        rule = DedupRule(
            name="full-rule",
            description="Full rule with all configs",
            field_configs=sample_field_configs,
            match_threshold=0.90,
            possible_threshold=0.70,
            blocking_strategy=BlockingStrategy.CANOPY,
            blocking_fields=["state", "zip"],
            merge_strategy=MergeStrategy.GOLDEN_RECORD,
            active=False,
        )
        assert rule.blocking_strategy == BlockingStrategy.CANOPY
        assert len(rule.field_configs) == 9
        assert rule.active is False

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            DedupRule(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            DedupRule(name="  ")

    def test_match_threshold_below_0_5_raises(self):
        with pytest.raises(ValidationError, match="match_threshold must be >= 0.5"):
            DedupRule(name="r", match_threshold=0.4)

    def test_match_threshold_at_0_5_ok(self):
        rule = DedupRule(name="r", match_threshold=0.5)
        assert rule.match_threshold == pytest.approx(0.5)

    def test_possible_threshold_below_0_1_raises(self):
        with pytest.raises(ValidationError, match="possible_threshold must be >= 0.1"):
            DedupRule(name="r", possible_threshold=0.05)

    def test_possible_threshold_at_0_1_ok(self):
        rule = DedupRule(name="r", possible_threshold=0.1)
        assert rule.possible_threshold == pytest.approx(0.1)

    def test_rule_id_default_is_uuid(self):
        rule = DedupRule(name="r")
        uuid.UUID(rule.rule_id)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupRule(name="r", extra="bad")

    def test_updated_at_default_none(self):
        rule = DedupRule(name="r")
        assert rule.updated_at is None


# -- DedupJob --

class TestDedupJob:
    def test_create_defaults(self):
        job = DedupJob()
        assert job.job_id  # UUID generated
        assert job.status == JobStatus.PENDING
        assert job.stage == PipelineStage.FINGERPRINT
        assert job.total_records == 0
        assert job.fingerprinted == 0
        assert job.compared == 0
        assert job.matched == 0
        assert job.clustered == 0
        assert job.merged == 0
        assert job.error_message is None
        assert job.started_at is None
        assert job.completed_at is None

    def test_is_active_pending(self):
        job = DedupJob(status=JobStatus.PENDING)
        assert job.is_active is True

    def test_is_active_running(self):
        job = DedupJob(status=JobStatus.RUNNING)
        assert job.is_active is True

    def test_is_active_completed(self):
        job = DedupJob(status=JobStatus.COMPLETED)
        assert job.is_active is False

    def test_is_active_failed(self):
        job = DedupJob(status=JobStatus.FAILED)
        assert job.is_active is False

    def test_is_active_cancelled(self):
        job = DedupJob(status=JobStatus.CANCELLED)
        assert job.is_active is False

    def test_progress_pct_fingerprint_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.FINGERPRINT)
        assert job.progress_pct == pytest.approx(10.0)

    def test_progress_pct_block_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.BLOCK)
        assert job.progress_pct == pytest.approx(25.0)

    def test_progress_pct_compare_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.COMPARE)
        assert job.progress_pct == pytest.approx(50.0)

    def test_progress_pct_classify_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.CLASSIFY)
        assert job.progress_pct == pytest.approx(65.0)

    def test_progress_pct_cluster_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.CLUSTER)
        assert job.progress_pct == pytest.approx(80.0)

    def test_progress_pct_merge_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.MERGE)
        assert job.progress_pct == pytest.approx(90.0)

    def test_progress_pct_complete_stage(self):
        job = DedupJob(status=JobStatus.RUNNING, stage=PipelineStage.COMPLETE)
        assert job.progress_pct == pytest.approx(100.0)

    def test_progress_pct_completed_status(self):
        job = DedupJob(status=JobStatus.COMPLETED, stage=PipelineStage.FINGERPRINT)
        assert job.progress_pct == pytest.approx(100.0)

    def test_progress_pct_failed_status(self):
        job = DedupJob(status=JobStatus.FAILED)
        assert job.progress_pct == pytest.approx(0.0)

    def test_progress_pct_cancelled_status(self):
        job = DedupJob(status=JobStatus.CANCELLED)
        assert job.progress_pct == pytest.approx(0.0)

    def test_duplicate_rate_with_records(self):
        job = DedupJob(total_records=100, matched=25)
        assert job.duplicate_rate == pytest.approx(0.25)

    def test_duplicate_rate_zero_records(self):
        job = DedupJob(total_records=0, matched=0)
        assert job.duplicate_rate == pytest.approx(0.0)

    def test_duplicate_rate_no_matches(self):
        job = DedupJob(total_records=100, matched=0)
        assert job.duplicate_rate == pytest.approx(0.0)

    def test_negative_total_records_raises(self):
        with pytest.raises(ValidationError):
            DedupJob(total_records=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupJob(extra="bad")


# -- PipelineCheckpoint --

class TestPipelineCheckpoint:
    def test_create_minimal(self):
        cp = PipelineCheckpoint(job_id="j1", stage=PipelineStage.COMPARE)
        assert cp.job_id == "j1"
        assert cp.stage == PipelineStage.COMPARE
        assert cp.records_processed == 0
        assert cp.state_data == {}

    def test_create_full(self):
        cp = PipelineCheckpoint(
            job_id="j2",
            stage=PipelineStage.CLUSTER,
            records_processed=5000,
            state_data={"cursor": 100, "batch": 50},
            provenance_hash="ph",
        )
        assert cp.records_processed == 5000
        assert cp.state_data["cursor"] == 100

    def test_empty_job_id_raises(self):
        with pytest.raises(ValidationError, match="job_id must be non-empty"):
            PipelineCheckpoint(job_id="", stage=PipelineStage.BLOCK)

    def test_negative_records_processed_raises(self):
        with pytest.raises(ValidationError):
            PipelineCheckpoint(job_id="j", stage=PipelineStage.BLOCK,
                               records_processed=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            PipelineCheckpoint(job_id="j", stage=PipelineStage.BLOCK, extra="x")

    def test_checkpoint_id_default_is_uuid(self):
        cp = PipelineCheckpoint(job_id="j", stage=PipelineStage.BLOCK)
        uuid.UUID(cp.checkpoint_id)


# -- DedupStatistics --

class TestDedupStatistics:
    def test_create_defaults(self):
        stats = DedupStatistics()
        assert stats.total_jobs == 0
        assert stats.total_records == 0
        assert stats.total_duplicates == 0
        assert stats.total_merges == 0
        assert stats.avg_similarity == 0.0
        assert stats.duplicate_rate == 0.0
        assert stats.avg_cluster_size == 0.0
        assert stats.by_status == {}
        assert stats.by_strategy == {}
        assert isinstance(stats.timestamp, datetime)

    def test_create_full(self):
        stats = DedupStatistics(
            total_jobs=10,
            total_records=50000,
            total_duplicates=2500,
            total_merges=1200,
            avg_similarity=0.87,
            duplicate_rate=0.05,
            avg_cluster_size=2.1,
            by_status={"completed": 8, "failed": 2},
            by_strategy={"sorted_neighborhood": 7, "canopy": 3},
        )
        assert stats.total_jobs == 10
        assert stats.avg_similarity == pytest.approx(0.87)

    def test_negative_total_jobs_raises(self):
        with pytest.raises(ValidationError):
            DedupStatistics(total_jobs=-1)

    def test_avg_similarity_above_one_raises(self):
        with pytest.raises(ValidationError):
            DedupStatistics(avg_similarity=1.1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupStatistics(extra="x")


# -- DedupIssueSummary --

class TestDedupIssueSummary:
    def test_create_minimal(self):
        issue = DedupIssueSummary(
            stage=PipelineStage.COMPARE,
            description="High comparison count in block",
        )
        assert issue.severity == IssueSeverity.MEDIUM
        assert issue.affected_records == 0
        assert issue.suggested_fix is None

    def test_create_full(self):
        issue = DedupIssueSummary(
            severity=IssueSeverity.CRITICAL,
            stage=PipelineStage.MERGE,
            description="Data loss detected during merge",
            affected_records=150,
            suggested_fix="Review merge strategy",
            provenance_hash="ph",
        )
        assert issue.severity == IssueSeverity.CRITICAL
        assert issue.affected_records == 150

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError, match="description must be non-empty"):
            DedupIssueSummary(stage=PipelineStage.BLOCK, description="")

    def test_whitespace_description_raises(self):
        with pytest.raises(ValidationError, match="description must be non-empty"):
            DedupIssueSummary(stage=PipelineStage.BLOCK, description="   ")

    def test_negative_affected_records_raises(self):
        with pytest.raises(ValidationError):
            DedupIssueSummary(stage=PipelineStage.BLOCK,
                              description="test", affected_records=-1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupIssueSummary(stage=PipelineStage.BLOCK,
                              description="test", extra="x")

    def test_all_severities(self):
        for sev in IssueSeverity:
            issue = DedupIssueSummary(
                severity=sev, stage=PipelineStage.BLOCK, description="test",
            )
            assert issue.severity == sev


# -- DedupJobSummary --

class TestDedupJobSummary:
    def test_create_minimal(self):
        js = DedupJobSummary(job_id="j1")
        assert js.job_id == "j1"
        assert js.status == JobStatus.COMPLETED
        assert js.stage == PipelineStage.COMPLETE
        assert js.total_records == 0

    def test_create_full(self):
        js = DedupJobSummary(
            job_id="j2",
            dataset_ids=["ds1", "ds2"],
            status=JobStatus.RUNNING,
            stage=PipelineStage.COMPARE,
            total_records=10000,
            matched=500,
            clustered=100,
            duplicate_rate=0.05,
            started_at=datetime.now(timezone.utc),
        )
        assert js.matched == 500

    def test_empty_job_id_raises(self):
        with pytest.raises(ValidationError, match="job_id must be non-empty"):
            DedupJobSummary(job_id="")

    def test_duplicate_rate_above_one_raises(self):
        with pytest.raises(ValidationError):
            DedupJobSummary(job_id="j", duplicate_rate=1.1)

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupJobSummary(job_id="j", extra="x")


# -- DedupReport --

class TestDedupReport:
    def test_create_minimal(self):
        rpt = DedupReport(job_id="j1")
        assert rpt.job_id == "j1"
        assert rpt.format == ReportFormat.JSON
        assert rpt.summary == ""
        assert rpt.clusters == []
        assert rpt.merge_decisions == []
        assert rpt.statistics is None
        assert rpt.issues == []
        assert isinstance(rpt.generated_at, datetime)

    def test_create_full(self):
        cluster = DuplicateCluster(
            cluster_id="c1",
            member_record_ids=["r1", "r2"],
            member_count=2,
        )
        stats = DedupStatistics(total_jobs=1, total_records=100)
        rpt = DedupReport(
            job_id="j2",
            format=ReportFormat.MARKDOWN,
            summary="Dedup completed successfully",
            clusters=[cluster],
            statistics=stats,
            provenance_hash="ph",
        )
        assert rpt.format == ReportFormat.MARKDOWN
        assert len(rpt.clusters) == 1

    def test_empty_job_id_raises(self):
        with pytest.raises(ValidationError, match="job_id must be non-empty"):
            DedupReport(job_id="")

    def test_all_report_formats(self):
        for fmt in ReportFormat:
            rpt = DedupReport(job_id="j", format=fmt)
            assert rpt.format == fmt

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            DedupReport(job_id="j", extra="x")


# =============================================================================
# 4. REQUEST MODEL TESTS (7 models)
# =============================================================================


class TestFingerprintRequest:
    def test_create_valid(self):
        req = FingerprintRequest(
            records=[{"name": "Alice"}],
            field_set=["name"],
        )
        assert len(req.records) == 1
        assert req.field_set == ["name"]
        assert req.algorithm == FingerprintAlgorithm.SHA256
        assert req.normalize is True

    def test_empty_records_raises(self):
        with pytest.raises(ValidationError):
            FingerprintRequest(records=[], field_set=["name"])

    def test_empty_field_set_raises(self):
        with pytest.raises(ValidationError):
            FingerprintRequest(records=[{"name": "A"}], field_set=[])

    def test_empty_string_in_field_set_raises(self):
        with pytest.raises(ValidationError, match="must not contain empty strings"):
            FingerprintRequest(records=[{"name": "A"}], field_set=[""])

    def test_whitespace_in_field_set_raises(self):
        with pytest.raises(ValidationError, match="must not contain empty strings"):
            FingerprintRequest(records=[{"name": "A"}], field_set=["  "])

    def test_custom_algorithm(self):
        req = FingerprintRequest(
            records=[{"name": "A"}],
            field_set=["name"],
            algorithm=FingerprintAlgorithm.SIMHASH,
            normalize=False,
        )
        assert req.algorithm == FingerprintAlgorithm.SIMHASH
        assert req.normalize is False

    def test_extra_field_raises(self):
        with pytest.raises(ValidationError):
            FingerprintRequest(
                records=[{"name": "A"}],
                field_set=["name"],
                extra="x",
            )


class TestBlockRequest:
    def test_create_valid(self):
        fp = RecordFingerprint(record_id="r1", fingerprint_hash="h1")
        req = BlockRequest(fingerprints=[fp])
        assert len(req.fingerprints) == 1
        assert req.strategy == BlockingStrategy.SORTED_NEIGHBORHOOD
        assert req.window_size == 10

    def test_empty_fingerprints_raises(self):
        with pytest.raises(ValidationError):
            BlockRequest(fingerprints=[])

    def test_custom_strategy(self):
        fp = RecordFingerprint(record_id="r1")
        req = BlockRequest(
            fingerprints=[fp],
            strategy=BlockingStrategy.CANOPY,
            key_fields=["state"],
            window_size=20,
        )
        assert req.strategy == BlockingStrategy.CANOPY
        assert req.window_size == 20

    def test_window_size_below_2_raises(self):
        fp = RecordFingerprint(record_id="r1")
        with pytest.raises(ValidationError):
            BlockRequest(fingerprints=[fp], window_size=1)

    def test_window_size_above_1000_raises(self):
        fp = RecordFingerprint(record_id="r1")
        with pytest.raises(ValidationError):
            BlockRequest(fingerprints=[fp], window_size=1001)

    def test_extra_field_raises(self):
        fp = RecordFingerprint(record_id="r1")
        with pytest.raises(ValidationError):
            BlockRequest(fingerprints=[fp], extra="x")


class TestCompareRequest:
    def test_create_valid(self):
        br = BlockResult(block_key="k", record_ids=["r1", "r2"])
        fcc = FieldComparisonConfig(field_name="name")
        req = CompareRequest(block_results=[br], field_configs=[fcc])
        assert req.sample_rate == 1.0

    def test_empty_block_results_raises(self):
        fcc = FieldComparisonConfig(field_name="name")
        with pytest.raises(ValidationError):
            CompareRequest(block_results=[], field_configs=[fcc])

    def test_empty_field_configs_raises(self):
        br = BlockResult(block_key="k")
        with pytest.raises(ValidationError):
            CompareRequest(block_results=[br], field_configs=[])

    def test_sample_rate_out_of_range_raises(self):
        br = BlockResult(block_key="k")
        fcc = FieldComparisonConfig(field_name="name")
        with pytest.raises(ValidationError):
            CompareRequest(block_results=[br], field_configs=[fcc], sample_rate=1.5)

    def test_extra_field_raises(self):
        br = BlockResult(block_key="k")
        fcc = FieldComparisonConfig(field_name="name")
        with pytest.raises(ValidationError):
            CompareRequest(block_results=[br], field_configs=[fcc], extra="x")


class TestClassifyRequest:
    def test_create_valid(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b", overall_score=0.9)
        req = ClassifyRequest(comparisons=[sr])
        assert req.match_threshold == pytest.approx(0.85)
        assert req.possible_threshold == pytest.approx(0.65)
        assert req.use_fellegi_sunter is False

    def test_empty_comparisons_raises(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(comparisons=[])

    def test_match_threshold_below_0_5_raises(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b")
        with pytest.raises(ValidationError, match="match_threshold must be >= 0.5"):
            ClassifyRequest(comparisons=[sr], match_threshold=0.4)

    def test_match_threshold_at_0_5_ok(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b")
        req = ClassifyRequest(comparisons=[sr], match_threshold=0.5)
        assert req.match_threshold == pytest.approx(0.5)

    def test_extra_field_raises(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b")
        with pytest.raises(ValidationError):
            ClassifyRequest(comparisons=[sr], extra="x")


class TestClusterRequest:
    def test_create_valid(self):
        mr = MatchResult(record_a_id="a", record_b_id="b",
                         classification=MatchClassification.MATCH)
        req = ClusterRequest(matches=[mr])
        assert req.algorithm == ClusterAlgorithm.UNION_FIND
        assert req.min_quality == pytest.approx(0.5)

    def test_empty_matches_raises(self):
        with pytest.raises(ValidationError):
            ClusterRequest(matches=[])

    def test_custom_algorithm(self):
        mr = MatchResult(record_a_id="a", record_b_id="b",
                         classification=MatchClassification.MATCH)
        req = ClusterRequest(
            matches=[mr],
            algorithm=ClusterAlgorithm.CONNECTED_COMPONENTS,
            min_quality=0.8,
        )
        assert req.algorithm == ClusterAlgorithm.CONNECTED_COMPONENTS

    def test_extra_field_raises(self):
        mr = MatchResult(record_a_id="a", record_b_id="b",
                         classification=MatchClassification.MATCH)
        with pytest.raises(ValidationError):
            ClusterRequest(matches=[mr], extra="x")


class TestMergeRequest:
    def test_create_valid(self):
        dc = DuplicateCluster(
            cluster_id="c1",
            member_record_ids=["r1", "r2"],
        )
        req = MergeRequest(clusters=[dc])
        assert req.strategy == MergeStrategy.KEEP_MOST_COMPLETE
        assert req.conflict_resolution == ConflictResolution.MOST_COMPLETE

    def test_empty_clusters_raises(self):
        with pytest.raises(ValidationError):
            MergeRequest(clusters=[])

    def test_custom_strategy(self):
        dc = DuplicateCluster(cluster_id="c1", member_record_ids=["r1"])
        req = MergeRequest(
            clusters=[dc],
            strategy=MergeStrategy.GOLDEN_RECORD,
            conflict_resolution=ConflictResolution.LONGEST,
            source_records={"r1": {"name": "Alice"}},
        )
        assert req.strategy == MergeStrategy.GOLDEN_RECORD

    def test_extra_field_raises(self):
        dc = DuplicateCluster(cluster_id="c1", member_record_ids=["r1"])
        with pytest.raises(ValidationError):
            MergeRequest(clusters=[dc], extra="x")


class TestPipelineRequest:
    def test_create_valid(self):
        rule = DedupRule(name="test")
        req = PipelineRequest(
            records=[{"name": "Alice"}],
            rule=rule,
        )
        assert len(req.records) == 1
        assert req.record_source == RecordSource.SINGLE_DATASET
        assert req.options == {}

    def test_empty_records_raises(self):
        rule = DedupRule(name="test")
        with pytest.raises(ValidationError):
            PipelineRequest(records=[], rule=rule)

    def test_cross_dataset(self):
        rule = DedupRule(name="test")
        req = PipelineRequest(
            records=[{"name": "Alice"}],
            rule=rule,
            dataset_ids=["ds1", "ds2"],
            record_source=RecordSource.CROSS_DATASET,
        )
        assert req.record_source == RecordSource.CROSS_DATASET

    def test_with_options(self):
        rule = DedupRule(name="test")
        req = PipelineRequest(
            records=[{"name": "Alice"}],
            rule=rule,
            options={"timeout": 300, "checkpoint": True},
        )
        assert req.options["timeout"] == 300

    def test_extra_field_raises(self):
        rule = DedupRule(name="test")
        with pytest.raises(ValidationError):
            PipelineRequest(
                records=[{"name": "Alice"}],
                rule=rule,
                extra="x",
            )


# =============================================================================
# 5. SERIALIZATION / DESERIALIZATION ROUNDTRIP TESTS
# =============================================================================


class TestSerializationRoundtrip:
    """Verify model_dump/model_validate for all 15 SDK models."""

    def test_record_fingerprint_roundtrip(self):
        obj = RecordFingerprint(record_id="r1", fingerprint_hash="h1")
        data = obj.model_dump()
        obj2 = RecordFingerprint(**data)
        assert obj.record_id == obj2.record_id
        assert obj.fingerprint_hash == obj2.fingerprint_hash

    def test_block_result_roundtrip(self):
        obj = BlockResult(block_key="k", record_ids=["r1", "r2"], record_count=2)
        data = obj.model_dump()
        obj2 = BlockResult(**data)
        assert obj.block_key == obj2.block_key
        assert obj.record_count == obj2.record_count

    def test_similarity_result_roundtrip(self):
        obj = SimilarityResult(record_a_id="a", record_b_id="b", overall_score=0.88)
        data = obj.model_dump()
        obj2 = SimilarityResult(**data)
        assert obj.overall_score == pytest.approx(obj2.overall_score)

    def test_field_comparison_config_roundtrip(self):
        obj = FieldComparisonConfig(field_name="name", weight=2.5)
        data = obj.model_dump()
        obj2 = FieldComparisonConfig(**data)
        assert obj.weight == pytest.approx(obj2.weight)

    def test_match_result_roundtrip(self):
        obj = MatchResult(record_a_id="a", record_b_id="b",
                          classification=MatchClassification.MATCH, confidence=0.95)
        data = obj.model_dump()
        obj2 = MatchResult(**data)
        assert obj.classification == obj2.classification

    def test_merge_conflict_roundtrip(self):
        obj = MergeConflict(field_name="email", chosen_value="a@b.com")
        data = obj.model_dump()
        obj2 = MergeConflict(**data)
        assert obj.chosen_value == obj2.chosen_value

    def test_duplicate_cluster_roundtrip(self):
        obj = DuplicateCluster(
            cluster_id="c1",
            member_record_ids=["r1", "r2"],
            cluster_quality=0.9,
        )
        data = obj.model_dump()
        obj2 = DuplicateCluster(**data)
        assert obj.cluster_quality == pytest.approx(obj2.cluster_quality)

    def test_merge_decision_roundtrip(self):
        obj = MergeDecision(cluster_id="c1", merged_record={"name": "Alice"})
        data = obj.model_dump()
        obj2 = MergeDecision(**data)
        assert obj.merged_record == obj2.merged_record

    def test_dedup_rule_roundtrip(self):
        obj = DedupRule(name="test", match_threshold=0.90)
        data = obj.model_dump()
        obj2 = DedupRule(**data)
        assert obj.match_threshold == pytest.approx(obj2.match_threshold)

    def test_dedup_job_roundtrip(self):
        obj = DedupJob(total_records=1000, matched=100)
        data = obj.model_dump()
        obj2 = DedupJob(**data)
        assert obj.total_records == obj2.total_records

    def test_pipeline_checkpoint_roundtrip(self):
        obj = PipelineCheckpoint(job_id="j1", stage=PipelineStage.BLOCK)
        data = obj.model_dump()
        obj2 = PipelineCheckpoint(**data)
        assert obj.stage == obj2.stage

    def test_dedup_statistics_roundtrip(self):
        obj = DedupStatistics(total_jobs=5, total_records=10000)
        data = obj.model_dump()
        obj2 = DedupStatistics(**data)
        assert obj.total_jobs == obj2.total_jobs

    def test_dedup_issue_summary_roundtrip(self):
        obj = DedupIssueSummary(
            stage=PipelineStage.COMPARE,
            description="too many comparisons",
        )
        data = obj.model_dump()
        obj2 = DedupIssueSummary(**data)
        assert obj.description == obj2.description

    def test_dedup_job_summary_roundtrip(self):
        obj = DedupJobSummary(job_id="j1", total_records=500)
        data = obj.model_dump()
        obj2 = DedupJobSummary(**data)
        assert obj.total_records == obj2.total_records

    def test_dedup_report_roundtrip(self):
        obj = DedupReport(job_id="j1", summary="All done")
        data = obj.model_dump()
        obj2 = DedupReport(**data)
        assert obj.summary == obj2.summary


# =============================================================================
# 6. JSON SERIALIZATION TESTS
# =============================================================================


class TestJsonSerialization:
    """Verify model_dump_json / model_validate_json roundtrip."""

    def test_record_fingerprint_json(self):
        obj = RecordFingerprint(record_id="r1")
        json_str = obj.model_dump_json()
        obj2 = RecordFingerprint.model_validate_json(json_str)
        assert obj.record_id == obj2.record_id

    def test_dedup_rule_json(self):
        obj = DedupRule(name="test")
        json_str = obj.model_dump_json()
        obj2 = DedupRule.model_validate_json(json_str)
        assert obj.name == obj2.name

    def test_dedup_job_json(self):
        obj = DedupJob(total_records=42)
        json_str = obj.model_dump_json()
        obj2 = DedupJob.model_validate_json(json_str)
        assert obj2.total_records == 42

    def test_similarity_result_json(self):
        obj = SimilarityResult(record_a_id="a", record_b_id="b", overall_score=0.77)
        json_str = obj.model_dump_json()
        obj2 = SimilarityResult.model_validate_json(json_str)
        assert obj2.overall_score == pytest.approx(0.77)

    def test_match_result_json(self):
        obj = MatchResult(
            record_a_id="a",
            record_b_id="b",
            classification=MatchClassification.NON_MATCH,
        )
        json_str = obj.model_dump_json()
        obj2 = MatchResult.model_validate_json(json_str)
        assert obj2.classification == MatchClassification.NON_MATCH


# =============================================================================
# 7. EDGE CASE AND BOUNDARY TESTS
# =============================================================================


class TestEdgeCases:
    """Test boundary values and edge cases across models."""

    def test_similarity_result_score_zero(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b", overall_score=0.0)
        assert sr.overall_score == 0.0

    def test_similarity_result_score_one(self):
        sr = SimilarityResult(record_a_id="a", record_b_id="b", overall_score=1.0)
        assert sr.overall_score == 1.0

    def test_cluster_quality_zero(self):
        dc = DuplicateCluster(cluster_quality=0.0)
        assert dc.cluster_quality == 0.0

    def test_cluster_quality_one(self):
        dc = DuplicateCluster(cluster_quality=1.0)
        assert dc.cluster_quality == 1.0

    def test_field_weight_boundary_zero(self):
        fcc = FieldComparisonConfig(field_name="f", weight=0.0)
        assert fcc.weight == 0.0

    def test_field_weight_boundary_ten(self):
        fcc = FieldComparisonConfig(field_name="f", weight=10.0)
        assert fcc.weight == 10.0

    def test_match_result_confidence_zero(self):
        mr = MatchResult(
            record_a_id="a", record_b_id="b",
            classification=MatchClassification.NON_MATCH,
            confidence=0.0,
        )
        assert mr.confidence == 0.0

    def test_match_result_confidence_one(self):
        mr = MatchResult(
            record_a_id="a", record_b_id="b",
            classification=MatchClassification.MATCH,
            confidence=1.0,
        )
        assert mr.confidence == 1.0

    def test_dedup_job_all_stages_progress(self):
        """All pipeline stages should have defined progress."""
        for stage in PipelineStage:
            job = DedupJob(status=JobStatus.RUNNING, stage=stage)
            assert 0.0 <= job.progress_pct <= 100.0

    def test_dedup_job_high_duplicate_rate(self):
        job = DedupJob(total_records=100, matched=100)
        assert job.duplicate_rate == pytest.approx(1.0)

    def test_large_record_ids_list(self):
        ids = [f"r-{i}" for i in range(1000)]
        dc = DuplicateCluster(member_record_ids=ids, member_count=1000)
        assert dc.member_count == 1000

    def test_unicode_field_name(self):
        fcc = FieldComparisonConfig(field_name="nombre_completo")
        assert fcc.field_name == "nombre_completo"

    def test_unicode_record_id(self):
        fp = RecordFingerprint(record_id="rec-\u00e9\u00e8\u00ea")
        assert "rec-" in fp.record_id

    def test_special_chars_in_block_key(self):
        br = BlockResult(block_key="key/with:special|chars")
        assert br.block_key == "key/with:special|chars"

    def test_dedup_rule_match_threshold_equals_possible(self):
        """match_threshold can equal possible_threshold."""
        rule = DedupRule(name="r", match_threshold=0.65, possible_threshold=0.65)
        assert rule.match_threshold == rule.possible_threshold

    def test_statistics_duplicate_rate_boundary(self):
        stats = DedupStatistics(duplicate_rate=1.0)
        assert stats.duplicate_rate == 1.0

    def test_pipeline_checkpoint_empty_state_data(self):
        cp = PipelineCheckpoint(job_id="j", stage=PipelineStage.BLOCK, state_data={})
        assert cp.state_data == {}

    def test_merge_conflict_none_chosen_value(self):
        mc = MergeConflict(field_name="f", chosen_value=None)
        assert mc.chosen_value is None

    def test_dedup_report_with_nested_models(self):
        cluster = DuplicateCluster(
            cluster_id="c1",
            member_record_ids=["r1"],
            member_count=1,
        )
        issue = DedupIssueSummary(
            stage=PipelineStage.MERGE,
            description="test issue",
        )
        decision = MergeDecision(cluster_id="c1")
        stats = DedupStatistics(total_jobs=1)
        rpt = DedupReport(
            job_id="j1",
            clusters=[cluster],
            merge_decisions=[decision],
            statistics=stats,
            issues=[issue],
        )
        assert len(rpt.clusters) == 1
        assert rpt.statistics.total_jobs == 1
        assert len(rpt.issues) == 1
        assert len(rpt.merge_decisions) == 1


# =============================================================================
# 8. __all__ EXPORTS TEST
# =============================================================================


class TestModelsExports:
    """Verify the models module exports all expected names."""

    def test_all_enums_exported(self):
        from greenlang.duplicate_detector import models
        enum_names = [
            "FingerprintAlgorithm", "BlockingStrategy", "SimilarityAlgorithm",
            "MatchClassification", "ClusterAlgorithm", "MergeStrategy",
            "ConflictResolution", "JobStatus", "PipelineStage",
            "RecordSource", "FieldType", "IssueSeverity", "ReportFormat",
        ]
        for name in enum_names:
            assert name in models.__all__, f"{name} missing from __all__"

    def test_all_sdk_models_exported(self):
        from greenlang.duplicate_detector import models
        model_names = [
            "RecordFingerprint", "BlockResult", "SimilarityResult",
            "FieldComparisonConfig", "MatchResult", "MergeConflict",
            "DuplicateCluster", "MergeDecision", "DedupRule", "DedupJob",
            "PipelineCheckpoint", "DedupStatistics", "DedupIssueSummary",
            "DedupJobSummary", "DedupReport",
        ]
        for name in model_names:
            assert name in models.__all__, f"{name} missing from __all__"

    def test_all_request_models_exported(self):
        from greenlang.duplicate_detector import models
        request_names = [
            "FingerprintRequest", "BlockRequest", "CompareRequest",
            "ClassifyRequest", "ClusterRequest", "MergeRequest",
            "PipelineRequest",
        ]
        for name in request_names:
            assert name in models.__all__, f"{name} missing from __all__"

    def test_all_constants_exported(self):
        from greenlang.duplicate_detector import models
        constant_names = [
            "DEFAULT_FIELD_WEIGHTS", "SIMILARITY_ALGORITHM_DEFAULTS",
            "BLOCKING_STRATEGY_DEFAULTS", "MAX_RECORDS_PER_COMPARISON",
            "DEFAULT_NGRAM_SIZE", "DEFAULT_MATCH_THRESHOLD",
            "DEFAULT_POSSIBLE_THRESHOLD", "DEFAULT_NON_MATCH_THRESHOLD",
            "MAX_BLOCK_COMPARISONS_WARN", "FINGERPRINT_ALGORITHMS",
            "BLOCKING_STRATEGIES", "SIMILARITY_ALGORITHMS",
            "MERGE_STRATEGIES", "CONFLICT_RESOLUTIONS",
            "PIPELINE_STAGE_ORDER", "FIELD_TYPES",
            "ISSUE_SEVERITY_ORDER", "REPORT_FORMAT_OPTIONS",
        ]
        for name in constant_names:
            assert name in models.__all__, f"{name} missing from __all__"
