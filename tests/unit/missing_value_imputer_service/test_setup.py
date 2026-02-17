# -*- coding: utf-8 -*-
"""
Unit tests for MissingValueImputerService facade - AGENT-DATA-012

Tests MissingValueImputerService, response models (AnalysisResponse,
ImputationResponse, BatchImputationResponse, ValidationResponse,
RuleResponse, TemplateResponse, PipelineResponse, StatsResponse),
_ProvenanceTracker, job management (create/list/get/delete),
analyze_missingness/_fallback_analyze, impute_values/_fallback_impute,
impute_batch, validate_imputation/_fallback_validate,
create_rule/list_rules/update_rule/delete_rule,
create_template/list_templates, run_pipeline, get_statistics,
health_check, get_metrics, configure_missing_value_imputer,
get_missing_value_imputer, get_router, _compute_hash, and edge cases.
Target: 85+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from greenlang.missing_value_imputer.config import MissingValueImputerConfig
from greenlang.missing_value_imputer.setup import (
    AnalysisResponse,
    BatchImputationResponse,
    ImputationResponse,
    MissingValueImputerService,
    PipelineResponse,
    RuleResponse,
    StatsResponse,
    TemplateResponse,
    ValidationResponse,
    _ProvenanceTracker,
    _compute_hash,
    _utcnow,
    configure_missing_value_imputer,
    get_missing_value_imputer,
    get_router,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a MissingValueImputerService instance."""
    return MissingValueImputerService()


@pytest.fixture
def started_service():
    """Create a started MissingValueImputerService instance."""
    svc = MissingValueImputerService()
    svc.startup()
    return svc


@pytest.fixture
def records_with_missing():
    """Create records with missing values for testing."""
    return [
        {"val": 1.0, "cat": "A"},
        {"val": 2.0, "cat": "A"},
        {"val": None, "cat": "B"},
        {"val": 4.0, "cat": None},
        {"val": 5.0, "cat": "B"},
    ]


@pytest.fixture
def complete_records():
    """Create complete records with no missing values."""
    return [
        {"val": float(i), "cat": "A"} for i in range(10)
    ]


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestResponseModels:
    def test_analysis_response_defaults(self):
        r = AnalysisResponse()
        assert r.total_records == 0
        assert r.total_columns == 0
        assert r.columns_with_missing == 0
        assert r.overall_missing_pct == 0.0
        assert r.analysis_id  # UUID auto-generated
        assert r.provenance_hash == ""

    def test_imputation_response_defaults(self):
        r = ImputationResponse()
        assert r.column_name == ""
        assert r.strategy == "mean"
        assert r.values_imputed == 0
        assert r.avg_confidence == 0.0
        assert r.result_id

    def test_batch_imputation_response_defaults(self):
        r = BatchImputationResponse()
        assert r.total_columns == 0
        assert r.total_values_imputed == 0
        assert r.results == []
        assert r.batch_id

    def test_validation_response_defaults(self):
        r = ValidationResponse()
        assert r.overall_passed is False
        assert r.columns_passed == 0
        assert r.columns_failed == 0
        assert r.validation_id

    def test_rule_response_defaults(self):
        r = RuleResponse()
        assert r.name == ""
        assert r.target_column == ""
        assert r.priority == "medium"
        assert r.is_active is True
        assert r.rule_id

    def test_template_response_defaults(self):
        r = TemplateResponse()
        assert r.name == ""
        assert r.column_strategies == {}
        assert r.default_strategy == "mean"
        assert r.confidence_threshold == 0.7
        assert r.template_id

    def test_pipeline_response_defaults(self):
        r = PipelineResponse()
        assert r.status == "completed"
        assert r.total_records == 0
        assert r.total_values_imputed == 0
        assert r.stages == {}
        assert r.pipeline_id

    def test_stats_response_defaults(self):
        r = StatsResponse()
        assert r.total_jobs == 0
        assert r.completed_jobs == 0
        assert r.failed_jobs == 0
        assert r.total_rules == 0
        assert r.by_strategy == {}

    def test_analysis_response_model_dump(self):
        r = AnalysisResponse(total_records=10, columns_with_missing=2)
        d = r.model_dump(mode="json")
        assert d["total_records"] == 10
        assert d["columns_with_missing"] == 2

    def test_imputation_response_with_values(self):
        r = ImputationResponse(
            column_name="test_col",
            strategy="median",
            values_imputed=5,
            avg_confidence=0.85,
            min_confidence=0.70,
        )
        assert r.column_name == "test_col"
        assert r.strategy == "median"
        assert r.values_imputed == 5


# ---------------------------------------------------------------------------
# _ProvenanceTracker tests
# ---------------------------------------------------------------------------


class TestProvenanceTracker:
    def test_initial_state(self):
        pt = _ProvenanceTracker()
        assert pt.entry_count == 0

    def test_record_returns_hash(self):
        pt = _ProvenanceTracker()
        h = pt.record("job", "123", "create", "abc123")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_entry_count_increments(self):
        pt = _ProvenanceTracker()
        pt.record("job", "1", "create", "h1")
        pt.record("job", "2", "create", "h2")
        assert pt.entry_count == 2

    def test_entries_stored(self):
        pt = _ProvenanceTracker()
        pt.record("analysis", "a1", "analyze", "h1")
        assert len(pt._entries) == 1
        assert pt._entries[0]["entity_type"] == "analysis"
        assert pt._entries[0]["entity_id"] == "a1"
        assert "timestamp" in pt._entries[0]
        assert "entry_hash" in pt._entries[0]

    def test_record_with_user_id(self):
        pt = _ProvenanceTracker()
        pt.record("rule", "r1", "create", "h1", user_id="test_user")
        assert pt._entries[0]["user_id"] == "test_user"

    def test_default_user_is_system(self):
        pt = _ProvenanceTracker()
        pt.record("rule", "r1", "create", "h1")
        assert pt._entries[0]["user_id"] == "system"


# ---------------------------------------------------------------------------
# _compute_hash / _utcnow tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_compute_hash_dict(self):
        h = _compute_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        data = {"a": 1, "b": 2}
        assert _compute_hash(data) == _compute_hash(data)

    def test_compute_hash_different_data(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_compute_hash_pydantic_model(self):
        model = AnalysisResponse(total_records=5)
        h = _compute_hash(model)
        assert len(h) == 64

    def test_utcnow_returns_datetime(self):
        dt = _utcnow()
        assert dt.microsecond == 0
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# Service initialization
# ---------------------------------------------------------------------------


class TestServiceInit:
    def test_default_config(self, service):
        assert service.config is not None
        assert isinstance(service.config, MissingValueImputerConfig)

    def test_custom_config(self):
        cfg = MissingValueImputerConfig(default_strategy="median")
        svc = MissingValueImputerService(config=cfg)
        assert svc.config.default_strategy == "median"

    def test_provenance_tracker_created(self, service):
        assert service.provenance is not None
        assert isinstance(service.provenance, _ProvenanceTracker)

    def test_engines_initialized(self, service):
        # All engines should be initialized (they are optional, may be None)
        assert service.analyzer_engine is not None
        assert service.statistical_engine is not None
        assert service.ml_engine is not None
        assert service.rule_based_engine is not None
        assert service.timeseries_engine is not None
        assert service.validation_engine is not None

    def test_stores_empty(self, service):
        assert len(service._jobs) == 0
        assert len(service._analysis_results) == 0
        assert len(service._imputation_results) == 0
        assert len(service._batch_results) == 0
        assert len(service._validation_results) == 0
        assert len(service._pipeline_results) == 0
        assert len(service._rules) == 0
        assert len(service._templates) == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_startup(self, service):
        assert service._started is False
        service.startup()
        assert service._started is True

    def test_shutdown(self, started_service):
        assert started_service._started is True
        started_service.shutdown()
        assert started_service._started is False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_not_started(self, service):
        result = service.health_check()
        assert result["status"] == "not_started"
        assert result["started"] is False

    def test_started(self, started_service):
        result = started_service.health_check()
        assert result["status"] == "healthy"
        assert result["started"] is True

    def test_engines_present(self, service):
        result = service.health_check()
        engines = result["engines"]
        assert "analyzer" in engines
        assert "statistical" in engines
        assert "ml" in engines
        assert "rule_based" in engines
        assert "timeseries" in engines
        assert "validation" in engines
        assert "pipeline" in engines

    def test_counts_reported(self, service):
        result = service.health_check()
        assert result["jobs"] == 0
        assert result["analyses"] == 0
        assert result["rules"] == 0
        assert result["templates"] == 0
        assert result["provenance_entries"] == 0
        assert "prometheus_available" in result


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------


class TestJobManagement:
    def test_create_job(self, service):
        job = service.create_job({"records": [{"a": 1}], "dataset_id": "ds1"})
        assert "job_id" in job
        assert job["status"] == "pending"
        assert job["dataset_id"] == "ds1"
        assert job["total_records"] == 1
        assert job["provenance_hash"]
        assert len(job["provenance_hash"]) == 64

    def test_create_job_updates_stats(self, service):
        service.create_job({"records": [{"a": 1}]})
        assert service._stats.total_jobs == 1
        assert service._stats.by_status.get("pending", 0) == 1

    def test_create_job_records_provenance(self, service):
        service.create_job({"records": [{"a": 1}]})
        assert service.provenance.entry_count == 1

    def test_list_jobs_empty(self, service):
        jobs = service.list_jobs()
        assert jobs == []

    def test_list_jobs_after_create(self, service):
        service.create_job({"records": [{"a": 1}]})
        service.create_job({"records": [{"b": 2}]})
        jobs = service.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_with_status_filter(self, service):
        service.create_job({"records": [{"a": 1}]})
        jobs = service.list_jobs(status="pending")
        assert len(jobs) == 1
        jobs = service.list_jobs(status="completed")
        assert len(jobs) == 0

    def test_list_jobs_with_limit_offset(self, service):
        for i in range(5):
            service.create_job({"records": [{"a": i}]})
        jobs = service.list_jobs(limit=2, offset=1)
        assert len(jobs) == 2

    def test_get_job(self, service):
        job = service.create_job({"records": [{"a": 1}]})
        result = service.get_job(job["job_id"])
        assert result is not None
        assert result["job_id"] == job["job_id"]

    def test_get_job_not_found(self, service):
        result = service.get_job("nonexistent")
        assert result is None

    def test_delete_job(self, service):
        job = service.create_job({"records": [{"a": 1}]})
        result = service.delete_job(job["job_id"])
        assert result is True
        assert service.get_job(job["job_id"])["status"] == "cancelled"

    def test_delete_job_not_found(self, service):
        result = service.delete_job("nonexistent")
        assert result is False

    def test_delete_job_provenance(self, service):
        job = service.create_job({"records": [{"a": 1}]})
        service.delete_job(job["job_id"])
        # 1 create + 1 cancel = 2 provenance entries
        assert service.provenance.entry_count == 2


# ---------------------------------------------------------------------------
# Missingness analysis
# ---------------------------------------------------------------------------


class TestAnalyzeMissingness:
    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.analyze_missingness([])

    def test_basic_analysis(self, service, records_with_missing):
        result = service.analyze_missingness(records_with_missing)
        assert isinstance(result, AnalysisResponse)
        assert result.total_records == 5
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_analysis_processing_time(self, service, records_with_missing):
        result = service.analyze_missingness(records_with_missing)
        assert result.processing_time_ms >= 0

    def test_analysis_stored(self, service, records_with_missing):
        result = service.analyze_missingness(records_with_missing)
        stored = service.get_analysis(result.analysis_id)
        assert stored is not None
        assert stored.analysis_id == result.analysis_id

    def test_analysis_not_found(self, service):
        result = service.get_analysis("nonexistent")
        assert result is None

    def test_analysis_updates_stats(self, service, records_with_missing):
        service.analyze_missingness(records_with_missing)
        assert service._stats.total_analyses >= 1

    def test_analysis_with_specific_columns(self, service, records_with_missing):
        result = service.analyze_missingness(records_with_missing, columns=["val"])
        assert isinstance(result, AnalysisResponse)

    def test_fallback_analyze_no_missing(self, service, complete_records):
        result = service.analyze_missingness(complete_records)
        assert isinstance(result, AnalysisResponse)
        assert result.columns_with_missing == 0


class TestFallbackAnalyze:
    def test_fallback_mode(self):
        """Test fallback analysis when analyzer engine is None."""
        svc = MissingValueImputerService()
        svc._analyzer_engine = None
        records = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        result = svc.analyze_missingness(records)
        assert isinstance(result, AnalysisResponse)
        assert result.total_records == 3
        assert result.columns_with_missing > 0
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_fallback_complete_records(self):
        svc = MissingValueImputerService()
        svc._analyzer_engine = None
        records = [{"x": 1.0}, {"x": 2.0}]
        result = svc.analyze_missingness(records)
        assert result.complete_records == 2
        assert result.complete_record_pct == 1.0

    def test_fallback_strategy_recommendations(self):
        svc = MissingValueImputerService()
        svc._analyzer_engine = None
        records = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        result = svc.analyze_missingness(records)
        # Should have at least one recommendation for the column with missing
        assert len(result.strategy_recommendations) > 0


class TestRecommendStrategy:
    def test_numeric_low_missing(self, service):
        result = service._recommend_strategy("col", [1.0, 2.0, 3.0], 0.05)
        assert result == "mean"

    def test_numeric_medium_missing(self, service):
        result = service._recommend_strategy("col", [1.0, 2.0, 3.0], 0.2)
        assert result == "median"

    def test_numeric_high_missing(self, service):
        result = service._recommend_strategy("col", [1.0, 2.0, 3.0], 0.4)
        assert result == "knn"

    def test_numeric_very_high_missing(self, service):
        result = service._recommend_strategy("col", [1.0, 2.0, 3.0], 0.6)
        assert result == "mice"

    def test_categorical_data(self, service):
        result = service._recommend_strategy("col", ["A", "B", "C"], 0.1)
        assert result == "mode"

    def test_empty_values(self, service):
        result = service._recommend_strategy("col", [], 1.0)
        assert result == "mode"


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


class TestImputeValues:
    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.impute_values([], "col")

    def test_missing_column_raises(self, service):
        with pytest.raises(ValueError, match="not found"):
            service.impute_values([{"a": 1}], "nonexistent")

    def test_basic_imputation(self, service, records_with_missing):
        result = service.impute_values(records_with_missing, "val")
        assert isinstance(result, ImputationResponse)
        assert result.column_name == "val"
        assert result.values_imputed > 0
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_imputation_stored(self, service, records_with_missing):
        result = service.impute_values(records_with_missing, "val")
        stored = service.get_results(result.result_id)
        assert stored is not None

    def test_get_results_not_found(self, service):
        assert service.get_results("nonexistent") is None

    def test_imputation_with_strategy(self, service, records_with_missing):
        result = service.impute_values(
            records_with_missing, "val", strategy="median",
        )
        assert isinstance(result, ImputationResponse)

    def test_imputation_updates_stats(self, service, records_with_missing):
        service.impute_values(records_with_missing, "val")
        assert service._stats.total_values_imputed > 0


class TestFallbackImpute:
    def test_fallback_mean(self):
        svc = MissingValueImputerService()
        svc._statistical_engine = None
        svc._ml_engine = None
        svc._timeseries_engine = None
        svc._rule_based_engine = None
        records = [{"x": 1.0}, {"x": 2.0}, {"x": None}, {"x": 4.0}]
        result = svc.impute_values(records, "x", strategy="mean")
        assert result.values_imputed == 1
        assert result.avg_confidence == 0.75
        # Mean of 1, 2, 4 = 7/3 ~= 2.333
        iv = result.imputed_values[0]
        assert abs(iv["imputed_value"] - 7.0 / 3.0) < 0.01

    def test_fallback_median(self):
        svc = MissingValueImputerService()
        svc._statistical_engine = None
        svc._ml_engine = None
        svc._timeseries_engine = None
        svc._rule_based_engine = None
        records = [{"x": 1.0}, {"x": 3.0}, {"x": None}, {"x": 5.0}]
        result = svc.impute_values(records, "x", strategy="median")
        assert result.values_imputed == 1
        iv = result.imputed_values[0]
        assert iv["imputed_value"] == pytest.approx(3.0, rel=0.01)

    def test_fallback_mode_categorical(self):
        svc = MissingValueImputerService()
        svc._statistical_engine = None
        svc._ml_engine = None
        svc._timeseries_engine = None
        svc._rule_based_engine = None
        records = [
            {"c": "A"}, {"c": "A"}, {"c": "B"}, {"c": None},
        ]
        result = svc.impute_values(records, "c", strategy="mode")
        assert result.values_imputed == 1
        iv = result.imputed_values[0]
        assert iv["imputed_value"] == "A"

    def test_fallback_all_missing(self):
        svc = MissingValueImputerService()
        svc._statistical_engine = None
        svc._ml_engine = None
        svc._timeseries_engine = None
        svc._rule_based_engine = None
        records = [{"x": None}, {"x": None}]
        result = svc.impute_values(records, "x", strategy="mean")
        assert result.values_imputed == 2
        # Should impute 0 when all missing
        assert result.imputed_values[0]["imputed_value"] == 0

    def test_fallback_completeness_tracking(self):
        svc = MissingValueImputerService()
        svc._statistical_engine = None
        svc._ml_engine = None
        svc._timeseries_engine = None
        svc._rule_based_engine = None
        records = [{"x": 1.0}, {"x": None}]
        result = svc.impute_values(records, "x", strategy="mean")
        assert result.completeness_before == pytest.approx(0.5, abs=0.01)
        assert result.completeness_after == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Batch imputation
# ---------------------------------------------------------------------------


class TestImputeBatch:
    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.impute_batch([])

    def test_basic_batch(self, service, records_with_missing):
        result = service.impute_batch(records_with_missing)
        assert isinstance(result, BatchImputationResponse)
        assert result.total_values_imputed > 0
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_batch_stored(self, service, records_with_missing):
        result = service.impute_batch(records_with_missing)
        assert result.batch_id in service._batch_results

    def test_batch_with_strategies(self, service, records_with_missing):
        result = service.impute_batch(
            records_with_missing,
            strategies={"val": "median"},
        )
        assert isinstance(result, BatchImputationResponse)

    def test_batch_no_missing_columns(self, service, complete_records):
        result = service.impute_batch(complete_records)
        assert result.total_values_imputed == 0
        assert result.total_columns == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateImputation:
    def test_empty_original_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.validate_imputation([], [{"a": 1}])

    def test_empty_imputed_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.validate_imputation([{"a": 1}], [])

    def test_basic_validation(self, service):
        orig = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        imp = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        result = service.validate_imputation(orig, imp)
        assert isinstance(result, ValidationResponse)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_validation_stored(self, service):
        orig = [{"x": 1.0}, {"x": None}]
        imp = [{"x": 1.0}, {"x": 2.0}]
        result = service.validate_imputation(orig, imp)
        stored = service.get_validation(result.validation_id)
        assert stored is not None

    def test_get_validation_not_found(self, service):
        assert service.get_validation("nonexistent") is None

    def test_validation_updates_stats(self, service):
        orig = [{"x": 1.0}, {"x": None}]
        imp = [{"x": 1.0}, {"x": 2.0}]
        service.validate_imputation(orig, imp)
        assert service._stats.total_validations >= 1


class TestFallbackValidate:
    def test_fallback_plausibility(self):
        svc = MissingValueImputerService()
        svc._validation_engine = None
        orig = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        imp = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        result = svc.validate_imputation(orig, imp)
        assert isinstance(result, ValidationResponse)
        assert result.overall_passed is True

    def test_fallback_out_of_range(self):
        svc = MissingValueImputerService()
        svc._validation_engine = None
        orig = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        imp = [{"x": 1.0}, {"x": 999.0}, {"x": 3.0}]
        result = svc.validate_imputation(orig, imp)
        # 999 is far outside [1,3] * 1.5 range
        assert result.columns_failed > 0

    def test_fallback_no_imputed_columns(self):
        svc = MissingValueImputerService()
        svc._validation_engine = None
        orig = [{"x": 1.0}, {"x": 2.0}]
        imp = [{"x": 1.0}, {"x": 2.0}]
        result = svc.validate_imputation(orig, imp)
        assert result.overall_passed is True
        assert result.total_columns == 0


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------


class TestRuleManagement:
    def test_create_rule(self, service):
        result = service.create_rule(
            name="test_rule",
            target_column="val",
            conditions=[{"field_name": "cat", "condition_type": "equals", "value": "A"}],
            impute_value=1.0,
            priority="high",
            justification="Test rule",
        )
        assert isinstance(result, RuleResponse)
        assert result.name == "test_rule"
        assert result.target_column == "val"
        assert result.priority == "high"
        assert result.is_active is True
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_create_rule_stored(self, service):
        result = service.create_rule(name="r1", target_column="c1")
        assert result.rule_id in service._rules

    def test_create_rule_updates_stats(self, service):
        service.create_rule(name="r1", target_column="c1")
        assert service._stats.total_rules == 1

    def test_list_rules_empty(self, service):
        rules = service.list_rules()
        assert rules == []

    def test_list_rules_after_create(self, service):
        service.create_rule(name="r1", target_column="c1")
        service.create_rule(name="r2", target_column="c2")
        rules = service.list_rules()
        assert len(rules) == 2

    def test_list_rules_active_filter(self, service):
        r = service.create_rule(name="r1", target_column="c1")
        service.delete_rule(r.rule_id)
        active_rules = service.list_rules(is_active=True)
        inactive_rules = service.list_rules(is_active=False)
        assert len(active_rules) == 0
        assert len(inactive_rules) == 1

    def test_update_rule(self, service):
        r = service.create_rule(name="r1", target_column="c1")
        updated = service.update_rule(r.rule_id, name="updated_r1", priority="critical")
        assert updated is not None
        assert updated.name == "updated_r1"
        assert updated.priority == "critical"

    def test_update_rule_not_found(self, service):
        result = service.update_rule("nonexistent", name="test")
        assert result is None

    def test_update_rule_ignores_unknown_fields(self, service):
        r = service.create_rule(name="r1", target_column="c1")
        updated = service.update_rule(r.rule_id, unknown_field="value")
        assert updated is not None
        # Unknown field is silently ignored

    def test_delete_rule(self, service):
        r = service.create_rule(name="r1", target_column="c1")
        result = service.delete_rule(r.rule_id)
        assert result is True
        # Rule should still exist but be inactive
        rule = service._rules[r.rule_id]
        assert rule["is_active"] is False

    def test_delete_rule_not_found(self, service):
        result = service.delete_rule("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Template management
# ---------------------------------------------------------------------------


class TestTemplateManagement:
    def test_create_template(self, service):
        result = service.create_template(
            name="test_template",
            description="A test template",
            strategies={"val": "median", "cat": "mode"},
        )
        assert isinstance(result, TemplateResponse)
        assert result.name == "test_template"
        assert result.description == "A test template"
        assert result.column_strategies == {"val": "median", "cat": "mode"}
        assert result.is_active is True
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_create_template_stored(self, service):
        result = service.create_template(name="t1")
        assert result.template_id in service._templates

    def test_create_template_updates_stats(self, service):
        service.create_template(name="t1")
        assert service._stats.total_templates == 1

    def test_list_templates_empty(self, service):
        templates = service.list_templates()
        assert templates == []

    def test_list_templates_after_create(self, service):
        service.create_template(name="t1")
        service.create_template(name="t2")
        templates = service.list_templates()
        assert len(templates) == 2

    def test_template_default_strategy(self, service):
        result = service.create_template(name="t1")
        assert result.default_strategy == service.config.default_strategy

    def test_template_confidence_threshold(self, service):
        result = service.create_template(name="t1")
        assert result.confidence_threshold == service.config.confidence_threshold


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="not be empty"):
            service.run_pipeline([])

    def test_basic_pipeline(self, service, records_with_missing):
        result = service.run_pipeline(records_with_missing)
        assert isinstance(result, PipelineResponse)
        assert result.status == "completed"
        assert result.total_records == 5
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_pipeline_stored(self, service, records_with_missing):
        result = service.run_pipeline(records_with_missing)
        assert result.pipeline_id in service._pipeline_results

    def test_pipeline_updates_stats(self, service, records_with_missing):
        service.run_pipeline(records_with_missing)
        assert service._stats.completed_jobs >= 1

    def test_pipeline_stages(self, service, records_with_missing):
        result = service.run_pipeline(records_with_missing)
        assert "analyze" in result.stages

    def test_pipeline_no_missing(self, service, complete_records):
        result = service.run_pipeline(complete_records)
        assert result.status == "completed"
        assert result.total_values_imputed == 0

    def test_pipeline_active_jobs_tracking(self, service, records_with_missing):
        # After pipeline completes, active_jobs should be back to 0
        service.run_pipeline(records_with_missing)
        assert service._active_jobs == 0

    def test_pipeline_processing_time(self, service, records_with_missing):
        result = service.run_pipeline(records_with_missing)
        assert result.processing_time_ms > 0


# ---------------------------------------------------------------------------
# Statistics and metrics
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_initial_stats(self, service):
        stats = service.get_statistics()
        assert isinstance(stats, StatsResponse)
        assert stats.total_jobs == 0
        assert stats.active_jobs == 0
        assert stats.provenance_entries == 0

    def test_stats_after_operations(self, service, records_with_missing):
        service.create_job({"records": records_with_missing})
        service.analyze_missingness(records_with_missing)
        stats = service.get_statistics()
        assert stats.total_jobs == 1
        assert stats.total_analyses >= 1
        assert stats.provenance_entries >= 2

    def test_stats_avg_confidence(self, service, records_with_missing):
        service.impute_values(records_with_missing, "val")
        stats = service.get_statistics()
        assert stats.avg_confidence > 0

    def test_stats_rules_templates(self, service):
        service.create_rule(name="r1", target_column="c1")
        service.create_template(name="t1")
        stats = service.get_statistics()
        assert stats.total_rules == 1
        assert stats.total_templates == 1


class TestGetMetrics:
    def test_metrics_dict(self, service):
        metrics = service.get_metrics()
        assert isinstance(metrics, dict)
        assert "prometheus_available" in metrics
        assert "started" in metrics
        assert "total_jobs" in metrics
        assert "total_values_imputed" in metrics
        assert "provenance_entries" in metrics

    def test_metrics_started_flag(self, started_service):
        metrics = started_service.get_metrics()
        assert metrics["started"] is True


# ---------------------------------------------------------------------------
# Engine properties
# ---------------------------------------------------------------------------


class TestEngineProperties:
    def test_analyzer_engine(self, service):
        assert service.analyzer_engine is not None

    def test_statistical_engine(self, service):
        assert service.statistical_engine is not None

    def test_ml_engine(self, service):
        assert service.ml_engine is not None

    def test_rule_based_engine(self, service):
        assert service.rule_based_engine is not None

    def test_timeseries_engine(self, service):
        assert service.timeseries_engine is not None

    def test_validation_engine_prop(self, service):
        assert service.validation_engine is not None

    def test_pipeline_engine(self, service):
        # Pipeline engine may or may not be set depending on imports
        # Just check the property is accessible
        _ = service.pipeline_engine

    def test_get_provenance(self, service):
        pt = service.get_provenance()
        assert isinstance(pt, _ProvenanceTracker)


# ---------------------------------------------------------------------------
# _extract_strategy
# ---------------------------------------------------------------------------


class TestExtractStrategy:
    def test_string_strategy(self, service):
        assert service._extract_strategy("mean") == "mean"

    def test_enum_strategy(self, service):
        """Test extracting strategy from an object with .value."""
        mock_enum = MagicMock()
        mock_enum.value = "median"
        assert service._extract_strategy(mock_enum) == "median"


# ---------------------------------------------------------------------------
# _update_confidence / _update_completeness
# ---------------------------------------------------------------------------


class TestInternalTracking:
    def test_update_confidence(self, service):
        service._update_confidence(0.8, 10)
        assert service._confidence_sum == pytest.approx(8.0)
        assert service._confidence_count == 10

    def test_update_confidence_zero_count(self, service):
        service._update_confidence(0.8, 0)
        assert service._confidence_count == 0

    def test_update_completeness(self, service):
        service._update_completeness(0.2)
        assert service._completeness_sum == pytest.approx(0.2)
        assert service._completeness_count == 1

    def test_update_completeness_zero(self, service):
        service._update_completeness(0.0)
        assert service._completeness_count == 0

    def test_update_completeness_negative(self, service):
        service._update_completeness(-0.1)
        assert service._completeness_count == 0


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_configure_missing_value_imputer(self):
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        result = asyncio.get_event_loop().run_until_complete(
            configure_missing_value_imputer(app)
        )
        assert isinstance(result, MissingValueImputerService)
        assert result._started is True
        # Service should be attached to app state
        assert app.state.missing_value_imputer_service == result

    def test_configure_with_custom_config(self):
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        cfg = MissingValueImputerConfig(default_strategy="median")
        result = asyncio.get_event_loop().run_until_complete(
            configure_missing_value_imputer(app, config=cfg)
        )
        assert result.config.default_strategy == "median"

    def test_get_missing_value_imputer(self):
        app = MagicMock()
        svc = MissingValueImputerService()
        app.state.missing_value_imputer_service = svc
        result = get_missing_value_imputer(app)
        assert result is svc

    def test_get_missing_value_imputer_not_configured(self):
        app = MagicMock()
        app.state = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="not configured"):
            get_missing_value_imputer(app)

    def test_get_router(self):
        result = get_router()
        # Router should be available since FastAPI is installed
        assert result is not None

    def test_get_router_with_service(self):
        svc = MissingValueImputerService()
        result = get_router(service=svc)
        assert result is not None
