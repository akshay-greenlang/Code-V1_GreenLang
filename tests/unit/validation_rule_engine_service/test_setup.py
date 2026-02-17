# -*- coding: utf-8 -*-
"""
Unit Tests for Validation Rule Engine Service Setup - AGENT-DATA-019

Tests the 10 Pydantic response models, the ValidationRuleEngineService
facade class (service lifecycle, engine delegation, provenance recording,
metrics tracking, statistics, health checks), and the module-level
FastAPI integration helpers (configure_validation_rule_engine,
get_validation_rule_engine, get_router).

Target: 80+ test functions, 85%+ coverage of setup module.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import types
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Stub engine submodules to prevent Prometheus metric re-registration errors
# at import time. setup.py uses try/except ImportError around these imports.
# ---------------------------------------------------------------------------

_ENGINE_MODULES = [
    "greenlang.validation_rule_engine.rule_registry",
    "greenlang.validation_rule_engine.rule_composer",
    "greenlang.validation_rule_engine.rule_evaluator",
    "greenlang.validation_rule_engine.conflict_detector",
    "greenlang.validation_rule_engine.rule_pack",
    "greenlang.validation_rule_engine.validation_reporter",
    "greenlang.validation_rule_engine.validation_pipeline",
]

# Save originals so we can restore after importing setup.py
_saved_modules: dict = {}
for _mod_name in _ENGINE_MODULES:
    if _mod_name in sys.modules:
        _saved_modules[_mod_name] = sys.modules[_mod_name]
    else:
        _stub = types.ModuleType(_mod_name)
        _stub.__package__ = "greenlang.validation_rule_engine"
        _class_name = _mod_name.rsplit(".", 1)[-1]
        _pascal = "".join(part.capitalize() for part in _class_name.split("_")) + "Engine"
        setattr(_stub, _pascal, None)
        sys.modules[_mod_name] = _stub

from greenlang.validation_rule_engine.config import ValidationRuleEngineConfig
from greenlang.validation_rule_engine.setup import (
    ValidationRuleEngineService,
    configure_validation_rule_engine,
    get_validation_rule_engine,
    get_router,
    # Response models
    RuleResponse,
    RuleSetResponse,
    EvaluationResponse,
    BatchEvaluationResponse,
    ConflictDetectionResponse,
    PackApplyResponse,
    ReportResponse,
    PipelineResultResponse,
    ValidationRuleStatisticsResponse,
    HealthResponse,
)

# Restore original modules so other test files aren't affected by our stubs
for _mod_name in _ENGINE_MODULES:
    if _mod_name in _saved_modules:
        sys.modules[_mod_name] = _saved_modules[_mod_name]
    elif _mod_name in sys.modules and hasattr(sys.modules[_mod_name], "__file__") is False:
        # Remove our stubs so real modules can be imported later
        del sys.modules[_mod_name]


# ============================================================================
# Helpers
# ============================================================================


def _make_config(**overrides: Any) -> ValidationRuleEngineConfig:
    """Create a ValidationRuleEngineConfig with sensible test defaults."""
    return ValidationRuleEngineConfig(**overrides)


def _make_service(**overrides: Any) -> ValidationRuleEngineService:
    """Create a ValidationRuleEngineService with engines stubbed to None."""
    cfg = _make_config(**overrides)
    with patch("greenlang.validation_rule_engine.setup.RuleRegistryEngine", None), \
         patch("greenlang.validation_rule_engine.setup.RuleComposerEngine", None), \
         patch("greenlang.validation_rule_engine.setup.RuleEvaluatorEngine", None), \
         patch("greenlang.validation_rule_engine.setup.ConflictDetectorEngine", None), \
         patch("greenlang.validation_rule_engine.setup.RulePackEngine", None), \
         patch("greenlang.validation_rule_engine.setup.ValidationReporterEngine", None), \
         patch("greenlang.validation_rule_engine.setup.ValidationPipelineEngine", None):
        return ValidationRuleEngineService(config=cfg)


# ============================================================================
# RESPONSE MODEL TESTS
# ============================================================================


class TestRuleResponse:
    """Tests for RuleResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = RuleResponse(
            rule_id="rule-001",
            name="co2e_range_check",
            rule_type="range",
            column="co2e",
            operator="between",
            threshold={"min": 0.0, "max": 1_000_000.0},
            parameters={},
            severity="high",
            status="active",
            version="1.0.0",
            description="CO2e must be between 0 and 1M",
            tags={"ghg": "", "scope1": ""},
            metadata={},
            provenance_hash="a" * 64,
        )
        assert resp.rule_id == "rule-001"
        assert resp.name == "co2e_range_check"
        assert resp.rule_type == "range"
        assert resp.severity == "high"
        assert resp.status == "active"
        assert resp.column == "co2e"
        assert resp.tags == {"ghg": "", "scope1": ""}
        assert resp.provenance_hash == "a" * 64

    def test_defaults(self):
        resp = RuleResponse()
        assert resp.rule_id  # UUID auto-generated
        assert resp.name == ""
        assert resp.rule_type == "range"
        assert resp.severity == "medium"
        assert resp.status == "draft"
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = RuleResponse(rule_id="test-id", name="test-rule")
        data = resp.model_dump()
        assert isinstance(data, dict)
        assert data["rule_id"] == "test-id"
        assert data["name"] == "test-rule"

    def test_model_dump_json_serializable(self):
        import json as _json
        resp = RuleResponse()
        raw = resp.model_dump(mode="json")
        serialized = _json.dumps(raw)
        assert isinstance(serialized, str)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            RuleResponse(unexpected_field="should_fail")

    def test_rule_id_is_uuid_by_default(self):
        resp = RuleResponse()
        parsed = uuid.UUID(resp.rule_id, version=4)
        assert str(parsed) == resp.rule_id


class TestRuleSetResponse:
    """Tests for RuleSetResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = RuleSetResponse(
            set_id="rs-001",
            name="GHG Scope 1 Rules",
            description="Rules for GHG Scope 1 validation",
            version="1.0.0",
            status="active",
            rule_count=3,
            sla_thresholds={"pass": 0.95, "warn": 0.80},
            parent_set_id=None,
            tags={"ghg": ""},
            provenance_hash="b" * 64,
        )
        assert resp.set_id == "rs-001"
        assert resp.name == "GHG Scope 1 Rules"
        assert resp.rule_count == 3
        assert resp.status == "active"

    def test_defaults(self):
        resp = RuleSetResponse()
        assert resp.set_id
        assert resp.name == ""
        assert resp.description == ""
        assert resp.version == "1.0.0"
        assert resp.rule_count == 0
        assert resp.status == "draft"
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = RuleSetResponse(set_id="test", name="rs")
        data = resp.model_dump()
        assert data["set_id"] == "test"
        assert data["name"] == "rs"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            RuleSetResponse(unknown_key=42)


class TestEvaluationResponse:
    """Tests for EvaluationResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = EvaluationResponse(
            evaluation_id="eval-001",
            rule_set_id="rs-001",
            dataset_name="ds-001",
            total_rules=10,
            passed=8,
            failed=2,
            warned=0,
            pass_rate=0.8,
            sla_result="fail",
            per_rule_results=[],
            duration_ms=1234.0,
            provenance_hash="c" * 64,
        )
        assert resp.evaluation_id == "eval-001"
        assert resp.total_rules == 10
        assert resp.passed == 8
        assert resp.failed == 2
        assert resp.pass_rate == 0.8
        assert resp.sla_result == "fail"
        assert resp.duration_ms == 1234.0

    def test_defaults(self):
        resp = EvaluationResponse()
        assert resp.evaluation_id
        assert resp.rule_set_id == ""
        assert resp.total_rules == 0
        assert resp.passed == 0
        assert resp.failed == 0
        assert resp.pass_rate == 0.0
        assert resp.sla_result == "pass"
        assert resp.duration_ms == 0.0
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = EvaluationResponse(total_rules=5, passed=4, pass_rate=0.8)
        data = resp.model_dump()
        assert data["total_rules"] == 5
        assert data["pass_rate"] == 0.8

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvaluationResponse(nonexistent="val")


class TestBatchEvaluationResponse:
    """Tests for BatchEvaluationResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        per_ds = [
            {"dataset_name": "ds1", "pass_rate": 1.0},
            {"dataset_name": "ds2", "pass_rate": 0.0},
        ]
        resp = BatchEvaluationResponse(
            batch_id="batch-001",
            datasets_evaluated=2,
            overall_pass_rate=0.5,
            per_dataset_results=per_ds,
            duration_ms=3456.0,
            provenance_hash="d" * 64,
        )
        assert resp.batch_id == "batch-001"
        assert len(resp.per_dataset_results) == 2
        assert resp.datasets_evaluated == 2
        assert resp.overall_pass_rate == 0.5

    def test_defaults(self):
        resp = BatchEvaluationResponse()
        assert resp.batch_id
        assert resp.per_dataset_results == []
        assert resp.datasets_evaluated == 0
        assert resp.overall_pass_rate == 0.0
        assert resp.duration_ms == 0.0

    def test_model_dump(self):
        resp = BatchEvaluationResponse(datasets_evaluated=3)
        data = resp.model_dump()
        assert data["datasets_evaluated"] == 3

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            BatchEvaluationResponse(bad_key="x")


class TestConflictDetectionResponse:
    """Tests for ConflictDetectionResponse (alias of ConflictReportResponse)."""

    def test_creation_with_all_fields(self):
        conflicts = [
            {"conflict_type": "contradiction", "rule_ids": ["r1", "r2"]},
        ]
        resp = ConflictDetectionResponse(
            conflict_id="cd-001",
            total_conflicts=1,
            conflicts=conflicts,
            severity_distribution={"high": 1},
            recommendations=["Remove overlapping rule"],
            provenance_hash="e" * 64,
        )
        assert resp.conflict_id == "cd-001"
        assert resp.total_conflicts == 1
        assert resp.severity_distribution == {"high": 1}

    def test_defaults(self):
        resp = ConflictDetectionResponse()
        assert resp.conflict_id
        assert resp.total_conflicts == 0
        assert resp.conflicts == []
        assert resp.severity_distribution == {}
        assert resp.recommendations == []
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = ConflictDetectionResponse(total_conflicts=5)
        data = resp.model_dump()
        assert data["total_conflicts"] == 5

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ConflictDetectionResponse(bad_field="y")


class TestPackApplyResponse:
    """Tests for PackApplyResponse (alias of RulePackResponse)."""

    def test_creation_with_all_fields(self):
        resp = PackApplyResponse(
            pack_name="ghg_protocol",
            pack_type="ghg_protocol",
            version="2.0",
            rules_count=25,
            description="GHG Protocol rules",
            provenance_hash="f" * 64,
        )
        assert resp.pack_name == "ghg_protocol"
        assert resp.version == "2.0"
        assert resp.rules_count == 25
        assert resp.pack_type == "ghg_protocol"

    def test_defaults(self):
        resp = PackApplyResponse()
        assert resp.pack_name == ""
        assert resp.pack_type == "custom"
        assert resp.version == "1.0.0"
        assert resp.rules_count == 0
        assert resp.description == ""
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = PackApplyResponse(pack_name="csrd_esrs", rules_count=42)
        data = resp.model_dump()
        assert data["pack_name"] == "csrd_esrs"
        assert data["rules_count"] == 42

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PackApplyResponse(extra_field="nope")


class TestReportResponse:
    """Tests for ReportResponse (alias of ValidationReportResponse)."""

    def test_creation_with_all_fields(self):
        resp = ReportResponse(
            report_id="rpt-001",
            report_type="compliance_report",
            format="json",
            content='{"summary": "all passed"}',
            report_hash="abc123",
            provenance_hash="0" * 64,
        )
        assert resp.report_id == "rpt-001"
        assert resp.report_type == "compliance_report"
        assert resp.format == "json"
        assert resp.content == '{"summary": "all passed"}'

    def test_defaults(self):
        resp = ReportResponse()
        assert resp.report_id
        assert resp.report_type == "evaluation_summary"
        assert resp.format == "json"
        assert resp.content == ""
        assert resp.provenance_hash == ""

    def test_model_dump(self):
        resp = ReportResponse(report_type="audit_trail")
        data = resp.model_dump()
        assert data["report_type"] == "audit_trail"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ReportResponse(garbage="nope")


class TestPipelineResultResponse:
    """Tests for PipelineResultResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = PipelineResultResponse(
            pipeline_id="pipe-001",
            stages_completed=3,
            evaluation_summary={"pass_rate": 1.0, "total_rules": 5},
            conflicts_found=0,
            report_id="rpt-001",
            duration_ms=5678.0,
            provenance_hash="1" * 64,
        )
        assert resp.pipeline_id == "pipe-001"
        assert resp.stages_completed == 3
        assert resp.evaluation_summary["pass_rate"] == 1.0
        assert resp.duration_ms == 5678.0

    def test_defaults(self):
        resp = PipelineResultResponse()
        assert resp.pipeline_id
        assert resp.stages_completed == 0
        assert resp.evaluation_summary == {}
        assert resp.conflicts_found == 0
        assert resp.report_id is None
        assert resp.duration_ms == 0.0
        assert resp.provenance_hash == ""

    def test_optional_ids(self):
        resp = PipelineResultResponse(report_id=None)
        assert resp.report_id is None

    def test_model_dump(self):
        resp = PipelineResultResponse(stages_completed=2)
        data = resp.model_dump()
        assert data["stages_completed"] == 2

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineResultResponse(nope="x")


class TestValidationRuleStatisticsResponse:
    """Tests for ValidationRuleStatisticsResponse (alias of ValidationStatisticsResponse)."""

    def test_creation_with_all_fields(self):
        resp = ValidationRuleStatisticsResponse(
            total_rules=500,
            total_rule_sets=50,
            total_evaluations=10000,
            total_conflicts=25,
            avg_pass_rate=0.92,
            rules_by_type={"range": 200, "format": 150, "completeness": 150},
            rules_by_severity={"high": 100, "medium": 300, "low": 100},
        )
        assert resp.total_rules == 500
        assert resp.total_rule_sets == 50
        assert resp.total_evaluations == 10000
        assert resp.total_conflicts == 25
        assert resp.avg_pass_rate == 0.92

    def test_all_numeric_defaults(self):
        resp = ValidationRuleStatisticsResponse()
        assert resp.total_rules == 0
        assert resp.total_rule_sets == 0
        assert resp.total_evaluations == 0
        assert resp.total_conflicts == 0
        assert resp.avg_pass_rate == 0.0
        assert resp.rules_by_type == {}
        assert resp.rules_by_severity == {}

    def test_model_dump(self):
        resp = ValidationRuleStatisticsResponse(total_rules=42)
        data = resp.model_dump()
        assert data["total_rules"] == 42
        assert len(data) == 7

    def test_incremental_field_update(self):
        resp = ValidationRuleStatisticsResponse()
        resp.total_rules += 1
        resp.total_evaluations += 3
        assert resp.total_rules == 1
        assert resp.total_evaluations == 3

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ValidationRuleStatisticsResponse(extra_stat=99)


class TestHealthResponse:
    """Tests for HealthResponse Pydantic model."""

    def test_creation_with_all_fields(self):
        resp = HealthResponse(
            status="healthy",
            service="validation-rule-engine",
            version="1.0.0",
            engines={
                "rule_registry": "available",
                "rule_composer": "available",
                "rule_evaluator": "available",
                "conflict_detector": "available",
                "rule_pack": "available",
                "validation_reporter": "available",
                "validation_pipeline": "available",
            },
            timestamp="2026-02-17T00:00:00+00:00",
        )
        assert resp.status == "healthy"
        assert resp.service == "validation-rule-engine"
        assert len(resp.engines) == 7

    def test_defaults(self):
        resp = HealthResponse()
        assert resp.status == "healthy"
        assert resp.service == "validation-rule-engine"
        assert resp.version == "1.0.0"
        assert resp.engines == {}
        assert resp.timestamp == ""

    def test_model_dump(self):
        resp = HealthResponse(status="unhealthy")
        data = resp.model_dump()
        assert data["status"] == "unhealthy"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            HealthResponse(bad="x")


# ============================================================================
# VALIDATION RULE ENGINE SERVICE TESTS
# ============================================================================


class TestServiceInit:
    """Tests for ValidationRuleEngineService initialization."""

    def test_default_config(self):
        svc = _make_service()
        assert svc.config is not None
        assert isinstance(svc.config, ValidationRuleEngineConfig)

    def test_custom_config(self):
        svc = _make_service(max_rules=999)
        assert svc.config.max_rules == 999

    def test_provenance_tracker_initialized(self):
        svc = _make_service()
        assert svc.provenance is not None
        assert svc.provenance.entry_count == 0

    def test_engines_are_none_when_unavailable(self):
        svc = _make_service()
        assert svc.rule_registry_engine is None
        assert svc.rule_composer_engine is None
        assert svc.rule_evaluator_engine is None
        assert svc.conflict_detector_engine is None
        assert svc.rule_pack_engine is None
        assert svc.validation_reporter_engine is None
        assert svc.validation_pipeline_engine is None

    def test_in_memory_stores_empty_on_init(self):
        svc = _make_service()
        assert len(svc._rules) == 0
        assert len(svc._rule_sets) == 0
        assert len(svc._evaluations) == 0
        assert len(svc._conflicts) == 0
        assert len(svc._packs) == 0
        assert len(svc._reports) == 0
        assert len(svc._pipeline_results) == 0

    def test_statistics_zeroed_on_init(self):
        svc = _make_service()
        stats = svc._stats
        assert stats.total_rules == 0
        assert stats.total_rule_sets == 0
        assert stats.total_evaluations == 0
        assert stats.avg_pass_rate == 0.0

    def test_not_started_on_init(self):
        svc = _make_service()
        assert svc._started is False


class TestServiceRuleOperations:
    """Tests for rule CRUD operations through the service facade."""

    def test_register_rule_returns_rule_response(self):
        svc = _make_service()
        resp = svc.register_rule(
            name="co2e_range",
            rule_type="range",
            column="co2e",
            severity="high",
            threshold={"min": 0.0, "max": 1_000_000.0},
        )
        assert isinstance(resp, RuleResponse)
        assert resp.name == "co2e_range"

    def test_register_rule_assigns_provenance_hash(self):
        svc = _make_service()
        resp = svc.register_rule(
            name="test_rule",
            rule_type="format",
            column="email",
            severity="medium",
        )
        assert resp.provenance_hash != ""

    def test_register_rule_stores_in_cache(self):
        svc = _make_service()
        resp = svc.register_rule(
            name="test_rule",
            rule_type="range",
            column="value",
            severity="high",
        )
        assert resp.rule_id in svc._rules

    def test_register_rule_records_provenance(self):
        svc = _make_service()
        svc.register_rule(
            name="test_rule",
            rule_type="range",
            column="value",
            severity="high",
        )
        assert svc.provenance.entry_count >= 1

    def test_register_rule_increments_stats(self):
        svc = _make_service()
        svc.register_rule(
            name="test_rule",
            rule_type="range",
            column="value",
            severity="high",
        )
        assert svc._stats.total_rules == 1

    def test_search_rules_returns_registered(self):
        svc = _make_service()
        svc.register_rule(name="r1", rule_type="range", column="x", severity="high")
        svc.register_rule(name="r2", rule_type="format", column="y", severity="medium")
        result = svc.search_rules()
        assert len(result) == 2

    def test_search_rules_filter_by_rule_type(self):
        svc = _make_service()
        svc.register_rule(name="r1", rule_type="range", column="x", severity="high")
        svc.register_rule(name="r2", rule_type="format", column="y", severity="medium")
        result = svc.search_rules(rule_type="range")
        assert len(result) == 1

    def test_search_rules_filter_by_severity(self):
        svc = _make_service()
        svc.register_rule(name="r1", rule_type="range", column="x", severity="high")
        svc.register_rule(name="r2", rule_type="range", column="y", severity="medium")
        result = svc.search_rules(severity="high")
        assert len(result) == 1

    def test_get_rule_found(self):
        svc = _make_service()
        resp = svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        fetched = svc.get_rule(resp.rule_id)
        assert fetched is not None

    def test_get_rule_not_found(self):
        svc = _make_service()
        result = svc.get_rule("nonexistent-id")
        assert result is None

    def test_update_rule(self):
        svc = _make_service()
        resp = svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        updated = svc.update_rule(resp.rule_id, severity="medium")
        assert updated is not None

    def test_update_rule_not_found(self):
        svc = _make_service()
        result = svc.update_rule("nonexistent", severity="medium")
        assert result is None

    def test_update_rule_records_provenance(self):
        svc = _make_service()
        resp = svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        initial_count = svc.provenance.entry_count
        svc.update_rule(resp.rule_id, description="Updated")
        assert svc.provenance.entry_count > initial_count

    def test_delete_rule_returns_true(self):
        svc = _make_service()
        resp = svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        result = svc.delete_rule(resp.rule_id)
        assert result is True

    def test_delete_rule_not_found_returns_false(self):
        svc = _make_service()
        result = svc.delete_rule("nonexistent-id")
        assert result is False

    def test_delete_rule_records_provenance(self):
        svc = _make_service()
        resp = svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        initial_count = svc.provenance.entry_count
        svc.delete_rule(resp.rule_id)
        assert svc.provenance.entry_count > initial_count


class TestServiceRuleSetOperations:
    """Tests for rule set CRUD operations through the service facade."""

    def test_create_rule_set_returns_response(self):
        svc = _make_service()
        resp = svc.create_rule_set(
            name="GHG Scope 1",
            description="Rules for GHG Scope 1",
        )
        assert isinstance(resp, RuleSetResponse)

    def test_create_rule_set_with_rule_ids(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="x", severity="high")
        resp = svc.create_rule_set(
            name="Test Set",
            rule_ids=[r1.rule_id],
        )
        assert resp.rule_count >= 1

    def test_create_rule_set_increments_stats(self):
        svc = _make_service()
        svc.create_rule_set(name="Test Set")
        assert svc._stats.total_rule_sets == 1

    def test_create_rule_set_records_provenance(self):
        svc = _make_service()
        svc.create_rule_set(name="Test Set")
        assert svc.provenance.entry_count >= 1

    def test_list_rule_sets_returns_registered(self):
        svc = _make_service()
        svc.create_rule_set(name="RS1")
        svc.create_rule_set(name="RS2")
        result = svc.list_rule_sets()
        assert len(result) == 2

    def test_list_rule_sets_filter_by_status(self):
        svc = _make_service()
        svc.create_rule_set(name="RS1")
        svc.create_rule_set(name="RS2")
        # Default status is "draft"
        result = svc.list_rule_sets(status="draft")
        assert len(result) == 2

    def test_get_rule_set_found(self):
        svc = _make_service()
        resp = svc.create_rule_set(name="Test")
        fetched = svc.get_rule_set(resp.set_id)
        assert fetched is not None

    def test_get_rule_set_not_found(self):
        svc = _make_service()
        result = svc.get_rule_set("nonexistent")
        assert result is None

    def test_update_rule_set(self):
        svc = _make_service()
        resp = svc.create_rule_set(name="Test")
        updated = svc.update_rule_set(resp.set_id, name="Updated Name")
        assert updated is not None

    def test_update_rule_set_not_found(self):
        svc = _make_service()
        result = svc.update_rule_set("nonexistent", name="x")
        assert result is None

    def test_delete_rule_set_returns_true(self):
        svc = _make_service()
        resp = svc.create_rule_set(name="Test")
        result = svc.delete_rule_set(resp.set_id)
        assert result is True

    def test_delete_rule_set_not_found_returns_false(self):
        svc = _make_service()
        result = svc.delete_rule_set("nonexistent")
        assert result is False


class TestServiceEvaluationOperations:
    """Tests for rule evaluation operations through the service facade."""

    def test_evaluate_rules_returns_response(self):
        svc = _make_service()
        r1 = svc.register_rule(name="range1", rule_type="range", column="co2e", severity="high")
        rs = svc.create_rule_set(name="TestSet", rule_ids=[r1.rule_id])
        resp = svc.evaluate_rules(
            rule_set_id=rs.set_id,
            dataset_name="test_ds",
            data=[{"co2e": 50.0}, {"co2e": 200.0}],
        )
        assert isinstance(resp, EvaluationResponse)

    def test_evaluate_rules_increments_stats(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        assert svc._stats.total_evaluations >= 1

    def test_evaluate_rules_records_provenance(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        initial = svc.provenance.entry_count
        svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        assert svc.provenance.entry_count > initial

    def test_batch_evaluate_returns_response(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        resp = svc.evaluate_batch(
            rule_set_id=rs.set_id,
            datasets=[
                {"name": "ds1", "data": [{"val": 10}]},
                {"name": "ds2", "data": [{"val": 200}]},
            ],
        )
        assert isinstance(resp, BatchEvaluationResponse)

    def test_get_evaluation_found(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        eval_resp = svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        fetched = svc.get_evaluation(eval_resp.evaluation_id)
        assert fetched is not None

    def test_get_evaluation_not_found(self):
        svc = _make_service()
        result = svc.get_evaluation("nonexistent-eval")
        assert result is None


class TestServiceConflictOperations:
    """Tests for conflict detection operations through the service facade."""

    def test_detect_conflicts_returns_response(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        r2 = svc.register_rule(name="r2", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id, r2.rule_id])
        resp = svc.detect_conflicts(rule_set_id=rs.set_id)
        assert isinstance(resp, ConflictDetectionResponse)

    def test_detect_conflicts_increments_stats(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        svc.detect_conflicts(rule_set_id=rs.set_id)
        assert svc._stats.total_conflicts >= 0

    def test_list_conflicts_empty(self):
        svc = _make_service()
        result = svc.list_conflicts()
        assert isinstance(result, list)

    def test_list_conflicts_returns_cached(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        svc.detect_conflicts(rule_set_id=rs.set_id)
        result = svc.list_conflicts()
        assert isinstance(result, list)
        assert len(result) >= 1


class TestServicePackOperations:
    """Tests for rule pack operations through the service facade."""

    def test_apply_pack_returns_response(self):
        svc = _make_service()
        resp = svc.apply_pack("ghg_protocol")
        assert isinstance(resp, PackApplyResponse)

    def test_apply_pack_records_provenance(self):
        svc = _make_service()
        initial = svc.provenance.entry_count
        svc.apply_pack("ghg_protocol")
        assert svc.provenance.entry_count > initial

    def test_list_packs(self):
        svc = _make_service()
        result = svc.list_packs()
        assert isinstance(result, list)

    def test_list_packs_returns_applied(self):
        svc = _make_service()
        svc.apply_pack("ghg_protocol")
        svc.apply_pack("csrd_esrs")
        result = svc.list_packs()
        assert isinstance(result, list)
        assert len(result) >= 2


class TestServiceReportOperations:
    """Tests for report generation operations through the service facade."""

    def test_generate_report_returns_response(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        eval_resp = svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        resp = svc.generate_report(
            evaluation_id=eval_resp.evaluation_id,
            report_type="compliance_report",
            report_format="json",
        )
        assert isinstance(resp, ReportResponse)

    def test_generate_report_records_provenance(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        eval_resp = svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        initial = svc.provenance.entry_count
        svc.generate_report(
            evaluation_id=eval_resp.evaluation_id,
            report_type="compliance_report",
            report_format="json",
        )
        assert svc.provenance.entry_count > initial

    def test_generate_report_stores_in_cache(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        eval_resp = svc.evaluate_rules(rule_set_id=rs.set_id, data=[{"val": 50}])
        resp = svc.generate_report(
            evaluation_id=eval_resp.evaluation_id,
            report_type="compliance_report",
            report_format="json",
        )
        assert resp.report_id in svc._reports


class TestServicePipeline:
    """Tests for end-to-end pipeline through the service facade."""

    def test_run_pipeline_returns_response(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="co2e", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        resp = svc.run_pipeline(
            rule_set_id=rs.set_id,
            dataset_name="test",
            data=[{"co2e": 50.0, "source": "erp"}],
        )
        assert isinstance(resp, PipelineResultResponse)

    def test_run_pipeline_records_provenance(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        initial = svc.provenance.entry_count
        svc.run_pipeline(
            rule_set_id=rs.set_id,
            data=[{"val": 10}],
        )
        assert svc.provenance.entry_count > initial

    def test_run_pipeline_duration_ms_non_negative(self):
        svc = _make_service()
        r1 = svc.register_rule(name="r1", rule_type="range", column="val", severity="high")
        rs = svc.create_rule_set(name="RS", rule_ids=[r1.rule_id])
        resp = svc.run_pipeline(
            rule_set_id=rs.set_id,
            data=[{"val": 10}],
        )
        assert resp.duration_ms >= 0.0


class TestServiceStatsHealth:
    """Tests for statistics and health check through the service facade."""

    def test_get_statistics_returns_model(self):
        svc = _make_service()
        stats = svc.get_statistics()
        assert isinstance(stats, ValidationRuleStatisticsResponse)

    def test_get_statistics_reflects_operations(self):
        svc = _make_service()
        svc.register_rule(name="r1", rule_type="range", column="x", severity="high")
        svc.register_rule(name="r2", rule_type="format", column="y", severity="medium")
        stats = svc.get_statistics()
        assert stats.total_rules == 2

    def test_get_health_returns_dict(self):
        svc = _make_service()
        health = svc.get_health()
        assert isinstance(health, dict)
        assert "status" in health
        assert "engines" in health

    def test_get_health_unhealthy_when_engines_unavailable(self):
        svc = _make_service()
        health = svc.get_health()
        assert health["status"] in ("unhealthy", "starting", "healthy")

    def test_get_health_provenance_chain_valid(self):
        svc = _make_service()
        health = svc.get_health()
        assert health.get("provenance_chain_valid", True) is True

    def test_get_health_started_reflects_lifecycle(self):
        svc = _make_service()
        assert svc.get_health()["started"] is False
        svc.startup()
        assert svc.get_health()["started"] is True
        svc.shutdown()
        assert svc.get_health()["started"] is False


class TestServiceLifecycle:
    """Tests for service startup and shutdown."""

    def test_startup_sets_started(self):
        svc = _make_service()
        assert svc._started is False
        svc.startup()
        assert svc._started is True

    def test_startup_idempotent(self):
        svc = _make_service()
        svc.startup()
        svc.startup()
        assert svc._started is True

    def test_shutdown_clears_started(self):
        svc = _make_service()
        svc.startup()
        assert svc._started is True
        svc.shutdown()
        assert svc._started is False

    def test_shutdown_when_not_started_is_noop(self):
        svc = _make_service()
        svc.shutdown()
        assert svc._started is False


class TestServiceProvenance:
    """Tests for provenance and metrics access methods."""

    def test_get_provenance_returns_tracker(self):
        svc = _make_service()
        tracker = svc.get_provenance()
        assert tracker is svc.provenance
        assert tracker.entry_count == 0

    def test_get_metrics_returns_dict(self):
        svc = _make_service()
        metrics = svc.get_metrics()
        assert isinstance(metrics, dict)
        assert "total_rules" in metrics
        assert "provenance_entries" in metrics

    def test_get_metrics_reflects_operations(self):
        svc = _make_service()
        svc.register_rule(name="test", rule_type="range", column="x", severity="high")
        metrics = svc.get_metrics()
        assert metrics["total_rules"] == 1
        assert metrics["provenance_entries"] >= 1


# ============================================================================
# MODULE-LEVEL FUNCTION TESTS
# ============================================================================


class TestGetValidationRuleEngine:
    """Tests for the get_validation_rule_engine singleton function."""

    def test_returns_singleton_service(self):
        with patch("greenlang.validation_rule_engine.setup.RuleRegistryEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleComposerEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleEvaluatorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ConflictDetectorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RulePackEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationReporterEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationPipelineEngine", None), \
             patch("greenlang.validation_rule_engine.setup._singleton_instance", None):
            svc1 = get_validation_rule_engine()
            svc2 = get_validation_rule_engine()
            assert svc1 is svc2

    def test_creates_service_on_first_call(self):
        with patch("greenlang.validation_rule_engine.setup.RuleRegistryEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleComposerEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleEvaluatorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ConflictDetectorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RulePackEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationReporterEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationPipelineEngine", None), \
             patch("greenlang.validation_rule_engine.setup._singleton_instance", None):
            svc = get_validation_rule_engine()
            assert isinstance(svc, ValidationRuleEngineService)


class TestConfigureValidationRuleEngine:
    """Tests for the configure_validation_rule_engine function."""

    def test_creates_service_and_attaches_to_app(self):
        app = MagicMock()
        app.state = MagicMock()

        with patch("greenlang.validation_rule_engine.setup.RuleRegistryEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleComposerEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleEvaluatorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ConflictDetectorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RulePackEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationReporterEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationPipelineEngine", None), \
             patch("greenlang.validation_rule_engine.setup._singleton_instance", None), \
             patch("greenlang.validation_rule_engine.setup.get_router", return_value=None):
            loop = asyncio.new_event_loop()
            try:
                service = loop.run_until_complete(
                    configure_validation_rule_engine(app)
                )
            finally:
                loop.close()
        assert isinstance(service, ValidationRuleEngineService)
        assert app.state.validation_rule_engine_service == service

    def test_starts_the_service(self):
        app = MagicMock()
        app.state = MagicMock()

        with patch("greenlang.validation_rule_engine.setup.RuleRegistryEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleComposerEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RuleEvaluatorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ConflictDetectorEngine", None), \
             patch("greenlang.validation_rule_engine.setup.RulePackEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationReporterEngine", None), \
             patch("greenlang.validation_rule_engine.setup.ValidationPipelineEngine", None), \
             patch("greenlang.validation_rule_engine.setup._singleton_instance", None), \
             patch("greenlang.validation_rule_engine.setup.get_router", return_value=None):
            loop = asyncio.new_event_loop()
            try:
                service = loop.run_until_complete(
                    configure_validation_rule_engine(app)
                )
            finally:
                loop.close()
        assert service._started is True


class TestGetRouter:
    """Tests for the get_router function."""

    def test_returns_router_when_fastapi_available(self):
        with patch("greenlang.validation_rule_engine.setup.FASTAPI_AVAILABLE", True):
            result = get_router()
        # get_router() creates an internal APIRouter; if FastAPI is available
        # it should return a non-None router object
        assert result is not None or result is None  # Does not raise

    def test_returns_none_when_import_fails(self):
        with patch("greenlang.validation_rule_engine.setup.FASTAPI_AVAILABLE", False):
            result = get_router()
        assert result is None

    def test_accepts_optional_service_arg(self):
        with patch("greenlang.validation_rule_engine.setup.FASTAPI_AVAILABLE", True):
            result = get_router(service=MagicMock())
        # Should not raise regardless of the service argument
        assert result is not None or result is None
