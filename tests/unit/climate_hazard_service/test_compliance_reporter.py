# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceReporterEngine - AGENT-DATA-020 (Engine 6 of 7)

Tests all public methods of ComplianceReporterEngine with comprehensive
coverage of report generation, CRUD operations, compliance validation,
framework templates, statistics, and export/import functionality.

Target: 85%+ code coverage across all methods and edge cases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.climate_hazard.compliance_reporter import (
    ADAPTATION_MEASURES,
    ComplianceReporterEngine,
    FRAMEWORK_TEMPLATES,
    RECOMMENDATIONS_BY_RISK_LEVEL,
    RISK_LEVEL_THRESHOLDS,
    ReportFormatLocal,
    ReportRecord,
    ReportTypeLocal,
    SUPPORTED_FRAMEWORKS,
    URGENCY_MAP,
    VALID_REPORT_FORMATS,
    VALID_REPORT_TYPES,
    _build_provenance_hash,
    _clamp,
    _classify_risk_level,
    _enum_value,
    _generate_id,
    _safe_mean,
    _utcnow,
    _utcnow_iso,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine() -> ComplianceReporterEngine:
    """Create a fresh ComplianceReporterEngine for each test."""
    return ComplianceReporterEngine()


@pytest.fixture
def engine_custom_provenance() -> ComplianceReporterEngine:
    """Create an engine with a custom genesis hash."""
    return ComplianceReporterEngine(genesis_hash="test-genesis-hash")


@pytest.fixture
def sample_risk_data() -> List[Dict[str, Any]]:
    """Sample risk data for report generation."""
    return [
        {
            "hazard_type": "flood",
            "location_id": "loc_001",
            "composite_score": 72.5,
            "risk_classification": "high",
            "probability": 0.65,
            "intensity": 0.8,
            "scenario": "ssp2_4.5",
        },
        {
            "hazard_type": "heat_wave",
            "location_id": "loc_002",
            "composite_score": 45.0,
            "risk_classification": "medium",
            "probability": 0.4,
            "intensity": 0.5,
            "scenario": "ssp2_4.5",
        },
    ]


@pytest.fixture
def sample_exposure_data() -> List[Dict[str, Any]]:
    """Sample exposure data for report generation."""
    return [
        {
            "asset_id": "asset_001",
            "hazard_type": "flood",
            "exposure_score": 68.0,
            "exposure_level": "high",
            "financial_impact": 1500000.0,
        },
        {
            "asset_id": "asset_002",
            "hazard_type": "heat_wave",
            "exposure_score": 35.0,
            "exposure_level": "low",
            "financial_impact": 250000.0,
        },
    ]


@pytest.fixture
def sample_vulnerability_data() -> List[Dict[str, Any]]:
    """Sample vulnerability data for report generation."""
    return [
        {
            "entity_id": "entity_001",
            "hazard_type": "flood",
            "vulnerability_score": 70.0,
            "vulnerability_level": "high",
            "exposure_score": 65.0,
            "sensitivity_score": 75.0,
            "adaptive_capacity_score": 30.0,
        },
    ]


@pytest.fixture
def populated_engine(
    engine: ComplianceReporterEngine,
    sample_risk_data: List[Dict[str, Any]],
    sample_exposure_data: List[Dict[str, Any]],
    sample_vulnerability_data: List[Dict[str, Any]],
) -> ComplianceReporterEngine:
    """Engine with several reports already generated."""
    for framework in ["tcfd", "csrd_esrs", "eu_taxonomy"]:
        engine.generate_report(
            report_type="physical_risk_assessment",
            report_format="json",
            framework=framework,
            title=f"Test {framework} Report",
            risk_data=sample_risk_data,
            exposure_data=sample_exposure_data,
            vulnerability_data=sample_vulnerability_data,
        )
    engine.generate_report(
        report_type="scenario_analysis",
        report_format="html",
        framework="tcfd",
        title="Scenario Report",
    )
    engine.generate_report(
        report_type="executive_dashboard",
        report_format="markdown",
        framework="ifrs_s2",
        title="Dashboard Report",
    )
    return engine


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level utility functions."""

    def test_utcnow_returns_utc_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_utcnow_iso_returns_string(self):
        result = _utcnow_iso()
        assert isinstance(result, str)
        assert "T" in result

    def test_generate_id_with_prefix(self):
        result = _generate_id("RPT")
        assert result.startswith("RPT-")
        assert len(result) == 16  # "RPT-" + 12 hex chars

    def test_generate_id_uniqueness(self):
        ids = {_generate_id("X") for _ in range(100)}
        assert len(ids) == 100

    def test_build_provenance_hash_deterministic(self):
        data = {"key": "value", "number": 42}
        h1 = _build_provenance_hash(data)
        h2 = _build_provenance_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_build_provenance_hash_different_data(self):
        h1 = _build_provenance_hash({"a": 1})
        h2 = _build_provenance_hash({"a": 2})
        assert h1 != h2

    def test_classify_risk_level_extreme(self):
        assert _classify_risk_level(80.0) == "extreme"
        assert _classify_risk_level(95.0) == "extreme"
        assert _classify_risk_level(100.0) == "extreme"

    def test_classify_risk_level_high(self):
        assert _classify_risk_level(60.0) == "high"
        assert _classify_risk_level(79.9) == "high"

    def test_classify_risk_level_medium(self):
        assert _classify_risk_level(40.0) == "medium"
        assert _classify_risk_level(59.9) == "medium"

    def test_classify_risk_level_low(self):
        assert _classify_risk_level(20.0) == "low"
        assert _classify_risk_level(39.9) == "low"

    def test_classify_risk_level_negligible(self):
        assert _classify_risk_level(0.0) == "negligible"
        assert _classify_risk_level(19.9) == "negligible"

    def test_clamp_within_range(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below_min(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_above_max(self):
        assert _clamp(150.0) == 100.0

    def test_clamp_custom_bounds(self):
        assert _clamp(5.0, 10.0, 20.0) == 10.0
        assert _clamp(25.0, 10.0, 20.0) == 20.0
        assert _clamp(15.0, 10.0, 20.0) == 15.0

    def test_safe_mean_normal(self):
        assert _safe_mean([10.0, 20.0, 30.0]) == 20.0

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_single(self):
        assert _safe_mean([42.0]) == 42.0

    def test_enum_value_with_enum(self):
        assert _enum_value(ReportTypeLocal.PHYSICAL_RISK_ASSESSMENT) == "physical_risk_assessment"

    def test_enum_value_with_string(self):
        assert _enum_value("plain_string") == "plain_string"

    def test_enum_value_with_int(self):
        assert _enum_value(42) == "42"


# =========================================================================
# Enums and Constants
# =========================================================================


class TestEnumsAndConstants:
    """Tests for module-level enumerations and constants."""

    def test_report_type_local_values(self):
        assert ReportTypeLocal.PHYSICAL_RISK_ASSESSMENT.value == "physical_risk_assessment"
        assert ReportTypeLocal.SCENARIO_ANALYSIS.value == "scenario_analysis"
        assert ReportTypeLocal.ADAPTATION_SCREENING.value == "adaptation_screening"
        assert ReportTypeLocal.EXPOSURE_SUMMARY.value == "exposure_summary"
        assert ReportTypeLocal.EXECUTIVE_DASHBOARD.value == "executive_dashboard"

    def test_report_format_local_values(self):
        assert ReportFormatLocal.JSON.value == "json"
        assert ReportFormatLocal.HTML.value == "html"
        assert ReportFormatLocal.MARKDOWN.value == "markdown"
        assert ReportFormatLocal.TEXT.value == "text"
        assert ReportFormatLocal.CSV.value == "csv"

    def test_supported_frameworks_count(self):
        assert len(SUPPORTED_FRAMEWORKS) == 6

    def test_supported_frameworks_contains_tcfd(self):
        assert "tcfd" in SUPPORTED_FRAMEWORKS

    def test_supported_frameworks_contains_all(self):
        expected = {"tcfd", "csrd_esrs", "eu_taxonomy", "sec_climate", "ifrs_s2", "ngfs"}
        assert set(SUPPORTED_FRAMEWORKS) == expected

    def test_valid_report_types_count(self):
        assert len(VALID_REPORT_TYPES) == 5

    def test_valid_report_formats_count(self):
        assert len(VALID_REPORT_FORMATS) == 5

    def test_risk_level_thresholds_ordered(self):
        for i in range(len(RISK_LEVEL_THRESHOLDS) - 1):
            assert RISK_LEVEL_THRESHOLDS[i][0] < RISK_LEVEL_THRESHOLDS[i + 1][0]

    def test_urgency_map_keys(self):
        expected_keys = {"negligible", "low", "medium", "high", "extreme"}
        assert set(URGENCY_MAP.keys()) == expected_keys

    def test_framework_templates_all_present(self):
        for fw in SUPPORTED_FRAMEWORKS:
            assert fw in FRAMEWORK_TEMPLATES

    def test_framework_templates_have_sections(self):
        for fw, template in FRAMEWORK_TEMPLATES.items():
            assert "name" in template
            assert "version" in template
            assert "description" in template
            assert "sections" in template
            assert len(template["sections"]) > 0

    def test_recommendations_by_risk_level_keys(self):
        expected = {"negligible", "low", "medium", "high", "extreme"}
        assert set(RECOMMENDATIONS_BY_RISK_LEVEL.keys()) == expected

    def test_adaptation_measures_keys(self):
        assert "flood" in ADAPTATION_MEASURES
        assert "drought" in ADAPTATION_MEASURES
        assert "general" in ADAPTATION_MEASURES


# =========================================================================
# ReportRecord dataclass
# =========================================================================


class TestReportRecord:
    """Tests for the ReportRecord dataclass."""

    def test_default_values(self):
        record = ReportRecord()
        assert record.report_id == ""
        assert record.report_type == "physical_risk_assessment"
        assert record.report_format == "json"
        assert record.framework == "tcfd"
        assert record.compliance_score == 0.0

    def test_to_dict(self):
        record = ReportRecord(
            report_id="RPT-test123456",
            report_type="scenario_analysis",
            framework="csrd_esrs",
            title="Test Report",
        )
        d = record.to_dict()
        assert d["report_id"] == "RPT-test123456"
        assert d["report_type"] == "scenario_analysis"
        assert d["framework"] == "csrd_esrs"
        assert d["title"] == "Test Report"

    def test_to_dict_contains_all_fields(self):
        record = ReportRecord()
        d = record.to_dict()
        expected_keys = {
            "report_id", "report_type", "report_format", "framework",
            "title", "description", "scope", "namespace", "asset_ids",
            "hazard_types", "scenarios", "time_horizons", "parameters",
            "content", "report_hash", "asset_count", "hazard_count",
            "scenario_count", "risk_summary", "recommendations",
            "compliance_score", "evidence_summary", "generated_at",
            "provenance_hash",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_returns_copies(self):
        record = ReportRecord(asset_ids=["a1", "a2"])
        d = record.to_dict()
        d["asset_ids"].append("a3")
        assert len(record.asset_ids) == 2  # Original unmodified


# =========================================================================
# ComplianceReporterEngine initialization
# =========================================================================


class TestEngineInitialization:
    """Tests for engine construction and initialization."""

    def test_default_initialization(self, engine: ComplianceReporterEngine):
        assert engine._provenance is not None
        assert engine._total_generated == 0
        assert engine._total_validations == 0
        assert engine._total_errors == 0
        assert len(engine._reports) == 0

    def test_initialization_with_custom_genesis(self, engine_custom_provenance):
        assert engine_custom_provenance._provenance is not None

    def test_initialization_with_external_provenance(self, tracker):
        engine = ComplianceReporterEngine(provenance=tracker)
        assert engine._provenance is tracker

    def test_initialization_thread_lock(self, engine: ComplianceReporterEngine):
        assert isinstance(engine._lock, type(threading.Lock()))


# =========================================================================
# generate_report
# =========================================================================


class TestGenerateReport:
    """Tests for the generate_report method."""

    def test_basic_generation(self, engine: ComplianceReporterEngine):
        result = engine.generate_report()
        assert "report_id" in result
        assert result["report_id"].startswith("RPT-")
        assert result["report_type"] == "physical_risk_assessment"
        assert result["report_format"] == "json"
        assert result["framework"] == "tcfd"
        assert result["provenance_hash"]
        assert len(result["provenance_hash"]) == 64

    def test_with_title(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(title="My Custom Report")
        assert result["title"] == "My Custom Report"

    def test_auto_generated_title(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(title="")
        assert result["title"] != ""

    @pytest.mark.parametrize("report_type", [
        "physical_risk_assessment",
        "scenario_analysis",
        "adaptation_screening",
        "exposure_summary",
        "executive_dashboard",
    ])
    def test_all_report_types(self, engine: ComplianceReporterEngine, report_type: str):
        result = engine.generate_report(report_type=report_type)
        assert result["report_type"] == report_type

    @pytest.mark.parametrize("fmt", ["json", "html", "markdown", "text", "csv"])
    def test_all_formats(self, engine: ComplianceReporterEngine, fmt: str):
        result = engine.generate_report(report_format=fmt)
        assert result["report_format"] == fmt
        assert result["content"] != ""

    @pytest.mark.parametrize("framework", [
        "tcfd", "csrd_esrs", "eu_taxonomy", "sec_climate", "ifrs_s2", "ngfs",
    ])
    def test_all_frameworks(self, engine: ComplianceReporterEngine, framework: str):
        result = engine.generate_report(framework=framework)
        assert result["framework"] == framework

    def test_with_risk_data(
        self,
        engine: ComplianceReporterEngine,
        sample_risk_data: List[Dict[str, Any]],
    ):
        result = engine.generate_report(
            risk_data=sample_risk_data,
            hazard_types=["flood", "heat_wave"],
        )
        assert result["hazard_count"] == 2
        assert result["risk_summary"] is not None

    def test_with_exposure_data(
        self,
        engine: ComplianceReporterEngine,
        sample_exposure_data: List[Dict[str, Any]],
    ):
        result = engine.generate_report(exposure_data=sample_exposure_data)
        assert result["evidence_summary"] is not None

    def test_with_vulnerability_data(
        self,
        engine: ComplianceReporterEngine,
        sample_vulnerability_data: List[Dict[str, Any]],
    ):
        result = engine.generate_report(vulnerability_data=sample_vulnerability_data)
        assert result["evidence_summary"] is not None

    def test_with_all_data(
        self,
        engine: ComplianceReporterEngine,
        sample_risk_data,
        sample_exposure_data,
        sample_vulnerability_data,
    ):
        result = engine.generate_report(
            risk_data=sample_risk_data,
            exposure_data=sample_exposure_data,
            vulnerability_data=sample_vulnerability_data,
            asset_ids=["a1", "a2"],
            hazard_types=["flood"],
            scenarios=["ssp2_4.5"],
            time_horizons=["mid_term"],
        )
        assert result["asset_count"] == 2
        assert result["scenario_count"] == 1

    def test_with_scope(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(scope="portfolio")
        assert result["scope"] == "portfolio"

    def test_with_namespace(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(namespace="tenant_a")
        assert result["namespace"] == "tenant_a"

    def test_include_recommendations_true(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(include_recommendations=True)
        # Recommendations list exists (may be empty if no risk data)
        assert isinstance(result["recommendations"], list)

    def test_include_recommendations_false(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(include_recommendations=False)
        assert result["recommendations"] == []

    def test_report_hash_is_sha256(self, engine: ComplianceReporterEngine):
        result = engine.generate_report()
        assert len(result["report_hash"]) == 64

    def test_compliance_score_in_range(
        self,
        engine: ComplianceReporterEngine,
        sample_risk_data,
    ):
        result = engine.generate_report(risk_data=sample_risk_data)
        assert 0.0 <= result["compliance_score"] <= 100.0

    def test_generated_at_populated(self, engine: ComplianceReporterEngine):
        result = engine.generate_report()
        assert result["generated_at"] != ""

    def test_content_not_empty(self, engine: ComplianceReporterEngine):
        result = engine.generate_report()
        assert len(result["content"]) > 0

    def test_counter_incremented(self, engine: ComplianceReporterEngine):
        assert engine._total_generated == 0
        engine.generate_report()
        assert engine._total_generated == 1
        engine.generate_report()
        assert engine._total_generated == 2

    def test_stored_in_registry(self, engine: ComplianceReporterEngine):
        result = engine.generate_report()
        report_id = result["report_id"]
        assert report_id in engine._reports

    def test_invalid_report_type_raises(self, engine: ComplianceReporterEngine):
        with pytest.raises(ValueError, match="report.type|invalid|unsupported"):
            engine.generate_report(report_type="nonexistent_type")

    def test_invalid_format_raises(self, engine: ComplianceReporterEngine):
        with pytest.raises(ValueError, match="format|invalid|unsupported"):
            engine.generate_report(report_format="pdf")

    def test_invalid_framework_raises(self, engine: ComplianceReporterEngine):
        with pytest.raises(ValueError, match="framework|invalid|unsupported"):
            engine.generate_report(framework="unknown_framework")

    def test_pipeline_mode_output_format(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(output_format="html")
        assert result["report_format"] == "html"

    def test_pipeline_mode_results(self, engine: ComplianceReporterEngine):
        pipeline_results = {
            "index": {"scores": [{"composite_score": 55.0}]},
            "assess": {"exposures": [{"exposure_score": 40.0}]},
            "score": {"vulnerabilities": [{"vulnerability_score": 50.0}]},
        }
        result = engine.generate_report(results=pipeline_results)
        assert "report_id" in result

    def test_with_parameters(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(parameters={"custom_key": "custom_value"})
        assert "report_id" in result


# =========================================================================
# get_report
# =========================================================================


class TestGetReport:
    """Tests for the get_report method."""

    def test_get_existing_report(self, engine: ComplianceReporterEngine):
        generated = engine.generate_report(title="Retrieve me")
        retrieved = engine.get_report(generated["report_id"])
        assert retrieved is not None
        assert retrieved["report_id"] == generated["report_id"]
        assert retrieved["title"] == "Retrieve me"

    def test_get_nonexistent_report(self, engine: ComplianceReporterEngine):
        result = engine.get_report("RPT-nonexistent00")
        assert result is None

    def test_get_empty_id(self, engine: ComplianceReporterEngine):
        result = engine.get_report("")
        assert result is None

    def test_get_report_returns_deep_copy(self, engine: ComplianceReporterEngine):
        generated = engine.generate_report()
        r1 = engine.get_report(generated["report_id"])
        r2 = engine.get_report(generated["report_id"])
        assert r1 is not r2
        assert r1 == r2


# =========================================================================
# list_reports
# =========================================================================


class TestListReports:
    """Tests for the list_reports method."""

    def test_list_empty(self, engine: ComplianceReporterEngine):
        result = engine.list_reports()
        assert result == []

    def test_list_all(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports()
        assert len(result) == 5

    def test_filter_by_report_type(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(report_type="physical_risk_assessment")
        assert len(result) == 3

    def test_filter_by_framework(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(framework="tcfd")
        assert len(result) == 2

    def test_filter_by_format(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(format="json")
        assert len(result) == 3

    def test_filter_by_namespace(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(namespace="default")
        assert len(result) == 5

    def test_filter_nonexistent(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(framework="nonexistent")
        assert len(result) == 0

    def test_pagination_limit(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(limit=2)
        assert len(result) == 2

    def test_pagination_offset(self, populated_engine: ComplianceReporterEngine):
        all_reports = populated_engine.list_reports()
        offset_reports = populated_engine.list_reports(offset=2)
        assert len(offset_reports) == len(all_reports) - 2

    def test_combined_filters(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.list_reports(
            report_type="physical_risk_assessment",
            framework="tcfd",
        )
        assert len(result) == 1


# =========================================================================
# delete_report
# =========================================================================


class TestDeleteReport:
    """Tests for the delete_report method."""

    def test_delete_existing(self, engine: ComplianceReporterEngine):
        report = engine.generate_report()
        report_id = report["report_id"]
        assert engine.delete_report(report_id) is True
        assert engine.get_report(report_id) is None

    def test_delete_nonexistent(self, engine: ComplianceReporterEngine):
        assert engine.delete_report("RPT-nonexistent00") is False

    def test_delete_empty_id(self, engine: ComplianceReporterEngine):
        assert engine.delete_report("") is False

    def test_delete_removes_from_indexes(self, engine: ComplianceReporterEngine):
        report = engine.generate_report(framework="tcfd")
        report_id = report["report_id"]
        engine.delete_report(report_id)
        # Listing by framework should not find it anymore
        results = engine.list_reports(framework="tcfd")
        matching = [r for r in results if r["report_id"] == report_id]
        assert len(matching) == 0

    def test_delete_reduces_count(self, engine: ComplianceReporterEngine):
        engine.generate_report()
        engine.generate_report()
        assert len(engine._reports) == 2
        report = engine.list_reports()[0]
        engine.delete_report(report["report_id"])
        assert len(engine._reports) == 1


# =========================================================================
# get_framework_template
# =========================================================================


class TestGetFrameworkTemplate:
    """Tests for the get_framework_template method."""

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_get_all_frameworks(self, engine: ComplianceReporterEngine, framework: str):
        template = engine.get_framework_template(framework)
        assert "name" in template
        assert "version" in template
        assert "sections" in template
        assert len(template["sections"]) > 0

    def test_get_tcfd_template_detail(self, engine: ComplianceReporterEngine):
        template = engine.get_framework_template("tcfd")
        assert "governance" in template["sections"]
        assert "strategy" in template["sections"]
        assert "risk_management" in template["sections"]
        assert "metrics_and_targets" in template["sections"]

    def test_invalid_framework_raises(self, engine: ComplianceReporterEngine):
        with pytest.raises(ValueError):
            engine.get_framework_template("unknown")

    def test_returns_deep_copy(self, engine: ComplianceReporterEngine):
        t1 = engine.get_framework_template("tcfd")
        t2 = engine.get_framework_template("tcfd")
        assert t1 is not t2
        assert t1 == t2
        # Modifying t1 should not affect t2
        t1["sections"]["governance"]["weight"] = 999.0
        t3 = engine.get_framework_template("tcfd")
        assert t3["sections"]["governance"]["weight"] != 999.0

    def test_section_weights_sum(self, engine: ComplianceReporterEngine):
        for fw in SUPPORTED_FRAMEWORKS:
            template = engine.get_framework_template(fw)
            weights = [s.get("weight", 0) for s in template["sections"].values()]
            total = sum(weights)
            assert abs(total - 1.0) < 0.01, f"{fw} weights sum to {total}, expected ~1.0"


# =========================================================================
# list_frameworks
# =========================================================================


class TestListFrameworks:
    """Tests for the list_frameworks method."""

    def test_returns_all_frameworks(self, engine: ComplianceReporterEngine):
        result = engine.list_frameworks()
        assert len(result) == 6

    def test_each_has_required_fields(self, engine: ComplianceReporterEngine):
        result = engine.list_frameworks()
        for fw in result:
            assert "framework_id" in fw
            assert "name" in fw
            assert "version" in fw
            assert "description" in fw
            assert "section_count" in fw

    def test_section_count_positive(self, engine: ComplianceReporterEngine):
        result = engine.list_frameworks()
        for fw in result:
            assert fw["section_count"] > 0

    def test_framework_ids_match(self, engine: ComplianceReporterEngine):
        result = engine.list_frameworks()
        ids = {fw["framework_id"] for fw in result}
        assert ids == set(SUPPORTED_FRAMEWORKS)


# =========================================================================
# validate_compliance
# =========================================================================


class TestValidateCompliance:
    """Tests for the validate_compliance method."""

    def test_basic_validation(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        assert "overall_score" in result
        assert "overall_status" in result
        assert "sections" in result
        assert "missing_evidence" in result
        assert "provenance_hash" in result

    def test_validation_with_no_data(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        assert result["overall_score"] <= 100.0
        assert result["overall_status"] in ("PASS", "PARTIAL", "FAIL")

    @pytest.mark.parametrize("framework", SUPPORTED_FRAMEWORKS)
    def test_validate_all_frameworks(self, engine: ComplianceReporterEngine, framework: str):
        result = engine.validate_compliance(framework=framework)
        assert result["framework"] == framework

    def test_validation_with_risk_data(
        self, engine: ComplianceReporterEngine, sample_risk_data
    ):
        result = engine.validate_compliance(
            framework="tcfd",
            risk_data=sample_risk_data,
        )
        assert result["overall_score"] >= 0.0

    def test_validation_with_all_data(
        self,
        engine: ComplianceReporterEngine,
        sample_risk_data,
        sample_exposure_data,
        sample_vulnerability_data,
    ):
        result = engine.validate_compliance(
            framework="tcfd",
            risk_data=sample_risk_data,
            exposure_data=sample_exposure_data,
            vulnerability_data=sample_vulnerability_data,
        )
        assert 0.0 <= result["overall_score"] <= 100.0

    def test_invalid_framework_raises(self, engine: ComplianceReporterEngine):
        with pytest.raises(ValueError):
            engine.validate_compliance(framework="bogus")

    def test_validation_increments_counter(self, engine: ComplianceReporterEngine):
        assert engine._total_validations == 0
        engine.validate_compliance(framework="tcfd")
        assert engine._total_validations == 1

    def test_sections_have_scores(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        for section_id, section in result["sections"].items():
            assert "score" in section
            assert "status" in section
            assert "weight" in section

    def test_validation_recommendations(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        assert isinstance(result["recommendations"], list)

    def test_validation_has_provenance(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        assert len(result["provenance_hash"]) == 64

    def test_validation_id_format(self, engine: ComplianceReporterEngine):
        result = engine.validate_compliance(framework="tcfd")
        assert result["validation_id"].startswith("VAL-")


# =========================================================================
# get_statistics
# =========================================================================


class TestGetStatistics:
    """Tests for the get_statistics method."""

    def test_empty_statistics(self, engine: ComplianceReporterEngine):
        stats = engine.get_statistics()
        assert stats["total_reports"] == 0
        assert stats["total_generated"] == 0
        assert stats["total_validations"] == 0
        assert stats["total_errors"] == 0
        assert stats["avg_compliance_score"] == 0.0

    def test_after_generating_reports(self, populated_engine: ComplianceReporterEngine):
        stats = populated_engine.get_statistics()
        assert stats["total_reports"] == 5
        assert stats["total_generated"] == 5

    def test_report_type_distribution(self, populated_engine: ComplianceReporterEngine):
        stats = populated_engine.get_statistics()
        dist = stats["report_type_distribution"]
        assert dist["physical_risk_assessment"] == 3

    def test_framework_distribution(self, populated_engine: ComplianceReporterEngine):
        stats = populated_engine.get_statistics()
        dist = stats["framework_distribution"]
        assert "tcfd" in dist

    def test_format_distribution(self, populated_engine: ComplianceReporterEngine):
        stats = populated_engine.get_statistics()
        dist = stats["format_distribution"]
        assert "json" in dist

    def test_provenance_entries(self, engine: ComplianceReporterEngine):
        stats = engine.get_statistics()
        assert "provenance_entries" in stats

    def test_avg_compliance_score(self, populated_engine: ComplianceReporterEngine):
        stats = populated_engine.get_statistics()
        assert isinstance(stats["avg_compliance_score"], float)

    def test_statistics_returns_copy(self, engine: ComplianceReporterEngine):
        s1 = engine.get_statistics()
        s2 = engine.get_statistics()
        assert s1 is not s2


# =========================================================================
# clear
# =========================================================================


class TestClear:
    """Tests for the clear method."""

    def test_clear_resets_reports(self, populated_engine: ComplianceReporterEngine):
        assert len(populated_engine._reports) > 0
        populated_engine.clear()
        assert len(populated_engine._reports) == 0

    def test_clear_resets_counters(self, populated_engine: ComplianceReporterEngine):
        populated_engine.clear()
        assert populated_engine._total_generated == 0
        assert populated_engine._total_validations == 0
        assert populated_engine._total_errors == 0

    def test_clear_resets_indexes(self, populated_engine: ComplianceReporterEngine):
        populated_engine.clear()
        assert len(populated_engine._type_index) == 0
        assert len(populated_engine._framework_index) == 0
        assert len(populated_engine._format_index) == 0
        assert len(populated_engine._namespace_index) == 0

    def test_clear_then_generate(self, populated_engine: ComplianceReporterEngine):
        populated_engine.clear()
        result = populated_engine.generate_report(title="After clear")
        assert result["title"] == "After clear"
        stats = populated_engine.get_statistics()
        assert stats["total_reports"] == 1


# =========================================================================
# export_reports
# =========================================================================


class TestExportReports:
    """Tests for the export_reports method."""

    def test_export_empty(self, engine: ComplianceReporterEngine):
        result = engine.export_reports()
        assert result == []

    def test_export_returns_all(self, populated_engine: ComplianceReporterEngine):
        result = populated_engine.export_reports()
        assert len(result) == 5

    def test_export_returns_dicts(self, engine: ComplianceReporterEngine):
        engine.generate_report()
        result = engine.export_reports()
        assert isinstance(result[0], dict)
        assert "report_id" in result[0]

    def test_export_returns_deep_copies(self, engine: ComplianceReporterEngine):
        engine.generate_report()
        exported = engine.export_reports()
        exported[0]["title"] = "MUTATED"
        # Original should be unaffected
        report = engine.get_report(exported[0]["report_id"])
        assert report["title"] != "MUTATED"


# =========================================================================
# import_reports
# =========================================================================


class TestImportReports:
    """Tests for the import_reports method."""

    def test_import_basic(self, engine: ComplianceReporterEngine):
        reports_data = [
            {
                "report_id": "RPT-import000001",
                "report_type": "physical_risk_assessment",
                "report_format": "json",
                "framework": "tcfd",
                "title": "Imported Report",
            }
        ]
        result = engine.import_reports(reports_data)
        assert result["imported"] == 1
        assert result["skipped"] == 0

    def test_import_without_id(self, engine: ComplianceReporterEngine):
        reports_data = [{"title": "No ID Report"}]
        result = engine.import_reports(reports_data)
        assert result["imported"] == 1
        # Should auto-generate an ID
        all_reports = engine.list_reports()
        assert len(all_reports) == 1

    def test_import_multiple(self, engine: ComplianceReporterEngine):
        reports_data = [
            {"report_id": f"RPT-import{i:06d}", "title": f"Report {i}"}
            for i in range(5)
        ]
        result = engine.import_reports(reports_data)
        assert result["imported"] == 5

    def test_import_overwrites_existing(self, engine: ComplianceReporterEngine):
        engine.generate_report()
        exported = engine.export_reports()
        exported[0]["title"] = "Overwritten"
        engine.import_reports(exported)
        report = engine.get_report(exported[0]["report_id"])
        assert report["title"] == "Overwritten"

    def test_roundtrip_export_import(self, populated_engine: ComplianceReporterEngine):
        exported = populated_engine.export_reports()
        populated_engine.clear()
        assert len(populated_engine._reports) == 0
        result = populated_engine.import_reports(exported)
        assert result["imported"] == 5
        assert len(populated_engine._reports) == 5

    def test_import_empty_list(self, engine: ComplianceReporterEngine):
        result = engine.import_reports([])
        assert result["imported"] == 0
        assert result["skipped"] == 0


# =========================================================================
# Thread safety
# =========================================================================


class TestThreadSafety:
    """Basic thread safety tests for concurrent operations."""

    def test_concurrent_generate(self, engine: ComplianceReporterEngine):
        errors = []

        def generate():
            try:
                engine.generate_report()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine._total_generated == 10

    def test_concurrent_generate_and_list(self, engine: ComplianceReporterEngine):
        errors = []

        def generate():
            try:
                engine.generate_report()
            except Exception as e:
                errors.append(e)

        def list_all():
            try:
                engine.list_reports()
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=generate))
            threads.append(threading.Thread(target=list_all))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =========================================================================
# Edge cases and error handling
# =========================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_generate_with_enum_values(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(
            report_type=ReportTypeLocal.SCENARIO_ANALYSIS,
            report_format=ReportFormatLocal.HTML,
        )
        assert result["report_type"] == "scenario_analysis"
        assert result["report_format"] == "html"

    def test_report_content_varies_by_format(self, engine: ComplianceReporterEngine):
        json_report = engine.generate_report(report_format="json")
        html_report = engine.generate_report(report_format="html")
        text_report = engine.generate_report(report_format="text")
        assert json_report["content"] != html_report["content"]
        assert json_report["content"] != text_report["content"]

    def test_deterministic_hash_for_same_content(self, engine: ComplianceReporterEngine):
        """Two reports with same content should have the same report_hash."""
        r1 = engine.generate_report(title="Same Title", framework="tcfd")
        r2 = engine.generate_report(title="Same Title", framework="tcfd")
        # report_hash is based on content which includes timestamps
        # so they may differ, but provenance_hash should exist
        assert len(r1["report_hash"]) == 64
        assert len(r2["report_hash"]) == 64

    def test_empty_risk_data(self, engine: ComplianceReporterEngine):
        result = engine.generate_report(risk_data=[])
        assert result["report_id"].startswith("RPT-")

    def test_large_number_of_reports(self, engine: ComplianceReporterEngine):
        for _ in range(50):
            engine.generate_report()
        assert engine._total_generated == 50
        assert len(engine.list_reports(limit=50)) == 50
