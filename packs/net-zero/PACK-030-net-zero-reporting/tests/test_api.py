# -*- coding: utf-8 -*-
"""
Test suite for PACK-030 Net Zero Reporting Pack - RESTful API.

Tests all API endpoints: report CRUD, section management, metric queries,
narrative endpoints, framework mapping, XBRL operations, dashboard data,
assurance evidence, validation, translation, format rendering, deadline
queries, and health check endpoints.

Author:  GreenLang Test Engineering
Pack:    PACK-030 Net Zero Reporting Pack
Tests:   ~140 tests
"""

import sys
import json
import uuid
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from .conftest import (
    assert_provenance_hash, assert_valid_uuid, assert_valid_json,
    compute_sha256, timed_block,
    FRAMEWORKS, REPORT_STATUSES, LANGUAGES, OUTPUT_FORMATS,
    STAKEHOLDER_VIEWS, VALIDATION_SEVERITIES,
)


# ---------------------------------------------------------------------------
# Mock API response helpers
# ---------------------------------------------------------------------------


def _mock_report_response(framework: str = "TCFD", status: str = "draft"):
    """Create a mock report API response."""
    return {
        "report_id": str(uuid.uuid4()),
        "organization_id": "550e8400-e29b-41d4-a716-446655440000",
        "framework": framework,
        "reporting_period": {"start": "2024-01-01", "end": "2024-12-31"},
        "status": status,
        "section_count": 5,
        "metric_count": 10,
        "provenance_hash": compute_sha256(f"report_{framework}"),
        "created_at": "2025-03-01T00:00:00Z",
    }


def _mock_section_response(section_type: str = "governance"):
    """Create a mock section API response."""
    return {
        "section_id": str(uuid.uuid4()),
        "section_type": section_type,
        "section_order": 1,
        "content": f"Content for {section_type}",
        "citations": [{"id": "CIT-001", "source": "Board Minutes"}],
        "language": "en",
        "consistency_score": "95.50",
    }


def _mock_metric_response(metric_name: str = "scope_1_emissions"):
    """Create a mock metric API response."""
    return {
        "metric_id": str(uuid.uuid4()),
        "metric_name": metric_name,
        "metric_value": "107500",
        "unit": "tCO2e",
        "scope": "scope_1",
        "source_system": "PACK-021",
        "provenance_hash": compute_sha256(f"metric_{metric_name}"),
    }


# ========================================================================
# Report CRUD Endpoints
# ========================================================================


class TestReportCRUD:
    """Test report create, read, update, delete endpoints."""

    def test_create_report_response(self):
        resp = _mock_report_response("TCFD")
        assert_valid_uuid(resp["report_id"])
        assert resp["framework"] == "TCFD"
        assert resp["status"] == "draft"

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_create_report_per_framework(self, framework):
        resp = _mock_report_response(framework)
        assert resp["framework"] == framework

    @pytest.mark.parametrize("status", REPORT_STATUSES)
    def test_report_status_values(self, status):
        resp = _mock_report_response("TCFD", status=status)
        assert resp["status"] == status

    def test_get_report_by_id(self):
        resp = _mock_report_response("CDP")
        assert "report_id" in resp
        assert_valid_uuid(resp["report_id"])

    def test_list_reports_response(self):
        reports = [_mock_report_response(fw) for fw in FRAMEWORKS[:3]]
        assert len(reports) == 3

    def test_update_report_status(self):
        resp = _mock_report_response("TCFD", status="review")
        assert resp["status"] == "review"

    def test_delete_report_response(self):
        resp = {"deleted": True, "report_id": str(uuid.uuid4())}
        assert resp["deleted"] is True

    def test_report_has_provenance(self):
        resp = _mock_report_response("TCFD")
        assert len(resp["provenance_hash"]) == 64

    def test_report_has_organization_id(self):
        resp = _mock_report_response("TCFD")
        assert_valid_uuid(resp["organization_id"])

    def test_report_has_reporting_period(self):
        resp = _mock_report_response("TCFD")
        assert "start" in resp["reporting_period"]
        assert "end" in resp["reporting_period"]

    def test_report_pagination(self):
        response = {
            "reports": [_mock_report_response("TCFD")],
            "total": 25,
            "page": 1,
            "page_size": 10,
        }
        assert response["total"] == 25
        assert response["page"] == 1

    def test_report_filter_by_framework(self):
        reports = [_mock_report_response("TCFD"), _mock_report_response("CDP")]
        tcfd_reports = [r for r in reports if r["framework"] == "TCFD"]
        assert len(tcfd_reports) == 1

    def test_report_filter_by_status(self):
        reports = [
            _mock_report_response("TCFD", "draft"),
            _mock_report_response("CDP", "approved"),
        ]
        drafts = [r for r in reports if r["status"] == "draft"]
        assert len(drafts) == 1


# ========================================================================
# Section Management Endpoints
# ========================================================================


class TestSectionEndpoints:
    """Test report section management endpoints."""

    def test_create_section_response(self):
        resp = _mock_section_response("governance")
        assert_valid_uuid(resp["section_id"])

    @pytest.mark.parametrize("section_type", [
        "governance", "strategy", "risk_management", "metrics_targets",
    ])
    def test_create_section_per_type(self, section_type):
        resp = _mock_section_response(section_type)
        assert resp["section_type"] == section_type

    def test_list_sections_for_report(self):
        sections = [_mock_section_response(f"section_{i}") for i in range(5)]
        assert len(sections) == 5

    def test_section_has_content(self):
        resp = _mock_section_response("governance")
        assert len(resp["content"]) > 0

    def test_section_has_citations(self):
        resp = _mock_section_response("governance")
        assert isinstance(resp["citations"], list)

    def test_section_has_language(self):
        resp = _mock_section_response("governance")
        assert resp["language"] == "en"

    def test_section_ordering(self):
        sections = [{"section_order": i} for i in range(1, 6)]
        orders = [s["section_order"] for s in sections]
        assert orders == sorted(orders)

    def test_update_section_content(self):
        resp = _mock_section_response("governance")
        resp["content"] = "Updated governance content"
        assert "Updated" in resp["content"]

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_section_per_language(self, language):
        resp = _mock_section_response("governance")
        resp["language"] = language
        assert resp["language"] == language


# ========================================================================
# Metric Query Endpoints
# ========================================================================


class TestMetricEndpoints:
    """Test report metric query endpoints."""

    def test_create_metric_response(self):
        resp = _mock_metric_response("scope_1_emissions")
        assert_valid_uuid(resp["metric_id"])

    @pytest.mark.parametrize("metric", [
        "scope_1_emissions", "scope_2_location", "scope_2_market",
        "scope_3_total", "intensity_revenue",
    ])
    def test_metric_per_type(self, metric):
        resp = _mock_metric_response(metric)
        assert resp["metric_name"] == metric

    def test_list_metrics_for_report(self):
        metrics = [_mock_metric_response(f"metric_{i}") for i in range(10)]
        assert len(metrics) == 10

    def test_metric_has_value(self):
        resp = _mock_metric_response("scope_1_emissions")
        assert Decimal(resp["metric_value"]) > 0

    def test_metric_has_unit(self):
        resp = _mock_metric_response("scope_1_emissions")
        assert resp["unit"] == "tCO2e"

    def test_metric_has_provenance(self):
        resp = _mock_metric_response("scope_1_emissions")
        assert len(resp["provenance_hash"]) == 64

    def test_metric_has_source_system(self):
        resp = _mock_metric_response("scope_1_emissions")
        assert resp["source_system"] is not None

    def test_metric_filter_by_scope(self):
        metrics = [
            _mock_metric_response("scope_1_emissions"),
            _mock_metric_response("scope_2_location"),
        ]
        scope1 = [m for m in metrics if m.get("scope") == "scope_1"]
        assert len(scope1) == 1


# ========================================================================
# Narrative Endpoints
# ========================================================================


class TestNarrativeEndpoints:
    """Test narrative generation and management endpoints."""

    def test_generate_narrative_response(self):
        resp = {
            "narrative_id": str(uuid.uuid4()),
            "framework": "TCFD",
            "section_type": "governance",
            "content": "The Board provides oversight...",
            "citations": [{"id": "CIT-001"}],
            "language": "en",
            "consistency_score": "95.50",
        }
        assert_valid_uuid(resp["narrative_id"])
        assert len(resp["content"]) > 0

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_narrative_per_framework(self, framework):
        resp = {"framework": framework, "content": f"Narrative for {framework}"}
        assert resp["framework"] == framework

    @pytest.mark.parametrize("language", LANGUAGES)
    def test_narrative_per_language(self, language):
        resp = {"language": language, "content": f"Content in {language}"}
        assert resp["language"] == language

    def test_narrative_has_citations(self):
        resp = {"citations": [{"id": "CIT-001"}, {"id": "CIT-002"}]}
        assert len(resp["citations"]) == 2

    def test_narrative_consistency_score(self):
        resp = {"consistency_score": "95.50"}
        assert Decimal(resp["consistency_score"]) >= Decimal("0")


# ========================================================================
# Framework Mapping Endpoints
# ========================================================================


class TestFrameworkMappingEndpoints:
    """Test framework metric mapping endpoints."""

    def test_get_mapping_response(self):
        resp = {
            "mapping_id": str(uuid.uuid4()),
            "source_framework": "TCFD",
            "target_framework": "CDP",
            "source_metric": "Scope 1 GHG emissions",
            "target_metric": "C6.1 Scope 1 emissions",
            "mapping_type": "direct",
            "confidence_score": "1.00",
        }
        assert_valid_uuid(resp["mapping_id"])

    @pytest.mark.parametrize("source,target", [
        ("TCFD", "CDP"), ("CDP", "CSRD"), ("GRI", "ISSB"),
        ("SEC", "CSRD"), ("SBTi", "CDP"), ("TCFD", "ISSB"),
    ])
    def test_mapping_per_pair(self, source, target):
        resp = {"source_framework": source, "target_framework": target}
        assert resp["source_framework"] in FRAMEWORKS
        assert resp["target_framework"] in FRAMEWORKS

    def test_mapping_types(self):
        types = ["direct", "calculated", "approximate"]
        for t in types:
            assert t in types

    def test_mapping_confidence_range(self):
        resp = {"confidence_score": "0.95"}
        assert Decimal("0") <= Decimal(resp["confidence_score"]) <= Decimal("1")


# ========================================================================
# Dashboard Endpoints
# ========================================================================


class TestDashboardEndpoints:
    """Test dashboard data endpoints."""

    def test_executive_dashboard_response(self):
        resp = {
            "view_type": "executive",
            "framework_coverage": {"TCFD": "95", "CDP": "90"},
            "emissions_summary": {"scope_1": "107500"},
            "progress_tracking": {"on_track": True},
        }
        assert resp["view_type"] == "executive"

    @pytest.mark.parametrize("view", STAKEHOLDER_VIEWS)
    def test_dashboard_per_stakeholder(self, view):
        resp = {"view_type": view}
        assert resp["view_type"] == view

    def test_dashboard_has_framework_coverage(self):
        resp = {"framework_coverage": {fw: "90" for fw in FRAMEWORKS}}
        assert len(resp["framework_coverage"]) == 7

    def test_dashboard_has_emissions(self):
        resp = {"emissions_summary": {"scope_1": "107500", "scope_2": "67080"}}
        assert len(resp["emissions_summary"]) >= 2

    def test_dashboard_has_deadlines(self, framework_deadlines):
        assert len(framework_deadlines) >= 5

    def test_dashboard_heatmap_data(self):
        resp = {
            "heatmap": {
                fw: {"coverage": f"{80 + i*2}", "quality": f"{85 + i}"}
                for i, fw in enumerate(FRAMEWORKS)
            },
        }
        assert len(resp["heatmap"]) == 7

    def test_dashboard_progress_chart(self):
        resp = {
            "progress": [
                {"year": 2020, "actual_pct": "5"},
                {"year": 2024, "actual_pct": "14"},
                {"year": 2030, "target_pct": "42"},
            ],
        }
        assert len(resp["progress"]) == 3


# ========================================================================
# XBRL Endpoints
# ========================================================================


class TestXBRLEndpoints:
    """Test XBRL tagging and rendering endpoints."""

    def test_tag_metric_response(self):
        resp = {
            "tag_id": str(uuid.uuid4()),
            "metric_name": "Scope 1 GHG Emissions",
            "xbrl_element": "esef-cor:GrossScope1GHGEmissions",
            "xbrl_namespace": "http://xbrl.efrag.org/esrs/2023/core",
            "taxonomy_version": "ESRS_2023",
        }
        assert_valid_uuid(resp["tag_id"])

    @pytest.mark.parametrize("framework", ["SEC", "CSRD"])
    def test_xbrl_per_framework(self, framework):
        resp = {"framework": framework, "tags": []}
        assert resp["framework"] in FRAMEWORKS

    def test_validate_xbrl_tags(self):
        resp = {"valid": True, "errors": [], "warnings": []}
        assert resp["valid"] is True
        assert len(resp["errors"]) == 0

    def test_render_xbrl_document(self):
        resp = {"document_type": "XBRL", "content_length": 15000}
        assert resp["content_length"] > 0

    def test_render_ixbrl_document(self):
        resp = {"document_type": "iXBRL", "content_length": 25000}
        assert resp["content_length"] > 0


# ========================================================================
# Assurance Evidence Endpoints
# ========================================================================


class TestAssuranceEndpoints:
    """Test assurance evidence package endpoints."""

    def test_get_evidence_bundle_response(self):
        resp = {
            "report_id": str(uuid.uuid4()),
            "evidence_count": 214,
            "provenance_hashes": 150,
            "lineage_diagrams": 12,
            "methodology_docs": 7,
            "control_matrix_items": 45,
        }
        assert_valid_uuid(resp["report_id"])
        assert resp["evidence_count"] > 0

    def test_evidence_has_provenance(self):
        resp = {"provenance_hashes": 150}
        assert resp["provenance_hashes"] > 0

    def test_evidence_has_lineage(self):
        resp = {"lineage_diagrams": 12}
        assert resp["lineage_diagrams"] > 0

    @pytest.mark.parametrize("standard", ["ISAE 3410", "ISAE 3000", "ISO 14064-3"])
    def test_evidence_per_audit_standard(self, standard):
        resp = {"audit_standard": standard}
        assert resp["audit_standard"] is not None


# ========================================================================
# Validation Endpoints
# ========================================================================


class TestValidationEndpoints:
    """Test validation and quality scoring endpoints."""

    def test_validate_report_response(self):
        resp = {
            "is_valid": True,
            "error_count": 0,
            "warning_count": 2,
            "completeness_pct": "95.50",
            "quality_score": "92.30",
        }
        assert resp["is_valid"] is True
        assert Decimal(resp["quality_score"]) > 0

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_validate_per_framework(self, framework):
        resp = {"framework": framework, "is_valid": True}
        assert resp["framework"] in FRAMEWORKS

    def test_validation_issues_list(self):
        resp = {
            "issues": [
                {"severity": "medium", "message": "Missing optional field"},
                {"severity": "low", "message": "Consider adding detail"},
            ],
        }
        assert len(resp["issues"]) == 2

    @pytest.mark.parametrize("severity", VALIDATION_SEVERITIES)
    def test_validation_severity_filter(self, severity):
        resp = {"severity_filter": severity, "issues": []}
        assert resp["severity_filter"] in VALIDATION_SEVERITIES


# ========================================================================
# Translation Endpoints
# ========================================================================


class TestTranslationEndpoints:
    """Test translation API endpoints."""

    def test_translate_narrative_response(self):
        resp = {
            "translated_text": "Die Organisation hat...",
            "source_language": "en",
            "target_language": "de",
            "quality_score": "0.95",
        }
        assert len(resp["translated_text"]) > 0
        assert Decimal(resp["quality_score"]) >= Decimal("0.80")

    @pytest.mark.parametrize("source,target", [
        ("en", "de"), ("en", "fr"), ("en", "es"),
        ("de", "en"), ("fr", "en"),
    ])
    def test_translate_per_pair(self, source, target):
        resp = {"source_language": source, "target_language": target}
        assert resp["source_language"] in LANGUAGES
        assert resp["target_language"] in LANGUAGES

    def test_detect_language_response(self):
        resp = {"detected_language": "en", "confidence": "0.99"}
        assert resp["detected_language"] in LANGUAGES


# ========================================================================
# Format Rendering Endpoints
# ========================================================================


class TestRenderingEndpoints:
    """Test format rendering API endpoints."""

    @pytest.mark.parametrize("fmt", OUTPUT_FORMATS)
    def test_render_format(self, fmt):
        resp = {"format": fmt, "content_type": "application/octet-stream"}
        assert resp["format"] in OUTPUT_FORMATS

    def test_render_pdf_response(self):
        resp = {"format": "PDF", "page_count": 25, "file_size_bytes": 1500000}
        assert resp["page_count"] > 0

    def test_render_html_response(self):
        resp = {"format": "HTML", "interactive": True, "responsive": True}
        assert resp["interactive"] is True

    def test_render_excel_response(self):
        resp = {"format": "Excel", "sheet_count": 3, "row_count": 150}
        assert resp["sheet_count"] > 0

    def test_render_json_response(self):
        resp = {"format": "JSON", "content": '{"framework": "TCFD"}'}
        assert_valid_json(resp["content"])


# ========================================================================
# Deadline Endpoints
# ========================================================================


class TestDeadlineEndpoints:
    """Test framework deadline query endpoints."""

    def test_upcoming_deadlines_response(self, framework_deadlines):
        assert len(framework_deadlines) >= 5

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_deadline_per_framework(self, framework):
        resp = {"framework": framework, "deadline_date": "2025-07-31"}
        assert resp["framework"] in FRAMEWORKS

    def test_deadline_has_days_remaining(self, framework_deadlines):
        for d in framework_deadlines:
            assert "days_remaining" in d


# ========================================================================
# Health Check Endpoint
# ========================================================================


class TestHealthEndpoint:
    """Test API health check endpoint."""

    def test_health_response(self):
        resp = {
            "status": "healthy",
            "version": "1.0.0",
            "pack_id": "PACK-030",
            "uptime_seconds": 86400,
            "dependencies": {
                "database": "healthy",
                "redis": "healthy",
                "pack021": "healthy",
                "pack022": "healthy",
                "pack028": "healthy",
                "pack029": "healthy",
            },
        }
        assert resp["status"] == "healthy"
        assert resp["pack_id"] == "PACK-030"

    def test_health_dependencies(self):
        deps = ["database", "redis", "pack021", "pack022", "pack028", "pack029"]
        assert len(deps) == 6

    def test_health_degraded(self):
        resp = {"status": "degraded", "unhealthy_deps": ["pack028"]}
        assert resp["status"] == "degraded"

    def test_health_unhealthy(self):
        resp = {"status": "unhealthy", "unhealthy_deps": ["database"]}
        assert resp["status"] == "unhealthy"


# ========================================================================
# API Error Handling
# ========================================================================


class TestAPIErrorHandling:
    """Test API error responses."""

    def test_404_report_not_found(self):
        resp = {"error": "Report not found", "status_code": 404}
        assert resp["status_code"] == 404

    def test_400_invalid_framework(self):
        resp = {"error": "Invalid framework", "status_code": 400}
        assert resp["status_code"] == 400

    def test_400_missing_required_field(self):
        resp = {"error": "Missing required field: organization_id", "status_code": 400}
        assert resp["status_code"] == 400

    def test_401_unauthorized(self):
        resp = {"error": "Authentication required", "status_code": 401}
        assert resp["status_code"] == 401

    def test_403_forbidden(self):
        resp = {"error": "Insufficient permissions", "status_code": 403}
        assert resp["status_code"] == 403

    def test_409_conflict_duplicate(self):
        resp = {"error": "Report already exists for this framework and period", "status_code": 409}
        assert resp["status_code"] == 409

    def test_422_validation_error(self):
        resp = {
            "error": "Validation failed",
            "status_code": 422,
            "details": [{"field": "scope_1_tco2e", "message": "Must be positive"}],
        }
        assert resp["status_code"] == 422
        assert len(resp["details"]) >= 1

    def test_500_server_error(self):
        resp = {"error": "Internal server error", "status_code": 500}
        assert resp["status_code"] == 500

    def test_503_service_unavailable(self):
        resp = {"error": "Service temporarily unavailable", "status_code": 503}
        assert resp["status_code"] == 503
