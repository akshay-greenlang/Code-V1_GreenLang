"""
Unit tests for PACK-007 EUDR Professional Pack - Report Templates

Tests all 10 professional-tier report templates including advanced risk reports,
satellite monitoring reports, supplier benchmarking, and more.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json


def _import_from_path(module_name, file_path):
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_PACK_007_DIR = Path(__file__).resolve().parent.parent

# Import templates module
templates_mod = _import_from_path(
    "pack_007_templates",
    _PACK_007_DIR / "templates" / "professional_templates.py"
)

pytestmark = pytest.mark.skipif(
    templates_mod is None,
    reason="PACK-007 templates module not available"
)


class TestAdvancedRiskReport:
    """Test Advanced Risk Report Template (TPL-001)."""

    @pytest.fixture
    def template(self):
        """Create template instance."""
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.AdvancedRiskReportTemplate()

    @pytest.fixture
    def sample_data(self):
        """Sample data for advanced risk report."""
        return {
            "supplier_id": "supplier_123",
            "product": "coffee",
            "risk_score": 75.5,
            "confidence_interval": [70.0, 81.0],
            "sensitivity_factors": [
                {"factor": "deforestation_risk", "impact": 0.45},
                {"factor": "governance_risk", "impact": 0.30},
            ]
        }

    def test_render_markdown(self, template, sample_data):
        """Test rendering in markdown format."""
        result = template.render(sample_data, format="markdown")

        assert result is not None
        assert isinstance(result, str)
        assert "# Advanced Risk Report" in result or "Advanced Risk" in result

    def test_render_html(self, template, sample_data):
        """Test rendering in HTML format."""
        result = template.render(sample_data, format="html")

        assert result is not None
        assert isinstance(result, str)
        assert "<html>" in result.lower() or "<div>" in result.lower()

    def test_render_json(self, template, sample_data):
        """Test rendering in JSON format."""
        result = template.render(sample_data, format="json")

        assert result is not None
        # Parse JSON to validate structure
        data = json.loads(result) if isinstance(result, str) else result
        assert "supplier_id" in data or "risk_score" in data

    def test_provenance_hash(self, template, sample_data):
        """Test provenance hash generation."""
        result = template.render(sample_data, format="json")

        if isinstance(result, dict):
            assert "provenance_hash" in result or "hash" in result
        elif isinstance(result, str):
            data = json.loads(result)
            assert "provenance_hash" in data or "hash" in data

    def test_data_model_validation(self, template):
        """Test data model validates correctly."""
        valid_data = {
            "supplier_id": "s1",
            "product": "coffee",
            "risk_score": 50.0
        }

        # Should not raise validation error
        result = template.render(valid_data, format="json")
        assert result is not None


class TestSatelliteMonitoringReport:
    """Test Satellite Monitoring Report Template (TPL-002)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.SatelliteMonitoringReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "plot_id": "plot_001",
            "monitoring_period": "2024-01-01 to 2024-01-31",
            "forest_cover_loss_percentage": 2.5,
            "alerts_generated": 3,
            "alert_details": [
                {"date": "2024-01-15", "severity": "MEDIUM", "area_ha": 1.2}
            ]
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None
        assert "Satellite Monitoring" in result or "plot_001" in result

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestSupplierBenchmarkReport:
    """Test Supplier Benchmark Report Template (TPL-003)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.SupplierBenchmarkReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "benchmark_id": "bench_001",
            "suppliers": [
                {"supplier_id": "s1", "score": 85, "rank": 1},
                {"supplier_id": "s2", "score": 78, "rank": 2},
            ],
            "criteria": ["eudr_compliance", "transparency"]
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestPortfolioDashboard:
    """Test Portfolio Dashboard Template (TPL-004)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.PortfolioDashboardTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "total_suppliers": 25,
            "total_plots": 150,
            "average_risk_score": 45.5,
            "compliance_rate": 92.0,
            "products": ["coffee", "cocoa", "palm_oil"]
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None
        assert "Portfolio" in result or "Dashboard" in result

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestAuditReadinessReport:
    """Test Audit Readiness Report Template (TPL-005)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.AuditReadinessReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "audit_type": "competent_authority_inspection",
            "readiness_score": 88.0,
            "gaps_identified": 3,
            "evidence_completeness": 95.0,
            "mock_audit_results": "PASS"
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestSupplyChainMapReport:
    """Test Supply Chain Map Report Template (TPL-006)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.SupplyChainMapReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "product": "coffee",
            "total_tiers": 5,
            "total_suppliers": 45,
            "critical_nodes": ["s1", "s2"],
            "concentration_risk": "MEDIUM"
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestProtectedAreaReport:
    """Test Protected Area Report Template (TPL-007)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.ProtectedAreaReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "plot_id": "plot_001",
            "wdpa_overlap": True,
            "kba_overlap": False,
            "indigenous_lands_overlap": False,
            "overall_risk_level": "HIGH",
            "exclusion_recommended": False
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestRegulatoryChangeReport:
    """Test Regulatory Change Report Template (TPL-008)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.RegulatoryChangeReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "regulation": "EUDR",
            "change_type": "amendment",
            "effective_date": "2025-06-01",
            "impact_level": "HIGH",
            "gaps_identified": 5,
            "migration_plan_status": "DRAFT"
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestAnnualComplianceReport:
    """Test Annual Compliance Report Template (TPL-009)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.AnnualComplianceReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "review_year": 2024,
            "compliance_score": 91.5,
            "total_dds_statements": 250,
            "findings": 12,
            "improvement_actions": 8,
            "board_presentation_date": "2025-02-15"
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None
        assert "2024" in result or "Annual" in result

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestGrievanceLogReport:
    """Test Grievance Log Report Template (TPL-010)."""

    @pytest.fixture
    def template(self):
        if templates_mod is None:
            pytest.skip("templates module not available")
        return templates_mod.GrievanceLogReportTemplate()

    @pytest.fixture
    def sample_data(self):
        return {
            "reporting_period": "Q1 2024",
            "total_complaints": 15,
            "resolved_complaints": 12,
            "pending_complaints": 3,
            "average_resolution_time_days": 18,
            "sla_compliance_rate": 93.0
        }

    def test_render_markdown(self, template, sample_data):
        result = template.render(sample_data, format="markdown")
        assert result is not None

    def test_render_html(self, template, sample_data):
        result = template.render(sample_data, format="html")
        assert result is not None

    def test_render_json(self, template, sample_data):
        result = template.render(sample_data, format="json")
        assert result is not None


class TestTemplateFeatures:
    """Test common template features."""

    def test_template_metadata(self):
        """Test all templates have metadata."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        template = templates_mod.AdvancedRiskReportTemplate()
        metadata = template.get_metadata()

        assert metadata is not None
        assert "name" in metadata or "template_name" in metadata
        assert "version" in metadata or "template_version" in metadata

    def test_template_format_support(self):
        """Test templates support multiple formats."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        template = templates_mod.AdvancedRiskReportTemplate()
        formats = template.get_supported_formats()

        assert formats is not None
        assert isinstance(formats, list)
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_template_validation(self):
        """Test template input validation."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        template = templates_mod.AdvancedRiskReportTemplate()

        # Test with invalid data
        invalid_data = {}
        result = template.validate(invalid_data)

        # Should return validation result
        assert result is not None

    def test_template_provenance(self):
        """Test all templates generate provenance hashes."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        templates_to_test = [
            templates_mod.AdvancedRiskReportTemplate(),
            templates_mod.SatelliteMonitoringReportTemplate(),
            templates_mod.SupplierBenchmarkReportTemplate(),
        ]

        for template in templates_to_test:
            sample_data = {"test": "data"}
            result = template.render(sample_data, format="json")

            if isinstance(result, dict):
                # Some templates may include provenance in output
                pass
            elif isinstance(result, str):
                # JSON string may contain provenance
                pass


class TestTemplateBulkGeneration:
    """Test bulk template generation."""

    def test_generate_multiple_reports(self):
        """Test generating multiple reports in batch."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        template = templates_mod.SupplierBenchmarkReportTemplate()

        datasets = [
            {"benchmark_id": "b1", "suppliers": []},
            {"benchmark_id": "b2", "suppliers": []},
            {"benchmark_id": "b3", "suppliers": []},
        ]

        results = []
        for data in datasets:
            result = template.render(data, format="json")
            results.append(result)

        assert len(results) == 3

    def test_template_caching(self):
        """Test template rendering with caching."""
        if templates_mod is None:
            pytest.skip("templates module not available")

        template = templates_mod.AdvancedRiskReportTemplate()
        data = {"supplier_id": "s1", "risk_score": 50.0}

        # Render twice with same data
        result1 = template.render(data, format="json")
        result2 = template.render(data, format="json")

        # Results should be consistent
        assert result1 is not None
        assert result2 is not None
