# -*- coding: utf-8 -*-
"""
Unit tests for RiskReportingEngine - AGENT-EUDR-017 Engine 8

Tests risk report generation in multiple formats (PDF, JSON, HTML, Excel, CSV)
with 6 report types, DDS package assembly, audit trail documentation, and
executive summaries.

Target: 50+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import pytest

from greenlang.agents.eudr.supplier_risk_scorer.risk_reporting_engine import (
    RiskReportingEngine,
)
from greenlang.agents.eudr.supplier_risk_scorer.models import (
    ReportType,
    ReportFormat,
    RiskLevel,
)


class TestRiskReportingEngineInit:
    """Tests for RiskReportingEngine initialization."""

    @pytest.mark.unit
    def test_initialization(self, mock_config):
        engine = RiskReportingEngine()
        assert engine._reports == {}


class TestGenerateReport:
    """Tests for generate_report method."""

    @pytest.mark.unit
    def test_generate_report_returns_result(
        self, risk_reporting_engine
    ):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        assert result is not None
        assert "report_id" in result
        assert result["report_type"] == ReportType.INDIVIDUAL


class TestAllReportTypes:
    """Tests for all 6 report types."""

    @pytest.mark.unit
    @pytest.mark.parametrize("report_type", [
        ReportType.INDIVIDUAL,
        ReportType.PORTFOLIO,
        ReportType.COMPARATIVE,
        ReportType.TREND,
        ReportType.AUDIT_PACKAGE,
        ReportType.EXECUTIVE,
    ])
    def test_generate_all_report_types(
        self, risk_reporting_engine, report_type
    ):
        """Test generation of all 6 report types."""
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001" if report_type == ReportType.INDIVIDUAL else None,
            supplier_ids=["SUPP-A", "SUPP-B"] if report_type in [
                ReportType.PORTFOLIO, ReportType.COMPARATIVE
            ] else None,
            report_type=report_type,
            format=ReportFormat.JSON,
        )
        assert result["report_type"] == report_type


class TestAllFormats:
    """Tests for all 5 output formats."""

    @pytest.mark.unit
    @pytest.mark.parametrize("format", [
        ReportFormat.JSON,
        ReportFormat.HTML,
        ReportFormat.PDF,
        ReportFormat.EXCEL,
        ReportFormat.CSV,
    ])
    def test_generate_all_formats(
        self, risk_reporting_engine, format
    ):
        """Test generation in all 5 formats."""
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=format,
        )
        assert result["format"] == format
        assert "content" in result or "file_path" in result


class TestIndividualReport:
    """Tests for individual supplier report."""

    @pytest.mark.unit
    def test_generate_individual_report(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        assert result["report_type"] == ReportType.INDIVIDUAL
        assert "risk_assessment" in result["content"]
        assert "dd_status" in result["content"]


class TestPortfolioReport:
    """Tests for portfolio-level report."""

    @pytest.mark.unit
    def test_generate_portfolio_report(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_ids=["SUPP-A", "SUPP-B", "SUPP-C"],
            report_type=ReportType.PORTFOLIO,
            format=ReportFormat.JSON,
        )
        assert result["report_type"] == ReportType.PORTFOLIO
        assert "total_suppliers" in result["content"]
        assert "risk_distribution" in result["content"]


class TestComparativeReport:
    """Tests for comparative analysis report."""

    @pytest.mark.unit
    def test_generate_comparative_report(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_ids=["SUPP-A", "SUPP-B"],
            report_type=ReportType.COMPARATIVE,
            format=ReportFormat.JSON,
        )
        assert result["report_type"] == ReportType.COMPARATIVE
        assert "comparisons" in result["content"]


class TestTrendReport:
    """Tests for trend analysis report."""

    @pytest.mark.unit
    def test_generate_trend_report(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.TREND,
            format=ReportFormat.JSON,
            date_range={
                "start": datetime.now(timezone.utc) - timedelta(days=365),
                "end": datetime.now(timezone.utc),
            },
        )
        assert result["report_type"] == ReportType.TREND
        assert "trend_analysis" in result["content"]


class TestAuditPackage:
    """Tests for audit package assembly."""

    @pytest.mark.unit
    def test_generate_audit_package(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.AUDIT_PACKAGE,
            format=ReportFormat.PDF,
        )
        assert result["report_type"] == ReportType.AUDIT_PACKAGE
        assert "documents" in result["content"]
        assert "audit_trail" in result["content"]
        assert "dds_reference" in result["content"]


class TestExecutiveSummary:
    """Tests for executive summary report."""

    @pytest.mark.unit
    def test_generate_executive_summary(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_ids=["SUPP-A", "SUPP-B", "SUPP-C"],
            report_type=ReportType.EXECUTIVE,
            format=ReportFormat.PDF,
        )
        assert result["report_type"] == ReportType.EXECUTIVE
        assert "key_metrics" in result["content"]
        assert "recommendations" in result["content"]


class TestSHA256Hash:
    """Tests for report SHA-256 hash generation."""

    @pytest.mark.unit
    def test_report_includes_sha256_hash(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        assert "content_hash" in result
        assert len(result["content_hash"]) == 64  # SHA-256

    @pytest.mark.unit
    def test_same_content_same_hash(self, risk_reporting_engine):
        result1 = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        result2 = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        # Same content should yield same hash
        if result1["content"] == result2["content"]:
            assert result1["content_hash"] == result2["content_hash"]


class TestKPICalculation:
    """Tests for KPI calculation in reports."""

    @pytest.mark.unit
    def test_calculate_portfolio_kpis(self, risk_reporting_engine):
        suppliers = [
            {"supplier_id": "SUPP-A", "risk_level": RiskLevel.LOW, "risk_score": 20.0},
            {"supplier_id": "SUPP-B", "risk_level": RiskLevel.HIGH, "risk_score": 70.0},
            {"supplier_id": "SUPP-C", "risk_level": RiskLevel.MEDIUM, "risk_score": 45.0},
        ]
        kpis = risk_reporting_engine.calculate_portfolio_kpis(suppliers)
        assert "average_risk_score" in kpis
        assert "high_risk_percentage" in kpis
        assert "total_suppliers" in kpis


class TestMultiLanguage:
    """Tests for multi-language report generation."""

    @pytest.mark.unit
    @pytest.mark.parametrize("language", ["en", "fr", "de", "es", "pt"])
    def test_generate_report_all_languages(
        self, risk_reporting_engine, language
    ):
        """Test report generation in all 5 supported languages."""
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.HTML,
            language=language,
        )
        assert result["language"] == language


class TestRetention:
    """Tests for report retention management."""

    @pytest.mark.unit
    def test_report_includes_expiry_date(
        self, risk_reporting_engine, mock_config
    ):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        assert "expiry_date" in result
        # Should be retention_days from now
        expected_expiry = datetime.now(timezone.utc) + timedelta(
            days=mock_config.report_retention_days
        )
        assert result["expiry_date"].date() == expected_expiry.date()


class TestDDSPackageGeneration:
    """Tests for DDS package generation."""

    @pytest.mark.unit
    def test_generate_dds_package(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_dds_package(
            supplier_id="SUPP-001",
            shipment_id="SHIP-001",
        )
        assert "dds_reference" in result
        assert "geolocation_data" in result
        assert "compliance_declaration" in result


class TestAuditTrailDocumentation:
    """Tests for audit trail documentation."""

    @pytest.mark.unit
    def test_generate_audit_trail(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_audit_trail(
            supplier_id="SUPP-001",
            date_range={
                "start": datetime.now(timezone.utc) - timedelta(days=90),
                "end": datetime.now(timezone.utc),
            },
        )
        assert "activities" in result
        assert "assessments" in result
        assert "changes" in result


class TestProvenance:
    """Tests for provenance tracking."""

    @pytest.mark.unit
    def test_report_includes_provenance_hash(self, risk_reporting_engine):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-PROV",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.JSON,
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.unit
    def test_invalid_report_type_raises_error(self, risk_reporting_engine):
        with pytest.raises(ValueError):
            risk_reporting_engine.generate_report(
                supplier_id="SUPP-001",
                report_type="invalid_type",  # Invalid
                format=ReportFormat.JSON,
            )

    @pytest.mark.unit
    def test_missing_required_params_raises_error(self, risk_reporting_engine):
        with pytest.raises(ValueError):
            # Individual report needs supplier_id
            risk_reporting_engine.generate_report(
                supplier_id=None,
                report_type=ReportType.INDIVIDUAL,
                format=ReportFormat.JSON,
            )


class TestFileSizeLimit:
    """Tests for report file size limits."""

    @pytest.mark.unit
    def test_report_respects_size_limit(
        self, risk_reporting_engine, mock_config
    ):
        result = risk_reporting_engine.generate_report(
            supplier_id="SUPP-001",
            report_type=ReportType.INDIVIDUAL,
            format=ReportFormat.PDF,
        )
        if "file_size_mb" in result:
            assert result["file_size_mb"] <= mock_config.max_report_size_mb
