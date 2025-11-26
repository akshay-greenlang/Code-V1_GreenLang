# -*- coding: utf-8 -*-
"""
Integration Tests for GL-010 EMISSIONWATCH Regulatory Reporting.

Tests EPA CEDRI report generation, EU ETS report generation,
report submission, and report validation.

Test Count: 15+ tests
Coverage Target: 90%+

Standards: EPA 40 CFR Part 75 ECMPS, EU E-PRTR

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, AsyncMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import EmissionsComplianceTools, RegulatoryReportResult


# =============================================================================
# TEST CLASS: REGULATORY REPORTING
# =============================================================================

@pytest.mark.integration
class TestRegulatoryReporting:
    """Integration tests for regulatory reporting."""

    # =========================================================================
    # EPA CEDRI/ECMPS REPORT TESTS
    # =========================================================================

    def test_epa_ecmps_report_generation(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EPA ECMPS report generation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert isinstance(result, RegulatoryReportResult)
        assert result.format_version == "ECMPS 3.0"
        assert result.jurisdiction == "EPA"

    def test_epa_ecmps_report_sections(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EPA ECMPS report has required sections."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert len(result.sections) > 0

        section_names = [s["section"] for s in result.sections]
        assert "Facility Information" in section_names
        assert "Emissions Summary" in section_names

    def test_epa_ecmps_report_certification(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EPA ECMPS report certification fields."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert result.certifier is not None
        assert result.certification_date is not None

    def test_epa_ecmps_submission_deadline(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EPA ECMPS submission deadline calculation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # Quarterly reports due Q+30 days
        assert result.submission_deadline == "Q+30 days"

    def test_epa_ecmps_data_availability(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EPA ECMPS data availability calculation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # Data availability should be calculated
        assert result.data_availability_percent >= 0

    # =========================================================================
    # EU ETS/E-PRTR REPORT TESTS
    # =========================================================================

    def test_eu_eled_report_generation(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EU E-PRTR report generation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EU_ELED",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert isinstance(result, RegulatoryReportResult)
        assert result.format_version == "E-PRTR 2023"
        assert result.jurisdiction == "EU"

    def test_eu_eled_submission_deadline(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EU ELED submission deadline."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EU_ELED",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # EU annual report due April 30
        assert result.submission_deadline == "April 30"

    def test_eu_eled_emissions_summary(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test EU ELED emissions summary."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EU_ELED",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert result.avg_nox_lb_mmbtu >= 0
        assert result.avg_sox_lb_mmbtu >= 0
        assert result.total_co2_tons >= 0

    # =========================================================================
    # CHINA MEE REPORT TESTS
    # =========================================================================

    def test_china_mee_report_generation(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test China MEE report generation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="CHINA_MEE",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert isinstance(result, RegulatoryReportResult)
        assert result.format_version == "GB 13223-2011"

    # =========================================================================
    # REPORT SUBMISSION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_report_submission_mock(self, mock_ecmps_api, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test report submission via mock API."""
        # Generate report
        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # Submit report
        submission_result = await mock_ecmps_api.submit_report(report.to_dict())

        assert submission_result["status"] == "accepted"
        assert submission_result["submission_id"] is not None

    @pytest.mark.asyncio
    async def test_report_submission_status(self, mock_ecmps_api):
        """Test report submission status checking."""
        status = await mock_ecmps_api.get_submission_status("SUB-12345")

        assert status["status"] == "processed"
        assert status["errors"] == []

    # =========================================================================
    # REPORT VALIDATION TESTS
    # =========================================================================

    def test_report_validation_complete(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test report validation for completeness."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # Verify all required fields present
        assert result.report_id is not None
        assert result.reporting_period is not None
        assert result.total_operating_hours >= 0
        assert result.compliance_rate_percent >= 0
        assert result.provenance_hash is not None

    def test_report_validation_attachments(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test report includes required attachments."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert len(result.attachments) > 0
        assert "hourly_emissions_data.csv" in result.attachments

    def test_report_validation_compliance_rate(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test report compliance rate calculation."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records,
        )

        # Compliance rate should account for exceedances
        assert 0 <= result.compliance_rate_percent <= 100
        assert result.exceedance_count >= 0

    def test_report_to_dict(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test report to_dict serialization."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "report_id" in result_dict
        assert "jurisdiction" in result_dict
        assert "sections" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# TEST CLASS: REPORT FORMATS
# =============================================================================

@pytest.mark.integration
class TestReportFormats:
    """Integration tests for different report formats."""

    @pytest.mark.parametrize("report_format,expected_version", [
        ("EPA_ECMPS", "ECMPS 3.0"),
        ("EU_ELED", "E-PRTR 2023"),
        ("CHINA_MEE", "GB 13223-2011"),
    ])
    def test_report_format_versions(
        self, emissions_tools, facility_data, reporting_period, emissions_records,
        report_format, expected_version
    ):
        """Test report format versions are correct."""
        result = emissions_tools.generate_regulatory_report(
            report_format=report_format,
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert result.format_version == expected_version
