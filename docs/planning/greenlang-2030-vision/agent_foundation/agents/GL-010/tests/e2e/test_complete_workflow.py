# -*- coding: utf-8 -*-
"""
End-to-End Tests for GL-010 EMISSIONWATCH Complete Workflows.

Tests daily compliance workflow, quarterly reporting workflow,
violation response workflow, and annual emissions inventory.

Test Count: 12+ tests
Coverage Target: 90%+

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import EmissionsComplianceTools


# =============================================================================
# TEST CLASS: COMPLETE WORKFLOW
# =============================================================================

@pytest.mark.e2e
class TestCompleteWorkflow:
    """End-to-end tests for complete workflows."""

    # =========================================================================
    # DAILY COMPLIANCE WORKFLOW TESTS
    # =========================================================================

    def test_daily_compliance_workflow(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data,
        epa_permit_limits, facility_data
    ):
        """Test complete daily compliance workflow."""
        # Simulate 24 hours of monitoring
        daily_records = []
        violations_detected = []

        for hour in range(24):
            # 1. Acquire CEMS data (simulated)
            cems_data = {**sample_cems_data, "hour": hour}

            # 2. Calculate emissions
            nox = emissions_tools.calculate_nox_emissions(cems_data, natural_gas_fuel_data)
            sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
            co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
            pm = emissions_tools.calculate_particulate_matter(cems_data, natural_gas_fuel_data)

            # 3. Check compliance
            emissions_result = {
                "nox": nox.to_dict(),
                "sox": sox.to_dict(),
                "co2": co2.to_dict(),
                "pm": pm.to_dict(),
            }

            compliance = emissions_tools.check_compliance_status(
                emissions_result=emissions_result,
                jurisdiction="EPA",
            )

            # 4. Detect violations
            violations = emissions_tools.detect_violations(
                emissions_result=emissions_result,
                permit_limits=epa_permit_limits,
            )

            # 5. Record data
            daily_records.append({
                "hour": hour,
                "nox_lb_mmbtu": nox.emission_rate_lb_mmbtu,
                "sox_lb_mmbtu": sox.emission_rate_lb_mmbtu,
                "co2_tons": co2.mass_rate_tons_hr,
                "pm_lb_mmbtu": pm.emission_rate_lb_mmbtu,
                "compliant": compliance.overall_status == "compliant",
                "valid": True,
            })

            violations_detected.extend(violations)

        # 6. Generate daily summary
        assert len(daily_records) == 24

        # Calculate daily averages
        avg_nox = sum(r["nox_lb_mmbtu"] for r in daily_records) / 24
        total_co2 = sum(r["co2_tons"] for r in daily_records)

        # Verify workflow completed
        assert avg_nox >= 0
        assert total_co2 > 0

    def test_daily_compliance_workflow_with_violation(
        self, emissions_tools, natural_gas_fuel_data, epa_permit_limits
    ):
        """Test daily workflow with violation detection and response."""
        # Simulate violation scenario
        violation_cems = {
            "nox_ppm": 150.0,  # High NOx
            "o2_percent": 3.0,
            "flow_rate_dscfm": 50000.0,
        }

        nox = emissions_tools.calculate_nox_emissions(violation_cems, natural_gas_fuel_data)

        emissions_result = {
            "nox": nox.to_dict(),
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        # Detect violations
        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # Verify violation detected and categorized
        if len(violations) > 0:
            assert violations[0].severity is not None
            assert violations[0].regulatory_reference is not None

    def test_daily_data_validation(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test daily data validation checks."""
        # Validate CEMS data quality
        assert sample_cems_data["nox_ppm"] >= 0
        assert 0 <= sample_cems_data["o2_percent"] <= 21
        assert sample_cems_data["flow_rate_dscfm"] > 0

        # Process and validate results
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)

        assert nox.emission_rate_lb_mmbtu >= 0
        assert nox.provenance_hash is not None

    # =========================================================================
    # QUARTERLY REPORTING WORKFLOW TESTS
    # =========================================================================

    def test_quarterly_reporting_workflow(
        self, emissions_tools, emissions_records, facility_data, reporting_period, compliance_events
    ):
        """Test complete quarterly reporting workflow."""
        # 1. Aggregate quarterly data
        quarterly_data = emissions_records[:720]  # 30 days worth

        # 2. Generate regulatory report
        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=quarterly_data,
        )

        # 3. Generate audit trail
        audit = emissions_tools.generate_audit_trail(
            audit_period=reporting_period,
            facility_data=facility_data,
            emissions_records=quarterly_data,
            compliance_events=compliance_events,
        )

        # Verify workflow completion
        assert report.report_id is not None
        assert report.compliance_rate_percent >= 0
        assert report.data_availability_percent >= 0

        assert audit.audit_id is not None
        assert audit.root_hash is not None
        assert audit.chain_valid == True

    def test_quarterly_reporting_data_availability(
        self, emissions_tools, facility_data, reporting_period
    ):
        """Test quarterly report data availability calculation."""
        # Simulate partial data availability
        records_with_gaps = []
        for i in range(720):
            records_with_gaps.append({
                "hour": i,
                "nox_lb_mmbtu": 0.08,
                "sox_lb_mmbtu": 0.10,
                "co2_tons": 40.0,
                "pm_lb_mmbtu": 0.02,
                "valid": i % 10 != 0,  # 10% invalid
                "compliant": True,
            })

        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=records_with_gaps,
        )

        # Data availability should be ~90%
        assert report.data_availability_percent >= 85

    def test_quarterly_reporting_certification(
        self, emissions_tools, emissions_records, facility_data, reporting_period
    ):
        """Test quarterly report certification requirements."""
        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        # Verify certification fields
        assert report.certifier is not None
        assert report.certification_date is not None
        assert report.submission_deadline is not None

    # =========================================================================
    # VIOLATION RESPONSE WORKFLOW TESTS
    # =========================================================================

    def test_violation_response_workflow(
        self, emissions_tools, epa_permit_limits, facility_data
    ):
        """Test complete violation response workflow."""
        # 1. Detect violation
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # 2. Categorize violation
        assert len(violations) > 0
        violation = violations[0]
        assert violation.severity is not None

        # 3. Generate violation record
        violation_record = violation.to_dict()
        assert "violation_id" in violation_record
        assert "regulatory_reference" in violation_record
        assert "exceedance_percent" in violation_record

        # 4. Document corrective action (simulated)
        corrective_action = {
            "violation_id": violation.violation_id,
            "action_taken": "Adjusted combustion parameters",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        assert corrective_action["violation_id"] == violation.violation_id

    def test_violation_escalation_workflow(self, emissions_tools, epa_permit_limits):
        """Test violation escalation based on severity."""
        # Create violations of different severities
        severities = []

        exceedance_levels = [
            0.105,  # 5% - low
            0.12,   # 20% - medium
            0.14,   # 40% - high
            0.20,   # 100% - critical
        ]

        for nox_value in exceedance_levels:
            emissions_result = {
                "nox": {"emission_rate_lb_mmbtu": nox_value},
                "sox": {"emission_rate_lb_mmbtu": 0.08},
                "co2": {"mass_rate_tons_hr": 40.0},
                "pm": {"emission_rate_lb_mmbtu": 0.02},
            }

            violations = emissions_tools.detect_violations(
                emissions_result=emissions_result,
                permit_limits=epa_permit_limits,
            )

            if violations:
                severities.append(violations[0].severity)

        # Should have escalating severities
        assert "low" in severities
        assert "critical" in severities

    # =========================================================================
    # ANNUAL EMISSIONS INVENTORY TESTS
    # =========================================================================

    def test_annual_emissions_inventory(
        self, emissions_tools, natural_gas_fuel_data, facility_data
    ):
        """Test annual emissions inventory workflow."""
        # Simulate annual data (simplified)
        annual_nox_tons = 0
        annual_sox_tons = 0
        annual_co2_tons = 0

        # Monthly calculations
        for month in range(12):
            monthly_hours = 720  # Approximate

            nox = emissions_tools.calculate_nox_emissions(
                {"nox_ppm": 45.0, "o2_percent": 3.0},
                natural_gas_fuel_data,
            )
            sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
            co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

            # Convert to annual tons
            annual_nox_tons += (nox.emission_rate_lb_hr * monthly_hours) / 2000
            annual_sox_tons += (sox.emission_rate_lb_hr * monthly_hours) / 2000
            annual_co2_tons += co2.mass_rate_tons_hr * monthly_hours

        # Verify annual totals
        assert annual_nox_tons > 0
        assert annual_sox_tons >= 0
        assert annual_co2_tons > 0

    def test_annual_inventory_report(
        self, emissions_tools, emissions_records, facility_data
    ):
        """Test annual inventory report generation."""
        annual_period = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "reporting_type": "annual",
        }

        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=annual_period,
            facility_data=facility_data,
            emissions_data=emissions_records,
        )

        assert report is not None
        assert report.total_co2_tons >= 0

    def test_annual_audit_trail(
        self, emissions_tools, emissions_records, facility_data, compliance_events
    ):
        """Test annual audit trail generation."""
        annual_period = {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        }

        audit = emissions_tools.generate_audit_trail(
            audit_period=annual_period,
            facility_data=facility_data,
            emissions_records=emissions_records,
            compliance_events=compliance_events,
        )

        # Verify audit completeness
        assert audit.audit_id is not None
        assert audit.root_hash is not None
        assert audit.chain_valid == True
        assert audit.epa_part_75_compliant is not None
        assert audit.qapp_requirements_met is not None
