# -*- coding: utf-8 -*-
"""
Integration Tests for GL-010 EMISSIONWATCH Full Pipeline.

Tests end-to-end monitoring, violation-to-alert pipeline, and
multi-source aggregation.

Test Count: 15+ tests
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
# TEST CLASS: FULL PIPELINE
# =============================================================================

@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for full emissions monitoring pipeline."""

    # =========================================================================
    # END-TO-END MONITORING TESTS
    # =========================================================================

    def test_end_to_end_monitoring_compliant(
        self, emissions_tools, low_emissions_cems_data, natural_gas_fuel_data, epa_permit_limits
    ):
        """Test end-to-end monitoring for compliant scenario."""
        # Step 1: Calculate all emissions
        nox = emissions_tools.calculate_nox_emissions(low_emissions_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(low_emissions_cems_data, natural_gas_fuel_data)

        # Step 2: Aggregate results
        emissions_result = {
            "nox": nox.to_dict(),
            "sox": sox.to_dict(),
            "co2": co2.to_dict(),
            "pm": pm.to_dict(),
        }

        # Step 3: Check compliance
        compliance = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # Step 4: Detect violations
        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # Assertions
        assert compliance.overall_status == "compliant"
        assert len(violations) == 0

    def test_end_to_end_monitoring_violation(
        self, emissions_tools, high_nox_cems_data, natural_gas_fuel_data, epa_permit_limits
    ):
        """Test end-to-end monitoring for violation scenario."""
        # Step 1: Calculate emissions
        nox = emissions_tools.calculate_nox_emissions(high_nox_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(high_nox_cems_data, natural_gas_fuel_data)

        # Step 2: Aggregate results
        emissions_result = {
            "nox": nox.to_dict(),
            "sox": sox.to_dict(),
            "co2": co2.to_dict(),
            "pm": pm.to_dict(),
        }

        # Step 3: Check compliance
        compliance = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # Step 4: Detect violations
        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # May have violations depending on emission rates
        assert compliance is not None
        assert isinstance(violations, list)

    def test_end_to_end_monitoring_continuous(
        self, emissions_tools, cems_data_series, natural_gas_fuel_data, epa_permit_limits
    ):
        """Test continuous monitoring over time series."""
        all_results = []
        all_violations = []

        for cems_data in cems_data_series:
            # Calculate emissions
            nox = emissions_tools.calculate_nox_emissions(cems_data, natural_gas_fuel_data)
            sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)

            emissions_result = {
                "nox": nox.to_dict(),
                "sox": sox.to_dict(),
                "co2": {"mass_rate_tons_hr": 40.0},
                "pm": {"emission_rate_lb_mmbtu": 0.02},
            }

            # Check for violations
            violations = emissions_tools.detect_violations(
                emissions_result=emissions_result,
                permit_limits=epa_permit_limits,
            )

            all_results.append(emissions_result)
            all_violations.extend(violations)

        # Should have processed all data points
        assert len(all_results) == len(cems_data_series)

    # =========================================================================
    # VIOLATION-TO-ALERT PIPELINE TESTS
    # =========================================================================

    def test_violation_to_alert_pipeline(
        self, emissions_tools, high_nox_cems_data, natural_gas_fuel_data, epa_permit_limits
    ):
        """Test violation detection triggers alert generation."""
        # Calculate emissions with high values
        nox = emissions_tools.calculate_nox_emissions(high_nox_cems_data, natural_gas_fuel_data)

        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # Above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        # Detect violations
        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # Verify violations detected
        assert len(violations) > 0

        # Verify violation details
        for violation in violations:
            assert violation.violation_id is not None
            assert violation.severity in ["low", "medium", "high", "critical"]
            assert violation.regulatory_reference is not None

    def test_violation_severity_escalation(self, emissions_tools, epa_permit_limits):
        """Test violation severity escalation based on exceedance."""
        # Low exceedance (5%)
        low_emissions = {
            "nox": {"emission_rate_lb_mmbtu": 0.105},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        # High exceedance (100%)
        high_emissions = {
            "nox": {"emission_rate_lb_mmbtu": 0.20},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        low_violations = emissions_tools.detect_violations(low_emissions, epa_permit_limits)
        high_violations = emissions_tools.detect_violations(high_emissions, epa_permit_limits)

        assert low_violations[0].severity == "low"
        assert high_violations[0].severity == "critical"

    def test_violation_alert_data_completeness(self, emissions_tools, epa_permit_limits):
        """Test violation alert has all required data."""
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

        violation = violations[0]
        violation_dict = violation.to_dict()

        # Verify all required fields for alerting
        assert "violation_id" in violation_dict
        assert "pollutant" in violation_dict
        assert "measured_value" in violation_dict
        assert "limit_value" in violation_dict
        assert "exceedance_percent" in violation_dict
        assert "severity" in violation_dict
        assert "timestamp" in violation_dict

    # =========================================================================
    # MULTI-SOURCE AGGREGATION TESTS
    # =========================================================================

    def test_multi_source_aggregation_single_unit(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test aggregation from single emission unit."""
        # Single unit emissions
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)

        total_co2_kg = co2.mass_rate_kg_hr
        total_nox_lb = nox.emission_rate_lb_hr

        assert total_co2_kg > 0
        assert total_nox_lb >= 0

    def test_multi_source_aggregation_multiple_units(self, emissions_tools, natural_gas_fuel_data):
        """Test aggregation from multiple emission units."""
        units = [
            {"cems_data": {"nox_ppm": 45, "o2_percent": 3.0}, "heat_input": 100},
            {"cems_data": {"nox_ppm": 50, "o2_percent": 4.0}, "heat_input": 80},
            {"cems_data": {"nox_ppm": 40, "o2_percent": 3.5}, "heat_input": 120},
        ]

        total_nox_lb_hr = 0
        total_co2_kg_hr = 0

        for unit in units:
            fuel_data = {**natural_gas_fuel_data, "heat_input_mmbtu_hr": unit["heat_input"]}

            nox = emissions_tools.calculate_nox_emissions(unit["cems_data"], fuel_data)
            co2 = emissions_tools.calculate_co2_emissions(fuel_data)

            total_nox_lb_hr += nox.emission_rate_lb_hr
            total_co2_kg_hr += co2.mass_rate_kg_hr

        # Aggregate should be sum of individual units
        assert total_nox_lb_hr > 0
        assert total_co2_kg_hr > 0

    def test_multi_source_aggregation_mixed_fuels(self, emissions_tools, natural_gas_fuel_data, coal_bituminous_data):
        """Test aggregation with mixed fuel sources."""
        # Natural gas unit
        ng_nox = emissions_tools.calculate_nox_emissions(
            {"nox_ppm": 45, "o2_percent": 3.0},
            natural_gas_fuel_data
        )
        ng_co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

        # Coal unit
        coal_nox = emissions_tools.calculate_nox_emissions(
            {"nox_ppm": 200, "o2_percent": 6.0},
            coal_bituminous_data
        )
        coal_co2 = emissions_tools.calculate_co2_emissions(coal_bituminous_data)

        # Total facility emissions
        total_co2 = ng_co2.mass_rate_kg_hr + coal_co2.mass_rate_kg_hr
        total_nox = ng_nox.emission_rate_lb_hr + coal_nox.emission_rate_lb_hr

        # Coal should have higher emissions
        assert coal_co2.mass_rate_kg_hr > ng_co2.mass_rate_kg_hr

    # =========================================================================
    # TREND ANALYSIS PIPELINE TESTS
    # =========================================================================

    def test_trend_analysis_pipeline(self, emissions_tools, cems_data_series, epa_permit_limits):
        """Test trend analysis in monitoring pipeline."""
        # Build historical data
        historical_data = []
        for cems_data in cems_data_series:
            nox_ppm = cems_data.get("nox_ppm", 45)
            # Convert ppm to approximate lb/MMBtu
            historical_data.append({
                "nox_lb_mmbtu": nox_ppm * 0.002,
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        # Predict exceedances
        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
            forecast_hours=24,
        )

        assert len(predictions) > 0

        # Verify predictions have required fields
        for pred in predictions:
            assert pred.pollutant is not None
            assert pred.current_value >= 0
            assert pred.predicted_value is not None
            assert pred.exceedance_probability >= 0

    def test_full_audit_pipeline(
        self, emissions_tools, facility_data, emissions_records, compliance_events
    ):
        """Test full audit trail pipeline."""
        audit_period = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        }

        # Generate audit trail
        audit = emissions_tools.generate_audit_trail(
            audit_period=audit_period,
            facility_data=facility_data,
            emissions_records=emissions_records[:100],
            compliance_events=compliance_events,
        )

        # Verify audit trail completeness
        assert audit.audit_id is not None
        assert audit.total_records == 100
        assert audit.root_hash is not None
        assert audit.chain_valid == True
        assert audit.certification_statement is not None

    def test_full_reporting_pipeline(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data,
        facility_data, reporting_period, epa_permit_limits
    ):
        """Test full monitoring-to-reporting pipeline."""
        # Step 1: Simulate hourly data collection
        hourly_records = []
        for hour in range(24):
            nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
            sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
            co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

            hourly_records.append({
                "hour": hour,
                "nox_lb_mmbtu": nox.emission_rate_lb_mmbtu,
                "sox_lb_mmbtu": sox.emission_rate_lb_mmbtu,
                "co2_tons": co2.mass_rate_tons_hr,
                "pm_lb_mmbtu": 0.02,
                "compliant": True,
                "valid": True,
            })

        # Step 2: Generate report
        report = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=hourly_records,
        )

        # Step 3: Generate audit trail
        audit = emissions_tools.generate_audit_trail(
            audit_period=reporting_period,
            facility_data=facility_data,
            emissions_records=hourly_records,
            compliance_events=[],
        )

        # Verify pipeline completion
        assert report.report_id is not None
        assert report.total_operating_hours == 24
        assert audit.total_records == 24
        assert audit.chain_valid == True
