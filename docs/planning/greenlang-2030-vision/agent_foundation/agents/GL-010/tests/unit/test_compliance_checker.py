# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH Compliance Checker.

Tests compliance status checking across EPA NSPS, EU IED, and China MEE
jurisdictions, including multi-jurisdiction handling, averaging periods,
and margin calculations.

Test Count: 25+ tests
Coverage Target: 90%+

Standards: 40 CFR Part 60, EU IED 2010/75/EU, China GB 13223-2011

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    ComplianceCheckResult,
    REGULATORY_LIMITS,
)


# =============================================================================
# TEST CLASS: COMPLIANCE CHECKER
# =============================================================================

@pytest.mark.unit
class TestComplianceChecker:
    """Test suite for compliance status checking."""

    # =========================================================================
    # BASIC COMPLIANCE CHECK TESTS
    # =========================================================================

    def test_check_compliance_status_basic(self, emissions_tools, epa_permit_limits):
        """Test basic compliance status check."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert isinstance(result, ComplianceCheckResult)
        assert result.overall_status in ["compliant", "non_compliant", "warning"]
        assert result.jurisdiction == "EPA"
        assert result.provenance_hash is not None

    def test_check_compliance_status_compliant(self, emissions_tools):
        """Test fully compliant emissions scenario."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},  # Below 0.10 limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},  # Below 0.15 limit
            "co2": {"mass_rate_tons_hr": 30.0},  # Below limit
            "pm": {"emission_rate_lb_mmbtu": 0.01},  # Below 0.03 limit
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.overall_status == "compliant"
        assert result.nox_status == "compliant"
        assert result.sox_status == "compliant"
        assert result.pm_status == "compliant"
        assert len(result.violations) == 0

    def test_check_compliance_status_non_compliant(self, emissions_tools):
        """Test non-compliant emissions scenario."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # Above 0.10 limit
            "sox": {"emission_rate_lb_mmbtu": 0.20},  # Above 0.15 limit
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.05},  # Above 0.03 limit
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.overall_status == "non_compliant"
        assert len(result.violations) > 0

    # =========================================================================
    # EPA NSPS COMPLIANCE TESTS
    # =========================================================================

    def test_epa_nsps_compliance_nox(self, emissions_tools):
        """Test EPA NSPS NOx compliance check."""
        # EPA NSPS Subpart Da: 0.10 lb/MMBtu
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.09},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.nox_status == "compliant"

    def test_epa_nsps_compliance_sox(self, emissions_tools):
        """Test EPA NSPS SOx compliance check."""
        # EPA NSPS Subpart Da: 0.15 lb/MMBtu
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.16},  # Exceeds limit
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.sox_status == "non_compliant"

    def test_epa_nsps_compliance_pm(self, emissions_tools):
        """Test EPA NSPS PM compliance check."""
        # EPA NSPS Subpart Da: 0.03 lb/MMBtu
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.04},  # Exceeds limit
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.pm_status == "non_compliant"

    # =========================================================================
    # EU IED COMPLIANCE TESTS
    # =========================================================================

    def test_eu_ied_compliance_check(self, emissions_tools):
        """Test EU IED compliance check."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.005},  # Low emissions
            "sox": {"emission_rate_lb_mmbtu": 0.008},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.0003},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EU_IED",
        )

        assert result.jurisdiction == "EU_IED"
        assert result.overall_status == "compliant"

    def test_eu_ied_bat_ael_limits(self, emissions_tools):
        """Test EU IED BAT-AEL limits are applied."""
        # EU IED uses mg/Nm3 limits
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.001},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EU_IED",
        )

        assert "nox_limit" in result.applicable_limits
        assert "sox_limit" in result.applicable_limits

    # =========================================================================
    # CHINA MEE COMPLIANCE TESTS
    # =========================================================================

    def test_china_mee_compliance_check(self, emissions_tools):
        """Test China MEE compliance check."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.003},  # Very low for ultra-low
            "sox": {"emission_rate_lb_mmbtu": 0.002},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.0005},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="CHINA_MEE",
        )

        assert result.jurisdiction == "CHINA_MEE"

    def test_china_ultra_low_emission_standards(self, emissions_tools):
        """Test China ultra-low emission standards."""
        # China GB 13223-2011 ultra-low: NOx 50 mg/Nm3, SOx 35 mg/Nm3
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.01},
            "sox": {"emission_rate_lb_mmbtu": 0.008},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.0002},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="CHINA_MEE",
        )

        assert result.applicable_limits is not None

    # =========================================================================
    # MULTI-JURISDICTION CHECK TESTS
    # =========================================================================

    def test_multi_jurisdiction_comparison(self, emissions_tools):
        """Test same emissions against multiple jurisdictions."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.09},
            "sox": {"emission_rate_lb_mmbtu": 0.12},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result_epa = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        result_eu = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EU_IED",
        )

        result_china = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="CHINA_MEE",
        )

        # Results may differ by jurisdiction
        assert result_epa.jurisdiction == "EPA"
        assert result_eu.jurisdiction == "EU_IED"
        assert result_china.jurisdiction == "CHINA_MEE"

    def test_jurisdiction_specific_limits_applied(self, emissions_tools):
        """Test jurisdiction-specific limits are correctly applied."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.01},
        }

        result_epa = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # EPA limit: 0.10 lb/MMBtu
        assert result_epa.applicable_limits["nox_limit"] == 0.10

    # =========================================================================
    # AVERAGING PERIOD HANDLING TESTS
    # =========================================================================

    def test_averaging_period_handling_hourly(self, emissions_tools):
        """Test hourly averaging period compliance check."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.09},
            "sox": {"emission_rate_lb_mmbtu": 0.12},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
            process_parameters={"averaging_period": "1-hour"},
        )

        assert result is not None

    def test_averaging_period_handling_30_day(self, emissions_tools):
        """Test 30-day rolling average compliance check."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.09},
            "sox": {"emission_rate_lb_mmbtu": 0.12},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
            process_parameters={"averaging_period": "30-day"},
        )

        assert result is not None

    # =========================================================================
    # MARGIN CALCULATION TESTS
    # =========================================================================

    def test_margin_calculation_positive(self, emissions_tools):
        """Test positive margin (below limit) calculation."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},  # 50% of 0.10 limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # 50% margin to limit
        assert result.margin_to_limits["nox_margin_percent"] == 50.0

    def test_margin_calculation_negative(self, emissions_tools):
        """Test negative margin (above limit) calculation."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # 150% of 0.10 limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # Negative margin (exceedance)
        assert result.margin_to_limits["nox_margin_percent"] < 0

    def test_margin_calculation_at_limit(self, emissions_tools):
        """Test margin at exactly the limit."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.10},  # Exactly at limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        # Zero margin
        assert result.margin_to_limits["nox_margin_percent"] == 0.0

    # =========================================================================
    # VIOLATION RECORDING TESTS
    # =========================================================================

    def test_violation_details_recorded(self, emissions_tools):
        """Test violation details are properly recorded."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # 50% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert len(result.violations) > 0
        violation = result.violations[0]
        assert "pollutant" in violation
        assert "measured" in violation
        assert "limit" in violation
        assert "exceedance_percent" in violation

    def test_multiple_violations_recorded(self, emissions_tools):
        """Test multiple violations are all recorded."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # Violation
            "sox": {"emission_rate_lb_mmbtu": 0.20},  # Violation
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.05},  # Violation
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert len(result.violations) >= 3

    # =========================================================================
    # DETERMINISM TESTS
    # =========================================================================

    def test_determinism_compliance_check(self, emissions_tools):
        """Test deterministic compliance checking."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        results = []
        for _ in range(10):
            result = emissions_tools.check_compliance_status(
                emissions_result=emissions_result,
                jurisdiction="EPA",
            )
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result.overall_status == first.overall_status
            assert result.nox_status == first.nox_status
            assert result.sox_status == first.sox_status

    def test_determinism_provenance_hash_compliance(self, emissions_tools):
        """Test compliance check provenance hash is deterministic."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        hashes = []
        for _ in range(5):
            result = emissions_tools.check_compliance_status(
                emissions_result=emissions_result,
                jurisdiction="EPA",
            )
            hashes.append(result.provenance_hash)

        assert len(set(hashes)) == 1

    # =========================================================================
    # TO_DICT CONVERSION TEST
    # =========================================================================

    def test_compliance_result_to_dict(self, emissions_tools):
        """Test ComplianceCheckResult to_dict conversion."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "overall_status" in result_dict
        assert "nox_status" in result_dict
        assert "sox_status" in result_dict
        assert "violations" in result_dict
        assert "jurisdiction" in result_dict
        assert "applicable_limits" in result_dict
        assert "margin_to_limits" in result_dict
        assert "provenance_hash" in result_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestComplianceCheckerParametrized:
    """Parametrized tests for compliance checker."""

    @pytest.mark.parametrize("nox_value,expected_status", [
        (0.05, "compliant"),
        (0.09, "compliant"),
        (0.10, "compliant"),  # At limit
        (0.11, "non_compliant"),
        (0.15, "non_compliant"),
    ])
    def test_nox_compliance_thresholds(
        self, emissions_tools, nox_value, expected_status
    ):
        """Test NOx compliance at various thresholds."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": nox_value},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction="EPA",
        )

        assert result.nox_status == expected_status

    @pytest.mark.parametrize("jurisdiction", ["EPA", "EU_IED", "CHINA_MEE"])
    def test_all_jurisdictions_supported(self, emissions_tools, jurisdiction):
        """Test all jurisdictions are supported."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.01},
        }

        result = emissions_tools.check_compliance_status(
            emissions_result=emissions_result,
            jurisdiction=jurisdiction,
        )

        assert result.jurisdiction == jurisdiction
