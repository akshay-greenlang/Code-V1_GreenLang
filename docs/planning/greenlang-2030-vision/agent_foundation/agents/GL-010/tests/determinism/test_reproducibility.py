# -*- coding: utf-8 -*-
"""
Determinism Tests for GL-010 EMISSIONWATCH Reproducibility.

Tests bit-perfect reproducibility of NOx, SOx, CO2 calculations,
compliance checks, report generation, cross-platform determinism,
and provenance hash consistency.

Test Count: 15+ tests
Coverage Target: 90%+

Critical: Zero-hallucination guarantee verification

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import hashlib
import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import EmissionsComplianceTools


# =============================================================================
# TEST CLASS: REPRODUCIBILITY
# =============================================================================

@pytest.mark.determinism
class TestReproducibility:
    """Test suite for calculation reproducibility and determinism."""

    # =========================================================================
    # NOX CALCULATION REPRODUCIBILITY TESTS
    # =========================================================================

    def test_nox_calculation_reproducibility(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test NOx calculation is bit-perfect reproducible."""
        results = []

        for _ in range(100):
            result = emissions_tools.calculate_nox_emissions(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        # All results must be identical
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.concentration_ppm == first.concentration_ppm, \
                f"NOx concentration differs at iteration {i}"
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu, \
                f"NOx emission rate differs at iteration {i}"
            assert result.emission_rate_lb_hr == first.emission_rate_lb_hr, \
                f"NOx mass rate differs at iteration {i}"
            assert result.thermal_nox_percent == first.thermal_nox_percent, \
                f"Thermal NOx percent differs at iteration {i}"
            assert result.fuel_nox_percent == first.fuel_nox_percent, \
                f"Fuel NOx percent differs at iteration {i}"
            assert result.prompt_nox_percent == first.prompt_nox_percent, \
                f"Prompt NOx percent differs at iteration {i}"
            assert result.correction_factor == first.correction_factor, \
                f"Correction factor differs at iteration {i}"

    def test_nox_calculation_different_inputs_different_outputs(
        self, emissions_tools, natural_gas_fuel_data
    ):
        """Test different inputs produce different outputs."""
        cems_low = {"nox_ppm": 25.0, "o2_percent": 3.0}
        cems_high = {"nox_ppm": 75.0, "o2_percent": 3.0}

        result_low = emissions_tools.calculate_nox_emissions(cems_low, natural_gas_fuel_data)
        result_high = emissions_tools.calculate_nox_emissions(cems_high, natural_gas_fuel_data)

        assert result_low.emission_rate_lb_mmbtu != result_high.emission_rate_lb_mmbtu

    # =========================================================================
    # SOX CALCULATION REPRODUCIBILITY TESTS
    # =========================================================================

    def test_sox_calculation_reproducibility(
        self, emissions_tools, fuel_oil_no2_data
    ):
        """Test SOx calculation is bit-perfect reproducible."""
        results = []

        for _ in range(100):
            result = emissions_tools.calculate_sox_emissions(
                fuel_data=fuel_oil_no2_data,
            )
            results.append(result)

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.concentration_ppm == first.concentration_ppm, \
                f"SOx concentration differs at iteration {i}"
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu, \
                f"SOx emission rate differs at iteration {i}"
            assert result.emission_rate_lb_hr == first.emission_rate_lb_hr, \
                f"SOx mass rate differs at iteration {i}"
            assert result.so2_so3_ratio == first.so2_so3_ratio, \
                f"SO2/SO3 ratio differs at iteration {i}"

    def test_sox_calculation_sulfur_sensitivity(self, emissions_tools):
        """Test SOx calculation responds to sulfur content changes."""
        fuel_low_s = {"fuel_type": "fuel_oil_no2", "heat_input_mmbtu_hr": 100.0,
                     "heating_value_btu_lb": 19500.0, "sulfur_percent": 0.25}
        fuel_high_s = {"fuel_type": "fuel_oil_no2", "heat_input_mmbtu_hr": 100.0,
                      "heating_value_btu_lb": 19500.0, "sulfur_percent": 0.50}

        result_low = emissions_tools.calculate_sox_emissions(fuel_low_s)
        result_high = emissions_tools.calculate_sox_emissions(fuel_high_s)

        # Double sulfur should approximately double emissions
        assert abs(result_high.emission_rate_lb_hr / result_low.emission_rate_lb_hr - 2.0) < 0.1

    # =========================================================================
    # CO2 CALCULATION REPRODUCIBILITY TESTS
    # =========================================================================

    def test_co2_calculation_reproducibility(
        self, emissions_tools, natural_gas_fuel_data
    ):
        """Test CO2 calculation is bit-perfect reproducible."""
        results = []

        for _ in range(100):
            result = emissions_tools.calculate_co2_emissions(
                fuel_data=natural_gas_fuel_data,
            )
            results.append(result)

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.concentration_percent == first.concentration_percent, \
                f"CO2 concentration differs at iteration {i}"
            assert result.emission_rate_lb_mmbtu == first.emission_rate_lb_mmbtu, \
                f"CO2 emission rate differs at iteration {i}"
            assert result.mass_rate_tons_hr == first.mass_rate_tons_hr, \
                f"CO2 mass rate differs at iteration {i}"
            assert result.mass_rate_kg_hr == first.mass_rate_kg_hr, \
                f"CO2 kg rate differs at iteration {i}"

    def test_co2_calculation_fuel_factor_accuracy(self, emissions_tools):
        """Test CO2 calculation uses correct fuel factors."""
        fuels = ["natural_gas", "fuel_oil_no2", "coal_bituminous"]
        expected_factors = {
            "natural_gas": 117.0,
            "fuel_oil_no2": 161.0,
            "coal_bituminous": 205.0,
        }

        for fuel_type in fuels:
            fuel_data = {"fuel_type": fuel_type, "heat_input_mmbtu_hr": 100.0}
            result = emissions_tools.calculate_co2_emissions(fuel_data)

            expected = expected_factors[fuel_type] * 0.99  # 99% efficiency
            assert abs(result.emission_rate_lb_mmbtu - expected) < 5.0, \
                f"CO2 factor incorrect for {fuel_type}"

    # =========================================================================
    # COMPLIANCE CHECK REPRODUCIBILITY TESTS
    # =========================================================================

    def test_compliance_check_reproducibility(self, emissions_tools):
        """Test compliance check is deterministic."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        results = []
        for _ in range(50):
            result = emissions_tools.check_compliance_status(
                emissions_result=emissions_result,
                jurisdiction="EPA",
            )
            results.append(result)

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.overall_status == first.overall_status, \
                f"Overall status differs at iteration {i}"
            assert result.nox_status == first.nox_status, \
                f"NOx status differs at iteration {i}"
            assert result.sox_status == first.sox_status, \
                f"SOx status differs at iteration {i}"
            assert result.margin_to_limits == first.margin_to_limits, \
                f"Margin to limits differs at iteration {i}"

    def test_compliance_threshold_consistency(self, emissions_tools):
        """Test compliance thresholds are consistently applied."""
        # Test at exactly the limit
        at_limit = {
            "nox": {"emission_rate_lb_mmbtu": 0.10},  # EPA limit
            "sox": {"emission_rate_lb_mmbtu": 0.15},  # EPA limit
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.03},  # EPA limit
        }

        results = []
        for _ in range(20):
            result = emissions_tools.check_compliance_status(
                emissions_result=at_limit,
                jurisdiction="EPA",
            )
            results.append(result.overall_status)

        # All results should be identical
        assert len(set(results)) == 1

    # =========================================================================
    # REPORT GENERATION REPRODUCIBILITY TESTS
    # =========================================================================

    def test_report_generation_reproducibility(
        self, emissions_tools, facility_data, reporting_period, emissions_records
    ):
        """Test report generation is deterministic."""
        reports = []

        for _ in range(10):
            report = emissions_tools.generate_regulatory_report(
                report_format="EPA_ECMPS",
                reporting_period=reporting_period,
                facility_data=facility_data,
                emissions_data=emissions_records[:100],
            )
            reports.append(report)

        first = reports[0]
        for i, report in enumerate(reports[1:], 2):
            assert report.avg_nox_lb_mmbtu == first.avg_nox_lb_mmbtu, \
                f"NOx average differs at iteration {i}"
            assert report.avg_sox_lb_mmbtu == first.avg_sox_lb_mmbtu, \
                f"SOx average differs at iteration {i}"
            assert report.total_co2_tons == first.total_co2_tons, \
                f"CO2 total differs at iteration {i}"
            assert report.compliance_rate_percent == first.compliance_rate_percent, \
                f"Compliance rate differs at iteration {i}"

    # =========================================================================
    # CROSS-PLATFORM DETERMINISM TESTS
    # =========================================================================

    def test_cross_platform_determinism_known_values(self, emissions_tools):
        """Test calculations match known reference values."""
        # Known test case: 50 ppm NOx at 3% O2 with natural gas
        cems_data = {"nox_ppm": 50.0, "o2_percent": 3.0}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        result = emissions_tools.calculate_nox_emissions(cems_data, fuel_data)

        # EPA Method 19 expected: E = C * Fd * Mw / (K * 10^6)
        # E = 50 * 8710 * 46.01 / (385.3 * 10^6) = 0.052
        # Allow small tolerance for rounding
        expected = 0.052
        assert abs(result.emission_rate_lb_mmbtu - expected) < 0.01

    def test_cross_platform_determinism_floating_point(self, emissions_tools):
        """Test floating point calculations are consistent."""
        # Use values that might cause floating point issues
        cems_data = {"nox_ppm": 33.33333, "o2_percent": 3.33333}
        fuel_data = {"fuel_type": "natural_gas", "heat_input_mmbtu_hr": 100.0}

        results = []
        for _ in range(50):
            result = emissions_tools.calculate_nox_emissions(cems_data, fuel_data)
            results.append(result.emission_rate_lb_mmbtu)

        # All results should be identical
        assert len(set(results)) == 1

    # =========================================================================
    # PROVENANCE HASH CONSISTENCY TESTS
    # =========================================================================

    def test_provenance_hash_consistency(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test provenance hash is consistent for same input."""
        hashes = []

        for _ in range(100):
            result = emissions_tools.calculate_nox_emissions(
                cems_data=sample_cems_data,
                fuel_data=natural_gas_fuel_data,
            )
            hashes.append(result.provenance_hash)

        # All hashes must be identical
        assert len(set(hashes)) == 1
        assert len(hashes[0]) == 64  # SHA-256

    def test_provenance_hash_changes_with_input(self, emissions_tools, natural_gas_fuel_data):
        """Test provenance hash changes with different input."""
        cems_1 = {"nox_ppm": 50.0, "o2_percent": 3.0}
        cems_2 = {"nox_ppm": 51.0, "o2_percent": 3.0}

        result_1 = emissions_tools.calculate_nox_emissions(cems_1, natural_gas_fuel_data)
        result_2 = emissions_tools.calculate_nox_emissions(cems_2, natural_gas_fuel_data)

        # Different inputs should produce different hashes
        assert result_1.provenance_hash != result_2.provenance_hash

    def test_provenance_hash_sha256_format(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test provenance hash is valid SHA-256 format."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        # SHA-256 is 64 hex characters
        assert len(result.provenance_hash) == 64
        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    def test_provenance_hash_all_calculators(
        self, emissions_tools, sample_cems_data, natural_gas_fuel_data
    ):
        """Test all calculators produce provenance hashes."""
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)
        ef = emissions_tools.calculate_emission_factors("natural_gas")
        fc = emissions_tools.analyze_fuel_composition("natural_gas")

        # All should have valid provenance hashes
        for result, name in [(nox, "NOx"), (sox, "SOx"), (co2, "CO2"),
                            (pm, "PM"), (ef, "EF"), (fc, "FC")]:
            assert hasattr(result, 'provenance_hash'), f"{name} missing provenance_hash"
            assert len(result.provenance_hash) == 64, f"{name} hash wrong length"

    # =========================================================================
    # AUDIT TRAIL HASH CHAIN TESTS
    # =========================================================================

    def test_audit_trail_hash_chain_consistency(
        self, emissions_tools, facility_data, emissions_records
    ):
        """Test audit trail hash chain is deterministic."""
        audit_period = {"start_date": "2024-01-01", "end_date": "2024-03-31"}

        audits = []
        for _ in range(10):
            audit = emissions_tools.generate_audit_trail(
                audit_period=audit_period,
                facility_data=facility_data,
                emissions_records=emissions_records[:50],
                compliance_events=[],
            )
            audits.append(audit)

        first = audits[0]
        for audit in audits[1:]:
            assert audit.root_hash == first.root_hash
            assert audit.record_hashes == first.record_hashes
            assert audit.chain_valid == first.chain_valid
