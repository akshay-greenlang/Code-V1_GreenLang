# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH Tools.

Tests all 12 deterministic tools including input validation,
output schema compliance, and error handling.

Test Count: 24+ tests
Coverage Target: 90%+

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
    EMISSIONS_TOOL_SCHEMAS,
    NOxEmissionsResult,
    SOxEmissionsResult,
    CO2EmissionsResult,
    PMEmissionsResult,
    ComplianceCheckResult,
    ViolationResult,
    RegulatoryReportResult,
    ExceedancePredictionResult,
    EmissionFactorResult,
    DispersionResult,
    AuditTrailResult,
    FuelAnalysisResult,
)


# =============================================================================
# TEST CLASS: TOOLS
# =============================================================================

@pytest.mark.unit
class TestTools:
    """Test suite for EmissionsComplianceTools."""

    # =========================================================================
    # TOOL SCHEMA TESTS
    # =========================================================================

    def test_tool_schemas_complete(self):
        """Test all 12 tool schemas are defined."""
        expected_tools = [
            "calculate_nox_emissions",
            "calculate_sox_emissions",
            "calculate_co2_emissions",
            "calculate_particulate_matter",
            "check_compliance_status",
            "generate_regulatory_report",
            "detect_violations",
            "predict_exceedances",
            "calculate_emission_factors",
            "analyze_fuel_composition",
            "calculate_dispersion",
            "generate_audit_trail",
        ]

        for tool in expected_tools:
            assert tool in EMISSIONS_TOOL_SCHEMAS, f"Missing tool schema: {tool}"

    def test_tool_schemas_have_required_fields(self):
        """Test tool schemas have required fields."""
        for tool_name, schema in EMISSIONS_TOOL_SCHEMAS.items():
            assert "name" in schema, f"{tool_name} missing 'name'"
            assert "description" in schema, f"{tool_name} missing 'description'"
            assert "parameters" in schema, f"{tool_name} missing 'parameters'"
            assert "deterministic" in schema, f"{tool_name} missing 'deterministic'"

    def test_all_tools_are_deterministic(self):
        """Test all tools are marked as deterministic."""
        for tool_name, schema in EMISSIONS_TOOL_SCHEMAS.items():
            assert schema["deterministic"] == True, f"{tool_name} not marked deterministic"

    # =========================================================================
    # TOOL 1: NOX EMISSIONS
    # =========================================================================

    def test_tool_calculate_nox_emissions(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test calculate_nox_emissions tool."""
        result = emissions_tools.calculate_nox_emissions(
            cems_data=sample_cems_data,
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, NOxEmissionsResult)
        assert result.concentration_ppm >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.calculation_method == "EPA_Method_19"

    def test_tool_calculate_nox_emissions_input_validation(self, emissions_tools):
        """Test NOx tool input validation."""
        # Should handle empty inputs gracefully
        result = emissions_tools.calculate_nox_emissions(
            cems_data={},
            fuel_data={},
        )
        assert result is not None

    # =========================================================================
    # TOOL 2: SOX EMISSIONS
    # =========================================================================

    def test_tool_calculate_sox_emissions(self, emissions_tools, fuel_oil_no2_data):
        """Test calculate_sox_emissions tool."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data=fuel_oil_no2_data,
        )

        assert isinstance(result, SOxEmissionsResult)
        assert result.concentration_ppm >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.calculation_method == "Stoichiometric_S_to_SO2"

    def test_tool_calculate_sox_emissions_input_validation(self, emissions_tools):
        """Test SOx tool input validation."""
        result = emissions_tools.calculate_sox_emissions(
            fuel_data={},
        )
        assert result is not None

    # =========================================================================
    # TOOL 3: CO2 EMISSIONS
    # =========================================================================

    def test_tool_calculate_co2_emissions(self, emissions_tools, natural_gas_fuel_data):
        """Test calculate_co2_emissions tool."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data=natural_gas_fuel_data,
        )

        assert isinstance(result, CO2EmissionsResult)
        assert result.concentration_percent >= 0
        assert result.emission_rate_lb_mmbtu >= 0
        assert result.calculation_method == "AP42_Emission_Factor"

    def test_tool_calculate_co2_emissions_input_validation(self, emissions_tools):
        """Test CO2 tool input validation."""
        result = emissions_tools.calculate_co2_emissions(
            fuel_data={},
        )
        assert result is not None

    # =========================================================================
    # TOOL 4: PARTICULATE MATTER
    # =========================================================================

    def test_tool_calculate_particulate_matter(self, emissions_tools, sample_cems_data, coal_bituminous_data):
        """Test calculate_particulate_matter tool."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data=sample_cems_data,
            fuel_data=coal_bituminous_data,
        )

        assert isinstance(result, PMEmissionsResult)
        assert result.concentration_mg_m3 >= 0
        assert result.pm10_fraction >= 0
        assert result.pm25_fraction >= 0
        assert result.calculation_method == "AP42_with_Control"

    def test_tool_calculate_particulate_matter_input_validation(self, emissions_tools):
        """Test PM tool input validation."""
        result = emissions_tools.calculate_particulate_matter(
            cems_data={},
            fuel_data={},
        )
        assert result is not None

    # =========================================================================
    # TOOL 5: COMPLIANCE STATUS
    # =========================================================================

    def test_tool_check_compliance_status(self, emissions_tools):
        """Test check_compliance_status tool."""
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

    def test_tool_check_compliance_status_input_validation(self, emissions_tools):
        """Test compliance tool input validation."""
        result = emissions_tools.check_compliance_status(
            emissions_result={},
            jurisdiction="EPA",
        )
        assert result is not None

    # =========================================================================
    # TOOL 6: REGULATORY REPORT
    # =========================================================================

    def test_tool_generate_regulatory_report(self, emissions_tools, facility_data, reporting_period, emissions_records):
        """Test generate_regulatory_report tool."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=emissions_records[:100],
        )

        assert isinstance(result, RegulatoryReportResult)
        assert result.report_id.startswith("RPT-")
        assert result.format_version is not None

    def test_tool_generate_regulatory_report_empty_data(self, emissions_tools, facility_data, reporting_period):
        """Test report tool with empty emissions data."""
        result = emissions_tools.generate_regulatory_report(
            report_format="EPA_ECMPS",
            reporting_period=reporting_period,
            facility_data=facility_data,
            emissions_data=[],
        )
        assert result is not None

    # =========================================================================
    # TOOL 7: DETECT VIOLATIONS
    # =========================================================================

    def test_tool_detect_violations(self, emissions_tools, epa_permit_limits):
        """Test detect_violations tool."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], ViolationResult)

    def test_tool_detect_violations_no_violations(self, emissions_tools, epa_permit_limits):
        """Test detect_violations with compliant emissions."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.05},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        result = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert isinstance(result, list)
        assert len(result) == 0

    # =========================================================================
    # TOOL 8: PREDICT EXCEEDANCES
    # =========================================================================

    def test_tool_predict_exceedances(self, emissions_tools, epa_permit_limits):
        """Test predict_exceedances tool."""
        historical_data = [
            {"nox_lb_mmbtu": 0.05 + i * 0.001, "sox_lb_mmbtu": 0.08, "pm_lb_mmbtu": 0.02}
            for i in range(24)
        ]

        result = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
            forecast_hours=24,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], ExceedancePredictionResult)

    def test_tool_predict_exceedances_short_history(self, emissions_tools, epa_permit_limits):
        """Test predict_exceedances with short history."""
        historical_data = [
            {"nox_lb_mmbtu": 0.05, "sox_lb_mmbtu": 0.08, "pm_lb_mmbtu": 0.02}
        ]

        result = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
        )

        # Should handle short history gracefully
        assert result is not None

    # =========================================================================
    # TOOL 9: EMISSION FACTORS
    # =========================================================================

    def test_tool_calculate_emission_factors(self, emissions_tools):
        """Test calculate_emission_factors tool."""
        result = emissions_tools.calculate_emission_factors("natural_gas")

        assert isinstance(result, EmissionFactorResult)
        assert result.fuel_type == "natural_gas"
        assert result.nox_factor > 0
        assert result.factor_source == "AP-42 Fifth Edition"

    def test_tool_calculate_emission_factors_all_fuels(self, emissions_tools):
        """Test emission factors for all supported fuels."""
        fuel_types = ["natural_gas", "fuel_oil_no2", "coal_bituminous", "biomass_wood"]

        for fuel_type in fuel_types:
            result = emissions_tools.calculate_emission_factors(fuel_type)
            assert result.fuel_type == fuel_type

    # =========================================================================
    # TOOL 10: FUEL COMPOSITION
    # =========================================================================

    def test_tool_analyze_fuel_composition(self, emissions_tools):
        """Test analyze_fuel_composition tool."""
        result = emissions_tools.analyze_fuel_composition("natural_gas")

        assert isinstance(result, FuelAnalysisResult)
        assert result.fuel_type == "natural_gas"
        assert result.carbon_percent > 0
        assert result.hhv_btu_lb > 0

    def test_tool_analyze_fuel_composition_with_analysis(self, emissions_tools):
        """Test analyze_fuel_composition with custom analysis."""
        ultimate = {"C": 80.0, "H": 12.0, "S": 1.0, "N": 0.5, "O": 3.0, "ash": 3.5}

        result = emissions_tools.analyze_fuel_composition(
            "coal_bituminous",
            ultimate_analysis=ultimate,
        )

        assert result.carbon_percent == 80.0
        assert result.sulfur_percent == 1.0

    # =========================================================================
    # TOOL 11: DISPERSION
    # =========================================================================

    def test_tool_calculate_dispersion(self, emissions_tools, stack_parameters, meteorological_data):
        """Test calculate_dispersion tool."""
        result = emissions_tools.calculate_dispersion(
            emission_rate_g_s=100.0,
            stack_parameters=stack_parameters,
            meteorological_data=meteorological_data,
        )

        assert isinstance(result, DispersionResult)
        assert result.max_ground_concentration >= 0
        assert result.distance_to_max_m > 0
        assert result.plume_rise_m >= 0
        assert result.calculation_method == "Gaussian_plume_Briggs"

    def test_tool_calculate_dispersion_stability_classes(self, emissions_tools, stack_parameters):
        """Test dispersion for different stability classes."""
        for stability in ["A", "B", "C", "D", "E", "F"]:
            met_data = {
                "wind_speed_m_s": 5.0,
                "stability_class": stability,
                "ambient_temperature_k": 298.0,
            }

            result = emissions_tools.calculate_dispersion(
                emission_rate_g_s=100.0,
                stack_parameters=stack_parameters,
                meteorological_data=met_data,
            )

            assert result.stability_class == stability

    # =========================================================================
    # TOOL 12: AUDIT TRAIL
    # =========================================================================

    def test_tool_generate_audit_trail(self, emissions_tools, facility_data, emissions_records, compliance_events):
        """Test generate_audit_trail tool."""
        audit_period = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        }

        result = emissions_tools.generate_audit_trail(
            audit_period=audit_period,
            facility_data=facility_data,
            emissions_records=emissions_records[:50],
            compliance_events=compliance_events,
        )

        assert isinstance(result, AuditTrailResult)
        assert result.audit_id.startswith("AUD-")
        assert result.root_hash is not None
        assert len(result.root_hash) == 64
        assert result.chain_valid == True

    def test_tool_generate_audit_trail_empty_records(self, emissions_tools, facility_data):
        """Test audit trail with empty records."""
        audit_period = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        }

        result = emissions_tools.generate_audit_trail(
            audit_period=audit_period,
            facility_data=facility_data,
            emissions_records=[],
            compliance_events=[],
        )

        assert result is not None
        assert result.total_records == 0

    # =========================================================================
    # OUTPUT SCHEMA COMPLIANCE TESTS
    # =========================================================================

    def test_all_results_have_provenance_hash(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test all result types include provenance hash."""
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)
        ef = emissions_tools.calculate_emission_factors("natural_gas")
        fc = emissions_tools.analyze_fuel_composition("natural_gas")

        assert hasattr(nox, 'provenance_hash') and nox.provenance_hash
        assert hasattr(sox, 'provenance_hash') and sox.provenance_hash
        assert hasattr(co2, 'provenance_hash') and co2.provenance_hash
        assert hasattr(pm, 'provenance_hash') and pm.provenance_hash
        assert hasattr(ef, 'provenance_hash') and ef.provenance_hash
        assert hasattr(fc, 'provenance_hash') and fc.provenance_hash

    def test_all_results_have_timestamp(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test all result types include timestamp."""
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)

        assert hasattr(nox, 'timestamp') and nox.timestamp
        assert hasattr(sox, 'timestamp') and sox.timestamp
        assert hasattr(co2, 'timestamp') and co2.timestamp

    def test_all_results_have_to_dict(self, emissions_tools, sample_cems_data, natural_gas_fuel_data):
        """Test all result types have to_dict method."""
        nox = emissions_tools.calculate_nox_emissions(sample_cems_data, natural_gas_fuel_data)
        sox = emissions_tools.calculate_sox_emissions(natural_gas_fuel_data)
        co2 = emissions_tools.calculate_co2_emissions(natural_gas_fuel_data)
        pm = emissions_tools.calculate_particulate_matter(sample_cems_data, natural_gas_fuel_data)

        assert hasattr(nox, 'to_dict') and callable(nox.to_dict)
        assert hasattr(sox, 'to_dict') and callable(sox.to_dict)
        assert hasattr(co2, 'to_dict') and callable(co2.to_dict)
        assert hasattr(pm, 'to_dict') and callable(pm.to_dict)

        # Verify to_dict returns dict
        assert isinstance(nox.to_dict(), dict)
        assert isinstance(sox.to_dict(), dict)
        assert isinstance(co2.to_dict(), dict)
        assert isinstance(pm.to_dict(), dict)
