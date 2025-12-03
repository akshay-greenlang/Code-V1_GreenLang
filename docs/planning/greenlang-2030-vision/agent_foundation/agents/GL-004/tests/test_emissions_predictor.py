# -*- coding: utf-8 -*-
"""
Test Suite for Emissions Predictor (GL-004 BURNMASTER).

Comprehensive tests for the emissions predictor module including:
- Unit tests for individual emission calculations
- NOx component tests (thermal, prompt, fuel)
- CO formation tests
- PM/UHC prediction tests
- EPA/EU compliance checking tests
- Emission credits calculation tests
- Load curve generation tests
- O2 correction tests
- Determinism and reproducibility tests
- Thread-safety tests

Reference Standards:
- EPA 40 CFR Part 60/63
- EPA AP-42 Emission Factors
- EU Industrial Emissions Directive

Test Coverage Target: 85%+
"""

import pytest
import math
import threading
import time
import hashlib
from decimal import Decimal
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculators.emissions_predictor import (
    EmissionsPredictor,
    EmissionsPredictorInput,
    EmissionsPredictorOutput,
    FuelComposition,
    CombustionConditions,
    NOxPrediction,
    COPrediction,
    UHCPrediction,
    PMPrediction,
    ComplianceResult,
    ComplianceStatus,
    EmissionCreditsResult,
    LoadEmissionCurve,
    RegulatoryLimit,
    RegulatoryStandard,
    EmissionType,
    CombustionMode,
    ProvenanceTracker,
    ThreadSafeCache,
    EPA_NSPS_LIMITS,
    EPA_MACT_LIMITS,
    EU_IED_LIMITS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def predictor():
    """Create an EmissionsPredictor instance."""
    return EmissionsPredictor()


@pytest.fixture
def natural_gas_composition():
    """Create natural gas fuel composition."""
    return FuelComposition(
        carbon=0.75,
        hydrogen=0.25,
        oxygen=0.0,
        nitrogen=0.0,
        sulfur=0.0,
        moisture=0.0,
        ash=0.0,
    )


@pytest.fixture
def fuel_oil_composition():
    """Create fuel oil composition."""
    return FuelComposition(
        carbon=0.87,
        hydrogen=0.13,
        oxygen=0.0,
        nitrogen=0.002,
        sulfur=0.005,
        moisture=0.0,
        ash=0.003,
    )


@pytest.fixture
def coal_composition():
    """Create coal fuel composition."""
    return FuelComposition(
        carbon=0.75,
        hydrogen=0.05,
        oxygen=0.08,
        nitrogen=0.015,
        sulfur=0.02,
        moisture=0.05,
        ash=0.055,
    )


@pytest.fixture
def standard_combustion_conditions():
    """Create standard combustion conditions."""
    return CombustionConditions(
        flame_temperature_c=1650.0,
        residence_time_s=2.0,
        excess_air_percent=15.0,
        o2_percent_dry=3.0,
        combustion_mode="lean",
        air_preheat_temp_c=150.0,
    )


@pytest.fixture
def high_temp_conditions():
    """Create high temperature combustion conditions."""
    return CombustionConditions(
        flame_temperature_c=1850.0,
        residence_time_s=2.5,
        excess_air_percent=12.0,
        o2_percent_dry=2.5,
        combustion_mode="lean",
    )


@pytest.fixture
def low_temp_conditions():
    """Create low temperature combustion conditions."""
    return CombustionConditions(
        flame_temperature_c=1400.0,
        residence_time_s=1.5,
        excess_air_percent=20.0,
        o2_percent_dry=4.5,
        combustion_mode="lean",
    )


@pytest.fixture
def natural_gas_input(natural_gas_composition, standard_combustion_conditions):
    """Create standard natural gas emissions input."""
    return EmissionsPredictorInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=500.0,
        fuel_composition=natural_gas_composition,
        fuel_hhv_mj_kg=55.5,
        fuel_lhv_mj_kg=50.0,
        combustion_conditions=standard_combustion_conditions,
        burner_type="nozzle_mix",
        low_nox_burner=False,
        flue_gas_recirculation=False,
        fgr_rate_percent=0.0,
        staged_combustion=False,
        scr_installed=False,
        load_percent=80.0,
        operating_hours_per_year=8000.0,
        regulatory_standard="epa_nsps",
    )


@pytest.fixture
def fuel_oil_input(fuel_oil_composition, standard_combustion_conditions):
    """Create fuel oil emissions input."""
    return EmissionsPredictorInput(
        fuel_type="fuel_oil_2",
        fuel_flow_kg_hr=400.0,
        fuel_composition=fuel_oil_composition,
        fuel_hhv_mj_kg=45.5,
        fuel_lhv_mj_kg=42.8,
        combustion_conditions=standard_combustion_conditions,
        burner_type="diffusion",
        load_percent=75.0,
        regulatory_standard="eu_ied",
    )


@pytest.fixture
def low_nox_input(natural_gas_composition, standard_combustion_conditions):
    """Create input with low NOx controls."""
    return EmissionsPredictorInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=500.0,
        fuel_composition=natural_gas_composition,
        fuel_hhv_mj_kg=55.5,
        fuel_lhv_mj_kg=50.0,
        combustion_conditions=standard_combustion_conditions,
        burner_type="low_nox",
        low_nox_burner=True,
        flue_gas_recirculation=True,
        fgr_rate_percent=15.0,
        staged_combustion=True,
        load_percent=80.0,
    )


@pytest.fixture
def scr_input(natural_gas_composition, standard_combustion_conditions):
    """Create input with SCR control."""
    return EmissionsPredictorInput(
        fuel_type="natural_gas",
        fuel_flow_kg_hr=500.0,
        fuel_composition=natural_gas_composition,
        fuel_hhv_mj_kg=55.5,
        fuel_lhv_mj_kg=50.0,
        combustion_conditions=standard_combustion_conditions,
        scr_installed=True,
        scr_efficiency_percent=85.0,
    )


# =============================================================================
# UNIT TESTS - FUEL COMPOSITION
# =============================================================================

class TestFuelComposition:
    """Tests for fuel composition dataclass."""

    def test_natural_gas_composition_valid(self, natural_gas_composition):
        """Test natural gas composition is valid."""
        total = (natural_gas_composition.carbon +
                natural_gas_composition.hydrogen +
                natural_gas_composition.oxygen +
                natural_gas_composition.nitrogen +
                natural_gas_composition.sulfur +
                natural_gas_composition.moisture +
                natural_gas_composition.ash)
        assert total == pytest.approx(1.0, rel=0.01)

    def test_fuel_composition_immutable(self, natural_gas_composition):
        """Test fuel composition is frozen."""
        with pytest.raises(AttributeError):
            natural_gas_composition.carbon = 0.80

    def test_zero_sulfur_composition(self, natural_gas_composition):
        """Test composition with zero sulfur."""
        assert natural_gas_composition.sulfur == 0.0


# =============================================================================
# UNIT TESTS - NOx PREDICTION
# =============================================================================

class TestNOxPrediction:
    """Tests for NOx emission prediction."""

    def test_thermal_nox_high_temp(self, predictor, natural_gas_composition, high_temp_conditions):
        """Test thermal NOx increases with temperature."""
        standard_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=15.0,
            o2_percent_dry=3.0,
        )
        high_cond = high_temp_conditions

        inputs_standard = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_cond,
        )
        inputs_high = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=high_cond,
        )

        result_standard = predictor.predict(inputs_standard)
        result_high = predictor.predict(inputs_high)

        # Higher temp should produce more thermal NOx
        assert result_high.nox_prediction.thermal_nox_ppm > result_standard.nox_prediction.thermal_nox_ppm

    def test_thermal_nox_low_temp_negligible(self, predictor, natural_gas_composition):
        """Test thermal NOx is negligible below 1500C."""
        low_temp_cond = CombustionConditions(
            flame_temperature_c=1400.0,
            residence_time_s=2.0,
            excess_air_percent=15.0,
            o2_percent_dry=3.0,
        )
        inputs = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=low_temp_cond,
        )

        result = predictor.predict(inputs)

        # Thermal NOx should be near zero below 1500C
        assert result.nox_prediction.thermal_nox_ppm < 5.0

    def test_prompt_nox_present(self, predictor, natural_gas_input):
        """Test prompt NOx is calculated."""
        result = predictor.predict(natural_gas_input)

        # Prompt NOx should be present for hydrocarbon fuels
        assert result.nox_prediction.prompt_nox_ppm > 0

    def test_fuel_nox_with_nitrogen(self, predictor, fuel_oil_composition, standard_combustion_conditions):
        """Test fuel NOx is calculated when fuel contains nitrogen."""
        inputs = EmissionsPredictorInput(
            fuel_type="fuel_oil_2",
            fuel_flow_kg_hr=400.0,
            fuel_composition=fuel_oil_composition,
            fuel_hhv_mj_kg=45.5,
            fuel_lhv_mj_kg=42.8,
            combustion_conditions=standard_combustion_conditions,
        )

        result = predictor.predict(inputs)

        # Fuel NOx should be present when fuel has nitrogen
        assert result.nox_prediction.fuel_nox_ppm > 0

    def test_fuel_nox_zero_without_nitrogen(self, predictor, natural_gas_input):
        """Test fuel NOx is zero without fuel nitrogen."""
        result = predictor.predict(natural_gas_input)

        # No fuel nitrogen means no fuel NOx
        assert result.nox_prediction.fuel_nox_ppm == 0.0

    def test_total_nox_components_sum(self, predictor, natural_gas_input):
        """Test total NOx equals sum of components."""
        result = predictor.predict(natural_gas_input)

        calculated_total = (result.nox_prediction.thermal_nox_ppm +
                          result.nox_prediction.prompt_nox_ppm +
                          result.nox_prediction.fuel_nox_ppm)
        assert result.nox_prediction.total_nox_ppm == pytest.approx(calculated_total, rel=0.01)

    def test_nox_units_conversion(self, predictor, natural_gas_input):
        """Test NOx unit conversions are correct."""
        result = predictor.predict(natural_gas_input)

        # ppm to mg/Nm3: ppm * MW_NO2 / 22.4
        expected_mg_nm3 = result.nox_prediction.total_nox_ppm * 46.006 / 22.4
        assert result.nox_prediction.total_nox_mg_nm3 == pytest.approx(expected_mg_nm3, rel=0.05)


# =============================================================================
# UNIT TESTS - NOx CONTROLS
# =============================================================================

class TestNOxControls:
    """Tests for NOx control technology effects."""

    def test_low_nox_burner_reduction(self, predictor, natural_gas_composition, standard_combustion_conditions):
        """Test low NOx burner reduces emissions."""
        base_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            low_nox_burner=False,
        )
        lnb_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            low_nox_burner=True,
        )

        base_result = predictor.predict(base_input)
        lnb_result = predictor.predict(lnb_input)

        # Low NOx burner should reduce NOx by ~50%
        assert lnb_result.nox_prediction.nox_after_controls_ppm < base_result.nox_prediction.nox_after_controls_ppm

    def test_fgr_reduction(self, predictor, natural_gas_composition, standard_combustion_conditions):
        """Test FGR reduces NOx emissions."""
        base_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            flue_gas_recirculation=False,
        )
        fgr_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            flue_gas_recirculation=True,
            fgr_rate_percent=20.0,
        )

        base_result = predictor.predict(base_input)
        fgr_result = predictor.predict(fgr_input)

        assert fgr_result.nox_prediction.nox_after_controls_ppm < base_result.nox_prediction.nox_after_controls_ppm

    def test_scr_reduction(self, predictor, scr_input):
        """Test SCR provides significant NOx reduction."""
        result = predictor.predict(scr_input)

        # SCR at 85% efficiency should reduce NOx by 85%
        # After controls should be much lower than total
        reduction = 1 - (result.nox_prediction.nox_after_controls_ppm / result.nox_prediction.total_nox_ppm)
        assert reduction >= 0.80

    def test_combined_controls(self, predictor, low_nox_input):
        """Test combined NOx controls stack."""
        result = predictor.predict(low_nox_input)

        # With multiple controls, reduction should be significant
        reduction = 1 - (result.nox_prediction.nox_after_controls_ppm / result.nox_prediction.total_nox_ppm)
        assert reduction >= 0.60


# =============================================================================
# UNIT TESTS - CO PREDICTION
# =============================================================================

class TestCOPrediction:
    """Tests for CO emission prediction."""

    def test_co_positive(self, predictor, natural_gas_input):
        """Test CO emissions are positive."""
        result = predictor.predict(natural_gas_input)
        assert result.co_prediction.co_ppm > 0

    def test_co_increases_low_excess_air(self, predictor, natural_gas_composition):
        """Test CO increases with low excess air."""
        high_ea_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=20.0,
            o2_percent_dry=4.0,
        )
        low_ea_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=3.0,
            o2_percent_dry=0.5,
        )

        high_ea_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=high_ea_cond,
        )
        low_ea_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=low_ea_cond,
        )

        high_ea_result = predictor.predict(high_ea_input)
        low_ea_result = predictor.predict(low_ea_input)

        # Low excess air should produce more CO
        assert low_ea_result.co_prediction.co_ppm > high_ea_result.co_prediction.co_ppm

    def test_co_increases_low_load(self, predictor, natural_gas_composition, standard_combustion_conditions):
        """Test CO increases at low load."""
        high_load_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            load_percent=100.0,
        )
        low_load_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
            load_percent=40.0,
        )

        high_load_result = predictor.predict(high_load_input)
        low_load_result = predictor.predict(low_load_input)

        assert low_load_result.co_prediction.co_ppm > high_load_result.co_prediction.co_ppm

    def test_co_efficiency_loss(self, predictor, natural_gas_input):
        """Test CO causes efficiency loss."""
        result = predictor.predict(natural_gas_input)

        # Efficiency loss should be calculated
        assert result.co_prediction.combustion_efficiency_loss_percent >= 0


# =============================================================================
# UNIT TESTS - PM PREDICTION
# =============================================================================

class TestPMPrediction:
    """Tests for particulate matter prediction."""

    def test_pm_natural_gas_low(self, predictor, natural_gas_input):
        """Test PM is low for natural gas."""
        result = predictor.predict(natural_gas_input)

        # Natural gas should have very low PM
        assert result.pm_prediction.pm_total_mg_nm3 < 10.0

    def test_pm_fuel_oil_higher(self, predictor, fuel_oil_input):
        """Test PM is higher for fuel oil."""
        result = predictor.predict(fuel_oil_input)

        # Fuel oil with ash should have higher PM
        assert result.pm_prediction.pm_total_mg_nm3 > 0

    def test_pm25_fraction(self, predictor, natural_gas_input):
        """Test PM2.5 is fraction of total PM."""
        result = predictor.predict(natural_gas_input)

        # PM2.5 should be <= total PM
        assert result.pm_prediction.pm25_mg_nm3 <= result.pm_prediction.pm_total_mg_nm3

    def test_pm10_fraction(self, predictor, natural_gas_input):
        """Test PM10 is between PM2.5 and total."""
        result = predictor.predict(natural_gas_input)

        assert result.pm_prediction.pm25_mg_nm3 <= result.pm_prediction.pm10_mg_nm3
        assert result.pm_prediction.pm10_mg_nm3 <= result.pm_prediction.pm_total_mg_nm3

    def test_pm_components(self, predictor, fuel_oil_input):
        """Test PM components (filterable vs condensable)."""
        result = predictor.predict(fuel_oil_input)

        total = result.pm_prediction.filterable_pm_mg_nm3 + result.pm_prediction.condensable_pm_mg_nm3
        assert total == pytest.approx(result.pm_prediction.pm_total_mg_nm3, rel=0.1)


# =============================================================================
# UNIT TESTS - UHC PREDICTION
# =============================================================================

class TestUHCPrediction:
    """Tests for unburned hydrocarbon prediction."""

    def test_uhc_positive(self, predictor, natural_gas_input):
        """Test UHC emissions are positive."""
        result = predictor.predict(natural_gas_input)
        assert result.uhc_prediction.uhc_ppm > 0

    def test_uhc_correlates_with_co(self, predictor, natural_gas_composition):
        """Test UHC correlates with CO (same sources)."""
        good_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=15.0,
            o2_percent_dry=3.0,
        )
        poor_cond = CombustionConditions(
            flame_temperature_c=1400.0,
            residence_time_s=1.0,
            excess_air_percent=3.0,
            o2_percent_dry=0.5,
        )

        good_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=good_cond,
        )
        poor_input = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=poor_cond,
        )

        good_result = predictor.predict(good_input)
        poor_result = predictor.predict(poor_input)

        # Poor combustion should have both higher CO and UHC
        assert poor_result.uhc_prediction.uhc_ppm > good_result.uhc_prediction.uhc_ppm

    def test_methane_slip_gas_fuel(self, predictor, natural_gas_input):
        """Test methane slip is calculated for gas fuels."""
        result = predictor.predict(natural_gas_input)

        # Methane slip should be present for natural gas
        assert result.uhc_prediction.methane_slip_ppm > 0


# =============================================================================
# UNIT TESTS - CO2 AND SOX
# =============================================================================

class TestCO2AndSOx:
    """Tests for CO2 and SOx calculations."""

    def test_co2_from_carbon(self, predictor, natural_gas_input):
        """Test CO2 is calculated from carbon content."""
        result = predictor.predict(natural_gas_input)

        # CO2 = C * 44/12 = C * 3.67
        expected_co2 = (natural_gas_input.fuel_flow_kg_hr *
                       natural_gas_input.fuel_composition.carbon *
                       (44.01 / 12.011))
        assert result.co2_kg_hr == pytest.approx(expected_co2, rel=0.05)

    def test_co2_annual_calculation(self, predictor, natural_gas_input):
        """Test annual CO2 calculation."""
        result = predictor.predict(natural_gas_input)

        expected_annual = result.co2_kg_hr * natural_gas_input.operating_hours_per_year / 1000
        assert result.co2_tonnes_year == pytest.approx(expected_annual, rel=0.01)

    def test_sox_zero_without_sulfur(self, predictor, natural_gas_input):
        """Test SOx is zero without fuel sulfur."""
        result = predictor.predict(natural_gas_input)
        assert result.sox_kg_hr == 0.0

    def test_sox_from_sulfur(self, predictor, fuel_oil_input):
        """Test SOx is calculated from sulfur content."""
        result = predictor.predict(fuel_oil_input)

        # SO2 = S * 64/32 = S * 2.0
        expected_sox = (fuel_oil_input.fuel_flow_kg_hr *
                       fuel_oil_input.fuel_composition.sulfur * 2.0)
        assert result.sox_kg_hr == pytest.approx(expected_sox, rel=0.05)


# =============================================================================
# UNIT TESTS - O2 CORRECTION
# =============================================================================

class TestO2Correction:
    """Tests for O2 reference correction."""

    def test_o2_correction_formula(self, predictor):
        """Test O2 correction uses correct formula."""
        # C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)

        # 100 ppm measured at 5% O2, corrected to 3% O2
        measured = 100.0
        measured_o2 = 5.0
        ref_o2 = 3.0

        expected = measured * (21 - ref_o2) / (21 - measured_o2)
        # 100 * 18 / 16 = 112.5

        # Use internal method via prediction
        corrected = predictor._correct_to_reference_o2(measured, measured_o2, ref_o2)
        assert corrected == pytest.approx(expected, rel=0.01)

    def test_corrected_values_in_output(self, predictor, natural_gas_input):
        """Test corrected values are in output."""
        result = predictor.predict(natural_gas_input)

        assert result.nox_corrected_ppm is not None
        assert result.co_corrected_ppm is not None
        assert result.reference_o2_percent == 3.0


# =============================================================================
# UNIT TESTS - COMPLIANCE CHECKING
# =============================================================================

class TestComplianceChecking:
    """Tests for regulatory compliance checking."""

    def test_compliance_results_present(self, predictor, natural_gas_input):
        """Test compliance results are generated."""
        result = predictor.predict(natural_gas_input)

        assert len(result.compliance_results) > 0

    def test_compliance_status_values(self, predictor, natural_gas_input):
        """Test compliance status is valid enum value."""
        result = predictor.predict(natural_gas_input)

        valid_statuses = [s.value for s in ComplianceStatus]
        for cr in result.compliance_results:
            assert cr.status in valid_statuses

    def test_compliance_margin_calculation(self, predictor, natural_gas_input):
        """Test margin to limit is calculated."""
        result = predictor.predict(natural_gas_input)

        for cr in result.compliance_results:
            # Margin = limit - corrected value
            expected_margin = cr.limit_value - cr.corrected_value
            assert cr.margin_to_limit == pytest.approx(expected_margin, rel=0.01)

    def test_percent_of_limit_calculation(self, predictor, natural_gas_input):
        """Test percent of limit is calculated."""
        result = predictor.predict(natural_gas_input)

        for cr in result.compliance_results:
            expected_percent = (cr.corrected_value / cr.limit_value * 100) if cr.limit_value > 0 else 0
            assert cr.percent_of_limit == pytest.approx(expected_percent, rel=0.1)

    def test_overall_compliance_status(self, predictor, natural_gas_input):
        """Test overall compliance status is determined."""
        result = predictor.predict(natural_gas_input)

        valid_statuses = [s.value for s in ComplianceStatus]
        assert result.overall_compliance_status in valid_statuses

    def test_violations_list(self, predictor, natural_gas_input):
        """Test violations list is populated for non-compliant cases."""
        result = predictor.predict(natural_gas_input)

        # If overall status is violation, violations list should have entries
        if result.overall_compliance_status == ComplianceStatus.VIOLATION.value:
            assert len(result.violations) > 0

    def test_eu_ied_limits_applied(self, predictor, fuel_oil_input):
        """Test EU IED limits are applied when requested."""
        result = predictor.predict(fuel_oil_input)

        # Check that standard is EU IED in results
        for cr in result.compliance_results:
            assert "EU IED" in cr.standard or "eu" in cr.standard.lower()


# =============================================================================
# UNIT TESTS - EMISSION CREDITS
# =============================================================================

class TestEmissionCredits:
    """Tests for emission credits calculation."""

    def test_emission_credits_calculated(self, predictor, natural_gas_input):
        """Test emission credits are calculated."""
        result = predictor.predict(natural_gas_input)

        assert result.emission_credits is not None

    def test_credit_values_decimal(self, predictor, natural_gas_input):
        """Test credit values use Decimal for precision."""
        result = predictor.predict(natural_gas_input)

        assert isinstance(result.emission_credits.nox_credits_tonnes, Decimal)
        assert isinstance(result.emission_credits.co2_credits_tonnes, Decimal)

    def test_credit_vintage_year(self, predictor, natural_gas_input):
        """Test credit vintage year is current year."""
        from datetime import datetime
        result = predictor.predict(natural_gas_input)

        current_year = datetime.utcnow().year
        assert result.emission_credits.credit_vintage_year == current_year


# =============================================================================
# UNIT TESTS - LOAD CURVES
# =============================================================================

class TestLoadCurves:
    """Tests for load-based emission curves."""

    def test_load_curve_generated(self, predictor, natural_gas_input):
        """Test load emission curve is generated."""
        result = predictor.predict(natural_gas_input)

        assert result.load_emission_curve is not None

    def test_load_curve_points(self, predictor, natural_gas_input):
        """Test load curve has multiple points."""
        result = predictor.predict(natural_gas_input)

        assert len(result.load_emission_curve.load_points_percent) >= 3

    def test_load_curve_nox_increases_with_load(self, predictor, natural_gas_input):
        """Test NOx generally increases with load in curve."""
        result = predictor.predict(natural_gas_input)

        curve = result.load_emission_curve
        # NOx at 100% should be higher than at 25%
        idx_25 = curve.load_points_percent.index(25.0)
        idx_100 = curve.load_points_percent.index(100.0)
        # This may not always hold due to turndown effects
        assert curve.nox_at_load_ppm[idx_100] >= curve.nox_at_load_ppm[idx_25]

    def test_load_curve_efficiency_decreases_at_low_load(self, predictor, natural_gas_input):
        """Test efficiency decreases at low load."""
        result = predictor.predict(natural_gas_input)

        curve = result.load_emission_curve
        idx_25 = curve.load_points_percent.index(25.0)
        idx_100 = curve.load_points_percent.index(100.0)

        # Efficiency at 100% should be higher than at 25%
        assert curve.efficiency_at_load[idx_100] > curve.efficiency_at_load[idx_25]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete prediction workflow."""

    def test_complete_prediction_natural_gas(self, predictor, natural_gas_input):
        """Test complete prediction for natural gas."""
        result = predictor.predict(natural_gas_input)

        # Verify all major components present
        assert result.nox_prediction is not None
        assert result.co_prediction is not None
        assert result.uhc_prediction is not None
        assert result.pm_prediction is not None
        assert result.co2_kg_hr > 0
        assert result.compliance_results is not None
        assert result.load_emission_curve is not None

    def test_complete_prediction_fuel_oil(self, predictor, fuel_oil_input):
        """Test complete prediction for fuel oil."""
        result = predictor.predict(fuel_oil_input)

        # Should have SOx due to sulfur content
        assert result.sox_kg_hr > 0
        # Should have higher PM due to ash
        assert result.pm_prediction.pm_total_mg_nm3 > 0

    def test_total_criteria_pollutants(self, predictor, natural_gas_input):
        """Test total criteria pollutants calculation."""
        result = predictor.predict(natural_gas_input)

        expected_total = (result.nox_prediction.total_nox_kg_hr +
                         result.co_prediction.co_kg_hr +
                         result.pm_prediction.pm_total_kg_hr +
                         result.sox_kg_hr)
        assert result.total_criteria_pollutants_kg_hr == pytest.approx(expected_total, rel=0.01)

    def test_recommendations_generated(self, predictor, natural_gas_input):
        """Test recommendations are generated."""
        result = predictor.predict(natural_gas_input)

        assert len(result.recommendations) > 0


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for calculation determinism and reproducibility."""

    def test_same_inputs_same_outputs(self, natural_gas_input):
        """Test same inputs produce same outputs."""
        predictor1 = EmissionsPredictor()
        predictor2 = EmissionsPredictor()

        result1 = predictor1.predict(natural_gas_input)
        result2 = predictor2.predict(natural_gas_input)

        assert result1.nox_prediction.total_nox_ppm == result2.nox_prediction.total_nox_ppm
        assert result1.co_prediction.co_ppm == result2.co_prediction.co_ppm
        assert result1.co2_kg_hr == result2.co2_kg_hr

    def test_repeated_calls_consistent(self, predictor, natural_gas_input):
        """Test repeated calls give consistent results."""
        results = [predictor.predict(natural_gas_input) for _ in range(5)]

        nox_values = [r.nox_prediction.total_nox_ppm for r in results]
        assert all(v == nox_values[0] for v in nox_values)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Tests for calculation performance."""

    def test_prediction_speed(self, predictor, natural_gas_input):
        """Test prediction completes within time limit."""
        import time

        start = time.perf_counter()
        result = predictor.predict(natural_gas_input)
        elapsed = time.perf_counter() - start

        # Should complete within 100ms
        assert elapsed < 0.1
        assert result is not None

    def test_multiple_predictions_performance(self, predictor, natural_gas_input):
        """Test multiple predictions maintain performance."""
        import time

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            predictor.predict(natural_gas_input)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        assert avg_time < 0.05


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_predictions(self, natural_gas_input):
        """Test concurrent predictions don't interfere."""
        results = []
        errors = []

        def predict():
            try:
                predictor = EmissionsPredictor()
                result = predictor.predict(natural_gas_input)
                results.append(result.nox_prediction.total_nox_ppm)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=predict) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == results[0] for r in results)


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Tests for provenance tracking."""

    def test_provenance_hash_generated(self, predictor, natural_gas_input):
        """Test provenance hash is generated."""
        result = predictor.predict(natural_gas_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    def test_calculation_steps_recorded(self, predictor, natural_gas_input):
        """Test calculation steps are recorded."""
        result = predictor.predict(natural_gas_input)

        assert result.calculation_steps > 0
        steps = predictor.get_calculation_steps()
        assert len(steps) == result.calculation_steps

    def test_timestamp_format(self, predictor, natural_gas_input):
        """Test timestamp is ISO format."""
        result = predictor.predict(natural_gas_input)

        assert result.calculation_timestamp.endswith("Z")
        assert "T" in result.calculation_timestamp


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_excess_air(self, predictor, natural_gas_composition):
        """Test handling of zero excess air."""
        zero_ea_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=0.0,
            o2_percent_dry=0.0,
        )
        inputs = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=zero_ea_cond,
        )

        result = predictor.predict(inputs)
        assert result is not None

    def test_high_excess_air(self, predictor, natural_gas_composition):
        """Test handling of high excess air."""
        high_ea_cond = CombustionConditions(
            flame_temperature_c=1650.0,
            residence_time_s=2.0,
            excess_air_percent=100.0,
            o2_percent_dry=10.0,
        )
        inputs = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=500.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=high_ea_cond,
        )

        result = predictor.predict(inputs)
        assert result is not None

    def test_zero_fuel_flow(self, predictor, natural_gas_composition, standard_combustion_conditions):
        """Test handling of zero fuel flow."""
        inputs = EmissionsPredictorInput(
            fuel_type="natural_gas",
            fuel_flow_kg_hr=0.0,
            fuel_composition=natural_gas_composition,
            fuel_hhv_mj_kg=55.5,
            fuel_lhv_mj_kg=50.0,
            combustion_conditions=standard_combustion_conditions,
        )

        result = predictor.predict(inputs)
        # Should handle gracefully
        assert result.co2_kg_hr == 0.0


# =============================================================================
# REGULATORY LIMITS DATABASE TESTS
# =============================================================================

class TestRegulatoryLimits:
    """Tests for regulatory limits database."""

    def test_epa_nsps_limits_present(self):
        """Test EPA NSPS limits are defined."""
        assert "nox_gas" in EPA_NSPS_LIMITS
        assert "nox_oil" in EPA_NSPS_LIMITS

    def test_epa_mact_limits_present(self):
        """Test EPA MACT limits are defined."""
        assert "co_gas_existing" in EPA_MACT_LIMITS

    def test_eu_ied_limits_present(self):
        """Test EU IED limits are defined."""
        assert "nox_gas" in EU_IED_LIMITS
        assert "co_gas" in EU_IED_LIMITS
        assert "pm_gas" in EU_IED_LIMITS

    def test_limit_has_required_fields(self):
        """Test limits have all required fields."""
        for key, limit in EPA_NSPS_LIMITS.items():
            assert hasattr(limit, "pollutant")
            assert hasattr(limit, "limit_value")
            assert hasattr(limit, "units")
            assert hasattr(limit, "reference_o2_percent")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
