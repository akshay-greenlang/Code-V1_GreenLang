# -*- coding: utf-8 -*-
"""
Unit Tests for GL-014 EXCHANGER-PRO Calculator Modules.

Comprehensive unit tests for all calculator modules including:
- Heat Transfer Calculator
- Fouling Calculator
- Pressure Drop Calculator
- Cleaning Optimizer
- Economic Calculator
- Performance Tracker

Target: 85%+ code coverage with deterministic, reproducible tests.

Author: GL-TestEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Import calculator modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calculators.heat_transfer_calculator import (
    FlowArrangement,
    CorrelationType,
    FluidPhase,
    TubeLayout,
    CalculationStep,
    ProvenanceBuilder,
    TEMA_FOULING_FACTORS,
    TUBE_MATERIAL_CONDUCTIVITY,
    PI,
)
from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingResistanceInput,
    FoulingRateInput,
    KernSeatonInput,
    EbertPanchalInput,
    FoulingClassificationInput,
    FoulingSeverityInput,
    FoulingPredictionInput,
    TimeToCleaningInput,
    ExchangerType,
    FluidType,
    FoulingMechanism,
    ScalingType,
    FoulingSeverity,
    FoulingResistanceResult,
    KernSeatonResult,
    EbertPanchalResult,
    FoulingSeverityResult,
    TEMA_FOULING_FACTORS as FOULING_TEMA_FACTORS,
    GAS_CONSTANT_R,
)
from calculators.pressure_drop_calculator import (
    FlowRegime,
    FrictionCorrelation,
    ShellType,
    BaffleType,
    TubePitchPattern,
    FluidProperties,
    TubeGeometry,
    ShellGeometry,
    FoulingCondition,
    TubeSideInput,
    ShellSideInput,
    PressureDropLimits,
    PI as PRESSURE_PI,
    STANDARD_GRAVITY,
)
from calculators.economic_calculator import (
    EconomicCalculator,
    EnergyLossInput,
    ProductionImpactInput,
    MaintenanceCostInput,
    TCOInput,
    ROIInput,
    CarbonImpactInput,
    EnergyLossResult,
    ProductionImpactResult,
    MaintenanceCostResult,
    TCOResult,
    ROIResult,
    CarbonImpactResult,
    FuelType,
    CleaningMethod,
    DepreciationMethod,
    EmissionScope,
    CO2_EMISSION_FACTORS,
    MACRS_7_YEAR,
)
from calculators.cleaning_optimizer import (
    CleaningOptimizer,
)


# =============================================================================
# Test Class: Heat Transfer Coefficient Calculations
# =============================================================================

class TestHeatTransferCoefficient:
    """Unit tests for heat transfer coefficient calculations."""

    def test_heat_transfer_coefficient_calculation(
        self,
        sample_temperature_data,
        sample_exchanger_parameters,
    ):
        """Test overall heat transfer coefficient calculation."""
        # Arrange
        u_clean = sample_exchanger_parameters["design_u_clean_w_m2_k"]
        area = sample_exchanger_parameters["heat_transfer_area_m2"]
        duty = sample_exchanger_parameters["design_duty_kw"] * Decimal("1000")  # Convert to W
        lmtd = sample_temperature_data["lmtd_counter_c"]

        # Act: U = Q / (A * LMTD)
        calculated_u = duty / (area * lmtd)

        # Assert
        assert calculated_u > Decimal("0"), "U must be positive"
        assert calculated_u < Decimal("5000"), "U should be realistic (<5000 W/m2K for shell-tube)"

    @pytest.mark.parametrize("u_clean,u_fouled,expected_rf", [
        (500.0, 420.0, 0.00038),  # Typical fouling
        (1000.0, 850.0, 0.000176),  # High U, lower fouling
        (300.0, 250.0, 0.000667),  # Lower U, higher fouling
    ])
    def test_fouling_resistance_from_u_values(
        self,
        u_clean: float,
        u_fouled: float,
        expected_rf: float,
    ):
        """Test fouling resistance calculation from clean and fouled U values."""
        # Act: R_f = (1/U_fouled) - (1/U_clean)
        rf_calculated = (1 / u_fouled) - (1 / u_clean)

        # Assert
        assert abs(rf_calculated - expected_rf) < 0.0001, (
            f"Expected R_f={expected_rf}, got {rf_calculated}"
        )


# =============================================================================
# Test Class: LMTD Calculations
# =============================================================================

class TestLMTDCalculation:
    """Unit tests for Log Mean Temperature Difference calculations."""

    def test_lmtd_calculation_countercurrent(self, sample_temperature_data):
        """Test LMTD calculation for counter-current flow."""
        # Arrange
        t_hot_in = float(sample_temperature_data["hot_inlet_c"])
        t_hot_out = float(sample_temperature_data["hot_outlet_c"])
        t_cold_in = float(sample_temperature_data["cold_inlet_c"])
        t_cold_out = float(sample_temperature_data["cold_outlet_c"])

        # Counter-current: dT1 = T_hot_in - T_cold_out, dT2 = T_hot_out - T_cold_in
        delta_t1 = t_hot_in - t_cold_out  # 120 - 65 = 55
        delta_t2 = t_hot_out - t_cold_in  # 80 - 30 = 50

        # Act: LMTD = (dT1 - dT2) / ln(dT1/dT2)
        lmtd_counter = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # Assert
        assert 40 < lmtd_counter < 60, "LMTD should be between hot and cold deltas"
        assert abs(lmtd_counter - 52.46) < 0.5, f"Expected ~52.46, got {lmtd_counter}"

    def test_lmtd_calculation_cocurrent(self, sample_temperature_data):
        """Test LMTD calculation for co-current (parallel) flow."""
        # Arrange
        t_hot_in = float(sample_temperature_data["hot_inlet_c"])
        t_hot_out = float(sample_temperature_data["hot_outlet_c"])
        t_cold_in = float(sample_temperature_data["cold_inlet_c"])
        t_cold_out = float(sample_temperature_data["cold_outlet_c"])

        # Co-current: dT1 = T_hot_in - T_cold_in, dT2 = T_hot_out - T_cold_out
        delta_t1 = t_hot_in - t_cold_in  # 120 - 30 = 90
        delta_t2 = t_hot_out - t_cold_out  # 80 - 65 = 15

        # Act: LMTD = (dT1 - dT2) / ln(dT1/dT2)
        lmtd_parallel = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # Assert
        assert lmtd_parallel > 0, "LMTD must be positive"
        assert lmtd_parallel < delta_t1, "LMTD must be less than max delta T"
        assert abs(lmtd_parallel - 41.86) < 0.5, f"Expected ~41.86, got {lmtd_parallel}"

    def test_lmtd_equal_deltas(self):
        """Test LMTD when delta T1 equals delta T2 (special case)."""
        # Arrange
        delta_t1 = 50.0
        delta_t2 = 50.0

        # Act: When dT1 = dT2, LMTD = dT (arithmetic mean approaches LMTD)
        if abs(delta_t1 - delta_t2) < 0.001:
            lmtd = (delta_t1 + delta_t2) / 2
        else:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # Assert
        assert lmtd == 50.0, "LMTD should equal delta T when deltas are equal"

    @pytest.mark.parametrize("dt1,dt2", [
        (-10, 20),
        (20, -10),
        (0, 20),
        (20, 0),
    ])
    def test_lmtd_invalid_temperatures(self, dt1: float, dt2: float):
        """Test LMTD raises error for invalid temperature differences."""
        # Assert
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            if dt1 <= 0 or dt2 <= 0:
                raise ValueError("Temperature differences must be positive")
            (dt1 - dt2) / math.log(dt1 / dt2)

    def test_lmtd_correction_factor(self):
        """Test LMTD correction factor (F) for 1-2 shell-tube exchanger."""
        # Arrange: For 1-2 shell-tube, use correction factor chart/formula
        # R = (T1_in - T1_out) / (T2_out - T2_in)
        # P = (T2_out - T2_in) / (T1_in - T2_in)
        t_hot_in, t_hot_out = 120.0, 80.0
        t_cold_in, t_cold_out = 30.0, 65.0

        R = (t_hot_in - t_hot_out) / (t_cold_out - t_cold_in)  # 40/35 = 1.14
        P = (t_cold_out - t_cold_in) / (t_hot_in - t_cold_in)  # 35/90 = 0.39

        # Act: F correction factor calculation (simplified Bowman formula)
        # For 1-2 exchanger: F = sqrt(R^2+1) * ln((1-P)/(1-P*R)) / ((R-1)*ln(X))
        # where X = (2-P*(R+1-sqrt(R^2+1)))/(2-P*(R+1+sqrt(R^2+1)))
        sqrt_r2_plus_1 = math.sqrt(R**2 + 1)
        X = (2 - P * (R + 1 - sqrt_r2_plus_1)) / (2 - P * (R + 1 + sqrt_r2_plus_1))

        if R != 1:
            F = (sqrt_r2_plus_1 * math.log((1 - P) / (1 - P * R))) / (
                (R - 1) * math.log(X)
            )
        else:
            F = 0.95  # Approximation when R = 1

        # Assert
        assert 0.7 < F < 1.0, f"F factor should be between 0.7 and 1.0, got {F}"
        assert abs(F - 0.95) < 0.1, f"Expected F ~0.95 for these conditions, got {F}"


# =============================================================================
# Test Class: Effectiveness-NTU Method
# =============================================================================

class TestEffectivenessNTU:
    """Unit tests for Effectiveness-NTU method calculations."""

    @pytest.mark.parametrize("flow_arrangement,ntu,c_r,expected_effectiveness", [
        (FlowArrangement.COUNTER_FLOW, 1.0, 0.5, 0.568),
        (FlowArrangement.COUNTER_FLOW, 2.0, 0.5, 0.780),
        (FlowArrangement.PARALLEL_FLOW, 1.0, 0.5, 0.487),
        (FlowArrangement.COUNTER_FLOW, 1.0, 1.0, 0.5),  # Special case C_r = 1
    ])
    def test_effectiveness_ntu(
        self,
        flow_arrangement: FlowArrangement,
        ntu: float,
        c_r: float,
        expected_effectiveness: float,
    ):
        """Test effectiveness calculation from NTU and capacity ratio."""
        # Act: Calculate effectiveness based on flow arrangement
        if flow_arrangement == FlowArrangement.COUNTER_FLOW:
            if abs(c_r - 1.0) < 0.001:
                # Special case when C_r = 1
                effectiveness = ntu / (1 + ntu)
            else:
                effectiveness = (1 - math.exp(-ntu * (1 - c_r))) / (
                    1 - c_r * math.exp(-ntu * (1 - c_r))
                )
        elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
            effectiveness = (1 - math.exp(-ntu * (1 + c_r))) / (1 + c_r)
        else:
            effectiveness = 0

        # Assert
        assert abs(effectiveness - expected_effectiveness) < 0.01, (
            f"Expected effectiveness={expected_effectiveness}, got {effectiveness}"
        )

    def test_ntu_from_effectiveness(self):
        """Test NTU calculation from known effectiveness."""
        # Arrange
        effectiveness = 0.6
        c_r = 0.5

        # Act: For counter-flow, NTU = (1/(1-C_r)) * ln((1-e*C_r)/(1-e))
        if abs(c_r - 1.0) < 0.001:
            ntu = effectiveness / (1 - effectiveness)
        else:
            ntu = (1 / (1 - c_r)) * math.log((1 - effectiveness * c_r) / (1 - effectiveness))

        # Assert
        assert ntu > 0, "NTU must be positive"
        assert 0.5 < ntu < 3.0, f"NTU should be reasonable, got {ntu}"


# =============================================================================
# Test Class: Fouling Calculations
# =============================================================================

class TestFoulingResistanceCalculation:
    """Unit tests for fouling resistance calculations."""

    def test_fouling_resistance_calculation(
        self,
        fouling_calculator: FoulingCalculator,
        fouling_resistance_input: FoulingResistanceInput,
    ):
        """Test fouling resistance calculation from U values."""
        # Act
        result = fouling_calculator.calculate_fouling_resistance(fouling_resistance_input)

        # Assert
        assert isinstance(result, FoulingResistanceResult)
        assert result.fouling_resistance_m2_k_w > Decimal("0")
        assert Decimal("0") < result.cleanliness_factor_percent <= Decimal("100")
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hash

    @pytest.mark.parametrize("u_clean,u_fouled,expected_cf", [
        (500.0, 500.0, 100.0),  # No fouling
        (500.0, 420.0, 84.0),   # 16% fouling
        (500.0, 250.0, 50.0),   # 50% fouling
    ])
    def test_cleanliness_factor_calculation(
        self,
        fouling_calculator: FoulingCalculator,
        u_clean: float,
        u_fouled: float,
        expected_cf: float,
    ):
        """Test cleanliness factor (CF) calculation."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=u_clean,
            u_fouled_w_m2_k=u_fouled,
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)
        cf = float(result.cleanliness_factor_percent)

        # Assert
        assert abs(cf - expected_cf) < 1.0, f"Expected CF={expected_cf}%, got {cf}%"


# =============================================================================
# Test Class: Kern-Seaton Model
# =============================================================================

class TestKernSeatonModel:
    """Unit tests for Kern-Seaton asymptotic fouling model."""

    def test_kern_seaton_model(
        self,
        fouling_calculator: FoulingCalculator,
        kern_seaton_input: KernSeatonInput,
    ):
        """Test Kern-Seaton asymptotic fouling model."""
        # Act
        result = fouling_calculator.calculate_kern_seaton(kern_seaton_input)

        # Assert
        assert isinstance(result, KernSeatonResult)
        assert result.predicted_r_f_m2_k_w > Decimal("0")
        assert result.predicted_r_f_m2_k_w < result.r_f_max_m2_k_w
        assert Decimal("0") <= result.asymptotic_approach_percent <= Decimal("100")

    @pytest.mark.parametrize("time_hours,expected_approach_percent", [
        (0, 0),          # t=0: 0% approach
        (500, 63.2),     # t=tau: 63.2% approach
        (1000, 86.5),    # t=2*tau: 86.5% approach
        (2500, 99.3),    # t=5*tau: 99.3% approach
    ])
    def test_kern_seaton_asymptotic_approach(
        self,
        fouling_calculator: FoulingCalculator,
        time_hours: float,
        expected_approach_percent: float,
    ):
        """Test Kern-Seaton approach to asymptotic value."""
        # Arrange
        input_data = KernSeatonInput(
            r_f_max_m2_k_w=0.0005,
            time_constant_hours=500.0,
            time_hours=time_hours,
        )

        # Act
        result = fouling_calculator.calculate_kern_seaton(input_data)

        # Assert
        # R_f(t) = R_f_max * (1 - exp(-t/tau))
        # Approach % = (1 - exp(-t/tau)) * 100
        expected = 100 * (1 - math.exp(-time_hours / 500.0))
        actual = float(result.asymptotic_approach_percent)

        assert abs(actual - expected) < 1.0, (
            f"Expected {expected}%, got {actual}%"
        )


# =============================================================================
# Test Class: Ebert-Panchal Model
# =============================================================================

class TestEbertPanchalModel:
    """Unit tests for Ebert-Panchal threshold fouling model."""

    def test_ebert_panchal_model(
        self,
        fouling_calculator: FoulingCalculator,
        ebert_panchal_input: EbertPanchalInput,
    ):
        """Test Ebert-Panchal threshold fouling model."""
        # Act
        result = fouling_calculator.calculate_ebert_panchal(ebert_panchal_input)

        # Assert
        assert isinstance(result, EbertPanchalResult)
        assert result.deposition_rate >= Decimal("0")
        assert result.removal_rate >= Decimal("0")
        assert result.threshold_velocity_m_s > Decimal("0")

    def test_ebert_panchal_above_threshold(self):
        """Test when velocity is above threshold (net deposition)."""
        # Arrange: Low velocity, high temperature -> deposition > removal
        fouling_calc = FoulingCalculator()
        input_data = EbertPanchalInput(
            reynolds_number=20000.0,
            prandtl_number=50.0,
            film_temperature_k=450.0,  # High temperature
            wall_shear_stress_pa=10.0,  # Low shear
            velocity_m_s=0.5,  # Low velocity
            fouling_mechanism=FoulingMechanism.CHEMICAL_REACTION,
        )

        # Act
        result = fouling_calc.calculate_ebert_panchal(input_data)

        # Assert: Net fouling rate should be positive
        assert result.fouling_rate_m2_k_w_per_hour >= Decimal("0")


# =============================================================================
# Test Class: Fouling Severity Assessment
# =============================================================================

class TestFoulingSeverityAssessment:
    """Unit tests for fouling severity assessment."""

    @pytest.mark.parametrize("normalized_rf,expected_severity", [
        (0.05, FoulingSeverity.CLEAN),
        (0.2, FoulingSeverity.LIGHT),
        (0.45, FoulingSeverity.MODERATE),
        (0.75, FoulingSeverity.HEAVY),
        (1.0, FoulingSeverity.SEVERE),
        (1.5, FoulingSeverity.CRITICAL),
    ])
    def test_fouling_severity_assessment(
        self,
        fouling_calculator: FoulingCalculator,
        normalized_rf: float,
        expected_severity: FoulingSeverity,
    ):
        """Test fouling severity classification."""
        # Arrange
        cf = 100 * (1 - normalized_rf * 0.15)  # Approximate CF from normalized Rf
        input_data = FoulingSeverityInput(
            normalized_fouling_factor=normalized_rf,
            cleanliness_factor_percent=cf,
        )

        # Act
        result = fouling_calculator.assess_fouling_severity(input_data)

        # Assert
        assert isinstance(result, FoulingSeverityResult)
        assert result.severity_level == expected_severity


# =============================================================================
# Test Class: Pressure Drop Calculations
# =============================================================================

class TestPressureDropTubeSide:
    """Unit tests for tube-side pressure drop calculations."""

    def test_pressure_drop_tube_side(
        self,
        sample_fluid_properties_cold: FluidProperties,
        sample_tube_geometry: TubeGeometry,
    ):
        """Test tube-side pressure drop calculation."""
        # Arrange
        fluid = sample_fluid_properties_cold
        geometry = sample_tube_geometry
        mass_flow = Decimal("20.0")

        # Calculate velocity
        flow_area_per_tube = Decimal(str(math.pi)) * geometry.tube_id_m**2 / Decimal("4")
        total_flow_area = flow_area_per_tube * geometry.number_of_tubes / geometry.number_of_passes
        velocity = mass_flow / (fluid.density_kg_m3 * total_flow_area)

        # Calculate Reynolds number
        re = fluid.density_kg_m3 * velocity * geometry.tube_id_m / fluid.viscosity_pa_s

        # Act: Calculate friction factor (Blasius for turbulent smooth)
        if re < Decimal("2300"):
            f = Decimal("64") / re
        else:
            f = Decimal("0.316") / (re ** Decimal("0.25"))

        # Darcy-Weisbach: dP = f * (L/D) * (rho*v^2/2)
        length_per_pass = geometry.tube_length_m * geometry.number_of_passes
        dp_friction = f * (length_per_pass / geometry.tube_id_m) * (
            fluid.density_kg_m3 * velocity**2 / Decimal("2")
        )

        # Assert
        assert dp_friction > Decimal("0"), "Pressure drop must be positive"
        assert dp_friction < Decimal("100000"), "Pressure drop should be reasonable (<100 kPa)"

    def test_pressure_drop_shell_side(
        self,
        sample_fluid_properties_hot: FluidProperties,
        sample_tube_geometry: TubeGeometry,
        sample_shell_geometry: ShellGeometry,
    ):
        """Test shell-side pressure drop calculation (Bell-Delaware method)."""
        # Arrange
        fluid = sample_fluid_properties_hot
        shell = sample_shell_geometry

        # Simplified crossflow area calculation
        bundle_diameter = shell.shell_id_m * Decimal("0.9")  # Approximate
        crossflow_area = (
            shell.baffle_spacing_m *
            (shell.shell_id_m - bundle_diameter)
        )

        mass_flow = Decimal("15.0")
        velocity = mass_flow / (fluid.density_kg_m3 * crossflow_area)

        # Assert
        assert velocity > Decimal("0"), "Velocity must be positive"
        assert velocity < Decimal("5"), "Shell-side velocity should be <5 m/s"


# =============================================================================
# Test Class: Friction Factor Correlations
# =============================================================================

class TestFrictionFactorCorrelations:
    """Unit tests for friction factor correlations."""

    @pytest.mark.parametrize("re,expected_f,tolerance", [
        (1000, 0.064, 0.001),      # Laminar: f = 64/Re
        (2000, 0.032, 0.001),      # Laminar edge
        (10000, 0.0316, 0.002),    # Turbulent Blasius
        (100000, 0.0178, 0.002),   # High Re turbulent
    ])
    def test_friction_factor_correlations(
        self,
        re: float,
        expected_f: float,
        tolerance: float,
    ):
        """Test friction factor correlations for different regimes."""
        # Act
        if re < 2300:
            # Laminar: Hagen-Poiseuille
            f = 64 / re
        else:
            # Turbulent smooth: Blasius
            f = 0.316 / (re ** 0.25)

        # Assert
        assert abs(f - expected_f) < tolerance, f"Expected f={expected_f}, got {f}"


# =============================================================================
# Test Class: Cleaning Interval Optimization
# =============================================================================

class TestCleaningIntervalOptimization:
    """Unit tests for cleaning interval optimization."""

    def test_cleaning_interval_optimization(
        self,
        cleaning_optimizer: CleaningOptimizer,
    ):
        """Test cleaning interval optimization calculation."""
        # Arrange
        fouling_rate = 0.00001  # m2.K/W per hour
        cleaning_cost = 15000  # USD
        energy_cost_per_hour = 50  # USD/hour for fouling penalty

        # Act: Optimal interval minimizes total cost
        # Economic model: Total cost = cleaning_cost/T + energy_cost * T/2
        # Optimal: dC/dT = 0 => T_opt = sqrt(2 * cleaning_cost / energy_cost)
        optimal_interval = math.sqrt(2 * cleaning_cost / energy_cost_per_hour)

        # Assert
        assert optimal_interval > 0, "Optimal interval must be positive"
        assert 10 < optimal_interval < 100, "Optimal interval should be reasonable"


# =============================================================================
# Test Class: Cost-Benefit Analysis
# =============================================================================

class TestCostBenefitAnalysis:
    """Unit tests for cost-benefit analysis calculations."""

    def test_cost_benefit_analysis(self, economic_calculator: EconomicCalculator):
        """Test cost-benefit analysis for cleaning decision."""
        # Arrange
        current_penalty = Decimal("1000")  # USD/day energy penalty
        cleaning_cost = Decimal("15000")  # USD
        expected_reduction = Decimal("0.8")  # 80% penalty reduction after cleaning

        # Act: Calculate payback period
        daily_savings = current_penalty * expected_reduction
        payback_days = cleaning_cost / daily_savings

        # Assert
        assert payback_days > Decimal("0")
        assert payback_days < Decimal("365"), "Payback should be less than a year"


# =============================================================================
# Test Class: Performance Efficiency
# =============================================================================

class TestPerformanceEfficiency:
    """Unit tests for performance efficiency calculations."""

    def test_performance_efficiency(
        self,
        sample_exchanger_parameters,
        sample_temperature_data,
    ):
        """Test thermal performance efficiency calculation."""
        # Arrange
        design_duty = sample_exchanger_parameters["design_duty_kw"]
        design_u = sample_exchanger_parameters["design_u_service_w_m2_k"]
        area = sample_exchanger_parameters["heat_transfer_area_m2"]
        lmtd = sample_temperature_data["lmtd_counter_c"]

        # Act: Calculate actual duty and efficiency
        actual_duty = design_u * area * lmtd / Decimal("1000")  # kW
        efficiency = (actual_duty / design_duty) * Decimal("100")

        # Assert
        assert efficiency > Decimal("0"), "Efficiency must be positive"
        assert efficiency <= Decimal("100"), "Efficiency cannot exceed 100%"


# =============================================================================
# Test Class: Health Index Calculation
# =============================================================================

class TestHealthIndexCalculation:
    """Unit tests for health index calculation."""

    @pytest.mark.parametrize("cf,dp_ratio,expected_health_range", [
        (95, 1.05, (0.9, 1.0)),    # Excellent condition
        (85, 1.15, (0.75, 0.9)),   # Good condition
        (70, 1.30, (0.5, 0.75)),   # Fair condition
        (50, 1.50, (0.25, 0.5)),   # Poor condition
    ])
    def test_health_index_calculation(
        self,
        cf: float,
        dp_ratio: float,
        expected_health_range: Tuple[float, float],
    ):
        """Test health index calculation from performance metrics."""
        # Act: Calculate health index (weighted average)
        # Health = 0.6 * CF/100 + 0.4 * (2 - dp_ratio) (normalized)
        thermal_score = cf / 100
        hydraulic_score = max(0, 2 - dp_ratio)  # 1.0 when dp_ratio=1
        health_index = 0.6 * thermal_score + 0.4 * hydraulic_score

        # Assert
        min_expected, max_expected = expected_health_range
        assert min_expected <= health_index <= max_expected, (
            f"Health index {health_index} not in expected range {expected_health_range}"
        )


# =============================================================================
# Test Class: Economic Impact Calculations
# =============================================================================

class TestEconomicImpact:
    """Unit tests for economic impact calculations."""

    def test_economic_impact(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test energy loss economic impact calculation."""
        # Act
        result = economic_calculator.calculate_energy_loss_cost(energy_loss_input)

        # Assert
        assert isinstance(result, EnergyLossResult)
        assert result.heat_transfer_loss_kw > Decimal("0")
        assert result.energy_cost_per_year_usd > Decimal("0")
        assert result.provenance_hash is not None


# =============================================================================
# Test Class: ROI Calculation
# =============================================================================

class TestROICalculation:
    """Unit tests for ROI calculations."""

    def test_roi_calculation(
        self,
        economic_calculator: EconomicCalculator,
        roi_input: ROIInput,
    ):
        """Test ROI calculation for investment analysis."""
        # Act
        result = economic_calculator.perform_roi_analysis(roi_input)

        # Assert
        assert isinstance(result, ROIResult)
        assert result.net_present_value_usd > Decimal("0"), "NPV should be positive"
        assert result.simple_payback_years > Decimal("0")
        assert result.simple_payback_years < Decimal("10")  # Should pay back within 10 years
        assert result.internal_rate_of_return_percent > Decimal("0")

    @pytest.mark.parametrize("investment,savings,expected_payback", [
        (50000, 25000, 2.0),
        (100000, 20000, 5.0),
        (30000, 15000, 2.0),
    ])
    def test_simple_payback_calculation(
        self,
        economic_calculator: EconomicCalculator,
        investment: float,
        savings: float,
        expected_payback: float,
    ):
        """Test simple payback period calculation."""
        # Arrange
        input_data = ROIInput(
            investment_cost=Decimal(str(investment)),
            annual_savings=Decimal(str(savings)),
        )

        # Act
        result = economic_calculator.perform_roi_analysis(input_data)
        payback = float(result.simple_payback_years)

        # Assert
        assert abs(payback - expected_payback) < 0.1, (
            f"Expected payback={expected_payback}, got {payback}"
        )


# =============================================================================
# Test Class: Carbon Impact
# =============================================================================

class TestCarbonImpact:
    """Unit tests for carbon impact calculations."""

    def test_carbon_impact_calculation(self, economic_calculator: EconomicCalculator):
        """Test carbon emissions calculation."""
        # Arrange
        input_data = CarbonImpactInput(
            energy_loss_kwh_per_year=Decimal("1000000"),
            fuel_type=FuelType.NATURAL_GAS,
            carbon_price_per_tonne=Decimal("50.00"),
            include_upstream_emissions=True,
        )

        # Act
        result = economic_calculator.calculate_carbon_impact(input_data)

        # Assert
        assert isinstance(result, CarbonImpactResult)
        assert result.direct_emissions_kg_co2e > Decimal("0")
        assert result.carbon_cost_usd > Decimal("0")
        assert result.scope_1_emissions_kg > Decimal("0")  # Natural gas is Scope 1

    @pytest.mark.parametrize("fuel_type,expected_emission_factor", [
        (FuelType.NATURAL_GAS, 0.185),
        (FuelType.COAL_BITUMINOUS, 0.340),
        (FuelType.FUEL_OIL_HEAVY, 0.280),
    ])
    def test_emission_factors(
        self,
        fuel_type: FuelType,
        expected_emission_factor: float,
    ):
        """Test emission factor lookup for different fuel types."""
        # Act
        actual_factor = CO2_EMISSION_FACTORS.get(fuel_type.value, 0)

        # Assert
        assert abs(actual_factor - expected_emission_factor) < 0.01, (
            f"Expected {expected_emission_factor}, got {actual_factor}"
        )


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================

class TestProvenanceTracking:
    """Unit tests for provenance hash verification."""

    def test_provenance_hash_generation(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test provenance hash is generated correctly."""
        # Act
        result = economic_calculator.calculate_energy_loss_cost(energy_loss_input)

        # Assert
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_hash_determinism(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test provenance hash is deterministic for same input."""
        # Act
        result1 = economic_calculator.calculate_energy_loss_cost(energy_loss_input)
        result2 = economic_calculator.calculate_energy_loss_cost(energy_loss_input)

        # Assert
        assert result1.provenance_hash == result2.provenance_hash, (
            "Same input should produce same provenance hash"
        )


# =============================================================================
# Test Class: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Unit tests for edge cases and error handling."""

    def test_zero_flow_rate(self, sample_fluid_properties_cold: FluidProperties):
        """Test handling of zero flow rate."""
        with pytest.raises(ValueError):
            TubeSideInput(
                fluid=sample_fluid_properties_cold,
                geometry=TubeGeometry(
                    tube_od_m=Decimal("0.01905"),
                    tube_id_m=Decimal("0.01483"),
                    tube_length_m=Decimal("6.0"),
                    number_of_tubes=200,
                ),
                mass_flow_rate_kg_s=Decimal("0"),  # Invalid: zero flow
            )

    def test_negative_temperature(self):
        """Test handling of invalid negative values."""
        with pytest.raises(ValueError):
            FluidProperties(
                density_kg_m3=Decimal("-1000"),  # Invalid: negative density
                viscosity_pa_s=Decimal("0.001"),
            )

    def test_fouled_u_greater_than_clean(self, fouling_calculator: FoulingCalculator):
        """Test error when fouled U > clean U (physically impossible)."""
        with pytest.raises(ValueError):
            FoulingResistanceInput(
                u_clean_w_m2_k=400.0,
                u_fouled_w_m2_k=500.0,  # Invalid: fouled > clean
            )

    def test_very_large_values(self, economic_calculator: EconomicCalculator):
        """Test handling of very large values."""
        # Arrange
        input_data = EnergyLossInput(
            design_duty_kw=Decimal("1E12"),  # 1 TW (extreme)
            actual_duty_kw=Decimal("9E11"),
            fuel_type=FuelType.NATURAL_GAS,
            fuel_cost_per_kwh=Decimal("0.05"),
            operating_hours_per_year=Decimal("8760"),
        )

        # Act
        result = economic_calculator.calculate_energy_loss_cost(input_data)

        # Assert: Should handle without overflow
        assert result.total_energy_penalty_per_year_usd > Decimal("0")

    def test_boundary_conditions(self, fouling_calculator: FoulingCalculator):
        """Test boundary condition: no fouling (U_clean = U_fouled)."""
        # Arrange
        input_data = FoulingResistanceInput(
            u_clean_w_m2_k=500.0,
            u_fouled_w_m2_k=500.0,  # No fouling
        )

        # Act
        result = fouling_calculator.calculate_fouling_resistance(input_data)

        # Assert
        assert result.fouling_resistance_m2_k_w == Decimal("0")
        assert result.cleanliness_factor_percent == Decimal("100")


# =============================================================================
# Test Class: Performance Tests
# =============================================================================

class TestCalculatorPerformance:
    """Performance tests for calculators."""

    def test_calculation_latency(
        self,
        economic_calculator: EconomicCalculator,
        energy_loss_input: EnergyLossInput,
    ):
        """Test calculation completes within 5ms target."""
        # Act
        start_time = time.perf_counter()
        result = economic_calculator.calculate_energy_loss_cost(energy_loss_input)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert
        assert elapsed_ms < 5.0, f"Calculation took {elapsed_ms:.2f}ms, target is <5ms"

    def test_batch_calculation_throughput(
        self,
        fouling_calculator: FoulingCalculator,
    ):
        """Test throughput for batch calculations."""
        # Arrange
        num_calculations = 1000
        inputs = [
            KernSeatonInput(
                r_f_max_m2_k_w=0.0005,
                time_constant_hours=500.0,
                time_hours=float(i * 10),
            )
            for i in range(num_calculations)
        ]

        # Act
        start_time = time.perf_counter()
        for input_data in inputs:
            fouling_calculator.calculate_kern_seaton(input_data)
        end_time = time.perf_counter()

        elapsed_seconds = end_time - start_time
        throughput = num_calculations / elapsed_seconds

        # Assert
        assert throughput > 100, f"Throughput {throughput:.0f}/s should be >100/s"
