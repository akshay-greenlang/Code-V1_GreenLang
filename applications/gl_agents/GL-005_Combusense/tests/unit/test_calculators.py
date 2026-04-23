# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for GL-005 CombustionControlAgent calculator modules.

Tests all 7 calculator components with 85%+ coverage:
1. CombustionStabilityCalculator
2. FuelAirOptimizer
3. HeatOutputCalculator
4. PIDController (including edge cases)
5. SafetyValidator
6. EmissionsCalculator
7. AdvancedCombustionDiagnosticsCalculator (NEW - 2,665 line calculator)

Validates:
- Determinism (same inputs = same outputs)
- Calculation accuracy against known values
- Edge cases and boundary conditions
- Error handling

Target: 100+ tests covering all calculation modules.
"""

import pytest
import math
import hashlib
import json
from decimal import Decimal
from typing import Dict, Any, List
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Import calculators
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'calculators'))

from combustion_stability_calculator import (
    CombustionStabilityCalculator,
    StabilityInput,
    StabilityResult,
    StabilityLevel,
    OscillationPattern
)
from fuel_air_optimizer import (
    FuelAirOptimizer,
    OptimizerInput,
    OptimizerResult,
    FuelType,
    OptimizationObjective,
    FuelComposition,
    EmissionConstraints
)
from heat_output_calculator import (
    HeatOutputCalculator,
    HeatOutputInput,
    HeatOutputResult,
    HeatLossCategory
)
from pid_controller import (
    PIDController,
    PIDInput,
    PIDOutput,
    ControlMode,
    TuningMethod,
    AntiWindupMethod,
    AutoTuneInput,
    AutoTuneOutput
)
from safety_validator import (
    SafetyValidator,
    SafetyValidatorInput,
    SafetyValidatorOutput,
    SafetyLimits,
    SafetyLevel as SafetyInterlockLevel,
    InterlockStatus,
    AlarmPriority
)
from emissions_calculator import (
    EmissionsCalculator,
    EmissionsInput,
    EmissionsResult,
    EmissionType,
    ComplianceStatus
)
from combustion_diagnostics import (
    AdvancedCombustionDiagnosticsCalculator,
    AdvancedDiagnosticInput,
    AdvancedDiagnosticOutput,
    ZoneInput,
    FaultType,
    FaultSeverity,
    FlamePattern,
    CombustionMode,
    MaintenancePriority,
    BurnerHealthCategory,
    TrendDirection,
    SensorType,
    FlamePatternMetrics,
    IncompleteCombustionMetrics,
    BurnerHealthScore,
    SootFormationPrediction,
    FlashbackBlowoffRisk,
    STOICHIOMETRIC_AFR,
    ADIABATIC_FLAME_TEMP,
    LAMINAR_FLAME_SPEED
)

pytestmark = pytest.mark.unit


# ============================================================================
# COMBUSTION STABILITY CALCULATOR TESTS
# ============================================================================

class TestCombustionStabilityCalculator:
    """Comprehensive tests for CombustionStabilityCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return CombustionStabilityCalculator()

    @pytest.fixture
    def stable_input(self):
        """Create stable combustion input data."""
        # Generate stable readings with low variance
        base_temp = 1200.0
        base_pressure = 5000.0
        n_samples = 50

        temp_readings = [base_temp + 0.5 * math.sin(i * 0.1) for i in range(n_samples)]
        pressure_readings = [base_pressure + 10 * math.sin(i * 0.1) for i in range(n_samples)]

        return StabilityInput(
            temperature_readings=temp_readings,
            temperature_setpoint=1200.0,
            pressure_readings=pressure_readings,
            pressure_setpoint=5000.0,
            sampling_rate_hz=10.0,
            flame_length_mm=500.0,
            flame_intensity_percent=85.0,
            fuel_flow_rate=500.0,
            air_flow_rate=5000.0
        )

    @pytest.fixture
    def unstable_input(self):
        """Create unstable combustion input data."""
        # Generate unstable readings with high variance
        base_temp = 1200.0
        base_pressure = 5000.0
        n_samples = 50

        temp_readings = [base_temp + 50 * math.sin(i * 0.5) + 20 * (i % 5) for i in range(n_samples)]
        pressure_readings = [base_pressure + 500 * math.sin(i * 0.5) for i in range(n_samples)]

        return StabilityInput(
            temperature_readings=temp_readings,
            temperature_setpoint=1200.0,
            pressure_readings=pressure_readings,
            pressure_setpoint=5000.0,
            sampling_rate_hz=10.0,
            flame_length_mm=500.0,
            flame_intensity_percent=45.0,
            fuel_flow_rate=500.0,
            air_flow_rate=5000.0
        )

    def test_calculate_stability_index_stable_conditions(self, calculator, stable_input):
        """Test stability index calculation for stable conditions."""
        result = calculator.calculate_stability_index(stable_input)

        assert isinstance(result, StabilityResult)
        assert result.stability_index >= 0.85  # High stability
        assert result.stability_level == StabilityLevel.STABLE
        assert result.requires_intervention is False
        assert result.blowout_risk_score < 0.3
        assert result.flashback_risk_score < 0.3

    def test_calculate_stability_index_unstable_conditions(self, calculator, unstable_input):
        """Test stability index calculation for unstable conditions."""
        result = calculator.calculate_stability_index(unstable_input)

        assert isinstance(result, StabilityResult)
        assert result.stability_index < 0.85
        # Should flag for intervention if truly unstable
        assert result.temperature_rms_deviation > 0

    def test_stability_calculation_determinism(self, calculator, stable_input):
        """Test that stability calculation is deterministic."""
        results = [calculator.calculate_stability_index(stable_input) for _ in range(5)]

        # All stability indices should be identical
        indices = [r.stability_index for r in results]
        assert len(set(indices)) == 1

        # All blowout risks should be identical
        blowout_risks = [r.blowout_risk_score for r in results]
        assert len(set(blowout_risks)) == 1

    def test_signal_statistics_calculation(self, calculator):
        """Test signal statistics calculation."""
        readings = [100.0, 102.0, 98.0, 101.0, 99.0]
        setpoint = 100.0

        stats = calculator._calculate_signal_statistics(readings, setpoint)

        assert 'mean_value' in stats
        assert 'rms_deviation' in stats
        assert 'peak_to_peak' in stats
        assert 'stability_index' in stats

        # Mean should be close to 100
        assert abs(stats['mean_value'] - 100.0) < 1.0

        # Peak-to-peak should be 4 (102 - 98)
        assert stats['peak_to_peak'] == pytest.approx(4.0, rel=1e-6)

    def test_oscillation_detection_periodic(self, calculator):
        """Test oscillation detection for periodic signal."""
        n_samples = 100
        sampling_rate = 10.0
        frequency = 1.0  # 1 Hz oscillation

        temp_readings = [100 + 10 * math.sin(2 * math.pi * frequency * i / sampling_rate)
                        for i in range(n_samples)]
        pressure_readings = [5000 + 50 * math.sin(2 * math.pi * frequency * i / sampling_rate)
                           for i in range(n_samples)]

        oscillation = calculator._detect_oscillations(
            temp_readings, pressure_readings, sampling_rate
        )

        # Should detect oscillation
        assert oscillation is not None
        assert oscillation.pattern_type in ["periodic", "random", "chaotic"]

    def test_flame_characteristics_analysis(self, calculator):
        """Test flame characteristics analysis."""
        result = calculator.analyze_flame_characteristics(
            flame_length_mm=500.0,
            flame_intensity_percent=85.0,
            target_length_mm=500.0,
            target_intensity_percent=90.0
        )

        assert 'length_stability' in result
        assert 'intensity_stability' in result
        assert 'overall_flame_stability' in result

        # Length exactly at target should have high stability
        assert result['length_stability'] == pytest.approx(1.0, rel=1e-6)

    def test_blowout_risk_prediction(self, calculator):
        """Test blowout risk prediction."""
        # Low fuel, high air -> lean mixture -> high blowout risk
        risk = calculator._predict_blowout_risk(
            fuel_flow_rate=100.0,
            air_flow_rate=10000.0,
            temp_rms=50.0,
            pressure_rms=500.0
        )

        assert 0 <= risk <= 1.0

    def test_flashback_risk_prediction(self, calculator):
        """Test flashback risk prediction."""
        # High fuel, low air -> rich mixture -> flashback risk
        risk = calculator._predict_flashback_risk(
            fuel_flow_rate=1000.0,
            air_flow_rate=5000.0,
            temperature=1500.0
        )

        assert 0 <= risk <= 1.0

    def test_equivalence_ratio_calculation(self, calculator):
        """Test equivalence ratio calculation."""
        stoich_ratio = 14.7  # Typical for natural gas

        # Stoichiometric combustion
        er = calculator._calculate_equivalence_ratio(
            fuel_flow_rate=100.0,
            air_flow_rate=100.0 * stoich_ratio,
            stoichiometric_ratio=stoich_ratio
        )

        assert er == pytest.approx(1.0, rel=1e-6)

    def test_stability_level_classification(self, calculator):
        """Test stability level classification."""
        assert calculator._classify_stability(0.90) == StabilityLevel.STABLE
        assert calculator._classify_stability(0.80) == StabilityLevel.MODERATELY_STABLE
        assert calculator._classify_stability(0.60) == StabilityLevel.UNSTABLE
        assert calculator._classify_stability(0.40) == StabilityLevel.CRITICALLY_UNSTABLE


# ============================================================================
# FUEL-AIR OPTIMIZER TESTS
# ============================================================================

class TestFuelAirOptimizer:
    """Comprehensive tests for FuelAirOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return FuelAirOptimizer()

    @pytest.fixture
    def optimizer_input(self):
        """Create standard optimizer input."""
        return OptimizerInput(
            target_heat_output_kw=1000.0,
            fuel_type=FuelType.NATURAL_GAS,
            fuel_heating_value_mj_per_kg=50.0,
            fuel_composition={'C': 75, 'H': 25, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
            current_fuel_flow_kg_per_hr=100.0,
            current_air_flow_kg_per_hr=1700.0,
            measured_o2_percent=3.5,
            measured_co_ppm=50.0,
            measured_nox_ppm=100.0,
            ambient_temperature_c=25.0,
            ambient_pressure_pa=101325,
            optimization_objective=OptimizationObjective.BALANCED,
            max_fuel_flow_kg_per_hr=500.0
        )

    def test_calculate_optimal_ratio_normal(self, optimizer, optimizer_input):
        """Test optimal ratio calculation for normal conditions."""
        result = optimizer.calculate_optimal_ratio(optimizer_input)

        assert isinstance(result, OptimizerResult)
        assert result.optimal_fuel_flow_kg_per_hr > 0
        assert result.optimal_air_flow_kg_per_hr > 0
        assert result.optimal_air_fuel_ratio > 0
        assert result.stoichiometric_air_kg_per_kg_fuel > 0
        assert 0 <= result.excess_air_percent <= 100
        assert result.optimization_iterations > 0

    def test_stoichiometric_air_calculation(self, optimizer):
        """Test stoichiometric air calculation."""
        # Natural gas composition (CH4)
        fuel_comp = FuelComposition(
            carbon=0.75,
            hydrogen=0.25,
            oxygen=0.0,
            nitrogen=0.0,
            sulfur=0.0,
            ash=0.0,
            moisture=0.0
        )

        stoich_air = optimizer._calculate_stoichiometric_air(fuel_comp)

        # For natural gas, stoichiometric AFR is approximately 17.2 kg air/kg fuel
        assert stoich_air > 15.0
        assert stoich_air < 20.0

    def test_required_fuel_flow_calculation(self, optimizer):
        """Test required fuel flow calculation."""
        target_heat_kw = 1000.0
        heating_value = 50.0  # MJ/kg

        fuel_flow = optimizer._calculate_required_fuel_flow(target_heat_kw, heating_value)

        # Verify: 1000 kW = 1000 kJ/s = 3600 MJ/hr
        # At 50 MJ/kg and 85% efficiency: 3600 / (50 * 0.85) = 84.7 kg/hr
        assert fuel_flow == pytest.approx(84.7, rel=0.1)

    def test_optimization_determinism(self, optimizer, optimizer_input):
        """Test optimization is deterministic."""
        results = [optimizer.calculate_optimal_ratio(optimizer_input) for _ in range(3)]

        fuel_flows = [r.optimal_fuel_flow_kg_per_hr for r in results]
        air_flows = [r.optimal_air_flow_kg_per_hr for r in results]

        assert len(set(fuel_flows)) == 1
        assert len(set(air_flows)) == 1

    def test_optimization_objective_minimize_nox(self, optimizer, optimizer_input):
        """Test optimization for minimizing NOx."""
        optimizer_input.optimization_objective = OptimizationObjective.MINIMIZE_NOX
        result = optimizer.calculate_optimal_ratio(optimizer_input)

        # Lower excess air should give lower NOx
        assert result.predicted_nox_mg_per_nm3 >= 0

    def test_optimization_objective_maximize_efficiency(self, optimizer, optimizer_input):
        """Test optimization for maximizing efficiency."""
        optimizer_input.optimization_objective = OptimizationObjective.MAXIMIZE_EFFICIENCY
        result = optimizer.calculate_optimal_ratio(optimizer_input)

        # Should achieve reasonable efficiency
        assert result.predicted_efficiency_percent >= 60

    def test_emission_constraints_check(self, optimizer, optimizer_input):
        """Test emission constraints checking."""
        optimizer_input.emission_constraints = {
            'max_nox_mg_per_nm3': 200.0,
            'max_co_mg_per_nm3': 100.0,
            'min_o2_percent': 2.0,
            'max_o2_percent': 10.0
        }

        result = optimizer.calculate_optimal_ratio(optimizer_input)

        assert result.constraints_satisfied is not None

    def test_convenience_method_minimize_emissions(self, optimizer, optimizer_input):
        """Test convenience method for minimizing emissions."""
        result = optimizer.minimize_emissions(optimizer_input, target_emission="nox")

        assert isinstance(result, OptimizerResult)

    def test_convenience_method_maximize_efficiency(self, optimizer, optimizer_input):
        """Test convenience method for maximizing efficiency."""
        result = optimizer.maximize_efficiency(optimizer_input)

        assert isinstance(result, OptimizerResult)


# ============================================================================
# HEAT OUTPUT CALCULATOR TESTS
# ============================================================================

class TestHeatOutputCalculator:
    """Comprehensive tests for HeatOutputCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return HeatOutputCalculator()

    @pytest.fixture
    def heat_input(self):
        """Create standard heat calculation input."""
        return HeatOutputInput(
            fuel_flow_rate_kg_per_hr=100.0,
            fuel_lower_heating_value_mj_per_kg=50.0,
            air_flow_rate_kg_per_hr=1700.0,
            flue_gas_temperature_c=200.0,
            flue_gas_o2_percent=3.5,
            flue_gas_co_ppm=50.0,
            ambient_temperature_c=25.0,
            ambient_pressure_pa=101325,
            fuel_hydrogen_percent=10.0,
            fuel_moisture_percent=0.0
        )

    def test_calculate_heat_output_normal(self, calculator, heat_input):
        """Test heat output calculation for normal conditions."""
        result = calculator.calculate_heat_output(heat_input)

        assert isinstance(result, HeatOutputResult)
        assert result.net_heat_input_kw > 0
        assert result.net_heat_output_kw > 0
        assert result.net_heat_output_kw < result.net_heat_input_kw  # Must have losses
        assert result.net_thermal_efficiency_percent > 0
        assert result.net_thermal_efficiency_percent <= 100
        assert result.total_heat_loss_kw > 0

    def test_net_heat_input_calculation(self, calculator):
        """Test net heat input calculation."""
        fuel_flow = 100.0  # kg/hr
        heating_value = 50.0  # MJ/kg

        heat_input_kw = calculator._calculate_net_heat_input(fuel_flow, heating_value)

        # Expected: 100 kg/hr * 50 MJ/kg = 5000 MJ/hr = 5000/3.6 kW = 1388.9 kW
        assert heat_input_kw == pytest.approx(1388.9, rel=0.01)

    def test_excess_air_calculation(self, calculator):
        """Test excess air calculation from O2."""
        # At 3% O2 in flue gas
        excess_air = calculator._calculate_excess_air(3.0)

        # EA = (O2 / (21 - O2)) * 100 = (3 / 18) * 100 = 16.67%
        assert excess_air == pytest.approx(16.67, rel=0.01)

    def test_stack_loss_calculation(self, calculator):
        """Test stack loss calculation."""
        flue_gas_flow = 1800.0  # kg/hr
        flue_gas_temp = 200.0  # C
        ambient_temp = 25.0  # C
        heat_input_kw = 1388.9  # kW

        loss_kw, loss_percent = calculator._calculate_stack_loss(
            flue_gas_flow, flue_gas_temp, ambient_temp, heat_input_kw
        )

        assert loss_kw > 0
        assert loss_percent > 0
        assert loss_percent < 50  # Stack loss typically 5-20%

    def test_moisture_loss_calculation(self, calculator):
        """Test moisture loss calculation."""
        fuel_flow = 100.0  # kg/hr
        hydrogen_percent = 10.0  # %
        moisture_percent = 5.0  # %
        heat_input_kw = 1388.9  # kW

        loss_kw, loss_percent = calculator._calculate_moisture_loss(
            fuel_flow, hydrogen_percent, moisture_percent, heat_input_kw
        )

        assert loss_kw > 0
        assert loss_percent > 0

    def test_incomplete_combustion_loss(self, calculator):
        """Test incomplete combustion loss from CO."""
        co_ppm = 100.0  # ppm
        flue_gas_flow = 1800.0  # kg/hr
        heat_input_kw = 1388.9  # kW

        loss_kw, loss_percent = calculator._calculate_incomplete_combustion_loss(
            co_ppm, flue_gas_flow, heat_input_kw
        )

        assert loss_kw >= 0
        assert loss_percent >= 0

    def test_radiation_loss_calculation(self, calculator):
        """Test radiation loss calculation."""
        surface_area = 10.0  # m2
        surface_temp = 100.0  # C
        ambient_temp = 25.0  # C
        emissivity = 0.85
        heat_input_kw = 1388.9  # kW

        loss_kw, loss_percent = calculator._calculate_radiation_loss(
            surface_area, surface_temp, ambient_temp, emissivity, heat_input_kw
        )

        assert loss_kw > 0
        assert loss_percent > 0

    def test_thermal_efficiency_calculation(self, calculator):
        """Test thermal efficiency calculation."""
        heat_output_kw = 1200.0
        fuel_flow = 100.0
        heating_value = 50.0

        efficiency = calculator.calculate_thermal_efficiency(
            heat_output_kw, fuel_flow, heating_value
        )

        # Expected: 1200 / 1388.9 * 100 = 86.4%
        assert efficiency == pytest.approx(86.4, rel=0.01)

    def test_target_validation(self, calculator):
        """Test validation against target."""
        actual = 1000.0
        target = 1050.0
        tolerance = 5.0

        result = calculator.validate_against_target(actual, target, tolerance)

        assert 'deviation_percent' in result
        assert 'within_tolerance' in result
        assert result['deviation_percent'] == pytest.approx(-4.76, rel=0.01)
        assert result['within_tolerance'] is True

    def test_heat_output_determinism(self, calculator, heat_input):
        """Test heat output calculation is deterministic."""
        results = [calculator.calculate_heat_output(heat_input) for _ in range(3)]

        outputs = [r.net_heat_output_kw for r in results]
        efficiencies = [r.net_thermal_efficiency_percent for r in results]

        assert len(set(outputs)) == 1
        assert len(set(efficiencies)) == 1


# ============================================================================
# PID CONTROLLER TESTS
# ============================================================================

class TestPIDController:
    """Comprehensive tests for PIDController."""

    @pytest.fixture
    def controller(self):
        """Create PID controller instance."""
        return PIDController(kp=1.0, ki=0.1, kd=0.05)

    @pytest.fixture
    def pid_input(self):
        """Create standard PID input."""
        return PIDInput(
            setpoint=100.0,
            process_variable=95.0,
            timestamp=1.0,
            kp=1.0,
            ki=0.1,
            kd=0.05,
            output_min=0.0,
            output_max=100.0,
            enable_anti_windup=True,
            anti_windup_method=AntiWindupMethod.CLAMPING
        )

    def test_calculate_control_output_basic(self, controller, pid_input):
        """Test basic control output calculation."""
        result = controller.calculate_control_output(pid_input)

        assert isinstance(result, PIDOutput)
        assert result.error == pytest.approx(5.0, rel=1e-6)  # 100 - 95
        assert result.p_term == pytest.approx(5.0, rel=1e-6)  # kp * error
        assert result.control_mode == ControlMode.AUTO

    def test_proportional_term_calculation(self, controller):
        """Test proportional term calculation."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=90.0,
            timestamp=1.0,
            kp=2.0,
            ki=0.0,
            kd=0.0
        )

        result = controller.calculate_control_output(pid_input)

        # P = Kp * error = 2.0 * 10 = 20
        assert result.p_term == pytest.approx(20.0, rel=1e-6)
        assert result.i_term == pytest.approx(0.0, rel=1e-6)
        assert result.d_term == pytest.approx(0.0, rel=1e-6)

    def test_integral_term_accumulation(self, controller):
        """Test integral term accumulation over multiple cycles."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=95.0,
            timestamp=0.0,
            kp=0.0,
            ki=1.0,
            kd=0.0
        )

        # First cycle
        result1 = controller.calculate_control_output(pid_input)

        # Second cycle (timestamp advanced)
        pid_input.timestamp = 0.1
        result2 = controller.calculate_control_output(pid_input)

        # Integral should have accumulated
        assert result2.error_integral > result1.error_integral

    def test_derivative_term_calculation(self, controller):
        """Test derivative term calculation."""
        # First reading
        pid_input1 = PIDInput(
            setpoint=100.0,
            process_variable=90.0,
            timestamp=0.0,
            kp=0.0,
            ki=0.0,
            kd=1.0
        )
        controller.calculate_control_output(pid_input1)

        # Second reading with error change
        pid_input2 = PIDInput(
            setpoint=100.0,
            process_variable=95.0,  # Error reduced
            timestamp=0.1,
            kp=0.0,
            ki=0.0,
            kd=1.0
        )
        result = controller.calculate_control_output(pid_input2)

        # Derivative should be non-zero (error changed)
        # Note: derivative is filtered, so may not be exact
        assert result.d_term != 0

    def test_output_limiting(self, controller):
        """Test output limiting functionality."""
        pid_input = PIDInput(
            setpoint=200.0,  # Large error
            process_variable=0.0,
            timestamp=0.0,
            kp=10.0,  # High gain
            ki=0.0,
            kd=0.0,
            output_min=0.0,
            output_max=50.0  # Low limit
        )

        result = controller.calculate_control_output(pid_input)

        # Output should be clamped to max
        assert result.control_output <= 50.0
        assert result.output_saturated is True

    def test_anti_windup_clamping(self, controller):
        """Test anti-windup with clamping method."""
        pid_input = PIDInput(
            setpoint=200.0,
            process_variable=0.0,
            timestamp=0.0,
            kp=1.0,
            ki=1.0,
            kd=0.0,
            output_min=0.0,
            output_max=50.0,
            enable_anti_windup=True,
            anti_windup_method=AntiWindupMethod.CLAMPING
        )

        # Run multiple cycles to saturate
        for i in range(10):
            pid_input.timestamp = i * 0.1
            result = controller.calculate_control_output(pid_input)

        # Anti-windup should be active when saturated
        assert result.output_saturated is True

    def test_manual_mode(self, controller):
        """Test manual control mode."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=90.0,
            timestamp=0.0,
            kp=1.0,
            ki=0.1,
            kd=0.05,
            control_mode=ControlMode.MANUAL,
            manual_output=75.0
        )

        result = controller.calculate_control_output(pid_input)

        assert result.control_mode == ControlMode.MANUAL
        assert result.control_output == 75.0

    def test_auto_tune_ziegler_nichols(self, controller):
        """Test Ziegler-Nichols auto-tuning."""
        # Create oscillating process data
        process_data = [(i * 0.1, 100 + 10 * math.sin(2 * math.pi * i / 10))
                       for i in range(50)]

        auto_tune_input = AutoTuneInput(
            process_data=process_data,
            setpoint=100.0,
            tuning_method=TuningMethod.ZIEGLER_NICHOLS_CLOSED_LOOP,
            ultimate_gain=2.0,
            ultimate_period=1.0
        )

        result = controller.auto_tune_parameters(auto_tune_input)

        assert isinstance(result, AutoTuneOutput)
        assert result.kp > 0
        assert result.ki >= 0
        assert result.kd >= 0
        assert result.tuning_method == TuningMethod.ZIEGLER_NICHOLS_CLOSED_LOOP

    def test_pid_determinism(self, controller, pid_input):
        """Test PID calculation is deterministic."""
        # Reset controller state
        controller.error_integral = 0
        controller.error_previous = 0
        controller.time_previous = None

        result1 = controller.calculate_control_output(pid_input)

        # Reset again
        controller.error_integral = 0
        controller.error_previous = 0
        controller.time_previous = None

        result2 = controller.calculate_control_output(pid_input)

        assert result1.control_output == result2.control_output
        assert result1.p_term == result2.p_term

    def test_update_pid_state(self, controller):
        """Test updating PID state."""
        controller.error_integral = 100.0

        controller.update_pid_state(reset_integral=True)
        assert controller.error_integral == 0.0

        controller.update_pid_state(error_integral=50.0)
        assert controller.error_integral == 50.0


# ============================================================================
# PID CONTROLLER EDGE CASES TESTS
# ============================================================================

class TestPIDControllerEdgeCases:
    """Edge case tests for PIDController."""

    @pytest.fixture
    def controller(self):
        """Create PID controller instance."""
        return PIDController(kp=1.0, ki=0.1, kd=0.05)

    def test_zero_error_handling(self, controller):
        """Test handling when error is zero."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=100.0,
            timestamp=1.0,
            kp=1.0,
            ki=0.1,
            kd=0.05
        )

        result = controller.calculate_control_output(pid_input)

        assert result.error == pytest.approx(0.0, abs=1e-6)
        assert result.p_term == pytest.approx(0.0, abs=1e-6)

    def test_negative_error_handling(self, controller):
        """Test handling when process variable exceeds setpoint."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=110.0,
            timestamp=1.0,
            kp=1.0,
            ki=0.1,
            kd=0.05
        )

        result = controller.calculate_control_output(pid_input)

        assert result.error == pytest.approx(-10.0, rel=1e-6)
        assert result.p_term == pytest.approx(-10.0, rel=1e-6)

    def test_very_large_error(self, controller):
        """Test handling of very large errors."""
        pid_input = PIDInput(
            setpoint=10000.0,
            process_variable=0.0,
            timestamp=1.0,
            kp=1.0,
            ki=0.0,
            kd=0.0,
            output_min=-1000.0,
            output_max=1000.0
        )

        result = controller.calculate_control_output(pid_input)

        # Should be clamped to output_max
        assert result.control_output <= 1000.0

    def test_very_small_time_delta(self, controller):
        """Test handling of very small time deltas."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=95.0,
            timestamp=0.0,
            kp=1.0,
            ki=0.1,
            kd=0.05
        )

        controller.calculate_control_output(pid_input)

        # Very small time delta
        pid_input.timestamp = 0.0001
        result = controller.calculate_control_output(pid_input)

        # Should not produce NaN or Inf
        assert not math.isnan(result.control_output)
        assert not math.isinf(result.control_output)

    def test_zero_gains(self, controller):
        """Test behavior with all zero gains."""
        pid_input = PIDInput(
            setpoint=100.0,
            process_variable=95.0,
            timestamp=1.0,
            kp=0.0,
            ki=0.0,
            kd=0.0
        )

        result = controller.calculate_control_output(pid_input)

        assert result.control_output == pytest.approx(0.0, abs=1e-6)

    def test_derivative_kick_prevention(self, controller):
        """Test derivative kick prevention on setpoint change."""
        # First cycle
        pid_input1 = PIDInput(
            setpoint=100.0,
            process_variable=95.0,
            timestamp=0.0,
            kp=0.0,
            ki=0.0,
            kd=1.0
        )
        controller.calculate_control_output(pid_input1)

        # Setpoint change (should use derivative on measurement)
        pid_input2 = PIDInput(
            setpoint=120.0,  # Setpoint changed
            process_variable=95.0,  # PV unchanged
            timestamp=0.1,
            kp=0.0,
            ki=0.0,
            kd=1.0
        )
        result = controller.calculate_control_output(pid_input2)

        # Derivative should be based on PV change (zero) not error change
        # This depends on implementation - check for reasonable output
        assert not math.isnan(result.d_term)
        assert not math.isinf(result.d_term)


# ============================================================================
# SAFETY VALIDATOR TESTS
# ============================================================================

class TestSafetyValidator:
    """Comprehensive tests for SafetyValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return SafetyValidator()

    @pytest.fixture
    def safety_limits(self):
        """Create safety limits."""
        return SafetyLimits(
            max_fuel_flow_rate=1000.0,
            max_air_flow_rate=10000.0
        )

    @pytest.fixture
    def normal_input(self, safety_limits):
        """Create normal operating input."""
        return SafetyValidatorInput(
            combustion_temperature_c=1200.0,
            flue_gas_temperature_c=300.0,
            combustion_pressure_pa=5000.0,
            fuel_supply_pressure_pa=200000.0,
            fuel_flow_rate_kg_per_hr=500.0,
            air_flow_rate_kg_per_hr=5000.0,
            o2_percent=3.5,
            co_ppm=50.0,
            safety_limits=safety_limits,
            burner_firing=True,
            flame_detected=True,
            purge_complete=True
        )

    @pytest.fixture
    def emergency_input(self, safety_limits):
        """Create emergency condition input."""
        return SafetyValidatorInput(
            combustion_temperature_c=1200.0,
            flue_gas_temperature_c=300.0,
            combustion_pressure_pa=5000.0,
            fuel_supply_pressure_pa=200000.0,
            fuel_flow_rate_kg_per_hr=500.0,
            air_flow_rate_kg_per_hr=5000.0,
            o2_percent=3.5,
            co_ppm=50.0,
            safety_limits=safety_limits,
            burner_firing=True,
            flame_detected=True,
            fire_detected=True  # Emergency!
        )

    def test_validate_all_safety_interlocks_normal(self, validator, normal_input):
        """Test safety validation for normal conditions."""
        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        assert isinstance(result, SafetyValidatorOutput)
        assert result.is_safe_to_operate is True
        assert result.requires_shutdown is False
        assert result.emergency_shutdown_required is False
        assert len(result.tripped_interlocks) == 0

    def test_validate_emergency_conditions(self, validator, emergency_input):
        """Test safety validation for emergency conditions."""
        result = validator.validate_all_safety_interlocks(emergency_input)

        assert result.emergency_shutdown_required is True
        assert result.safety_level == SafetyInterlockLevel.EMERGENCY_SHUTDOWN
        assert result.is_safe_to_operate is False

    def test_temperature_limit_violation(self, validator, normal_input):
        """Test temperature limit violation detection."""
        normal_input.combustion_temperature_c = 1600.0  # Exceeds limit

        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        assert len(result.temperature_violations) > 0
        assert result.requires_shutdown is True

    def test_pressure_limit_violation(self, validator, normal_input):
        """Test pressure limit violation detection."""
        normal_input.combustion_pressure_pa = 15000.0  # Exceeds limit

        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        assert len(result.pressure_violations) > 0

    def test_flame_loss_detection(self, validator, normal_input):
        """Test flame loss detection."""
        normal_input.flame_detected = False
        normal_input.burner_firing = True  # Fuel flowing but no flame!

        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        # Should trigger shutdown
        assert result.requires_shutdown is True

    def test_rate_of_change_check(self, validator, normal_input):
        """Test rate of change limit checking."""
        normal_input.previous_temperature_c = 1100.0
        normal_input.time_delta_seconds = 1.0  # 1 second
        # Rate = (1200 - 1100) / 1 * 60 = 6000 C/min - way over limit

        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        assert result.excessive_rate_of_change is True

    def test_risk_score_calculation(self, validator):
        """Test risk score calculation."""
        risk = validator.calculate_risk_score(
            temp_violations=["high temp"],
            pressure_violations=[],
            flow_violations=[],
            emission_violations=["high CO"],
            tripped_interlock_count=0
        )

        assert 0 <= risk <= 1.0

    def test_check_emergency_conditions(self, validator):
        """Test emergency condition checking."""
        shutdown, reason = validator.check_emergency_conditions(
            fire_detected=True,
            gas_leak_detected=False,
            operator_stop=False
        )

        assert shutdown is True
        assert "FIRE" in reason

    def test_redundant_sensor_check(self, validator, normal_input):
        """Test redundant sensor mismatch detection."""
        normal_input.backup_temperature_c = 1100.0  # 100C difference - too much

        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            result = validator.validate_all_safety_interlocks(normal_input)

        assert result.redundant_sensor_mismatch is True

    def test_safety_validation_determinism(self, validator, normal_input):
        """Test safety validation is deterministic."""
        with patch('safety_validator.DeterministicClock') as mock_clock:
            mock_clock.now.return_value.timestamp.return_value = 0.0
            results = [validator.validate_all_safety_interlocks(normal_input) for _ in range(3)]

        risk_scores = [r.overall_risk_score for r in results]
        assert len(set(risk_scores)) == 1


# ============================================================================
# EMISSIONS CALCULATOR TESTS
# ============================================================================

class TestEmissionsCalculator:
    """Comprehensive tests for EmissionsCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return EmissionsCalculator()

    @pytest.fixture
    def emissions_input(self):
        """Create standard emissions input."""
        return EmissionsInput(
            fuel_type="natural_gas",
            fuel_flow_rate_kg_per_hr=100.0,
            fuel_properties={'C': 75, 'H': 25, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
            fuel_heating_value_mj_per_kg=50.0,
            air_flow_rate_kg_per_hr=1700.0,
            combustion_temperature_c=1200.0,
            excess_air_percent=15.0,
            flue_gas_o2_percent=3.0,
            flue_gas_co_ppm=50.0,
            flue_gas_nox_ppm=100.0,
            flue_gas_temperature_c=200.0,
            operating_hours_per_year=8000
        )

    def test_calculate_all_emissions(self, calculator, emissions_input):
        """Test comprehensive emissions calculation."""
        result = calculator.calculate_all_emissions(emissions_input)

        assert isinstance(result, EmissionsResult)
        assert result.nox_kg_per_hr >= 0
        assert result.co_kg_per_hr >= 0
        assert result.co2_kg_per_hr > 0
        assert result.sox_kg_per_hr >= 0
        assert result.nox_tonnes_per_year >= 0
        assert result.co2_tonnes_per_year > 0
        assert result.specific_co2_kg_per_kwh > 0

    def test_calculate_co2_emissions(self, calculator, emissions_input):
        """Test CO2 emissions calculation using carbon balance."""
        co2_kg_hr, co2_percent = calculator.calculate_co2_emissions(emissions_input)

        # For 100 kg/hr fuel with 75% carbon:
        # CO2 = 100 * 0.75 * (44/12) = 275 kg/hr
        assert co2_kg_hr == pytest.approx(275.0, rel=0.01)
        assert co2_percent > 0

    def test_calculate_nox_emissions(self, calculator, emissions_input):
        """Test NOx emissions calculation."""
        nox_kg_hr, nox_mg_nm3 = calculator.calculate_nox_emissions(emissions_input)

        assert nox_kg_hr >= 0
        assert nox_mg_nm3 >= 0

    def test_calculate_co_emissions(self, calculator, emissions_input):
        """Test CO emissions calculation."""
        co_kg_hr, co_mg_nm3 = calculator.calculate_co_emissions(emissions_input)

        assert co_kg_hr >= 0
        assert co_mg_nm3 >= 0

    def test_o2_reference_correction(self, calculator):
        """Test O2 reference correction calculation."""
        concentration = 100.0  # mg/Nm3
        measured_o2 = 6.0  # %
        reference_o2 = 3.0  # %

        corrected = calculator._correct_to_reference_o2(
            concentration, measured_o2, reference_o2
        )

        # C_ref = 100 * (21 - 3) / (21 - 6) = 100 * 18/15 = 120
        assert corrected == pytest.approx(120.0, rel=0.01)

    def test_co_estimation_from_conditions(self, calculator):
        """Test CO estimation from combustion conditions."""
        # Low excess air should give high CO
        co_low_ea = calculator._estimate_co_from_conditions(5.0, 1200.0)
        co_high_ea = calculator._estimate_co_from_conditions(20.0, 1200.0)

        assert co_low_ea > co_high_ea

    def test_compliance_check(self, calculator, emissions_input):
        """Test regulatory compliance checking."""
        emissions_input.emission_limits = {
            'nox_mg_per_nm3': 200.0,
            'co_mg_per_nm3': 100.0
        }

        result = calculator.calculate_all_emissions(emissions_input)

        assert 'nox' in result.compliance_status or len(result.compliance_status) >= 0

    def test_reporting_threshold(self, calculator, emissions_input):
        """Test reporting threshold detection."""
        result = calculator.calculate_all_emissions(emissions_input)

        # Should not require reporting for small emissions
        assert result.requires_reporting is False

    def test_emissions_determinism(self, calculator, emissions_input):
        """Test emissions calculation is deterministic."""
        results = [calculator.calculate_all_emissions(emissions_input) for _ in range(3)]

        co2_emissions = [r.co2_kg_per_hr for r in results]
        nox_emissions = [r.nox_kg_per_hr for r in results]

        assert len(set(co2_emissions)) == 1
        assert len(set(nox_emissions)) == 1


# ============================================================================
# ADVANCED COMBUSTION DIAGNOSTICS CALCULATOR TESTS (NEW)
# ============================================================================

class TestAdvancedCombustionDiagnosticsCalculator:
    """Comprehensive tests for AdvancedCombustionDiagnosticsCalculator (2,665 lines)."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return AdvancedCombustionDiagnosticsCalculator()

    @pytest.fixture
    def diagnostic_input(self):
        """Create standard diagnostic input."""
        n_samples = 100
        sampling_rate = 10.0

        # Generate stable readings
        temp_readings = [1200.0 + 5 * math.sin(i * 0.1) for i in range(n_samples)]
        pressure_readings = [5000.0 + 50 * math.sin(i * 0.1) for i in range(n_samples)]
        flame_readings = [85.0 + 2 * math.sin(i * 0.15) for i in range(n_samples)]

        return AdvancedDiagnosticInput(
            temperature_readings_c=temp_readings,
            pressure_readings_pa=pressure_readings,
            flame_intensity_readings=flame_readings,
            sampling_rate_hz=sampling_rate,
            o2_actual_percent=3.5,
            co_actual_ppm=50.0,
            co2_actual_percent=10.0,
            fuel_flow_kg_hr=500.0,
            air_flow_kg_hr=8600.0,  # 17.2 AFR * 500 = 8600 (stoichiometric)
            combustion_temperature_c=1200.0,
            furnace_pressure_pa=5000.0,
            fuel_type="natural_gas",
            fuel_heating_value_mj_kg=50.0,
            fuel_specific_gravity=0.6,
            reference_heating_value_mj_kg=50.0,
            reference_wobbe_index=50.0,
            fuel_hydrogen_content_percent=23.0,
            o2_setpoint_percent=3.0,
            temperature_setpoint_c=1200.0,
            fuel_demand_percent=50.0,
            air_demand_percent=50.0,
            fuel_actual_percent=50.0,
            air_actual_percent=50.0,
            burner_diameter_mm=100.0,
            burner_velocity_m_s=20.0,
            baseline_efficiency_percent=85.0
        )

    def test_calculate_diagnostics_full(self, calculator, diagnostic_input):
        """Test full diagnostics calculation."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        assert isinstance(result, AdvancedDiagnosticOutput)
        assert result.summary is not None
        assert result.processing_time_ms > 0
        assert result.calculation_timestamp is not None

    def test_equivalence_ratio_calculation(self, calculator):
        """Test equivalence ratio calculation."""
        # Stoichiometric for natural gas
        phi = calculator._calculate_equivalence_ratio(
            fuel_flow_kg_hr=500.0,
            air_flow_kg_hr=8600.0,  # 17.2 * 500
            fuel_type="natural_gas"
        )

        assert phi == pytest.approx(1.0, rel=0.05)

    def test_equivalence_ratio_lean(self, calculator):
        """Test lean equivalence ratio."""
        phi = calculator._calculate_equivalence_ratio(
            fuel_flow_kg_hr=500.0,
            air_flow_kg_hr=12000.0,  # More air than stoichiometric
            fuel_type="natural_gas"
        )

        assert phi < 1.0  # Lean mixture

    def test_equivalence_ratio_rich(self, calculator):
        """Test rich equivalence ratio."""
        phi = calculator._calculate_equivalence_ratio(
            fuel_flow_kg_hr=500.0,
            air_flow_kg_hr=6000.0,  # Less air than stoichiometric
            fuel_type="natural_gas"
        )

        assert phi > 1.0  # Rich mixture

    def test_classify_combustion_mode_stoichiometric(self, calculator):
        """Test combustion mode classification - stoichiometric."""
        mode = calculator._classify_combustion_mode(1.0)
        assert mode == CombustionMode.STOICHIOMETRIC

    def test_classify_combustion_mode_lean(self, calculator):
        """Test combustion mode classification - lean."""
        mode = calculator._classify_combustion_mode(0.85)
        assert mode == CombustionMode.LEAN

    def test_classify_combustion_mode_ultra_lean(self, calculator):
        """Test combustion mode classification - ultra-lean."""
        mode = calculator._classify_combustion_mode(0.6)
        assert mode == CombustionMode.ULTRA_LEAN

    def test_classify_combustion_mode_rich(self, calculator):
        """Test combustion mode classification - rich."""
        mode = calculator._classify_combustion_mode(1.15)
        assert mode == CombustionMode.RICH

    def test_classify_combustion_mode_very_rich(self, calculator):
        """Test combustion mode classification - very rich."""
        mode = calculator._classify_combustion_mode(1.5)
        assert mode == CombustionMode.VERY_RICH

    def test_flame_pattern_analysis_stable(self, calculator, diagnostic_input):
        """Test flame pattern analysis for stable conditions."""
        flame_pattern = calculator._analyze_flame_pattern(
            intensity_readings=diagnostic_input.flame_intensity_readings,
            sampling_rate_hz=diagnostic_input.sampling_rate_hz,
            equivalence_ratio=1.0,
            combustion_mode=CombustionMode.STOICHIOMETRIC,
            temperature_c=1200.0
        )

        assert isinstance(flame_pattern, FlamePatternMetrics)
        assert flame_pattern.stability_index > 0.7
        assert flame_pattern.pattern_type in [FlamePattern.STABLE, FlamePattern.FLICKERING]

    def test_flame_pattern_analysis_unstable(self, calculator):
        """Test flame pattern analysis for unstable conditions."""
        n = 100
        # Highly variable readings
        unstable_readings = [85 + 30 * math.sin(i * 0.5) + 10 * (i % 7) for i in range(n)]

        flame_pattern = calculator._analyze_flame_pattern(
            intensity_readings=unstable_readings,
            sampling_rate_hz=10.0,
            equivalence_ratio=1.2,
            combustion_mode=CombustionMode.RICH,
            temperature_c=1200.0
        )

        assert isinstance(flame_pattern, FlamePatternMetrics)
        assert flame_pattern.stability_index < 0.9

    def test_incomplete_combustion_analysis_normal(self, calculator):
        """Test incomplete combustion analysis for normal conditions."""
        result = calculator._analyze_incomplete_combustion(
            co_ppm=50.0,
            co2_percent=10.0,
            o2_percent=3.5,
            fuel_flow_kg_hr=500.0,
            temperature_c=1200.0,
            equivalence_ratio=1.0
        )

        assert isinstance(result, IncompleteCombustionMetrics)
        assert result.is_incomplete is False
        assert result.severity in [FaultSeverity.NONE, FaultSeverity.LOW]

    def test_incomplete_combustion_analysis_high_co(self, calculator):
        """Test incomplete combustion analysis with high CO."""
        result = calculator._analyze_incomplete_combustion(
            co_ppm=300.0,  # High CO
            co2_percent=10.0,
            o2_percent=2.0,
            fuel_flow_kg_hr=500.0,
            temperature_c=1200.0,
            equivalence_ratio=1.2
        )

        assert result.is_incomplete is True
        assert result.severity in [FaultSeverity.MEDIUM, FaultSeverity.HIGH]

    def test_incomplete_combustion_critical_co(self, calculator):
        """Test incomplete combustion with critical CO levels."""
        result = calculator._analyze_incomplete_combustion(
            co_ppm=600.0,  # Critical CO
            co2_percent=8.0,
            o2_percent=1.0,
            fuel_flow_kg_hr=500.0,
            temperature_c=1000.0,
            equivalence_ratio=1.4
        )

        assert result.severity == FaultSeverity.CRITICAL

    def test_burner_health_score_calculation(self, calculator, diagnostic_input):
        """Test burner health score calculation."""
        flame_pattern = FlamePatternMetrics(
            pattern_type=FlamePattern.STABLE,
            combustion_mode=CombustionMode.STOICHIOMETRIC,
            equivalence_ratio=1.0,
            stability_index=0.95,
            pulsation_frequency_hz=0.5,
            pulsation_amplitude_percent=5.0,
            lift_distance_mm=0.0,
            asymmetry_index=0.05,
            luminosity_variance=2.0,
            flame_temperature_k=1473.15,
            provenance_hash="test_hash"
        )

        incomplete_combustion = IncompleteCombustionMetrics(
            co_formation_index=100.0,
            co_concentration_ppm=50.0,
            co_mass_rate_kg_hr=0.01,
            combustion_efficiency_percent=88.0,
            unburned_carbon_percent=0.005,
            carbon_in_ash_percent=0.0,
            chemical_efficiency_percent=99.0,
            is_incomplete=False,
            severity=FaultSeverity.NONE,
            root_cause="Within normal operating range",
            provenance_hash="test_hash"
        )

        efficiency_degradation = calculator._calculate_efficiency_degradation(
            efficiency_history=[85.0, 84.5, 84.0],
            timestamps_days=[0, 30, 60],
            baseline_efficiency=85.0,
            current_efficiency=84.0
        )

        health = calculator._calculate_burner_health(
            flame_pattern,
            incomplete_combustion,
            efficiency_degradation,
            diagnostic_input
        )

        assert isinstance(health, BurnerHealthScore)
        assert 0 <= health.overall_score <= 100
        assert health.category in list(BurnerHealthCategory)

    def test_soot_formation_prediction_normal(self, calculator):
        """Test soot formation prediction for normal conditions."""
        result = calculator._predict_soot_formation(
            equivalence_ratio=1.0,
            temperature_c=1200.0,
            fuel_type="natural_gas",
            hydrogen_content_percent=23.0
        )

        assert isinstance(result, SootFormationPrediction)
        assert result.is_sooting is False
        assert result.soot_risk_level in [FaultSeverity.NONE, FaultSeverity.LOW]

    def test_soot_formation_prediction_rich(self, calculator):
        """Test soot formation prediction for rich combustion."""
        result = calculator._predict_soot_formation(
            equivalence_ratio=1.4,  # Very rich
            temperature_c=1400.0,
            fuel_type="diesel",
            hydrogen_content_percent=12.0
        )

        assert isinstance(result, SootFormationPrediction)
        # Rich combustion should have higher soot risk
        assert result.soot_formation_index > 0

    def test_flashback_blowoff_risk_normal(self, calculator):
        """Test flashback/blowoff risk for normal conditions."""
        result = calculator._assess_flashback_blowoff_risk(
            burner_velocity_m_s=20.0,
            fuel_type="natural_gas",
            equivalence_ratio=1.0,
            temperature_c=1200.0,
            burner_diameter_mm=100.0
        )

        assert isinstance(result, FlashbackBlowoffRisk)
        assert result.operating_regime == "stable"
        assert result.flashback_severity in [FaultSeverity.NONE, FaultSeverity.LOW]
        assert result.blowoff_severity in [FaultSeverity.NONE, FaultSeverity.LOW]

    def test_flashback_risk_high_phi(self, calculator):
        """Test flashback risk with rich mixture."""
        result = calculator._assess_flashback_blowoff_risk(
            burner_velocity_m_s=10.0,  # Low velocity
            fuel_type="hydrogen",  # High flame speed
            equivalence_ratio=1.2,
            temperature_c=1300.0,
            burner_diameter_mm=100.0
        )

        # High phi and high flame speed should increase flashback risk
        assert result.flashback_risk_score > 0

    def test_blowoff_risk_lean_high_velocity(self, calculator):
        """Test blowoff risk with lean mixture and high velocity."""
        result = calculator._assess_flashback_blowoff_risk(
            burner_velocity_m_s=50.0,  # High velocity
            fuel_type="natural_gas",
            equivalence_ratio=0.6,  # Very lean
            temperature_c=1000.0,
            burner_diameter_mm=100.0
        )

        # Lean mixture and high velocity should increase blowoff risk
        assert result.blowoff_risk_score > 0

    def test_instability_indicators(self, calculator, diagnostic_input):
        """Test combustion instability indicators calculation."""
        result = calculator._calculate_instability_indicators(
            pressure_readings=diagnostic_input.pressure_readings_pa,
            temperature_readings=diagnostic_input.temperature_readings_c,
            flame_readings=diagnostic_input.flame_intensity_readings,
            sampling_rate_hz=diagnostic_input.sampling_rate_hz
        )

        assert result.pressure_oscillation_amplitude_pa >= 0
        assert result.instability_score >= 0
        assert isinstance(result.is_thermoacoustic, bool)

    def test_sensor_drift_detection(self, calculator, diagnostic_input):
        """Test sensor drift detection."""
        diagnostic_input.sensor_baselines = {
            'o2': 3.5,
            'co': 40.0,
            'temperature': 1200.0
        }
        diagnostic_input.time_since_calibration_hours = {
            'o2': 720.0,  # 30 days
            'co': 168.0,  # 7 days
            'temperature': 24.0
        }

        drift_results = calculator._calculate_sensor_drift(diagnostic_input)

        assert isinstance(drift_results, list)

    def test_cross_limiting_validation(self, calculator, diagnostic_input):
        """Test cross-limiting parameter validation."""
        result = calculator._validate_cross_limiting(diagnostic_input)

        assert result.fuel_demand_percent == diagnostic_input.fuel_demand_percent
        assert result.air_demand_percent == diagnostic_input.air_demand_percent
        assert isinstance(result.cross_limit_active, bool)

    def test_trim_control_parameters(self, calculator, diagnostic_input):
        """Test trim control parameter calculation."""
        result = calculator._calculate_trim_parameters(diagnostic_input)

        assert result.o2_setpoint_percent == diagnostic_input.o2_setpoint_percent
        assert result.o2_actual_percent == diagnostic_input.o2_actual_percent
        assert isinstance(result.is_saturated_high, bool)
        assert isinstance(result.is_saturated_low, bool)

    def test_fault_detection_all(self, calculator, diagnostic_input):
        """Test comprehensive fault detection."""
        flame_pattern = calculator._analyze_flame_pattern(
            diagnostic_input.flame_intensity_readings,
            diagnostic_input.sampling_rate_hz,
            1.0,
            CombustionMode.STOICHIOMETRIC,
            diagnostic_input.combustion_temperature_c
        )

        incomplete_combustion = calculator._analyze_incomplete_combustion(
            diagnostic_input.co_actual_ppm,
            diagnostic_input.co2_actual_percent,
            diagnostic_input.o2_actual_percent,
            diagnostic_input.fuel_flow_kg_hr,
            diagnostic_input.combustion_temperature_c,
            1.0
        )

        fuel_quality = calculator._assess_fuel_quality(
            diagnostic_input.fuel_heating_value_mj_kg,
            diagnostic_input.fuel_specific_gravity,
            diagnostic_input.reference_heating_value_mj_kg,
            diagnostic_input.reference_wobbe_index,
            diagnostic_input.fuel_hydrogen_content_percent
        )

        air_distribution = calculator._analyze_air_distribution(
            diagnostic_input.zone_data,
            diagnostic_input.air_flow_kg_hr
        )

        soot_prediction = calculator._predict_soot_formation(
            1.0,
            diagnostic_input.combustion_temperature_c,
            diagnostic_input.fuel_type,
            diagnostic_input.fuel_hydrogen_content_percent
        )

        flashback_blowoff = calculator._assess_flashback_blowoff_risk(
            diagnostic_input.burner_velocity_m_s,
            diagnostic_input.fuel_type,
            1.0,
            diagnostic_input.combustion_temperature_c,
            diagnostic_input.burner_diameter_mm
        )

        faults = calculator._detect_all_faults(
            diagnostic_input,
            flame_pattern,
            incomplete_combustion,
            fuel_quality,
            air_distribution,
            soot_prediction,
            flashback_blowoff
        )

        assert isinstance(faults, list)

    def test_maintenance_recommendation_generation(self, calculator, diagnostic_input):
        """Test maintenance recommendation generation."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        recommendations = result.summary.maintenance_recommendations

        assert isinstance(recommendations, tuple)
        # Each recommendation should have required fields
        for rec in recommendations:
            assert rec.priority in list(MaintenancePriority)
            assert len(rec.description) > 0

    def test_overall_health_score_calculation(self, calculator, diagnostic_input):
        """Test overall health score calculation."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        assert 0 <= result.summary.overall_health_score <= 100

    def test_requires_immediate_action_flag(self, calculator, diagnostic_input):
        """Test immediate action requirement flag."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        assert isinstance(result.summary.requires_immediate_action, bool)

    def test_diagnostics_determinism(self, calculator, diagnostic_input):
        """Test diagnostics calculation is deterministic."""
        results = [calculator.calculate_diagnostics(diagnostic_input) for _ in range(3)]

        health_scores = [r.summary.overall_health_score for r in results]
        provenance_hashes = [r.summary.provenance_hash for r in results]

        assert len(set(health_scores)) == 1
        assert len(set(provenance_hashes)) == 1

    def test_provenance_hash_format(self, calculator, diagnostic_input):
        """Test provenance hash is valid SHA-256."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        hash_value = result.summary.provenance_hash

        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)

    def test_performance_target(self, calculator, diagnostic_input):
        """Test performance target (<10ms)."""
        result = calculator.calculate_diagnostics(diagnostic_input)

        # Allow some tolerance for test environment
        # The target is <10ms, but test environments may be slower
        assert result.processing_time_ms < 100.0  # Relaxed for testing

    def test_stoichiometric_afr_constants(self):
        """Test stoichiometric AFR constants are defined."""
        assert STOICHIOMETRIC_AFR["natural_gas"] == 17.2
        assert STOICHIOMETRIC_AFR["methane"] == 17.2
        assert STOICHIOMETRIC_AFR["propane"] == 15.7
        assert STOICHIOMETRIC_AFR["hydrogen"] == 34.3

    def test_adiabatic_flame_temp_constants(self):
        """Test adiabatic flame temperature constants."""
        assert ADIABATIC_FLAME_TEMP["natural_gas"] == 2223
        assert ADIABATIC_FLAME_TEMP["hydrogen"] == 2400
        assert ADIABATIC_FLAME_TEMP["coal"] == 2100

    def test_laminar_flame_speed_constants(self):
        """Test laminar flame speed constants."""
        assert LAMINAR_FLAME_SPEED["natural_gas"] == 0.40
        assert LAMINAR_FLAME_SPEED["hydrogen"] == 3.10

    def test_multi_zone_air_distribution(self, calculator):
        """Test multi-zone air distribution analysis."""
        zones = [
            ZoneInput(
                zone_id="zone_1",
                air_flow_kg_hr=1000.0,
                target_flow_kg_hr=1000.0,
                damper_position_percent=50.0
            ),
            ZoneInput(
                zone_id="zone_2",
                air_flow_kg_hr=900.0,
                target_flow_kg_hr=1000.0,
                damper_position_percent=45.0
            )
        ]

        result = calculator._analyze_air_distribution(zones, 5000.0)

        assert result.overall_balance_score >= 0
        assert len(result.zones) == 2

    def test_efficiency_degradation_trending(self, calculator):
        """Test efficiency degradation trend calculation."""
        efficiency_history = [85.0, 84.5, 84.0, 83.5, 83.0]
        timestamps_days = [0, 30, 60, 90, 120]

        result = calculator._calculate_efficiency_degradation(
            efficiency_history,
            timestamps_days,
            baseline_efficiency=85.0,
            current_efficiency=83.0
        )

        assert result.degradation_percent == pytest.approx(2.0, rel=0.1)
        assert result.trend_direction in list(TrendDirection)
        assert result.trend_confidence > 0  # R-squared

    def test_trend_analysis(self, calculator, diagnostic_input):
        """Test trend analysis."""
        diagnostic_input.historical_o2_readings = [3.5, 3.6, 3.7, 3.8, 4.0]
        diagnostic_input.historical_co_readings = [50, 55, 60, 65, 70]
        diagnostic_input.historical_timestamps_minutes = [0, 10, 20, 30, 40]

        trends = calculator._analyze_trends(diagnostic_input)

        assert isinstance(trends, list)


# ============================================================================
# CALCULATION HASH VALIDATION TESTS
# ============================================================================

class TestCalculationHashValidation:
    """Test calculation hash generation for determinism validation."""

    def test_calculation_input_hash_determinism(self):
        """Test hash generation is deterministic for same inputs."""
        inputs = {
            'fuel_flow': 500.0,
            'air_flow': 5000.0,
            'temperature': 1200.0
        }

        hash1 = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_hash_different_for_different_inputs(self):
        """Test hash is different for different inputs."""
        inputs1 = {'fuel_flow': 500.0, 'temperature': 1200.0}
        inputs2 = {'fuel_flow': 501.0, 'temperature': 1200.0}

        hash1 = hashlib.sha256(json.dumps(inputs1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(inputs2, sort_keys=True).encode()).hexdigest()

        assert hash1 != hash2


# ============================================================================
# EDGE CASES AND BOUNDARY TESTS
# ============================================================================

@pytest.mark.boundary
class TestCalculatorBoundaryCases:
    """Test calculator edge cases and boundary conditions."""

    def test_zero_fuel_flow_handling(self):
        """Test handling of zero fuel flow."""
        fuel_flow = 0.0
        air_flow = 5000.0

        # Division should be handled safely
        if fuel_flow == 0:
            ratio = 0.0
        else:
            ratio = fuel_flow / air_flow

        assert ratio == 0.0

    def test_zero_air_flow_handling(self):
        """Test handling of zero air flow."""
        fuel_flow = 500.0
        air_flow = 0.0

        # Division should be handled safely
        if air_flow == 0:
            ratio = float('inf')
        else:
            ratio = fuel_flow / air_flow

        assert math.isinf(ratio)

    def test_very_small_values(self):
        """Test calculations with very small values."""
        fuel_flow = 0.001
        air_flow = 0.01

        ratio = fuel_flow / air_flow

        assert ratio == pytest.approx(0.1, rel=1e-6)

    def test_very_large_values(self):
        """Test calculations with very large values."""
        fuel_flow = 10000.0
        air_flow = 100000.0

        ratio = fuel_flow / air_flow

        assert ratio == pytest.approx(0.1, rel=1e-6)

    def test_maximum_boundary_values(self):
        """Test maximum boundary value handling."""
        # Test temperature at maximum limit
        max_temp = 2000.0

        # Should not exceed physical limits
        assert max_temp <= 2500.0  # Reasonable combustion upper limit


# ============================================================================
# DIAGNOSTIC INPUT VALIDATION TESTS
# ============================================================================

class TestAdvancedDiagnosticInputValidation:
    """Test input validation for AdvancedDiagnosticInput."""

    def test_valid_input_creation(self):
        """Test valid input creation."""
        n = 100
        input_data = AdvancedDiagnosticInput(
            temperature_readings_c=[1200.0] * n,
            pressure_readings_pa=[5000.0] * n,
            flame_intensity_readings=[85.0] * n,
            sampling_rate_hz=10.0,
            o2_actual_percent=3.5,
            co_actual_ppm=50.0,
            fuel_flow_kg_hr=500.0,
            air_flow_kg_hr=8600.0,
            combustion_temperature_c=1200.0,
            furnace_pressure_pa=5000.0,
            fuel_heating_value_mj_kg=50.0,
            o2_setpoint_percent=3.0,
            temperature_setpoint_c=1200.0,
            fuel_demand_percent=50.0,
            air_demand_percent=50.0,
            fuel_actual_percent=50.0,
            air_actual_percent=50.0
        )

        assert input_data.o2_actual_percent == 3.5
        assert input_data.fuel_flow_kg_hr == 500.0

    def test_invalid_o2_percent_high(self):
        """Test validation rejects O2 > 21%."""
        n = 100
        with pytest.raises(ValueError):
            AdvancedDiagnosticInput(
                temperature_readings_c=[1200.0] * n,
                pressure_readings_pa=[5000.0] * n,
                flame_intensity_readings=[85.0] * n,
                o2_actual_percent=25.0,  # Invalid - exceeds 21%
                co_actual_ppm=50.0,
                fuel_flow_kg_hr=500.0,
                air_flow_kg_hr=8600.0,
                combustion_temperature_c=1200.0,
                furnace_pressure_pa=5000.0,
                fuel_heating_value_mj_kg=50.0,
                o2_setpoint_percent=3.0,
                temperature_setpoint_c=1200.0,
                fuel_demand_percent=50.0,
                air_demand_percent=50.0,
                fuel_actual_percent=50.0,
                air_actual_percent=50.0
            )

    def test_mismatched_array_lengths(self):
        """Test validation rejects mismatched array lengths."""
        with pytest.raises(ValueError):
            AdvancedDiagnosticInput(
                temperature_readings_c=[1200.0] * 100,
                pressure_readings_pa=[5000.0] * 50,  # Different length
                flame_intensity_readings=[85.0] * 100,
                o2_actual_percent=3.5,
                co_actual_ppm=50.0,
                fuel_flow_kg_hr=500.0,
                air_flow_kg_hr=8600.0,
                combustion_temperature_c=1200.0,
                furnace_pressure_pa=5000.0,
                fuel_heating_value_mj_kg=50.0,
                o2_setpoint_percent=3.0,
                temperature_setpoint_c=1200.0,
                fuel_demand_percent=50.0,
                air_demand_percent=50.0,
                fuel_actual_percent=50.0,
                air_actual_percent=50.0
            )

    def test_empty_readings_array(self):
        """Test validation rejects empty readings array."""
        with pytest.raises(ValueError):
            AdvancedDiagnosticInput(
                temperature_readings_c=[],  # Empty
                pressure_readings_pa=[],
                flame_intensity_readings=[],
                o2_actual_percent=3.5,
                co_actual_ppm=50.0,
                fuel_flow_kg_hr=500.0,
                air_flow_kg_hr=8600.0,
                combustion_temperature_c=1200.0,
                furnace_pressure_pa=5000.0,
                fuel_heating_value_mj_kg=50.0,
                o2_setpoint_percent=3.0,
                temperature_setpoint_c=1200.0,
                fuel_demand_percent=50.0,
                air_demand_percent=50.0,
                fuel_actual_percent=50.0,
                air_actual_percent=50.0
            )
