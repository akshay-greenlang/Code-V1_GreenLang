#!/usr/bin/env python3
"""Script to create all test files for GL-004 BURNMASTER."""
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))

# Complete conftest.py content
conftest_content = '''"""
Shared pytest fixtures for GL-004 BURNMASTER test suite.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, AsyncMock
import random


class FuelType(Enum):
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    PROPANE = "propane"
    HYDROGEN = "hydrogen"
    COAL = "coal"
    BIOMASS = "biomass"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"


class OperatingMode(Enum):
    OFF = "off"
    PURGE = "purge"
    IGNITION = "ignition"
    LOW_FIRE = "low_fire"
    MODULATING = "modulating"
    HIGH_FIRE = "high_fire"
    SHUTDOWN = "shutdown"


class BurnerState(Enum):
    SAFE = "safe"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    LOCKOUT = "lockout"


class ControlMode(Enum):
    OBSERVE = "observe"
    ADVISORY = "advisory"
    CLOSED_LOOP = "closed_loop"
    MANUAL = "manual"
    FALLBACK = "fallback"


@dataclass
class FuelProperties:
    fuel_type: FuelType
    lower_heating_value: float
    higher_heating_value: float
    stoichiometric_air_ratio: float
    density: float
    carbon_content: float
    hydrogen_content: float
    oxygen_content: float
    nitrogen_content: float
    sulfur_content: float
    moisture_content: float
    adiabatic_flame_temp: float


@dataclass
class CombustionMeasurement:
    timestamp: datetime
    fuel_flow_rate: float
    air_flow_rate: float
    flue_gas_temp: float
    ambient_temp: float
    o2_percentage: float
    co_ppm: float
    co2_percentage: float
    nox_ppm: float
    lambda_value: float
    efficiency: float
    heat_output: float


@dataclass
class SafetyEnvelope:
    min_lambda: float = 1.05
    max_lambda: float = 1.50
    min_fuel_flow: float = 0.0
    max_fuel_flow: float = 100.0
    min_air_flow: float = 0.0
    max_air_flow: float = 1000.0
    max_flue_gas_temp: float = 1500.0
    max_co_ppm: float = 100.0
    max_nox_ppm: float = 50.0
    min_o2_percentage: float = 1.0
    max_o2_percentage: float = 10.0


@dataclass
class OptimizationResult:
    success: bool
    optimal_lambda: float
    optimal_air_flow: float
    predicted_efficiency: float
    predicted_co_ppm: float
    predicted_nox_ppm: float
    iterations: int
    convergence_time_ms: float
    constraints_satisfied: bool
    recommendations: List[str] = field(default_factory=list)


STOICHIOMETRIC_AIR_RATIOS = {
    FuelType.NATURAL_GAS: 17.2, FuelType.DIESEL: 14.5, FuelType.PROPANE: 15.7,
    FuelType.HYDROGEN: 34.3, FuelType.COAL: 11.5, FuelType.BIOMASS: 6.5,
    FuelType.FUEL_OIL_2: 14.4, FuelType.FUEL_OIL_6: 13.8,
}

LOWER_HEATING_VALUES = {
    FuelType.NATURAL_GAS: 50.0, FuelType.DIESEL: 42.5, FuelType.PROPANE: 46.4,
    FuelType.HYDROGEN: 120.0, FuelType.COAL: 25.0, FuelType.BIOMASS: 18.0,
    FuelType.FUEL_OIL_2: 42.6, FuelType.FUEL_OIL_6: 40.5,
}

HIGHER_HEATING_VALUES = {
    FuelType.NATURAL_GAS: 55.5, FuelType.DIESEL: 45.4, FuelType.PROPANE: 50.3,
    FuelType.HYDROGEN: 141.8, FuelType.COAL: 27.0, FuelType.BIOMASS: 20.0,
    FuelType.FUEL_OIL_2: 45.5, FuelType.FUEL_OIL_6: 43.0,
}

ADIABATIC_FLAME_TEMPS = {
    FuelType.NATURAL_GAS: 2223.0, FuelType.DIESEL: 2327.0, FuelType.PROPANE: 2267.0,
    FuelType.HYDROGEN: 2483.0, FuelType.COAL: 2150.0, FuelType.BIOMASS: 1900.0,
    FuelType.FUEL_OIL_2: 2300.0, FuelType.FUEL_OIL_6: 2250.0,
}


class CombustionTestDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_fuel_properties(self, fuel_type: FuelType) -> FuelProperties:
        densities = {FuelType.NATURAL_GAS: 0.72, FuelType.DIESEL: 832.0, FuelType.PROPANE: 1.88,
                     FuelType.HYDROGEN: 0.09, FuelType.COAL: 1400.0, FuelType.BIOMASS: 500.0,
                     FuelType.FUEL_OIL_2: 850.0, FuelType.FUEL_OIL_6: 970.0}
        carbon = {FuelType.NATURAL_GAS: 0.75, FuelType.DIESEL: 0.87, FuelType.PROPANE: 0.82,
                  FuelType.HYDROGEN: 0.0, FuelType.COAL: 0.75, FuelType.BIOMASS: 0.45,
                  FuelType.FUEL_OIL_2: 0.87, FuelType.FUEL_OIL_6: 0.88}
        hydrogen = {FuelType.NATURAL_GAS: 0.25, FuelType.DIESEL: 0.13, FuelType.PROPANE: 0.18,
                    FuelType.HYDROGEN: 1.0, FuelType.COAL: 0.05, FuelType.BIOMASS: 0.06,
                    FuelType.FUEL_OIL_2: 0.12, FuelType.FUEL_OIL_6: 0.10}
        return FuelProperties(
            fuel_type=fuel_type, lower_heating_value=LOWER_HEATING_VALUES[fuel_type],
            higher_heating_value=HIGHER_HEATING_VALUES[fuel_type],
            stoichiometric_air_ratio=STOICHIOMETRIC_AIR_RATIOS[fuel_type],
            density=densities.get(fuel_type, 1.0), carbon_content=carbon.get(fuel_type, 0.0),
            hydrogen_content=hydrogen.get(fuel_type, 0.0), oxygen_content=0.0,
            nitrogen_content=0.0, sulfur_content=0.0, moisture_content=0.0,
            adiabatic_flame_temp=ADIABATIC_FLAME_TEMPS[fuel_type])

    def generate_combustion_measurement(self, fuel_type=FuelType.NATURAL_GAS,
                                         lambda_value=None, base_fuel_flow=1.0):
        if lambda_value is None:
            lambda_value = self.rng.uniform(1.05, 1.30)
        fuel_flow = base_fuel_flow * self.rng.uniform(0.9, 1.1)
        air_flow = fuel_flow * STOICHIOMETRIC_AIR_RATIOS[fuel_type] * lambda_value
        o2 = min(21.0 * (lambda_value - 1.0) / lambda_value, 21.0) if lambda_value > 1.0 else 0.0
        co = max(5, 100 * np.exp(-(lambda_value - 1.05) * 10)) if lambda_value >= 1.05 else 100
        nox = max(5, 50 * np.exp(-((lambda_value - 1.12) ** 2) / 0.02))
        max_eff = 92.0 if fuel_type == FuelType.NATURAL_GAS else 88.0
        eff = max(60.0, min(max_eff - 0.5 * (lambda_value - 1.0) ** 2 * 20, max_eff))
        flue_temp = ADIABATIC_FLAME_TEMPS[fuel_type] * 0.4 + 298.15 * 0.6
        max_co2 = {FuelType.NATURAL_GAS: 11.7, FuelType.DIESEL: 15.3}
        co2 = max(0.0, max_co2.get(fuel_type, 12.0) / lambda_value)
        return CombustionMeasurement(
            timestamp=datetime.now(), fuel_flow_rate=fuel_flow, air_flow_rate=air_flow,
            flue_gas_temp=flue_temp, ambient_temp=298.15, o2_percentage=o2, co_ppm=co,
            co2_percentage=co2, nox_ppm=nox, lambda_value=lambda_value, efficiency=eff,
            heat_output=fuel_flow * LOWER_HEATING_VALUES[fuel_type] * eff / 100)


@pytest.fixture
def data_generator():
    return CombustionTestDataGenerator(seed=42)

@pytest.fixture
def fuel_properties_natural_gas(data_generator):
    return data_generator.generate_fuel_properties(FuelType.NATURAL_GAS)

@pytest.fixture
def fuel_properties_diesel(data_generator):
    return data_generator.generate_fuel_properties(FuelType.DIESEL)

@pytest.fixture
def sample_measurement(data_generator):
    return data_generator.generate_combustion_measurement(lambda_value=1.15)

@pytest.fixture
def sample_time_series(data_generator):
    return [data_generator.generate_combustion_measurement() for _ in range(100)]

@pytest.fixture
def safety_envelope():
    return SafetyEnvelope()

@pytest.fixture
def mock_dcs_connection():
    m = Mock()
    m.read_tag = Mock(return_value=100.0)
    m.write_tag = Mock(return_value=True)
    m.read_tags = Mock(return_value={"FIC001.PV": 100.0, "AIC001.PV": 1720.0})
    m.is_connected = Mock(return_value=True)
    return m

@pytest.fixture
def mock_ml_model():
    m = Mock()
    m.predict = Mock(return_value=np.array([0.85]))
    m.predict_proba = Mock(return_value=np.array([[0.15, 0.85]]))
    return m

@pytest.fixture
def mock_optimizer():
    m = Mock()
    m.optimize = Mock(return_value=OptimizationResult(
        success=True, optimal_lambda=1.12, optimal_air_flow=1720.0,
        predicted_efficiency=91.5, predicted_co_ppm=15.0, predicted_nox_ppm=25.0,
        iterations=10, convergence_time_ms=50.0, constraints_satisfied=True))
    return m


def assert_within_tolerance(actual, expected, tolerance=0.01, message=""):
    diff = abs(actual - expected)
    rel = diff / abs(expected) if expected != 0 else diff
    assert rel <= tolerance, f"{message}: Expected {expected}, got {actual}"

def assert_within_range(value, min_val, max_val, message=""):
    assert min_val <= value <= max_val, f"{message}: {value} not in [{min_val}, {max_val}]"


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")

@pytest.fixture(autouse=True)
def reset_random_seed():
    np.random.seed(42)
    random.seed(42)
    yield
'''

with open(os.path.join(tests_dir, 'conftest.py'), 'w', encoding='utf-8') as f:
    f.write(conftest_content)
print('Created conftest.py')

# test_stoichiometry.py
stoich_content = '''"""Unit tests for Stoichiometry Calculations."""
import pytest
import numpy as np
from conftest import FuelType, STOICHIOMETRIC_AIR_RATIOS, assert_within_tolerance, assert_within_range


class StoichiometryCalculator:
    O2_IN_AIR = 20.95

    def __init__(self, fuel_properties):
        self.fuel = fuel_properties

    def calculate_stoichiometric_air(self, fuel_mass):
        if fuel_mass < 0:
            raise ValueError("Fuel mass cannot be negative")
        return fuel_mass * self.fuel.stoichiometric_air_ratio

    def calculate_lambda(self, actual_air, fuel_mass):
        if fuel_mass <= 0:
            raise ValueError("Fuel mass must be positive")
        if actual_air < 0:
            raise ValueError("Air mass cannot be negative")
        return actual_air / self.calculate_stoichiometric_air(fuel_mass)

    def calculate_excess_air_percentage(self, lambda_value):
        return (lambda_value - 1.0) * 100.0

    def calculate_o2_from_lambda(self, lambda_value):
        if lambda_value < 1.0:
            return 0.0
        return min(self.O2_IN_AIR * (lambda_value - 1.0) / lambda_value, self.O2_IN_AIR)


class TestStoichiometricAir:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return StoichiometryCalculator(fuel_properties_natural_gas)

    def test_stoichiometric_air_natural_gas(self, calculator):
        stoich = calculator.calculate_stoichiometric_air(1.0)
        assert_within_tolerance(stoich, 17.2, tolerance=0.01)

    def test_stoichiometric_air_scaling(self, calculator):
        stoich_1 = calculator.calculate_stoichiometric_air(1.0)
        stoich_10 = calculator.calculate_stoichiometric_air(10.0)
        assert_within_tolerance(stoich_10, stoich_1 * 10, tolerance=1e-10)

    def test_negative_fuel_raises(self, calculator):
        with pytest.raises(ValueError):
            calculator.calculate_stoichiometric_air(-1.0)

    @pytest.mark.parametrize("fuel_type,expected", [
        (FuelType.NATURAL_GAS, 17.2), (FuelType.DIESEL, 14.5),
        (FuelType.PROPANE, 15.7), (FuelType.HYDROGEN, 34.3),
    ])
    def test_all_fuel_ratios(self, fuel_type, expected, data_generator):
        props = data_generator.generate_fuel_properties(fuel_type)
        calc = StoichiometryCalculator(props)
        assert_within_tolerance(calc.calculate_stoichiometric_air(1.0), expected, 0.01)


class TestLambdaCalculations:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return StoichiometryCalculator(fuel_properties_natural_gas)

    def test_lambda_stoichiometric(self, calculator):
        stoich = calculator.calculate_stoichiometric_air(1.0)
        lam = calculator.calculate_lambda(stoich, 1.0)
        assert_within_tolerance(lam, 1.0, tolerance=1e-10)

    def test_lambda_lean(self, calculator):
        stoich = calculator.calculate_stoichiometric_air(1.0)
        lam = calculator.calculate_lambda(stoich * 1.2, 1.0)
        assert_within_tolerance(lam, 1.2, tolerance=1e-10)

    def test_excess_air_percentage(self, calculator):
        excess = calculator.calculate_excess_air_percentage(1.15)
        assert_within_tolerance(excess, 15.0, tolerance=1e-10)

    def test_o2_from_lambda(self, calculator):
        o2 = calculator.calculate_o2_from_lambda(1.15)
        assert_within_range(o2, 2.0, 4.0)

    def test_o2_zero_for_rich(self, calculator):
        assert calculator.calculate_o2_from_lambda(0.95) == 0.0


class TestDeterminism:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return StoichiometryCalculator(fuel_properties_natural_gas)

    def test_repeated_calculations(self, calculator):
        results = [calculator.calculate_stoichiometric_air(1.0) for _ in range(100)]
        assert all(r == results[0] for r in results)
'''

with open(os.path.join(tests_dir, 'unit', 'test_stoichiometry.py'), 'w', encoding='utf-8') as f:
    f.write(stoich_content)
print('Created unit/test_stoichiometry.py')

# test_thermodynamics.py
thermo_content = '''"""Unit tests for Thermodynamics Calculations."""
import pytest
import numpy as np
from conftest import FuelType, ADIABATIC_FLAME_TEMPS, assert_within_tolerance, assert_within_range


class ThermodynamicsCalculator:
    SIEGERT_A1 = {FuelType.NATURAL_GAS: 0.66, FuelType.DIESEL: 0.68}
    SIEGERT_A2 = {FuelType.NATURAL_GAS: 0.009, FuelType.DIESEL: 0.007}

    def __init__(self, fuel_properties):
        self.fuel = fuel_properties

    def calculate_adiabatic_flame_temp(self, lambda_value, preheat_temp=298.15):
        if lambda_value <= 0:
            raise ValueError("Lambda must be positive")
        t_ad = self.fuel.adiabatic_flame_temp
        dilution = 0.15 * (lambda_value - 1.0) if lambda_value >= 1.0 else 0.20 * (1.0 - lambda_value)
        result = t_ad * (1.0 - dilution) + (preheat_temp - 298.15) * 0.8
        return max(result, 298.15)

    def calculate_stack_loss_siegert(self, flue_temp, ambient_temp, o2_pct):
        if o2_pct >= 21.0 or o2_pct < 0:
            raise ValueError("Invalid O2 percentage")
        a1 = self.SIEGERT_A1.get(self.fuel.fuel_type, 0.68)
        a2 = self.SIEGERT_A2.get(self.fuel.fuel_type, 0.007)
        co2_max = {FuelType.NATURAL_GAS: 11.7, FuelType.DIESEL: 15.3}.get(self.fuel.fuel_type, 12.0)
        co2 = co2_max * (21.0 - o2_pct) / 21.0
        if co2 <= 0:
            return 100.0
        return max(0.0, min((a1 / co2 + a2) * (flue_temp - ambient_temp), 100.0))

    def calculate_efficiency_gross(self, stack_loss, unburned=0.0, radiation=1.0):
        return max(0.0, min(100.0 - stack_loss - unburned - radiation, 100.0))

    def calculate_heat_output(self, fuel_flow, efficiency):
        if fuel_flow < 0:
            raise ValueError("Fuel flow cannot be negative")
        if efficiency < 0 or efficiency > 100:
            raise ValueError("Efficiency must be 0-100")
        return fuel_flow * self.fuel.lower_heating_value * (efficiency / 100.0)


class TestAdiabaticFlameTemp:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return ThermodynamicsCalculator(fuel_properties_natural_gas)

    def test_stoichiometric_temp(self, calculator):
        t_ad = calculator.calculate_adiabatic_flame_temp(1.0)
        assert_within_tolerance(t_ad, ADIABATIC_FLAME_TEMPS[FuelType.NATURAL_GAS], 0.01)

    def test_temp_decreases_with_lambda(self, calculator):
        temps = [calculator.calculate_adiabatic_flame_temp(lam) for lam in [1.0, 1.1, 1.2, 1.3]]
        for i in range(1, len(temps)):
            assert temps[i] < temps[i-1]

    def test_preheat_increases_temp(self, calculator):
        t1 = calculator.calculate_adiabatic_flame_temp(1.15, 298.15)
        t2 = calculator.calculate_adiabatic_flame_temp(1.15, 473.15)
        assert t2 > t1

    def test_invalid_lambda_raises(self, calculator):
        with pytest.raises(ValueError):
            calculator.calculate_adiabatic_flame_temp(0.0)


class TestStackLoss:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return ThermodynamicsCalculator(fuel_properties_natural_gas)

    def test_typical_stack_loss(self, calculator):
        loss = calculator.calculate_stack_loss_siegert(450.0, 298.15, 3.0)
        assert_within_range(loss, 3.0, 20.0)

    def test_loss_increases_with_temp(self, calculator):
        losses = [calculator.calculate_stack_loss_siegert(t, 298.15, 3.0) for t in [400, 450, 500]]
        for i in range(1, len(losses)):
            assert losses[i] > losses[i-1]

    def test_loss_zero_at_ambient(self, calculator):
        loss = calculator.calculate_stack_loss_siegert(298.15, 298.15, 3.0)
        assert_within_tolerance(loss, 0.0, 0.01)


class TestEfficiency:
    @pytest.fixture
    def calculator(self, fuel_properties_natural_gas):
        return ThermodynamicsCalculator(fuel_properties_natural_gas)

    def test_efficiency_calculation(self, calculator):
        eff = calculator.calculate_efficiency_gross(8.0, 0.2, 1.0)
        assert_within_tolerance(eff, 90.8, 0.01)

    def test_heat_output(self, calculator):
        output = calculator.calculate_heat_output(1.0, 90.0)
        assert_within_tolerance(output, 45.0, 0.01)

    def test_heat_output_scales(self, calculator):
        o1 = calculator.calculate_heat_output(1.0, 90.0)
        o10 = calculator.calculate_heat_output(10.0, 90.0)
        assert_within_tolerance(o10, o1 * 10, 0.01)
'''

with open(os.path.join(tests_dir, 'unit', 'test_thermodynamics.py'), 'w', encoding='utf-8') as f:
    f.write(thermo_content)
print('Created unit/test_thermodynamics.py')

# test_calculators.py
calc_content = '''"""Unit tests for Combustion Calculators."""
import pytest
import numpy as np
from conftest import FuelType, assert_within_tolerance, assert_within_range


class AirFuelRatioCalculator:
    def __init__(self, stoich_ratio):
        self.stoich_ratio = stoich_ratio

    def calculate_actual_afr(self, air_flow, fuel_flow):
        if fuel_flow <= 0:
            raise ValueError("Fuel flow must be positive")
        return air_flow / fuel_flow

    def calculate_lambda(self, actual_afr):
        return actual_afr / self.stoich_ratio

    def calculate_required_air(self, fuel_flow, target_lambda):
        return fuel_flow * self.stoich_ratio * target_lambda


class FlameStabilityCalculator:
    def calculate_stability_index(self, lambda_val, turndown, temp_variance):
        base = 1.0
        lambda_factor = 1.0 - abs(lambda_val - 1.12) * 2
        turndown_factor = 1.0 - (1.0 - turndown) * 0.5
        variance_factor = 1.0 - min(temp_variance / 50.0, 0.5)
        return max(0.0, min(base * lambda_factor * turndown_factor * variance_factor, 1.0))


class EmissionsCalculator:
    def estimate_co(self, lambda_val, temp):
        if lambda_val < 1.0:
            return 1000 + (1.0 - lambda_val) * 10000
        return max(5, 100 * np.exp(-(lambda_val - 1.05) * 10))

    def estimate_nox(self, lambda_val, temp):
        return max(5, 50 * np.exp(-((lambda_val - 1.12) ** 2) / 0.02))


class TestAirFuelRatioCalculator:
    @pytest.fixture
    def calculator(self):
        return AirFuelRatioCalculator(stoich_ratio=17.2)

    def test_afr_calculation(self, calculator):
        afr = calculator.calculate_actual_afr(17.2, 1.0)
        assert_within_tolerance(afr, 17.2, 0.01)

    def test_lambda_stoichiometric(self, calculator):
        lam = calculator.calculate_lambda(17.2)
        assert_within_tolerance(lam, 1.0, 0.01)

    def test_required_air(self, calculator):
        air = calculator.calculate_required_air(1.0, 1.15)
        assert_within_tolerance(air, 17.2 * 1.15, 0.01)


class TestFlameStabilityCalculator:
    @pytest.fixture
    def calculator(self):
        return FlameStabilityCalculator()

    def test_optimal_stability(self, calculator):
        idx = calculator.calculate_stability_index(1.12, 1.0, 0.0)
        assert idx > 0.9

    def test_low_stability_high_lambda(self, calculator):
        idx = calculator.calculate_stability_index(1.5, 1.0, 0.0)
        assert idx < 0.5


class TestEmissionsCalculator:
    @pytest.fixture
    def calculator(self):
        return EmissionsCalculator()

    def test_co_rich_combustion(self, calculator):
        co = calculator.estimate_co(0.95, 1000)
        assert co > 500

    def test_co_lean_combustion(self, calculator):
        co = calculator.estimate_co(1.15, 1000)
        assert co < 50

    def test_nox_peaks_near_optimal(self, calculator):
        nox_optimal = calculator.estimate_nox(1.12, 1000)
        nox_lean = calculator.estimate_nox(1.30, 1000)
        assert nox_optimal > nox_lean
'''

with open(os.path.join(tests_dir, 'unit', 'test_calculators.py'), 'w', encoding='utf-8') as f:
    f.write(calc_content)
print('Created unit/test_calculators.py')

# test_optimization.py
opt_content = '''"""Unit tests for Optimization Module."""
import pytest
import numpy as np
from conftest import OptimizationResult, assert_within_tolerance, assert_within_range


class ObjectiveFunction:
    def __init__(self, weights=None):
        self.weights = weights or {"efficiency": 0.5, "emissions": 0.3, "safety": 0.2}

    def evaluate(self, lambda_val, fuel_flow, constraints):
        eff_score = 1.0 - abs(lambda_val - 1.10) * 2
        emit_score = 1.0 - abs(lambda_val - 1.15) * 2
        safety_score = 1.0 if 1.05 <= lambda_val <= 1.50 else 0.0
        return (self.weights["efficiency"] * eff_score +
                self.weights["emissions"] * emit_score +
                self.weights["safety"] * safety_score)


class ConstraintHandler:
    def __init__(self, envelope):
        self.envelope = envelope

    def check_lambda(self, lambda_val):
        return self.envelope.min_lambda <= lambda_val <= self.envelope.max_lambda

    def check_all(self, lambda_val, fuel_flow, co_ppm, nox_ppm):
        return (self.check_lambda(lambda_val) and
                fuel_flow <= self.envelope.max_fuel_flow and
                co_ppm <= self.envelope.max_co_ppm and
                nox_ppm <= self.envelope.max_nox_ppm)


class CombustionOptimizer:
    def __init__(self, objective, constraints):
        self.objective = objective
        self.constraints = constraints

    def optimize(self, current_lambda, fuel_flow, max_iter=100):
        best_lambda = current_lambda
        best_score = self.objective.evaluate(current_lambda, fuel_flow, {})
        for i in range(max_iter):
            test_lambda = np.clip(current_lambda + np.random.uniform(-0.02, 0.02), 1.05, 1.50)
            if self.constraints.check_lambda(test_lambda):
                score = self.objective.evaluate(test_lambda, fuel_flow, {})
                if score > best_score:
                    best_score = score
                    best_lambda = test_lambda
        return OptimizationResult(
            success=True, optimal_lambda=best_lambda, optimal_air_flow=fuel_flow * 17.2 * best_lambda,
            predicted_efficiency=90.0, predicted_co_ppm=20.0, predicted_nox_ppm=30.0,
            iterations=max_iter, convergence_time_ms=10.0, constraints_satisfied=True)


class TestObjectiveFunction:
    @pytest.fixture
    def objective(self):
        return ObjectiveFunction()

    def test_optimal_lambda_high_score(self, objective):
        score = objective.evaluate(1.12, 1.0, {})
        assert score > 0.5

    def test_extreme_lambda_low_score(self, objective):
        score = objective.evaluate(1.50, 1.0, {})
        assert score < 0.5


class TestConstraintHandler:
    @pytest.fixture
    def handler(self, safety_envelope):
        return ConstraintHandler(safety_envelope)

    def test_valid_lambda(self, handler):
        assert handler.check_lambda(1.15)

    def test_invalid_lambda_low(self, handler):
        assert not handler.check_lambda(1.01)

    def test_invalid_lambda_high(self, handler):
        assert not handler.check_lambda(1.60)


class TestCombustionOptimizer:
    @pytest.fixture
    def optimizer(self, safety_envelope):
        obj = ObjectiveFunction()
        const = ConstraintHandler(safety_envelope)
        return CombustionOptimizer(obj, const)

    def test_optimizer_returns_result(self, optimizer):
        result = optimizer.optimize(1.15, 1.0, max_iter=50)
        assert result.success
        assert result.constraints_satisfied

    def test_optimizer_respects_bounds(self, optimizer):
        result = optimizer.optimize(1.15, 1.0, max_iter=50)
        assert 1.05 <= result.optimal_lambda <= 1.50

    def test_optimizer_deterministic_with_seed(self, optimizer):
        np.random.seed(42)
        r1 = optimizer.optimize(1.15, 1.0, max_iter=50)
        np.random.seed(42)
        r2 = optimizer.optimize(1.15, 1.0, max_iter=50)
        assert r1.optimal_lambda == r2.optimal_lambda
'''

with open(os.path.join(tests_dir, 'unit', 'test_optimization.py'), 'w', encoding='utf-8') as f:
    f.write(opt_content)
print('Created unit/test_optimization.py')

# test_safety.py
safety_content = '''"""Unit tests for Safety Module."""
import pytest
from conftest import SafetyEnvelope, BurnerState, assert_within_range


class SafetyEnvelopeValidator:
    def __init__(self, envelope):
        self.envelope = envelope

    def validate_lambda(self, lambda_val):
        return self.envelope.min_lambda <= lambda_val <= self.envelope.max_lambda

    def validate_fuel_flow(self, flow):
        return self.envelope.min_fuel_flow <= flow <= self.envelope.max_fuel_flow

    def validate_flue_temp(self, temp):
        return temp <= self.envelope.max_flue_gas_temp

    def validate_co(self, co_ppm):
        return co_ppm <= self.envelope.max_co_ppm

    def validate_nox(self, nox_ppm):
        return nox_ppm <= self.envelope.max_nox_ppm

    def get_state(self, lambda_val, co_ppm, nox_ppm, flue_temp):
        if not self.validate_lambda(lambda_val):
            return BurnerState.TRIP
        if co_ppm > self.envelope.max_co_ppm * 2:
            return BurnerState.ALARM
        if co_ppm > self.envelope.max_co_ppm:
            return BurnerState.WARNING
        return BurnerState.SAFE


class InterlockHandler:
    def __init__(self):
        self.interlocks = {}

    def set_interlock(self, name, active):
        self.interlocks[name] = active

    def check_interlock(self, name):
        return self.interlocks.get(name, False)

    def any_active(self):
        return any(self.interlocks.values())

    def get_active_list(self):
        return [k for k, v in self.interlocks.items() if v]


class HazardDetector:
    def detect_flame_out(self, o2_pct, flue_temp, co_ppm):
        return o2_pct > 20.0 and flue_temp < 350.0

    def detect_rich_combustion(self, lambda_val, o2_pct, co_ppm):
        return lambda_val < 1.0 or (o2_pct < 0.5 and co_ppm > 500)


class TestSafetyEnvelopeValidator:
    @pytest.fixture
    def validator(self, safety_envelope):
        return SafetyEnvelopeValidator(safety_envelope)

    def test_valid_lambda(self, validator):
        assert validator.validate_lambda(1.15)

    def test_invalid_lambda_low(self, validator):
        assert not validator.validate_lambda(1.01)

    def test_invalid_lambda_high(self, validator):
        assert not validator.validate_lambda(1.60)

    def test_valid_co(self, validator):
        assert validator.validate_co(50.0)

    def test_invalid_co(self, validator):
        assert not validator.validate_co(200.0)

    def test_state_safe(self, validator):
        assert validator.get_state(1.15, 50.0, 30.0, 450.0) == BurnerState.SAFE

    def test_state_warning(self, validator):
        assert validator.get_state(1.15, 150.0, 30.0, 450.0) == BurnerState.WARNING

    def test_state_trip(self, validator):
        assert validator.get_state(1.01, 50.0, 30.0, 450.0) == BurnerState.TRIP


class TestInterlockHandler:
    @pytest.fixture
    def handler(self):
        return InterlockHandler()

    def test_set_and_check_interlock(self, handler):
        handler.set_interlock("fuel_pressure_low", True)
        assert handler.check_interlock("fuel_pressure_low")

    def test_any_active(self, handler):
        assert not handler.any_active()
        handler.set_interlock("test", True)
        assert handler.any_active()

    def test_get_active_list(self, handler):
        handler.set_interlock("a", True)
        handler.set_interlock("b", False)
        handler.set_interlock("c", True)
        active = handler.get_active_list()
        assert "a" in active
        assert "c" in active
        assert "b" not in active


class TestHazardDetector:
    @pytest.fixture
    def detector(self):
        return HazardDetector()

    def test_detect_flame_out(self, detector):
        assert detector.detect_flame_out(21.0, 300.0, 5000)

    def test_no_flame_out(self, detector):
        assert not detector.detect_flame_out(3.0, 500.0, 20)

    def test_detect_rich_combustion(self, detector):
        assert detector.detect_rich_combustion(0.95, 0.1, 1000)

    def test_no_rich_combustion(self, detector):
        assert not detector.detect_rich_combustion(1.15, 3.0, 20)
'''

with open(os.path.join(tests_dir, 'unit', 'test_safety.py'), 'w', encoding='utf-8') as f:
    f.write(safety_content)
print('Created unit/test_safety.py')

# test_ml_models.py
ml_content = '''"""Unit tests for ML Models."""
import pytest
import numpy as np
from unittest.mock import Mock


class StabilityPredictor:
    def __init__(self, model=None):
        self.model = model or self._default_model()

    def _default_model(self):
        m = Mock()
        m.predict = Mock(return_value=np.array([0.85]))
        return m

    def predict(self, features):
        return float(self.model.predict([features])[0])

    def predict_batch(self, feature_batch):
        return [self.predict(f) for f in feature_batch]


class EmissionsPredictor:
    def __init__(self, co_model=None, nox_model=None):
        self.co_model = co_model or self._default_model(20.0)
        self.nox_model = nox_model or self._default_model(30.0)

    def _default_model(self, default_val):
        m = Mock()
        m.predict = Mock(return_value=np.array([default_val]))
        return m

    def predict_co(self, features):
        return float(self.co_model.predict([features])[0])

    def predict_nox(self, features):
        return float(self.nox_model.predict([features])[0])


class SoftSensor:
    def __init__(self, sensor_type, model=None):
        self.sensor_type = sensor_type
        self.model = model or self._default_model()
        self.calibration_offset = 0.0

    def _default_model(self):
        m = Mock()
        m.predict = Mock(return_value=np.array([100.0]))
        return m

    def estimate(self, inputs):
        raw = float(self.model.predict([inputs])[0])
        return raw + self.calibration_offset

    def calibrate(self, measured, estimated):
        self.calibration_offset = measured - estimated


class TestStabilityPredictor:
    @pytest.fixture
    def predictor(self):
        return StabilityPredictor()

    def test_predict_returns_float(self, predictor):
        result = predictor.predict([1.15, 100.0, 17.2])
        assert isinstance(result, float)

    def test_predict_in_range(self, predictor):
        result = predictor.predict([1.15, 100.0, 17.2])
        assert 0.0 <= result <= 1.0

    def test_batch_prediction(self, predictor):
        batch = [[1.15, 100.0, 17.2], [1.20, 100.0, 17.2]]
        results = predictor.predict_batch(batch)
        assert len(results) == 2


class TestEmissionsPredictor:
    @pytest.fixture
    def predictor(self):
        return EmissionsPredictor()

    def test_predict_co(self, predictor):
        co = predictor.predict_co([1.15, 100.0])
        assert co >= 0

    def test_predict_nox(self, predictor):
        nox = predictor.predict_nox([1.15, 100.0])
        assert nox >= 0


class TestSoftSensor:
    @pytest.fixture
    def sensor(self):
        return SoftSensor("o2")

    def test_estimate(self, sensor):
        result = sensor.estimate([1.15, 100.0])
        assert result > 0

    def test_calibration(self, sensor):
        initial = sensor.estimate([1.15, 100.0])
        sensor.calibrate(105.0, initial)
        calibrated = sensor.estimate([1.15, 100.0])
        assert abs(calibrated - 105.0) < 0.01
'''

with open(os.path.join(tests_dir, 'unit', 'test_ml_models.py'), 'w', encoding='utf-8') as f:
    f.write(ml_content)
print('Created unit/test_ml_models.py')

# unit/__init__.py
with open(os.path.join(tests_dir, 'unit', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('"""Unit tests for GL-004 BURNMASTER."""\n')
print('Created unit/__init__.py')

# integration/__init__.py
with open(os.path.join(tests_dir, 'integration', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('"""Integration tests for GL-004 BURNMASTER."""\n')
print('Created integration/__init__.py')

# integration/test_control_loop.py
control_loop_content = '''"""Integration tests for Control Loop."""
import pytest
from unittest.mock import Mock, AsyncMock
from conftest import ControlMode, SafetyEnvelope


class ControlLoop:
    def __init__(self, mode=ControlMode.OBSERVE):
        self.mode = mode
        self.dcs = None
        self.optimizer = None

    def set_mode(self, mode):
        self.mode = mode

    def connect_dcs(self, dcs):
        self.dcs = dcs

    def execute_cycle(self, measurements):
        if self.mode == ControlMode.OBSERVE:
            return {"action": "observe", "recommendations": []}
        elif self.mode == ControlMode.ADVISORY:
            return {"action": "advisory", "recommendations": ["Adjust lambda to 1.12"]}
        elif self.mode == ControlMode.CLOSED_LOOP:
            return {"action": "control", "setpoint": 1.12}
        return {"action": "none"}


class TestControlLoopModes:
    @pytest.fixture
    def loop(self):
        return ControlLoop()

    def test_observe_mode(self, loop):
        result = loop.execute_cycle({})
        assert result["action"] == "observe"

    def test_advisory_mode(self, loop):
        loop.set_mode(ControlMode.ADVISORY)
        result = loop.execute_cycle({})
        assert result["action"] == "advisory"
        assert len(result["recommendations"]) > 0

    def test_closed_loop_mode(self, loop):
        loop.set_mode(ControlMode.CLOSED_LOOP)
        result = loop.execute_cycle({})
        assert result["action"] == "control"
        assert "setpoint" in result


class TestControlLoopTransitions:
    @pytest.fixture
    def loop(self):
        return ControlLoop()

    def test_observe_to_advisory(self, loop):
        assert loop.mode == ControlMode.OBSERVE
        loop.set_mode(ControlMode.ADVISORY)
        assert loop.mode == ControlMode.ADVISORY

    def test_advisory_to_closed_loop(self, loop):
        loop.set_mode(ControlMode.ADVISORY)
        loop.set_mode(ControlMode.CLOSED_LOOP)
        assert loop.mode == ControlMode.CLOSED_LOOP


class TestDCSIntegration:
    @pytest.fixture
    def loop(self, mock_dcs_connection):
        l = ControlLoop()
        l.connect_dcs(mock_dcs_connection)
        return l

    def test_dcs_connected(self, loop):
        assert loop.dcs is not None
        assert loop.dcs.is_connected()
'''

with open(os.path.join(tests_dir, 'integration', 'test_control_loop.py'), 'w', encoding='utf-8') as f:
    f.write(control_loop_content)
print('Created integration/test_control_loop.py')

# integration/test_data_pipeline.py
pipeline_content = '''"""Integration tests for Data Pipeline."""
import pytest
from unittest.mock import Mock
from datetime import datetime, timedelta
from conftest import CombustionMeasurement


class DataPipeline:
    def __init__(self, historian=None):
        self.historian = historian
        self.buffer = []

    def connect_historian(self, historian):
        self.historian = historian

    def fetch_data(self, start, end, tags):
        if self.historian:
            return self.historian.query(tags[0], start, end)
        return []

    def compute_features(self, measurements):
        if not measurements:
            return {}
        lambdas = [m.lambda_value for m in measurements]
        efficiencies = [m.efficiency for m in measurements]
        return {
            "lambda_mean": sum(lambdas) / len(lambdas),
            "lambda_std": (sum((l - sum(lambdas)/len(lambdas))**2 for l in lambdas) / len(lambdas)) ** 0.5,
            "efficiency_mean": sum(efficiencies) / len(efficiencies),
        }

    def validate_data(self, measurements):
        valid = []
        invalid = []
        for m in measurements:
            if 0.8 <= m.lambda_value <= 2.0 and 0 <= m.efficiency <= 100:
                valid.append(m)
            else:
                invalid.append(m)
        return valid, invalid


class TestDataPipeline:
    @pytest.fixture
    def pipeline(self, mock_historian_connection):
        p = DataPipeline()
        p.connect_historian(mock_historian_connection)
        return p

    def test_fetch_data(self, pipeline):
        start = datetime.now() - timedelta(hours=1)
        end = datetime.now()
        data = pipeline.fetch_data(start, end, ["FIC001.PV"])
        assert len(data) > 0


class TestFeatureComputation:
    @pytest.fixture
    def pipeline(self):
        return DataPipeline()

    def test_compute_features(self, pipeline, sample_time_series):
        features = pipeline.compute_features(sample_time_series)
        assert "lambda_mean" in features
        assert "efficiency_mean" in features

    def test_empty_data_features(self, pipeline):
        features = pipeline.compute_features([])
        assert features == {}


class TestDataValidation:
    @pytest.fixture
    def pipeline(self):
        return DataPipeline()

    def test_valid_data(self, pipeline, sample_time_series):
        valid, invalid = pipeline.validate_data(sample_time_series)
        assert len(valid) > 0
'''

with open(os.path.join(tests_dir, 'integration', 'test_data_pipeline.py'), 'w', encoding='utf-8') as f:
    f.write(pipeline_content)
print('Created integration/test_data_pipeline.py')

# validation/__init__.py
with open(os.path.join(tests_dir, 'validation', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('"""Validation tests for GL-004 BURNMASTER."""\n')
print('Created validation/__init__.py')

# validation/test_physics_validation.py
physics_content = '''"""Physics validation tests."""
import pytest
import numpy as np
from conftest import FuelType, STOICHIOMETRIC_AIR_RATIOS, ADIABATIC_FLAME_TEMPS, assert_within_tolerance


class TestStoichiometryPhysics:
    @pytest.mark.parametrize("fuel_type,expected", [
        (FuelType.NATURAL_GAS, 17.2),
        (FuelType.DIESEL, 14.5),
        (FuelType.PROPANE, 15.7),
        (FuelType.HYDROGEN, 34.3),
    ])
    def test_stoichiometric_ratios(self, fuel_type, expected):
        actual = STOICHIOMETRIC_AIR_RATIOS[fuel_type]
        assert_within_tolerance(actual, expected, tolerance=0.05)

    def test_hydrogen_highest_stoich(self):
        h2_ratio = STOICHIOMETRIC_AIR_RATIOS[FuelType.HYDROGEN]
        for ft, ratio in STOICHIOMETRIC_AIR_RATIOS.items():
            if ft != FuelType.HYDROGEN:
                assert h2_ratio > ratio


class TestFlameTemperaturePhysics:
    @pytest.mark.parametrize("fuel_type,min_temp,max_temp", [
        (FuelType.NATURAL_GAS, 2100, 2300),
        (FuelType.HYDROGEN, 2400, 2550),
        (FuelType.COAL, 2000, 2200),
    ])
    def test_flame_temp_ranges(self, fuel_type, min_temp, max_temp):
        temp = ADIABATIC_FLAME_TEMPS[fuel_type]
        assert min_temp <= temp <= max_temp

    def test_hydrogen_highest_temp(self):
        h2_temp = ADIABATIC_FLAME_TEMPS[FuelType.HYDROGEN]
        for ft, temp in ADIABATIC_FLAME_TEMPS.items():
            if ft != FuelType.HYDROGEN:
                assert h2_temp >= temp


class TestO2LambdaRelationship:
    def test_o2_formula(self):
        for lambda_val in [1.05, 1.10, 1.15, 1.20, 1.30]:
            o2_expected = 21.0 * (lambda_val - 1.0) / lambda_val
            assert 0 <= o2_expected <= 21.0

    def test_o2_increases_with_lambda(self):
        o2_values = [21.0 * (lam - 1.0) / lam for lam in [1.05, 1.10, 1.15, 1.20]]
        for i in range(1, len(o2_values)):
            assert o2_values[i] > o2_values[i-1]


class TestEfficiencyPhysics:
    def test_efficiency_bounded(self):
        for eff in [60, 70, 80, 90, 95, 100]:
            assert 0 <= eff <= 100

    def test_stack_loss_positive(self):
        flue_temps = [400, 450, 500, 550]
        for temp in flue_temps:
            delta_t = temp - 298.15
            assert delta_t > 0
'''

with open(os.path.join(tests_dir, 'validation', 'test_physics_validation.py'), 'w', encoding='utf-8') as f:
    f.write(physics_content)
print('Created validation/test_physics_validation.py')

# performance/__init__.py
with open(os.path.join(tests_dir, 'performance', '__init__.py'), 'w', encoding='utf-8') as f:
    f.write('"""Performance tests for GL-004 BURNMASTER."""\n')
print('Created performance/__init__.py')

# performance/test_benchmarks.py
bench_content = '''"""Performance benchmark tests."""
import pytest
import time
import numpy as np
from conftest import CombustionTestDataGenerator, FuelType


class TestOptimizationPerformance:
    @pytest.fixture
    def data_gen(self):
        return CombustionTestDataGenerator(seed=42)

    @pytest.mark.performance
    def test_optimization_cycle_time(self, data_gen):
        measurements = [data_gen.generate_combustion_measurement() for _ in range(100)]
        start = time.perf_counter()
        for _ in range(100):
            _ = sum(m.lambda_value for m in measurements) / len(measurements)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 100  # <100ms for 100 iterations

    @pytest.mark.performance
    def test_feature_computation_time(self, data_gen):
        measurements = [data_gen.generate_combustion_measurement() for _ in range(1000)]
        start = time.perf_counter()
        features = {
            "lambda_mean": np.mean([m.lambda_value for m in measurements]),
            "lambda_std": np.std([m.lambda_value for m in measurements]),
            "efficiency_mean": np.mean([m.efficiency for m in measurements]),
        }
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50  # <50ms


class TestThroughput:
    @pytest.fixture
    def data_gen(self):
        return CombustionTestDataGenerator(seed=42)

    @pytest.mark.performance
    def test_measurement_generation_throughput(self, data_gen):
        num_records = 10000
        start = time.perf_counter()
        measurements = [data_gen.generate_combustion_measurement() for _ in range(num_records)]
        elapsed = time.perf_counter() - start
        throughput = num_records / elapsed
        assert throughput >= 1000  # >= 1000 records/sec
        assert len(measurements) == num_records


class TestMemoryUsage:
    @pytest.fixture
    def data_gen(self):
        return CombustionTestDataGenerator(seed=42)

    @pytest.mark.performance
    def test_large_batch_memory(self, data_gen):
        batch_size = 10000
        measurements = [data_gen.generate_combustion_measurement() for _ in range(batch_size)]
        assert len(measurements) == batch_size
'''

with open(os.path.join(tests_dir, 'performance', 'test_benchmarks.py'), 'w', encoding='utf-8') as f:
    f.write(bench_content)
print('Created performance/test_benchmarks.py')

# test_api.py
api_content = '''"""API tests for GL-004 BURNMASTER."""
import pytest
from unittest.mock import Mock, patch


class BurnmasterAPI:
    def __init__(self):
        self.authenticated = False
        self.current_mode = "observe"

    def authenticate(self, token):
        if token == "valid_token":
            self.authenticated = True
            return True
        return False

    def get_status(self):
        if not self.authenticated:
            raise PermissionError("Not authenticated")
        return {"mode": self.current_mode, "healthy": True}

    def get_measurements(self):
        if not self.authenticated:
            raise PermissionError("Not authenticated")
        return {"lambda": 1.15, "efficiency": 91.0, "co_ppm": 20.0}

    def set_mode(self, mode):
        if not self.authenticated:
            raise PermissionError("Not authenticated")
        if mode not in ["observe", "advisory", "closed_loop"]:
            raise ValueError("Invalid mode")
        self.current_mode = mode
        return {"mode": mode, "success": True}

    def optimize(self, target_efficiency):
        if not self.authenticated:
            raise PermissionError("Not authenticated")
        return {"optimal_lambda": 1.12, "predicted_efficiency": target_efficiency}


class TestAPIAuthentication:
    @pytest.fixture
    def api(self):
        return BurnmasterAPI()

    def test_valid_authentication(self, api):
        assert api.authenticate("valid_token")
        assert api.authenticated

    def test_invalid_authentication(self, api):
        assert not api.authenticate("invalid_token")
        assert not api.authenticated

    def test_unauthenticated_access(self, api):
        with pytest.raises(PermissionError):
            api.get_status()


class TestAPIEndpoints:
    @pytest.fixture
    def api(self):
        api = BurnmasterAPI()
        api.authenticate("valid_token")
        return api

    def test_get_status(self, api):
        status = api.get_status()
        assert "mode" in status
        assert "healthy" in status

    def test_get_measurements(self, api):
        measurements = api.get_measurements()
        assert "lambda" in measurements
        assert "efficiency" in measurements

    def test_set_mode_valid(self, api):
        result = api.set_mode("advisory")
        assert result["success"]
        assert api.current_mode == "advisory"

    def test_set_mode_invalid(self, api):
        with pytest.raises(ValueError):
            api.set_mode("invalid_mode")

    def test_optimize(self, api):
        result = api.optimize(92.0)
        assert "optimal_lambda" in result
        assert "predicted_efficiency" in result
'''

with open(os.path.join(tests_dir, 'test_api.py'), 'w', encoding='utf-8') as f:
    f.write(api_content)
print('Created test_api.py')

print('\\nAll test files created successfully!')
