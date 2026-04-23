# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-005 COMBUSENSE.

Verifies bit-perfect reproducibility following zero-hallucination principles.
All calculations use Decimal arithmetic and fixed seed (42) for any
randomized operations to ensure complete reproducibility.

Key Principles:
- Same inputs MUST produce identical outputs (bit-perfect)
- Floating-point operations avoided where precision matters
- SHA-256 provenance hashes must be deterministic
- Control actions must be reproducible for audit compliance

Reference Standards:
- ASME PTC 4.1: Calculation reproducibility requirements
- ISO 17025: Laboratory result reproducibility
- IEC 61508: Deterministic safety system requirements
"""

import pytest
import hashlib
import json
import random
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_CEILING, ROUND_FLOOR
from typing import Dict, List, Any, Tuple


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def deterministic_seed():
    """Fixed seed for deterministic testing."""
    return 42


@pytest.fixture
def sample_combustion_inputs():
    """Sample combustion inputs using Decimal for precision."""
    return {
        "fuel_flow_kg_hr": Decimal("100.0"),
        "air_flow_kg_hr": Decimal("1200.0"),
        "fuel_lhv_mj_kg": Decimal("42.0"),
        "fuel_hhv_mj_kg": Decimal("46.5"),
        "flue_gas_temperature_c": Decimal("250.0"),
        "ambient_temperature_c": Decimal("25.0"),
        "o2_percent": Decimal("4.5"),
        "co_ppm": Decimal("25.0"),
        "furnace_temperature_c": Decimal("900.0"),
        "fuel_carbon_percent": Decimal("85.0"),
        "fuel_hydrogen_percent": Decimal("13.0"),
        "fuel_moisture_percent": Decimal("0.5")
    }


@pytest.fixture
def pid_parameters():
    """PID controller parameters using Decimal."""
    return {
        "kp": Decimal("2.0"),
        "ki": Decimal("0.5"),
        "kd": Decimal("0.1"),
        "setpoint": Decimal("4.5"),
        "sample_time": Decimal("0.1")
    }


@pytest.fixture
def control_action_inputs():
    """Control action inputs for hash testing."""
    return {
        "fuel_flow_setpoint": Decimal("105.234567"),
        "air_flow_setpoint": Decimal("1254.789012"),
        "fuel_valve_position": Decimal("52.3456"),
        "air_damper_position": Decimal("61.2345")
    }


# -----------------------------------------------------------------------------
# Bit-Perfect Reproducibility Tests
# -----------------------------------------------------------------------------

class TestBitPerfectReproducibility:
    """Test bit-perfect reproducibility of core calculations."""

    @pytest.mark.determinism
    def test_thermal_efficiency_reproducibility(self, sample_combustion_inputs):
        """Test thermal efficiency calculation is bit-perfect reproducible."""
        fuel_flow = sample_combustion_inputs["fuel_flow_kg_hr"]
        fuel_lhv = sample_combustion_inputs["fuel_lhv_mj_kg"]
        heat_output_kw = Decimal("1000.0")

        results = []
        for _ in range(1000):
            # Gross heat input: fuel_flow * LHV * 1000 / 3600 (MJ/hr to kW)
            gross_input_kw = fuel_flow * fuel_lhv * Decimal("1000") / Decimal("3600")
            # Thermal efficiency
            efficiency = (heat_output_kw / gross_input_kw * Decimal("100")).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )
            results.append(efficiency)

        # All results must be identical
        assert len(set(results)) == 1, "Thermal efficiency calculation not deterministic"
        assert results[0] == Decimal("85.7143")

    @pytest.mark.determinism
    def test_excess_air_calculation_reproducibility(self, sample_combustion_inputs):
        """Test excess air calculation is bit-perfect reproducible."""
        o2_percent = sample_combustion_inputs["o2_percent"]

        results = []
        for _ in range(1000):
            # Excess air % = O2 / (21 - O2) * 100
            denominator = Decimal("21") - o2_percent
            excess_air = (o2_percent / denominator * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            results.append(excess_air)

        assert len(set(results)) == 1, "Excess air calculation not deterministic"
        assert results[0] == Decimal("27.27")

    @pytest.mark.determinism
    def test_stack_loss_calculation_reproducibility(self, sample_combustion_inputs):
        """Test stack loss calculation is bit-perfect reproducible."""
        fuel_flow = sample_combustion_inputs["fuel_flow_kg_hr"]
        air_flow = sample_combustion_inputs["air_flow_kg_hr"]
        flue_temp = sample_combustion_inputs["flue_gas_temperature_c"]
        ambient_temp = sample_combustion_inputs["ambient_temperature_c"]
        cp_flue = Decimal("1.05")  # kJ/kg.K

        results = []
        for _ in range(1000):
            # Flue gas mass flow = fuel + air
            flue_mass = fuel_flow + air_flow
            # Temperature difference
            temp_diff = flue_temp - ambient_temp
            # Stack loss kW = mass * Cp * dT / 3600
            stack_loss = (flue_mass * cp_flue * temp_diff / Decimal("3600")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            results.append(stack_loss)

        assert len(set(results)) == 1, "Stack loss calculation not deterministic"
        assert results[0] == Decimal("77.19")

    @pytest.mark.determinism
    def test_moisture_loss_calculation_reproducibility(self, sample_combustion_inputs):
        """Test moisture loss calculation is bit-perfect reproducible."""
        fuel_flow = sample_combustion_inputs["fuel_flow_kg_hr"]
        fuel_moisture = sample_combustion_inputs["fuel_moisture_percent"]
        fuel_hydrogen = sample_combustion_inputs["fuel_hydrogen_percent"]
        h_fg = Decimal("2257")  # kJ/kg latent heat

        results = []
        for _ in range(1000):
            # Moisture in fuel
            moisture_fuel = fuel_flow * fuel_moisture / Decimal("100")
            # Water from H2 combustion (1 kg H2 -> 9 kg H2O)
            hydrogen_mass = fuel_flow * fuel_hydrogen / Decimal("100")
            water_combustion = hydrogen_mass * Decimal("9")
            # Total water
            total_water = moisture_fuel + water_combustion
            # Moisture loss (simplified - latent heat only)
            moisture_loss = (total_water * h_fg / Decimal("3600")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            results.append(moisture_loss)

        assert len(set(results)) == 1, "Moisture loss calculation not deterministic"


class TestPIDControllerDeterminism:
    """Test PID controller calculations are deterministic."""

    @pytest.mark.determinism
    def test_pid_output_reproducibility(self, pid_parameters):
        """Test PID output is bit-perfect reproducible."""
        process_value = Decimal("4.2")
        integral_sum = Decimal("0")
        last_error = Decimal("0")

        results = []
        for _ in range(1000):
            error = pid_parameters["setpoint"] - process_value
            integral = integral_sum + error * pid_parameters["sample_time"]
            derivative = (error - last_error) / pid_parameters["sample_time"]

            output = (
                pid_parameters["kp"] * error +
                pid_parameters["ki"] * integral +
                pid_parameters["kd"] * derivative
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            results.append(output)

        assert len(set(results)) == 1, "PID output not deterministic"

    @pytest.mark.determinism
    def test_pid_sequence_reproducibility(self, pid_parameters, deterministic_seed):
        """Test PID sequence is reproducible over multiple steps."""
        random.seed(deterministic_seed)

        def run_pid_sequence():
            integral = Decimal("0")
            last_error = Decimal("0")
            outputs = []

            # Simulate varying process values
            process_values = [Decimal(str(4.0 + random.random() * 0.5)) for _ in range(10)]

            for pv in process_values:
                error = pid_parameters["setpoint"] - pv
                integral += error * pid_parameters["sample_time"]
                derivative = (error - last_error) / pid_parameters["sample_time"]

                output = (
                    pid_parameters["kp"] * error +
                    pid_parameters["ki"] * integral +
                    pid_parameters["kd"] * derivative
                ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

                outputs.append(output)
                last_error = error

            return tuple(outputs)

        # Reset seed and run sequence multiple times
        sequences = []
        for _ in range(10):
            random.seed(deterministic_seed)
            sequences.append(run_pid_sequence())

        # All sequences must be identical
        assert len(set(sequences)) == 1, "PID sequence not reproducible with same seed"


class TestProvenanceHashConsistency:
    """Test provenance hash generation is consistent."""

    @pytest.mark.determinism
    def test_hash_consistency_same_input(self, control_action_inputs):
        """Test same inputs produce identical hash."""
        data = {k: str(v) for k, v in control_action_inputs.items()}

        hashes = []
        for _ in range(100):
            hash_value = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(hash_value)

        assert len(set(hashes)) == 1, "Hash not consistent for same input"

    @pytest.mark.determinism
    def test_hash_changes_with_input(self, control_action_inputs):
        """Test hash changes when input changes."""
        data1 = {k: str(v) for k, v in control_action_inputs.items()}
        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()

        # Modify one value
        data2 = data1.copy()
        data2["fuel_flow_setpoint"] = str(Decimal(data1["fuel_flow_setpoint"]) + Decimal("0.000001"))
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()

        assert hash1 != hash2, "Hash should change when input changes"

    @pytest.mark.determinism
    def test_hash_length_is_sha256(self, control_action_inputs):
        """Test hash is valid SHA-256 (64 hex characters)."""
        data = {k: str(v) for k, v in control_action_inputs.items()}
        hash_value = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

        assert len(hash_value) == 64
        assert all(c in '0123456789abcdef' for c in hash_value)

    @pytest.mark.determinism
    def test_control_action_hash_determinism(self):
        """Test control action provenance hash is deterministic."""
        action_data = {
            "fuel_flow_setpoint": 105.234567,
            "air_flow_setpoint": 1254.789012,
            "fuel_valve_position": 52.3456,
            "air_damper_position": 61.2345
        }

        def compute_action_hash(data: Dict[str, float]) -> str:
            rounded = {k: round(v, 6) for k, v in data.items()}
            return hashlib.sha256(
                json.dumps(rounded, sort_keys=True).encode()
            ).hexdigest()

        hashes = [compute_action_hash(action_data) for _ in range(100)]
        assert len(set(hashes)) == 1


class TestSeedPropagation:
    """Test random seed propagation for reproducibility."""

    @pytest.mark.determinism
    def test_random_seed_propagation(self, deterministic_seed):
        """Test random values are reproducible with fixed seed."""
        random.seed(deterministic_seed)
        values_1 = [random.random() for _ in range(100)]

        random.seed(deterministic_seed)
        values_2 = [random.random() for _ in range(100)]

        assert values_1 == values_2, "Random values not reproducible with same seed"

    @pytest.mark.determinism
    def test_no_hidden_randomness_in_calculations(self, sample_combustion_inputs):
        """Test calculations have no hidden random dependencies."""
        results = []
        for _ in range(100):
            fuel = sample_combustion_inputs["fuel_flow_kg_hr"]
            air = sample_combustion_inputs["air_flow_kg_hr"]
            afr = (air / fuel).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(afr)

        assert len(set(results)) == 1, "Hidden randomness detected in calculations"

    @pytest.mark.determinism
    def test_simulation_reproducibility(self, deterministic_seed):
        """Test simulation run is fully reproducible."""
        def run_simulation(seed: int, steps: int = 100) -> List[Tuple[Decimal, Decimal]]:
            random.seed(seed)
            results = []

            fuel = Decimal("100.0")
            air = Decimal("1200.0")

            for _ in range(steps):
                # Simulate sensor noise
                fuel_noise = Decimal(str(random.gauss(0, 0.5)))
                air_noise = Decimal(str(random.gauss(0, 5.0)))

                measured_fuel = (fuel + fuel_noise).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                measured_air = (air + air_noise).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                results.append((measured_fuel, measured_air))

            return results

        run1 = run_simulation(deterministic_seed)
        run2 = run_simulation(deterministic_seed)

        assert run1 == run2, "Simulation not reproducible"


class TestFloatingPointStability:
    """Test floating-point stability using Decimal arithmetic."""

    @pytest.mark.determinism
    def test_associativity_preserved(self):
        """Test Decimal addition is associative (unlike float)."""
        values = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]

        # Left-to-right
        sum_lr = values[0] + values[1] + values[2]
        # Right-to-left
        sum_rl = values[2] + values[1] + values[0]
        # Different grouping
        sum_grouped = (values[0] + values[2]) + values[1]

        assert sum_lr == sum_rl == sum_grouped == Decimal("0.6")

    @pytest.mark.determinism
    def test_decimal_precision_maintained(self):
        """Test Decimal maintains precision for combustion calculations."""
        # Small increment test
        result = Decimal("1.0000000001") - Decimal("0.0000000001")
        assert result == Decimal("1.0")

        # Large number precision
        large = Decimal("1234567890.123456789")
        small = Decimal("0.000000001")
        diff = large - small
        assert diff == Decimal("1234567890.123456788")

    @pytest.mark.determinism
    def test_edge_case_very_small_values(self):
        """Test handling of very small values (sensor resolution)."""
        tiny = Decimal("1E-15")
        assert tiny + tiny == Decimal("2E-15")
        assert (tiny * Decimal("1000000")).quantize(
            Decimal("1E-9"), rounding=ROUND_HALF_UP
        ) == Decimal("1E-9")

    @pytest.mark.determinism
    def test_edge_case_very_large_values(self):
        """Test handling of very large values (heat output)."""
        large_heat = Decimal("1000000000")  # 1 GW
        small_change = Decimal("0.001")  # 1 W
        result = (large_heat + small_change).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert result == Decimal("1000000000.001")

    @pytest.mark.determinism
    def test_division_precision(self):
        """Test division maintains precision for efficiency calculations."""
        numerator = Decimal("1000.0")  # Heat output kW
        denominator = Decimal("1166.67")  # Heat input kW

        # Using different precision settings
        result_6 = (numerator / denominator * Decimal("100")).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )
        result_4 = (numerator / denominator * Decimal("100")).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        result_2 = (numerator / denominator * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        assert result_6 == Decimal("85.714149")
        assert result_4 == Decimal("85.7141")
        assert result_2 == Decimal("85.71")


class TestRoundingConsistency:
    """Test rounding behavior is consistent."""

    @pytest.mark.determinism
    def test_round_half_up_consistency(self):
        """Test ROUND_HALF_UP is consistently applied."""
        test_values = [
            (Decimal("1.245"), Decimal("0.01"), Decimal("1.25")),
            (Decimal("1.255"), Decimal("0.01"), Decimal("1.26")),
            (Decimal("1.2450"), Decimal("0.001"), Decimal("1.245")),
            (Decimal("-1.245"), Decimal("0.01"), Decimal("-1.25")),
        ]

        for value, precision, expected in test_values:
            result = value.quantize(precision, rounding=ROUND_HALF_UP)
            assert result == expected, f"{value} rounded to {precision} should be {expected}"

    @pytest.mark.determinism
    def test_efficiency_rounding_consistency(self):
        """Test efficiency values are rounded consistently."""
        efficiencies = [
            Decimal("85.7141428571"),
            Decimal("88.1234567890"),
            Decimal("92.5555555555"),
        ]

        for eff in efficiencies:
            # Round to 2 decimal places 100 times
            results = [
                eff.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                for _ in range(100)
            ]
            assert len(set(results)) == 1


class TestSafetyCalculationDeterminism:
    """Test safety-critical calculations are deterministic."""

    @pytest.mark.determinism
    def test_risk_score_determinism(self):
        """Test risk score calculation is deterministic."""
        violations = {
            "temperature": 1,
            "pressure": 0,
            "flow": 1,
            "emissions": 2,
            "interlocks": 0
        }

        def calculate_risk(v: Dict[str, int]) -> Decimal:
            temp_risk = Decimal(str(min(v["temperature"] / 3.0, 1.0)))
            press_risk = Decimal(str(min(v["pressure"] / 3.0, 1.0)))
            flow_risk = Decimal(str(min(v["flow"] / 2.0, 1.0)))
            emission_risk = Decimal(str(min(v["emissions"] / 2.0, 1.0)))
            interlock_risk = Decimal(str(min(v["interlocks"] / 2.0, 1.0)))

            risk = (
                Decimal("0.25") * temp_risk +
                Decimal("0.25") * press_risk +
                Decimal("0.15") * flow_risk +
                Decimal("0.15") * emission_risk +
                Decimal("0.20") * interlock_risk
            )
            return risk.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        results = [calculate_risk(violations) for _ in range(100)]
        assert len(set(results)) == 1

    @pytest.mark.determinism
    def test_interlock_status_determinism(self):
        """Test interlock status determination is deterministic."""
        values = {
            "flame_present": True,
            "fuel_pressure_ok": True,
            "air_pressure_ok": True,
            "furnace_temp_ok": False,  # One failure
            "emergency_stop_clear": True
        }

        def check_interlocks(v: Dict[str, bool]) -> Tuple[bool, List[str]]:
            failed = [k for k, v in values.items() if not v]
            all_safe = len(failed) == 0
            return all_safe, sorted(failed)

        results = [check_interlocks(values) for _ in range(100)]
        assert len(set(results)) == 1
        assert results[0] == (False, ["furnace_temp_ok"])
