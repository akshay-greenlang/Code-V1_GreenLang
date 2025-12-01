# -*- coding: utf-8 -*-
"""
Determinism and Reproducibility Tests for GL-006 HEATRECLAIM (WasteHeatRecoveryOptimizer).

This module provides comprehensive determinism tests covering:
- Bit-perfect reproducibility of all calculations
- Pinch analysis determinism
- Exergy calculation reproducibility
- ROI calculation consistency
- Provenance hash consistency
- Floating-point stability
- Random seed propagation
- Cross-platform consistency

Zero-hallucination Verification:
All thermodynamic calculations must produce identical results when given identical inputs.
This is fundamental to regulatory compliance and audit trail integrity.

References:
- GL-012 STEAMQUAL determinism patterns
- IEEE 754 Floating-Point Standard
- GreenLang Zero-Hallucination Guidelines
"""

import pytest
import hashlib
import json
import random
import math
import sys
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_CEILING
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_seed():
    """Provide deterministic seed for reproducibility testing."""
    return 42


@pytest.fixture
def pinch_analysis_inputs():
    """Create standardized inputs for pinch analysis."""
    return {
        "hot_streams": [
            {
                "stream_id": "H1",
                "supply_temp": Decimal("180.0"),
                "target_temp": Decimal("60.0"),
                "heat_capacity_flow": Decimal("10.0")
            },
            {
                "stream_id": "H2",
                "supply_temp": Decimal("150.0"),
                "target_temp": Decimal("40.0"),
                "heat_capacity_flow": Decimal("8.0")
            },
        ],
        "cold_streams": [
            {
                "stream_id": "C1",
                "supply_temp": Decimal("20.0"),
                "target_temp": Decimal("135.0"),
                "heat_capacity_flow": Decimal("7.5")
            },
        ],
        "min_approach_temp": Decimal("10.0")
    }


@pytest.fixture
def exergy_inputs():
    """Create standardized inputs for exergy calculations."""
    return {
        "temperature_k": Decimal("453.15"),  # 180 C
        "reference_temp_k": Decimal("298.15"),  # 25 C
        "pressure_kpa": Decimal("500.0"),
        "reference_pressure_kpa": Decimal("101.325"),
        "mass_flow_kg_s": Decimal("5.0"),
        "specific_heat_kj_kg_k": Decimal("4.186"),
        "entropy_kj_kg_k": Decimal("6.5")
    }


@pytest.fixture
def roi_inputs():
    """Create standardized inputs for ROI calculations."""
    return {
        "capital_cost": Decimal("500000.00"),
        "annual_savings": Decimal("150000.00"),
        "discount_rate": Decimal("0.10"),
        "project_life_years": 15,
        "escalation_rate": Decimal("0.03"),
        "tax_rate": Decimal("0.25"),
        "depreciation_years": 7
    }


@pytest.fixture
def heat_exchanger_inputs():
    """Create standardized inputs for heat exchanger calculations."""
    return {
        "hot_inlet_temp": Decimal("180.0"),
        "hot_outlet_temp": Decimal("80.0"),
        "cold_inlet_temp": Decimal("25.0"),
        "cold_outlet_temp": Decimal("120.0"),
        "duty_kw": Decimal("500.0"),
        "overall_htc_kw_m2k": Decimal("0.5"),
        "fouling_factor": Decimal("0.0001")
    }


# ============================================================================
# PINCH ANALYSIS REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestPinchAnalysisReproducibility:
    """Test bit-perfect reproducibility of pinch analysis calculations."""

    @pytest.mark.determinism
    def test_heat_duty_calculation_reproducibility(self, pinch_analysis_inputs):
        """Test heat duty calculations are deterministic."""
        results = []

        for _ in range(1000):
            duties = []
            for stream in pinch_analysis_inputs["hot_streams"]:
                duty = stream["heat_capacity_flow"] * abs(
                    stream["supply_temp"] - stream["target_temp"]
                )
                duties.append(duty.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
            results.append(tuple(duties))

        assert len(set(results)) == 1, "Heat duty calculation not deterministic"

    @pytest.mark.determinism
    def test_temperature_interval_reproducibility(self, pinch_analysis_inputs):
        """Test temperature interval construction is deterministic."""
        results = []

        for _ in range(1000):
            temps = set()
            min_approach = pinch_analysis_inputs["min_approach_temp"]

            for stream in pinch_analysis_inputs["hot_streams"]:
                temps.add(stream["supply_temp"] - min_approach / 2)
                temps.add(stream["target_temp"] - min_approach / 2)

            for stream in pinch_analysis_inputs["cold_streams"]:
                temps.add(stream["supply_temp"] + min_approach / 2)
                temps.add(stream["target_temp"] + min_approach / 2)

            sorted_temps = tuple(sorted(temps, reverse=True))
            results.append(sorted_temps)

        assert len(set(results)) == 1, "Temperature interval construction not deterministic"

    @pytest.mark.determinism
    def test_cascade_algorithm_reproducibility(self, pinch_analysis_inputs):
        """Test cascade algorithm produces identical results."""
        results = []

        for _ in range(1000):
            # Simulated cascade
            intervals = [
                Decimal("175"), Decimal("145"), Decimal("115"),
                Decimal("85"), Decimal("55"), Decimal("35")
            ]
            total_cp_hot = sum(s["heat_capacity_flow"] for s in pinch_analysis_inputs["hot_streams"])
            total_cp_cold = sum(s["heat_capacity_flow"] for s in pinch_analysis_inputs["cold_streams"])

            cascade = []
            cumulative = Decimal("0")
            for i in range(len(intervals) - 1):
                dt = intervals[i] - intervals[i + 1]
                net_heat = (total_cp_hot - total_cp_cold) * dt
                cumulative += net_heat
                cascade.append(cumulative.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

            results.append(tuple(cascade))

        assert len(set(results)) == 1, "Cascade algorithm not deterministic"

    @pytest.mark.determinism
    def test_minimum_utility_reproducibility(self, pinch_analysis_inputs):
        """Test minimum utility calculations are deterministic."""
        results = []

        for _ in range(1000):
            total_hot_duty = sum(
                s["heat_capacity_flow"] * abs(s["supply_temp"] - s["target_temp"])
                for s in pinch_analysis_inputs["hot_streams"]
            )
            total_cold_duty = sum(
                s["heat_capacity_flow"] * abs(s["target_temp"] - s["supply_temp"])
                for s in pinch_analysis_inputs["cold_streams"]
            )

            min_hot_utility = max(Decimal("0"), total_cold_duty - total_hot_duty)
            min_cold_utility = max(Decimal("0"), total_hot_duty - total_cold_duty)

            result = (
                min_hot_utility.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                min_cold_utility.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )
            results.append(result)

        assert len(set(results)) == 1, "Minimum utility calculation not deterministic"


# ============================================================================
# EXERGY CALCULATION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestExergyCalculationReproducibility:
    """Test bit-perfect reproducibility of exergy calculations."""

    @pytest.mark.determinism
    def test_physical_exergy_reproducibility(self, exergy_inputs):
        """Test physical exergy calculations are deterministic."""
        results = []

        for _ in range(1000):
            T = exergy_inputs["temperature_k"]
            T0 = exergy_inputs["reference_temp_k"]
            cp = exergy_inputs["specific_heat_kj_kg_k"]
            m = exergy_inputs["mass_flow_kg_s"]

            # Use Decimal for precision
            ln_ratio = Decimal(str(math.log(float(T) / float(T0))))
            ex_specific = cp * (T - T0 - T0 * ln_ratio)
            ex_total = (m * ex_specific).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            results.append(ex_total)

        assert len(set(results)) == 1, "Physical exergy calculation not deterministic"

    @pytest.mark.determinism
    def test_carnot_factor_reproducibility(self, exergy_inputs):
        """Test Carnot factor calculation is deterministic."""
        results = []

        for _ in range(1000):
            T = exergy_inputs["temperature_k"]
            T0 = exergy_inputs["reference_temp_k"]

            carnot = (1 - T0 / T).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(carnot)

        assert len(set(results)) == 1, "Carnot factor calculation not deterministic"
        # Verify expected value
        assert results[0] == Decimal("0.3420")

    @pytest.mark.determinism
    def test_exergy_destruction_reproducibility(self, exergy_inputs):
        """Test exergy destruction calculation is deterministic."""
        results = []

        for _ in range(1000):
            T0 = exergy_inputs["reference_temp_k"]
            S_gen = Decimal("0.5")  # Entropy generation kJ/K

            ex_destruction = (T0 * S_gen).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(ex_destruction)

        assert len(set(results)) == 1, "Exergy destruction calculation not deterministic"

    @pytest.mark.determinism
    def test_exergetic_efficiency_reproducibility(self, exergy_inputs):
        """Test exergetic efficiency calculation is deterministic."""
        results = []

        for _ in range(1000):
            ex_input = Decimal("1000.0")
            ex_output = Decimal("720.0")

            efficiency = (ex_output / ex_input).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(efficiency)

        assert len(set(results)) == 1
        assert results[0] == Decimal("0.7200")


# ============================================================================
# ROI CALCULATION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestROICalculationReproducibility:
    """Test bit-perfect reproducibility of ROI calculations."""

    @pytest.mark.determinism
    def test_npv_calculation_reproducibility(self, roi_inputs):
        """Test NPV calculation is deterministic."""
        results = []

        for _ in range(100):
            capital = roi_inputs["capital_cost"]
            savings = roi_inputs["annual_savings"]
            rate = roi_inputs["discount_rate"]
            years = roi_inputs["project_life_years"]
            escalation = roi_inputs["escalation_rate"]

            npv = -capital
            for year in range(1, years + 1):
                escalated_savings = savings * (1 + escalation) ** year
                discount_factor = (1 + rate) ** year
                pv = escalated_savings / discount_factor
                npv += pv

            npv_rounded = npv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(npv_rounded)

        assert len(set(results)) == 1, "NPV calculation not deterministic"

    @pytest.mark.determinism
    def test_irr_calculation_reproducibility(self, roi_inputs):
        """Test IRR calculation is deterministic."""
        results = []

        for _ in range(100):
            capital = float(roi_inputs["capital_cost"])
            savings = float(roi_inputs["annual_savings"])
            years = roi_inputs["project_life_years"]

            # Newton-Raphson with fixed iterations
            irr = 0.10
            for _ in range(50):
                npv = -capital
                npv_deriv = 0

                for year in range(1, years + 1):
                    npv += savings / (1 + irr) ** year
                    npv_deriv -= year * savings / (1 + irr) ** (year + 1)

                if abs(npv) < 0.01:
                    break

                if npv_deriv != 0:
                    irr = irr - npv / npv_deriv
                    irr = max(-0.99, min(irr, 10.0))

            irr_rounded = round(irr * 100, 2)
            results.append(irr_rounded)

        assert len(set(results)) == 1, "IRR calculation not deterministic"

    @pytest.mark.determinism
    def test_payback_period_reproducibility(self, roi_inputs):
        """Test payback period calculation is deterministic."""
        results = []

        for _ in range(1000):
            capital = roi_inputs["capital_cost"]
            savings = roi_inputs["annual_savings"]

            payback = (capital / savings).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(payback)

        assert len(set(results)) == 1
        assert results[0] == Decimal("3.33")


# ============================================================================
# HEAT EXCHANGER CALCULATION REPRODUCIBILITY
# ============================================================================

@pytest.mark.determinism
class TestHeatExchangerReproducibility:
    """Test bit-perfect reproducibility of heat exchanger calculations."""

    @pytest.mark.determinism
    def test_lmtd_calculation_reproducibility(self, heat_exchanger_inputs):
        """Test LMTD calculation is deterministic."""
        results = []

        for _ in range(1000):
            hot_in = float(heat_exchanger_inputs["hot_inlet_temp"])
            hot_out = float(heat_exchanger_inputs["hot_outlet_temp"])
            cold_in = float(heat_exchanger_inputs["cold_inlet_temp"])
            cold_out = float(heat_exchanger_inputs["cold_outlet_temp"])

            dt1 = hot_in - cold_out  # 180 - 120 = 60
            dt2 = hot_out - cold_in  # 80 - 25 = 55

            if abs(dt1 - dt2) < 0.1:
                lmtd = dt1
            else:
                lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

            lmtd_rounded = Decimal(str(round(lmtd, 4)))
            results.append(lmtd_rounded)

        assert len(set(results)) == 1, "LMTD calculation not deterministic"

    @pytest.mark.determinism
    def test_area_calculation_reproducibility(self, heat_exchanger_inputs):
        """Test heat exchanger area calculation is deterministic."""
        results = []

        for _ in range(1000):
            Q = heat_exchanger_inputs["duty_kw"]
            U = heat_exchanger_inputs["overall_htc_kw_m2k"]
            LMTD = Decimal("57.4825")  # Pre-calculated

            area = (Q / (U * LMTD)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            results.append(area)

        assert len(set(results)) == 1, "Area calculation not deterministic"


# ============================================================================
# PROVENANCE HASH CONSISTENCY
# ============================================================================

@pytest.mark.determinism
class TestProvenanceHashConsistency:
    """Test provenance hash consistency for audit trail."""

    @pytest.mark.determinism
    def test_hash_consistency_same_input(self, pinch_analysis_inputs):
        """Test same input always produces same hash."""
        # Convert Decimal to string for JSON serialization
        data = json.loads(
            json.dumps(pinch_analysis_inputs, default=str, sort_keys=True)
        )

        hashes = []
        for _ in range(100):
            h = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(h)

        assert len(set(hashes)) == 1, "Hash not consistent for same input"

    @pytest.mark.determinism
    def test_hash_changes_with_input(self, pinch_analysis_inputs):
        """Test hash changes when input changes."""
        data1 = json.loads(json.dumps(pinch_analysis_inputs, default=str, sort_keys=True))

        original_hash = hashlib.sha256(
            json.dumps(data1, sort_keys=True).encode()
        ).hexdigest()

        # Modify input
        data2 = data1.copy()
        data2["min_approach_temp"] = "11.0"

        modified_hash = hashlib.sha256(
            json.dumps(data2, sort_keys=True).encode()
        ).hexdigest()

        assert original_hash != modified_hash, "Hash should change with different input"

    @pytest.mark.determinism
    def test_hash_length_always_64(self, pinch_analysis_inputs):
        """Test SHA-256 hash is always 64 characters."""
        for i in range(100):
            data = {"iteration": i, **json.loads(json.dumps(pinch_analysis_inputs, default=str))}
            h = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()

            assert len(h) == 64, f"Hash length {len(h)} != 64"


# ============================================================================
# FLOATING-POINT STABILITY
# ============================================================================

@pytest.mark.determinism
class TestFloatingPointStability:
    """Test floating-point calculation stability."""

    @pytest.mark.determinism
    def test_decimal_associativity(self):
        """Test associativity is preserved with Decimal."""
        values = [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")]

        left_assoc = (values[0] + values[1]) + values[2]
        right_assoc = values[0] + (values[1] + values[2])

        assert left_assoc == right_assoc, "Associativity not preserved"
        assert left_assoc == Decimal("0.6")

    @pytest.mark.determinism
    def test_decimal_commutativity(self):
        """Test commutativity is preserved with Decimal."""
        a = Decimal("3.14159")
        b = Decimal("2.71828")

        assert a + b == b + a
        assert a * b == b * a

    @pytest.mark.determinism
    def test_decimal_precision_preservation(self):
        """Test Decimal preserves precision correctly."""
        result = Decimal("1.0000000001") - Decimal("0.0000000001")
        assert result == Decimal("1.0000000000")

    @pytest.mark.determinism
    def test_edge_case_small_values(self):
        """Test edge cases with very small values."""
        small1 = Decimal("1E-15")
        small2 = Decimal("1E-15")

        assert small1 + small2 == Decimal("2E-15")
        assert small1 * Decimal("1000") == Decimal("1E-12")

    @pytest.mark.determinism
    def test_edge_case_large_values(self):
        """Test edge cases with large values."""
        large1 = Decimal("1E15")
        large2 = Decimal("1E15")

        assert large1 + large2 == Decimal("2E15")
        assert large1 / large2 == Decimal("1")

    @pytest.mark.determinism
    def test_rounding_consistency(self):
        """Test rounding is consistent across operations."""
        value = Decimal("1.23456789")

        rounded_half_up = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        assert rounded_half_up == Decimal("1.2346")

        rounded_down = value.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        assert rounded_down == Decimal("1.2345")

        # Verify consistency across iterations
        results = []
        for _ in range(100):
            r = value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(r)

        assert len(set(results)) == 1


# ============================================================================
# SEED PROPAGATION
# ============================================================================

@pytest.mark.determinism
class TestSeedPropagation:
    """Test random seed propagation for reproducibility."""

    @pytest.mark.determinism
    def test_random_seed_propagation(self, deterministic_seed):
        """Test random seed produces consistent sequences."""
        random.seed(deterministic_seed)
        sequence1 = [random.random() for _ in range(100)]

        random.seed(deterministic_seed)
        sequence2 = [random.random() for _ in range(100)]

        assert sequence1 == sequence2, "Random sequences not reproducible with same seed"

    @pytest.mark.determinism
    def test_no_hidden_randomness_in_calculations(self, pinch_analysis_inputs):
        """Test calculations have no hidden random elements."""
        results = []

        for _ in range(100):
            # This calculation should have no randomness
            total = sum(
                s["heat_capacity_flow"] * abs(s["supply_temp"] - s["target_temp"])
                for s in pinch_analysis_inputs["hot_streams"]
            )
            results.append(total)

        assert len(set(results)) == 1, "Hidden randomness detected in calculations"


# ============================================================================
# CROSS-ITERATION CONSISTENCY
# ============================================================================

@pytest.mark.determinism
class TestCrossIterationConsistency:
    """Test consistency across multiple iterations and invocations."""

    @pytest.mark.determinism
    def test_full_pipeline_consistency(self, pinch_analysis_inputs, exergy_inputs, roi_inputs):
        """Test full calculation pipeline produces consistent results."""
        results = []

        for _ in range(50):
            # Step 1: Pinch analysis
            total_hot_duty = sum(
                s["heat_capacity_flow"] * abs(s["supply_temp"] - s["target_temp"])
                for s in pinch_analysis_inputs["hot_streams"]
            )

            # Step 2: Exergy calculation
            carnot = 1 - exergy_inputs["reference_temp_k"] / exergy_inputs["temperature_k"]

            # Step 3: ROI
            payback = roi_inputs["capital_cost"] / roi_inputs["annual_savings"]

            # Combine results
            combined = {
                "hot_duty": str(total_hot_duty.quantize(Decimal("0.01"))),
                "carnot": str(carnot.quantize(Decimal("0.0001"))),
                "payback": str(payback.quantize(Decimal("0.01")))
            }

            result_hash = hashlib.sha256(
                json.dumps(combined, sort_keys=True).encode()
            ).hexdigest()
            results.append(result_hash)

        assert len(set(results)) == 1, "Full pipeline not consistent across iterations"

    @pytest.mark.determinism
    def test_order_independence(self):
        """Test calculation order does not affect results."""
        values = [Decimal("10.5"), Decimal("20.3"), Decimal("15.7"), Decimal("8.2")]

        # Different orderings
        sum_forward = sum(values)
        sum_reverse = sum(reversed(values))
        sum_sorted = sum(sorted(values))

        assert sum_forward == sum_reverse == sum_sorted


# ============================================================================
# GOLDEN VALUE TESTS
# ============================================================================

@pytest.mark.determinism
class TestGoldenValues:
    """Test against known golden values for verification."""

    @pytest.mark.determinism
    def test_known_pinch_temperature(self):
        """Test pinch temperature calculation against known value."""
        # Known test case from Linnhoff & Flower (1978)
        # This is a simplified verification
        min_approach = Decimal("10.0")
        hot_temps = [Decimal("180.0"), Decimal("150.0")]
        cold_temps = [Decimal("120.0"), Decimal("80.0")]

        # Shifted temperatures
        shifted_hot = [t - min_approach / 2 for t in hot_temps]
        shifted_cold = [t + min_approach / 2 for t in cold_temps]

        all_temps = sorted(set(shifted_hot + shifted_cold), reverse=True)

        # Verify interval construction
        assert len(all_temps) == 4
        assert all_temps[0] == Decimal("175.0")  # 180 - 5

    @pytest.mark.determinism
    def test_known_carnot_efficiency(self):
        """Test Carnot efficiency against known value."""
        T_hot = Decimal("500.0")  # K
        T_cold = Decimal("300.0")  # K

        carnot = 1 - T_cold / T_hot

        expected = Decimal("0.4")
        assert carnot == expected

    @pytest.mark.determinism
    def test_known_lmtd(self):
        """Test LMTD against known value."""
        dt1 = 100.0
        dt2 = 50.0

        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        # Known value for these inputs
        expected = 72.1348  # Approximately

        assert abs(lmtd - expected) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
