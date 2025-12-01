# -*- coding: utf-8 -*-
"""
Unit Tests for SteamQualityCalculator.

This module provides comprehensive tests for the SteamQualityCalculator class,
covering dryness fraction calculations, superheat degree calculations, steam
state determination, and thermodynamic property calculations.

Coverage Target: 95%+
Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ASME PTC 19.11: Steam and Water Sampling

Test Categories:
1. Dryness fraction calculations (x = 0.0, 0.5, 0.9, 0.95, 1.0)
2. Superheat degree calculations (subcooled, saturated, superheated)
3. Steam quality index calculation
4. Steam state determination
5. Specific volume/enthalpy/entropy calculations
6. Edge cases: very low/high pressure, extreme temperatures
7. Determinism verification

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import hashlib
import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import test fixtures from conftest
from conftest import (
    SteamState,
    SteamProperties,
    generate_provenance_hash,
    assert_within_tolerance,
    assert_deterministic,
    calculate_saturation_temperature,
)


# =============================================================================
# MOCK CALCULATOR CLASSES FOR TESTING
# =============================================================================

@dataclass
class SteamQualityInput:
    """Input data for steam quality calculation."""
    pressure_bar: float
    temperature_c: float
    enthalpy_kj_kg: float = None
    entropy_kj_kg_k: float = None
    specific_volume_m3_kg: float = None


@dataclass
class SteamQualityOutput:
    """Output data from steam quality calculation."""
    dryness_fraction: float
    wetness_percent: float
    superheat_c: float
    state: SteamState
    quality_index: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    provenance_hash: str
    calculation_time_ms: float


class SteamQualityCalculator:
    """
    Steam quality calculator based on IAPWS-IF97.

    This is a test implementation that provides deterministic calculations
    for steam thermodynamic properties.
    """

    # IAPWS-IF97 Constants
    CRITICAL_PRESSURE_BAR = 220.64
    CRITICAL_TEMPERATURE_C = 373.946
    CRITICAL_DENSITY_KG_M3 = 322.0

    # Saturation property lookup table (simplified for testing)
    SATURATION_TABLE = {
        1.01325: {'Tsat': 100.0, 'hf': 419.1, 'hg': 2675.5, 'hfg': 2256.4,
                  'sf': 1.3069, 'sg': 7.3549, 'vf': 0.001044, 'vg': 1.6729},
        5.0: {'Tsat': 151.83, 'hf': 640.1, 'hg': 2747.5, 'hfg': 2107.4,
              'sf': 1.8604, 'sg': 6.8207, 'vf': 0.001093, 'vg': 0.3749},
        10.0: {'Tsat': 179.88, 'hf': 762.6, 'hg': 2776.2, 'hfg': 2013.6,
               'sf': 2.1382, 'sg': 6.5828, 'vf': 0.001127, 'vg': 0.1943},
        20.0: {'Tsat': 212.37, 'hf': 908.6, 'hg': 2797.2, 'hfg': 1888.6,
               'sf': 2.4467, 'sg': 6.3367, 'vf': 0.001177, 'vg': 0.09954},
        40.0: {'Tsat': 250.33, 'hf': 1087.4, 'hg': 2799.4, 'hfg': 1712.0,
               'sf': 2.7966, 'sg': 6.0670, 'vf': 0.001252, 'vg': 0.04978},
        100.0: {'Tsat': 311.0, 'hf': 1408.0, 'hg': 2725.5, 'hfg': 1317.5,
                'sf': 3.3603, 'sg': 5.6198, 'vf': 0.001452, 'vg': 0.01803},
    }

    def __init__(self):
        """Initialize steam quality calculator."""
        self.calculation_count = 0

    def _get_saturation_properties(self, pressure_bar: float) -> Dict[str, float]:
        """
        Get saturation properties for given pressure.

        Uses linear interpolation between table values.
        """
        if pressure_bar <= 0:
            raise ValueError("Pressure must be positive")

        if pressure_bar >= self.CRITICAL_PRESSURE_BAR:
            raise ValueError(f"Pressure {pressure_bar} bar exceeds critical pressure")

        # Find bounding pressures
        pressures = sorted(self.SATURATION_TABLE.keys())

        if pressure_bar <= pressures[0]:
            return self.SATURATION_TABLE[pressures[0]].copy()
        if pressure_bar >= pressures[-1]:
            return self.SATURATION_TABLE[pressures[-1]].copy()

        # Linear interpolation
        for i in range(len(pressures) - 1):
            if pressures[i] <= pressure_bar <= pressures[i + 1]:
                p1, p2 = pressures[i], pressures[i + 1]
                t1, t2 = self.SATURATION_TABLE[p1], self.SATURATION_TABLE[p2]
                ratio = (pressure_bar - p1) / (p2 - p1)

                result = {}
                for key in t1:
                    result[key] = t1[key] + ratio * (t2[key] - t1[key])
                return result

        return self.SATURATION_TABLE[pressures[0]].copy()

    def calculate_dryness_fraction(
        self,
        pressure_bar: float,
        enthalpy_kj_kg: float
    ) -> float:
        """
        Calculate dryness fraction (quality) from pressure and enthalpy.

        x = (h - hf) / hfg

        Args:
            pressure_bar: Steam pressure in bar
            enthalpy_kj_kg: Specific enthalpy in kJ/kg

        Returns:
            Dryness fraction (0.0 to 1.0)
        """
        sat_props = self._get_saturation_properties(pressure_bar)
        hf = sat_props['hf']
        hfg = sat_props['hfg']
        hg = sat_props['hg']

        if enthalpy_kj_kg <= hf:
            return 0.0  # Subcooled or saturated liquid
        elif enthalpy_kj_kg >= hg:
            return 1.0  # Saturated vapor or superheated
        else:
            # Wet steam region
            x = (enthalpy_kj_kg - hf) / hfg
            return max(0.0, min(1.0, x))

    def calculate_superheat(
        self,
        pressure_bar: float,
        temperature_c: float
    ) -> float:
        """
        Calculate superheat degree.

        Superheat = T - Tsat

        Args:
            pressure_bar: Steam pressure in bar
            temperature_c: Actual temperature in Celsius

        Returns:
            Superheat in degrees Celsius (negative for subcooled)
        """
        sat_props = self._get_saturation_properties(pressure_bar)
        tsat = sat_props['Tsat']
        return temperature_c - tsat

    def determine_state(
        self,
        pressure_bar: float,
        temperature_c: float,
        enthalpy_kj_kg: float = None
    ) -> SteamState:
        """
        Determine the thermodynamic state of steam.

        Args:
            pressure_bar: Steam pressure in bar
            temperature_c: Steam temperature in Celsius
            enthalpy_kj_kg: Specific enthalpy (optional)

        Returns:
            SteamState enumeration value
        """
        if pressure_bar >= self.CRITICAL_PRESSURE_BAR:
            return SteamState.SUPERCRITICAL

        sat_props = self._get_saturation_properties(pressure_bar)
        tsat = sat_props['Tsat']

        superheat = temperature_c - tsat

        if superheat < -0.1:  # Small tolerance for numerical precision
            return SteamState.SUBCOOLED
        elif superheat > 0.1:
            return SteamState.SUPERHEATED
        else:
            # At saturation - check enthalpy or assume vapor
            if enthalpy_kj_kg is not None:
                x = self.calculate_dryness_fraction(pressure_bar, enthalpy_kj_kg)
                if x <= 0.0:
                    return SteamState.SATURATED_LIQUID
                elif x >= 1.0:
                    return SteamState.SATURATED_VAPOR
                else:
                    return SteamState.WET_STEAM
            return SteamState.SATURATED_VAPOR

    def calculate_quality_index(
        self,
        dryness_fraction: float,
        superheat_c: float
    ) -> float:
        """
        Calculate steam quality index (0-100).

        Combines dryness and superheat into single quality metric.
        """
        if superheat_c > 0:
            # Superheated steam - full quality
            return 100.0
        else:
            # Based on dryness fraction
            return dryness_fraction * 100.0

    def calculate_specific_volume(
        self,
        pressure_bar: float,
        dryness_fraction: float
    ) -> float:
        """
        Calculate specific volume for wet steam.

        v = vf + x * vfg
        """
        sat_props = self._get_saturation_properties(pressure_bar)
        vf = sat_props['vf']
        vg = sat_props['vg']
        vfg = vg - vf

        return vf + dryness_fraction * vfg

    def calculate_specific_entropy(
        self,
        pressure_bar: float,
        dryness_fraction: float
    ) -> float:
        """
        Calculate specific entropy for wet steam.

        s = sf + x * sfg
        """
        sat_props = self._get_saturation_properties(pressure_bar)
        sf = sat_props['sf']
        sg = sat_props['sg']
        sfg = sg - sf

        return sf + dryness_fraction * sfg

    def calculate(self, input_data: SteamQualityInput) -> SteamQualityOutput:
        """
        Perform complete steam quality calculation.

        Args:
            input_data: SteamQualityInput with pressure, temperature, etc.

        Returns:
            SteamQualityOutput with all calculated properties
        """
        import time
        start_time = time.perf_counter()

        self.calculation_count += 1

        pressure = input_data.pressure_bar
        temperature = input_data.temperature_c

        # Calculate superheat
        superheat = self.calculate_superheat(pressure, temperature)

        # Determine state
        state = self.determine_state(
            pressure, temperature, input_data.enthalpy_kj_kg
        )

        # Calculate dryness fraction
        if input_data.enthalpy_kj_kg is not None:
            dryness = self.calculate_dryness_fraction(
                pressure, input_data.enthalpy_kj_kg
            )
        elif superheat > 0:
            dryness = 1.0  # Superheated
        elif superheat < 0:
            dryness = 0.0  # Subcooled
        else:
            dryness = 1.0  # Assume saturated vapor

        # Calculate quality index
        quality_index = self.calculate_quality_index(dryness, superheat)

        # Calculate wetness
        wetness = (1.0 - dryness) * 100.0

        # Get saturation properties
        sat_props = self._get_saturation_properties(pressure)

        # Calculate specific properties
        if state == SteamState.SUPERHEATED:
            enthalpy = sat_props['hg'] + 2.0 * superheat  # Simplified
            entropy = sat_props['sg'] + 0.005 * superheat  # Simplified
            volume = sat_props['vg'] * (1 + 0.003 * superheat)  # Simplified
        elif state == SteamState.SUBCOOLED:
            enthalpy = sat_props['hf'] + 4.2 * superheat  # Simplified (cp * dT)
            entropy = sat_props['sf'] + 0.01 * superheat
            volume = sat_props['vf']
        else:
            enthalpy = sat_props['hf'] + dryness * sat_props['hfg']
            entropy = self.calculate_specific_entropy(pressure, dryness)
            volume = self.calculate_specific_volume(pressure, dryness)

        # Override with input values if provided
        if input_data.enthalpy_kj_kg is not None:
            enthalpy = input_data.enthalpy_kj_kg
        if input_data.entropy_kj_kg_k is not None:
            entropy = input_data.entropy_kj_kg_k
        if input_data.specific_volume_m3_kg is not None:
            volume = input_data.specific_volume_m3_kg

        # Generate provenance hash
        hash_data = {
            'pressure_bar': pressure,
            'temperature_c': temperature,
            'dryness_fraction': round(dryness, 10),
            'state': state.value,
        }
        provenance_hash = generate_provenance_hash(hash_data)

        end_time = time.perf_counter()
        calc_time_ms = (end_time - start_time) * 1000

        return SteamQualityOutput(
            dryness_fraction=dryness,
            wetness_percent=wetness,
            superheat_c=superheat,
            state=state,
            quality_index=quality_index,
            specific_enthalpy_kj_kg=enthalpy,
            specific_entropy_kj_kg_k=entropy,
            specific_volume_m3_kg=volume,
            provenance_hash=provenance_hash,
            calculation_time_ms=calc_time_ms
        )


# =============================================================================
# TEST CLASS: DRYNESS FRACTION CALCULATIONS
# =============================================================================

class TestDrynessFractionCalculations:
    """Test suite for dryness fraction (quality) calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_dryness_fraction_x0_saturated_liquid(self, calculator):
        """Test dryness fraction = 0.0 for saturated liquid."""
        # At 10 bar, hf = 762.6 kJ/kg
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=762.6
        )

        assert x == 0.0, f"Expected x=0.0 for saturated liquid, got {x}"

    @pytest.mark.unit
    def test_dryness_fraction_x0_subcooled(self, calculator):
        """Test dryness fraction = 0.0 for subcooled water."""
        # Enthalpy below hf should give x = 0
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=500.0  # Below hf = 762.6
        )

        assert x == 0.0, f"Expected x=0.0 for subcooled, got {x}"

    @pytest.mark.unit
    def test_dryness_fraction_x0_5(self, calculator):
        """Test dryness fraction = 0.5 for 50% quality steam."""
        # At 10 bar: hf = 762.6, hfg = 2013.6
        # h = hf + 0.5 * hfg = 762.6 + 0.5 * 2013.6 = 1769.4 kJ/kg
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=1769.4
        )

        assert_within_tolerance(x, 0.5, 0.01, "Dryness fraction x=0.5")

    @pytest.mark.unit
    def test_dryness_fraction_x0_9(self, calculator):
        """Test dryness fraction = 0.9 for 90% quality steam."""
        # h = hf + 0.9 * hfg = 762.6 + 0.9 * 2013.6 = 2574.84 kJ/kg
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=2574.84
        )

        assert_within_tolerance(x, 0.9, 0.01, "Dryness fraction x=0.9")

    @pytest.mark.unit
    def test_dryness_fraction_x0_95(self, calculator):
        """Test dryness fraction = 0.95 for 95% quality steam."""
        # h = hf + 0.95 * hfg = 762.6 + 0.95 * 2013.6 = 2675.52 kJ/kg
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=2675.52
        )

        assert_within_tolerance(x, 0.95, 0.01, "Dryness fraction x=0.95")

    @pytest.mark.unit
    def test_dryness_fraction_x1_saturated_vapor(self, calculator):
        """Test dryness fraction = 1.0 for saturated vapor."""
        # At 10 bar, hg = 2776.2 kJ/kg
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=2776.2
        )

        assert x == 1.0, f"Expected x=1.0 for saturated vapor, got {x}"

    @pytest.mark.unit
    def test_dryness_fraction_x1_superheated(self, calculator):
        """Test dryness fraction = 1.0 for superheated steam."""
        # Enthalpy above hg should give x = 1.0
        x = calculator.calculate_dryness_fraction(
            pressure_bar=10.0,
            enthalpy_kj_kg=2900.0  # Above hg = 2776.2
        )

        assert x == 1.0, f"Expected x=1.0 for superheated, got {x}"

    @pytest.mark.unit
    @pytest.mark.parametrize("pressure_bar,enthalpy_kj_kg,expected_x", [
        (1.01325, 419.1, 0.0),    # Saturated liquid at 1 atm
        (1.01325, 1547.3, 0.5),   # 50% quality at 1 atm
        (1.01325, 2675.5, 1.0),   # Saturated vapor at 1 atm
        (5.0, 640.1, 0.0),        # Saturated liquid at 5 bar
        (5.0, 1693.8, 0.5),       # 50% quality at 5 bar
        (5.0, 2747.5, 1.0),       # Saturated vapor at 5 bar
        (40.0, 1087.4, 0.0),      # Saturated liquid at 40 bar
        (40.0, 1943.4, 0.5),      # 50% quality at 40 bar
        (40.0, 2799.4, 1.0),      # Saturated vapor at 40 bar
    ])
    def test_dryness_fraction_parametrized(
        self, calculator, pressure_bar, enthalpy_kj_kg, expected_x
    ):
        """Parametrized test for dryness fraction at various conditions."""
        x = calculator.calculate_dryness_fraction(pressure_bar, enthalpy_kj_kg)

        assert_within_tolerance(
            x, expected_x, 0.02,
            f"Dryness at {pressure_bar} bar, h={enthalpy_kj_kg}"
        )


# =============================================================================
# TEST CLASS: SUPERHEAT DEGREE CALCULATIONS
# =============================================================================

class TestSuperheatDegreeCalculations:
    """Test suite for superheat degree calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_superheat_subcooled(self, calculator):
        """Test negative superheat for subcooled water."""
        # At 10 bar, Tsat = 179.88C
        superheat = calculator.calculate_superheat(
            pressure_bar=10.0,
            temperature_c=150.0  # Below saturation
        )

        expected = 150.0 - 179.88
        assert_within_tolerance(superheat, expected, 0.5, "Subcooled superheat")
        assert superheat < 0, "Subcooled should have negative superheat"

    @pytest.mark.unit
    def test_superheat_saturated(self, calculator):
        """Test zero superheat at saturation."""
        # At 10 bar, Tsat = 179.88C
        superheat = calculator.calculate_superheat(
            pressure_bar=10.0,
            temperature_c=179.88
        )

        assert_within_tolerance(superheat, 0.0, 0.5, "Saturated superheat")

    @pytest.mark.unit
    def test_superheat_superheated_50k(self, calculator):
        """Test 50K superheat."""
        # Tsat + 50 at 10 bar
        superheat = calculator.calculate_superheat(
            pressure_bar=10.0,
            temperature_c=229.88
        )

        assert_within_tolerance(superheat, 50.0, 0.5, "50K superheat")

    @pytest.mark.unit
    def test_superheat_superheated_100k(self, calculator):
        """Test 100K superheat."""
        superheat = calculator.calculate_superheat(
            pressure_bar=10.0,
            temperature_c=279.88
        )

        assert_within_tolerance(superheat, 100.0, 0.5, "100K superheat")

    @pytest.mark.unit
    @pytest.mark.parametrize("pressure_bar,temp_c,expected_superheat", [
        (1.01325, 100.0, 0.0),     # Saturation at 1 atm
        (1.01325, 150.0, 50.0),    # 50K superheat at 1 atm
        (1.01325, 80.0, -20.0),    # 20K subcooled at 1 atm
        (10.0, 179.88, 0.0),       # Saturation at 10 bar
        (10.0, 250.0, 70.12),      # Superheat at 10 bar
        (40.0, 250.33, 0.0),       # Saturation at 40 bar
        (40.0, 350.33, 100.0),     # 100K superheat at 40 bar
    ])
    def test_superheat_parametrized(
        self, calculator, pressure_bar, temp_c, expected_superheat
    ):
        """Parametrized test for superheat at various conditions."""
        superheat = calculator.calculate_superheat(pressure_bar, temp_c)

        assert_within_tolerance(
            superheat, expected_superheat, 1.0,
            f"Superheat at {pressure_bar} bar, T={temp_c}C"
        )


# =============================================================================
# TEST CLASS: STEAM STATE DETERMINATION
# =============================================================================

class TestSteamStateDetermination:
    """Test suite for steam state determination."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_state_subcooled(self, calculator):
        """Test subcooled water state detection."""
        state = calculator.determine_state(
            pressure_bar=10.0,
            temperature_c=150.0  # Below Tsat = 179.88C
        )

        assert state == SteamState.SUBCOOLED

    @pytest.mark.unit
    def test_state_saturated_liquid(self, calculator):
        """Test saturated liquid state detection."""
        state = calculator.determine_state(
            pressure_bar=10.0,
            temperature_c=179.88,
            enthalpy_kj_kg=762.6  # hf at 10 bar
        )

        assert state == SteamState.SATURATED_LIQUID

    @pytest.mark.unit
    def test_state_wet_steam(self, calculator):
        """Test wet steam state detection."""
        state = calculator.determine_state(
            pressure_bar=10.0,
            temperature_c=179.88,
            enthalpy_kj_kg=1769.4  # 50% quality
        )

        assert state == SteamState.WET_STEAM

    @pytest.mark.unit
    def test_state_saturated_vapor(self, calculator):
        """Test saturated vapor state detection."""
        state = calculator.determine_state(
            pressure_bar=10.0,
            temperature_c=179.88,
            enthalpy_kj_kg=2776.2  # hg at 10 bar
        )

        assert state == SteamState.SATURATED_VAPOR

    @pytest.mark.unit
    def test_state_superheated(self, calculator):
        """Test superheated steam state detection."""
        state = calculator.determine_state(
            pressure_bar=10.0,
            temperature_c=250.0  # Above Tsat = 179.88C
        )

        assert state == SteamState.SUPERHEATED

    @pytest.mark.unit
    def test_state_supercritical(self, calculator):
        """Test supercritical state detection."""
        state = calculator.determine_state(
            pressure_bar=250.0,  # Above critical pressure
            temperature_c=400.0
        )

        assert state == SteamState.SUPERCRITICAL


# =============================================================================
# TEST CLASS: QUALITY INDEX CALCULATIONS
# =============================================================================

class TestQualityIndexCalculations:
    """Test suite for steam quality index calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_quality_index_saturated_liquid(self, calculator):
        """Test quality index for saturated liquid (0%)."""
        qi = calculator.calculate_quality_index(dryness_fraction=0.0, superheat_c=0.0)
        assert qi == 0.0

    @pytest.mark.unit
    def test_quality_index_50pct_quality(self, calculator):
        """Test quality index for 50% quality wet steam."""
        qi = calculator.calculate_quality_index(dryness_fraction=0.5, superheat_c=0.0)
        assert qi == 50.0

    @pytest.mark.unit
    def test_quality_index_90pct_quality(self, calculator):
        """Test quality index for 90% quality wet steam."""
        qi = calculator.calculate_quality_index(dryness_fraction=0.9, superheat_c=0.0)
        assert qi == 90.0

    @pytest.mark.unit
    def test_quality_index_saturated_vapor(self, calculator):
        """Test quality index for saturated vapor (100%)."""
        qi = calculator.calculate_quality_index(dryness_fraction=1.0, superheat_c=0.0)
        assert qi == 100.0

    @pytest.mark.unit
    def test_quality_index_superheated(self, calculator):
        """Test quality index for superheated steam (100%)."""
        qi = calculator.calculate_quality_index(dryness_fraction=1.0, superheat_c=50.0)
        assert qi == 100.0


# =============================================================================
# TEST CLASS: SPECIFIC PROPERTY CALCULATIONS
# =============================================================================

class TestSpecificPropertyCalculations:
    """Test suite for specific volume, enthalpy, and entropy calculations."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_specific_volume_saturated_liquid(self, calculator):
        """Test specific volume at x=0 (saturated liquid)."""
        v = calculator.calculate_specific_volume(pressure_bar=10.0, dryness_fraction=0.0)

        # vf at 10 bar = 0.001127 m3/kg
        assert_within_tolerance(v, 0.001127, 0.0001, "Specific volume at x=0")

    @pytest.mark.unit
    def test_specific_volume_saturated_vapor(self, calculator):
        """Test specific volume at x=1 (saturated vapor)."""
        v = calculator.calculate_specific_volume(pressure_bar=10.0, dryness_fraction=1.0)

        # vg at 10 bar = 0.1943 m3/kg
        assert_within_tolerance(v, 0.1943, 0.01, "Specific volume at x=1")

    @pytest.mark.unit
    def test_specific_volume_wet_steam(self, calculator):
        """Test specific volume for wet steam (x=0.9)."""
        v = calculator.calculate_specific_volume(pressure_bar=10.0, dryness_fraction=0.9)

        # v = vf + x*vfg = 0.001127 + 0.9*(0.1943-0.001127) = 0.1749 m3/kg
        expected = 0.001127 + 0.9 * (0.1943 - 0.001127)
        assert_within_tolerance(v, expected, 0.01, "Specific volume at x=0.9")

    @pytest.mark.unit
    def test_specific_entropy_saturated_liquid(self, calculator):
        """Test specific entropy at x=0."""
        s = calculator.calculate_specific_entropy(pressure_bar=10.0, dryness_fraction=0.0)

        # sf at 10 bar = 2.1382 kJ/kg.K
        assert_within_tolerance(s, 2.1382, 0.01, "Specific entropy at x=0")

    @pytest.mark.unit
    def test_specific_entropy_saturated_vapor(self, calculator):
        """Test specific entropy at x=1."""
        s = calculator.calculate_specific_entropy(pressure_bar=10.0, dryness_fraction=1.0)

        # sg at 10 bar = 6.5828 kJ/kg.K
        assert_within_tolerance(s, 6.5828, 0.01, "Specific entropy at x=1")

    @pytest.mark.unit
    @pytest.mark.parametrize("x,expected_v,expected_s", [
        (0.0, 0.001127, 2.1382),
        (0.25, 0.0495, 3.25),
        (0.5, 0.0977, 4.36),
        (0.75, 0.146, 5.47),
        (1.0, 0.1943, 6.5828),
    ])
    def test_specific_properties_parametrized(
        self, calculator, x, expected_v, expected_s
    ):
        """Parametrized test for specific properties at various qualities."""
        v = calculator.calculate_specific_volume(10.0, x)
        s = calculator.calculate_specific_entropy(10.0, x)

        # Allow larger tolerance for interpolated values
        assert_within_tolerance(v, expected_v, 0.02, f"Specific volume at x={x}")
        assert_within_tolerance(s, expected_s, 0.2, f"Specific entropy at x={x}")


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_very_low_pressure(self, calculator):
        """Test calculations at very low pressure (near vacuum)."""
        # 0.1 bar - uses minimum table value
        sat_props = calculator._get_saturation_properties(0.1)
        assert sat_props is not None
        assert sat_props['Tsat'] > 0

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_zero_pressure_raises_error(self, calculator):
        """Test that zero pressure raises ValueError."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculator._get_saturation_properties(0.0)

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_negative_pressure_raises_error(self, calculator):
        """Test that negative pressure raises ValueError."""
        with pytest.raises(ValueError, match="Pressure must be positive"):
            calculator._get_saturation_properties(-1.0)

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_above_critical_pressure_raises_error(self, calculator):
        """Test that pressure above critical point raises ValueError."""
        with pytest.raises(ValueError, match="exceeds critical pressure"):
            calculator._get_saturation_properties(250.0)

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_extreme_high_temperature(self, calculator):
        """Test superheat calculation at extreme temperature."""
        superheat = calculator.calculate_superheat(
            pressure_bar=10.0,
            temperature_c=500.0  # Very high superheat
        )

        expected = 500.0 - 179.88
        assert_within_tolerance(superheat, expected, 1.0, "Extreme superheat")

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_near_critical_pressure(self, calculator):
        """Test calculations near critical pressure."""
        # 200 bar - should still work
        sat_props = calculator._get_saturation_properties(150.0)
        assert sat_props is not None
        assert sat_props['hfg'] < 1000  # hfg decreases near critical

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_enthalpy_exactly_at_hf(self, calculator):
        """Test dryness when enthalpy is exactly hf."""
        x = calculator.calculate_dryness_fraction(10.0, 762.6)
        assert x == 0.0

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_enthalpy_exactly_at_hg(self, calculator):
        """Test dryness when enthalpy is exactly hg."""
        x = calculator.calculate_dryness_fraction(10.0, 2776.2)
        assert x == 1.0


# =============================================================================
# TEST CLASS: DETERMINISM VERIFICATION
# =============================================================================

class TestDeterminism:
    """Test suite for verifying calculation determinism."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_dryness_determinism_100_runs(self, calculator):
        """Test dryness fraction is deterministic across 100 runs."""
        results = []
        for _ in range(100):
            x = calculator.calculate_dryness_fraction(10.0, 2574.84)
            results.append(x)

        assert_deterministic(results, "Dryness fraction calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_superheat_determinism_100_runs(self, calculator):
        """Test superheat is deterministic across 100 runs."""
        results = []
        for _ in range(100):
            sh = calculator.calculate_superheat(10.0, 250.0)
            results.append(sh)

        assert_deterministic(results, "Superheat calculation")

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_full_calculation_determinism(self, calculator):
        """Test full calculation is deterministic."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0,
            enthalpy_kj_kg=2900.0
        )

        results = [calculator.calculate(input_data) for _ in range(100)]

        # Check all outputs are identical
        first = results[0]
        for i, r in enumerate(results[1:], 2):
            assert r.dryness_fraction == first.dryness_fraction, f"Run {i} differs"
            assert r.superheat_c == first.superheat_c, f"Run {i} superheat differs"
            assert r.state == first.state, f"Run {i} state differs"
            assert r.provenance_hash == first.provenance_hash, f"Run {i} hash differs"

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_provenance_hash_reproducibility(self, calculator):
        """Test provenance hash is reproducible for same inputs."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0,
            enthalpy_kj_kg=2900.0
        )

        result1 = calculator.calculate(input_data)
        result2 = calculator.calculate(input_data)

        assert result1.provenance_hash == result2.provenance_hash
        assert len(result1.provenance_hash) == 64  # SHA-256 hex length

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_no_random_in_calculations(self, calculator):
        """Test that calculations don't use unseeded random."""
        import random

        # Set random state
        random.seed(12345)
        state_before = random.getstate()

        # Run calculations
        for _ in range(10):
            calculator.calculate_dryness_fraction(10.0, 2000.0)
            calculator.calculate_superheat(10.0, 250.0)

        state_after = random.getstate()

        # State should be unchanged (no random usage)
        assert state_before == state_after, "Random state was modified"

    @pytest.mark.unit
    @pytest.mark.determinism
    def test_calculation_count_increments(self, calculator):
        """Test calculation counter increments correctly."""
        initial = calculator.calculation_count

        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0
        )

        calculator.calculate(input_data)
        calculator.calculate(input_data)
        calculator.calculate(input_data)

        assert calculator.calculation_count == initial + 3


# =============================================================================
# TEST CLASS: FULL CALCULATION WORKFLOW
# =============================================================================

class TestFullCalculationWorkflow:
    """Test suite for complete calculation workflow."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    def test_complete_calculation_saturated_vapor(self, calculator):
        """Test complete calculation for saturated vapor."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=179.88,
            enthalpy_kj_kg=2776.2
        )

        result = calculator.calculate(input_data)

        assert result.dryness_fraction == 1.0
        assert result.wetness_percent == 0.0
        assert abs(result.superheat_c) < 1.0  # Near zero
        assert result.state == SteamState.SATURATED_VAPOR
        assert result.quality_index == 100.0
        assert result.provenance_hash is not None
        assert result.calculation_time_ms >= 0

    @pytest.mark.unit
    def test_complete_calculation_wet_steam(self, calculator):
        """Test complete calculation for wet steam."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=179.88,
            enthalpy_kj_kg=2574.84  # x = 0.9
        )

        result = calculator.calculate(input_data)

        assert_within_tolerance(result.dryness_fraction, 0.9, 0.01, "Dryness")
        assert_within_tolerance(result.wetness_percent, 10.0, 1.0, "Wetness")
        assert result.state == SteamState.WET_STEAM
        assert_within_tolerance(result.quality_index, 90.0, 1.0, "Quality index")

    @pytest.mark.unit
    def test_complete_calculation_superheated(self, calculator):
        """Test complete calculation for superheated steam."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0
        )

        result = calculator.calculate(input_data)

        assert result.dryness_fraction == 1.0
        assert result.wetness_percent == 0.0
        assert result.superheat_c > 50.0
        assert result.state == SteamState.SUPERHEATED
        assert result.quality_index == 100.0

    @pytest.mark.unit
    def test_complete_calculation_subcooled(self, calculator):
        """Test complete calculation for subcooled water."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=150.0
        )

        result = calculator.calculate(input_data)

        assert result.dryness_fraction == 0.0
        assert result.wetness_percent == 100.0
        assert result.superheat_c < 0  # Subcooled
        assert result.state == SteamState.SUBCOOLED
        assert result.quality_index == 0.0


# =============================================================================
# TEST CLASS: IAPWS-IF97 COMPLIANCE
# =============================================================================

class TestIAPWSCompliance:
    """Test suite for IAPWS-IF97 standard compliance."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    @pytest.mark.iapws
    def test_saturation_temperature_1bar(self, calculator):
        """Test saturation temperature at 1.01325 bar (100C reference)."""
        sat_props = calculator._get_saturation_properties(1.01325)
        assert_within_tolerance(sat_props['Tsat'], 100.0, 0.5, "Tsat at 1 atm")

    @pytest.mark.unit
    @pytest.mark.iapws
    def test_saturation_enthalpy_10bar(self, calculator):
        """Test saturation enthalpies at 10 bar against IAPWS values."""
        sat_props = calculator._get_saturation_properties(10.0)

        # IAPWS-IF97 reference values for 10 bar
        assert_within_tolerance(sat_props['hf'], 762.6, 5.0, "hf at 10 bar")
        assert_within_tolerance(sat_props['hg'], 2776.2, 5.0, "hg at 10 bar")
        assert_within_tolerance(sat_props['hfg'], 2013.6, 10.0, "hfg at 10 bar")

    @pytest.mark.unit
    @pytest.mark.iapws
    def test_saturation_entropy_10bar(self, calculator):
        """Test saturation entropies at 10 bar against IAPWS values."""
        sat_props = calculator._get_saturation_properties(10.0)

        assert_within_tolerance(sat_props['sf'], 2.1382, 0.05, "sf at 10 bar")
        assert_within_tolerance(sat_props['sg'], 6.5828, 0.05, "sg at 10 bar")

    @pytest.mark.unit
    @pytest.mark.iapws
    def test_critical_point_values(self, calculator):
        """Test critical point constants are correct."""
        assert_within_tolerance(
            calculator.CRITICAL_PRESSURE_BAR, 220.64, 0.1, "Critical pressure"
        )
        assert_within_tolerance(
            calculator.CRITICAL_TEMPERATURE_C, 373.946, 0.1, "Critical temperature"
        )


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================

class TestPerformance:
    """Test suite for performance benchmarks."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_calculation_under_5ms(self, calculator, benchmark_targets):
        """Test single calculation completes under 5ms."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0,
            enthalpy_kj_kg=2900.0
        )

        result = calculator.calculate(input_data)

        assert result.calculation_time_ms < benchmark_targets['steam_quality_calculation_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_100_calculations(self, calculator, performance_timer, benchmark_targets):
        """Test 100 calculations complete within target time."""
        input_data = SteamQualityInput(
            pressure_bar=10.0,
            temperature_c=250.0,
            enthalpy_kj_kg=2900.0
        )

        with performance_timer() as timer:
            for _ in range(100):
                calculator.calculate(input_data)

        assert timer.elapsed_ms < benchmark_targets['batch_100_calculations_ms']

    @pytest.mark.unit
    @pytest.mark.performance
    def test_dryness_calculation_performance(self, calculator, performance_timer):
        """Test dryness fraction calculation is fast."""
        with performance_timer() as timer:
            for _ in range(1000):
                calculator.calculate_dryness_fraction(10.0, 2000.0)

        # 1000 calculations should complete in under 50ms
        assert timer.elapsed_ms < 50.0


# =============================================================================
# TEST CLASS: GOLDEN TEST CASES
# =============================================================================

class TestGoldenTestCases:
    """Test suite for golden test cases with known-good values."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return SteamQualityCalculator()

    @pytest.mark.unit
    @pytest.mark.golden
    def test_golden_saturated_10bar(self, calculator, golden_test_cases):
        """Golden test: saturated vapor at 10 bar."""
        case = golden_test_cases[0]  # steam_quality_10bar_saturated

        input_data = SteamQualityInput(
            pressure_bar=case['input']['pressure_bar'],
            temperature_c=case['input']['temperature_c'],
            enthalpy_kj_kg=case['input']['enthalpy_kj_kg']
        )

        result = calculator.calculate(input_data)

        assert_within_tolerance(
            result.dryness_fraction,
            case['expected']['dryness_fraction'],
            case['tolerance']['dryness_fraction'],
            "Golden: dryness fraction"
        )
        assert result.state == case['expected']['state']
        assert_within_tolerance(
            result.quality_index,
            case['expected']['quality_index'],
            case['tolerance']['quality_index'],
            "Golden: quality index"
        )

    @pytest.mark.unit
    @pytest.mark.golden
    def test_golden_wet_steam_90pct(self, calculator, golden_test_cases):
        """Golden test: wet steam at 90% quality."""
        case = golden_test_cases[1]  # steam_quality_10bar_wet_90pct

        input_data = SteamQualityInput(
            pressure_bar=case['input']['pressure_bar'],
            temperature_c=case['input']['temperature_c'],
            enthalpy_kj_kg=case['input']['enthalpy_kj_kg']
        )

        result = calculator.calculate(input_data)

        assert_within_tolerance(
            result.dryness_fraction,
            case['expected']['dryness_fraction'],
            case['tolerance']['dryness_fraction'],
            "Golden: dryness fraction"
        )
        assert result.state == case['expected']['state']
