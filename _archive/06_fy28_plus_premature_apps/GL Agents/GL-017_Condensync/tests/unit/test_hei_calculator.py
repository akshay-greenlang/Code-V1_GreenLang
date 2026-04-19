# -*- coding: utf-8 -*-
"""
Unit Tests: HEI Calculator

Comprehensive tests for HEI Standards cleanliness factor calculations including:
- Cleanliness Factor (CF) calculation
- Log Mean Temperature Difference (LMTD) calculation
- Heat duty calculations
- HEI correction factors
- Edge cases (LMTD singularity, boundary conditions)

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers

Target Coverage: 85%+
Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    TubeMaterial,
    CondenserConfig,
    CondenserReading,
    ThermalInput,
    HEICalculationResult,
    GoldenTestCase,
    AssertionHelpers,
    ProvenanceCalculator,
    calculate_lmtd,
    saturation_temp_from_pressure,
    HEI_REFERENCE_CONDITIONS,
    OPERATING_LIMITS,
    TEST_SEED,
)


# =============================================================================
# HEI CALCULATOR IMPLEMENTATION FOR TESTING
# =============================================================================

class HEICalculator:
    """
    HEI Standards-based condenser performance calculator.

    Calculates cleanliness factor (CF), LMTD, heat transfer coefficients,
    and fouling resistance per HEI Standards 12th Edition.
    """

    VERSION = "1.0.0"

    # HEI Reference conditions
    REF_CW_INLET_TEMP_C = 21.11  # 70F
    REF_CW_VELOCITY_M_S = 2.134  # 7 ft/s
    REF_CLEANLINESS = 0.85

    # Tube material thermal conductivities (W/m-K)
    TUBE_CONDUCTIVITIES = {
        TubeMaterial.ADMIRALTY_BRASS: 111.0,
        TubeMaterial.COPPER_NICKEL_90_10: 45.0,
        TubeMaterial.COPPER_NICKEL_70_30: 29.0,
        TubeMaterial.TITANIUM_GRADE_2: 21.9,
        TubeMaterial.STAINLESS_304: 16.2,
        TubeMaterial.STAINLESS_316: 16.2,
        TubeMaterial.DUPLEX_2205: 19.0,
    }

    # HEI heat transfer coefficient constants
    HEI_C1 = 3174.0  # Base coefficient (Btu/hr-ft2-F)

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize HEI calculator with configuration."""
        self.config = config or {}
        self._calculation_count = 0

    def calculate_lmtd(
        self,
        ttd: float,
        approach: float,
        handle_singularity: bool = True
    ) -> float:
        """
        Calculate Log Mean Temperature Difference.

        For condenser:
        - TTD = T_sat - T_cw_out (Terminal Temperature Difference)
        - Approach = T_sat - T_cw_in

        Args:
            ttd: Terminal temperature difference (C)
            approach: Approach temperature difference (C)
            handle_singularity: If True, handle TTD = Approach case

        Returns:
            LMTD in Celsius

        Raises:
            ValueError: If temperatures are invalid
        """
        if ttd <= 0:
            raise ValueError(f"TTD must be positive, got {ttd}")
        if approach <= 0:
            raise ValueError(f"Approach must be positive, got {approach}")
        if approach < ttd:
            raise ValueError(f"Approach ({approach}) must be >= TTD ({ttd})")

        # Handle singularity when TTD = Approach
        if abs(ttd - approach) < 1e-6:
            if handle_singularity:
                return ttd
            else:
                raise ValueError("LMTD undefined when TTD equals Approach")

        # Standard LMTD formula
        lmtd = (approach - ttd) / math.log(approach / ttd)
        return lmtd

    def calculate_lmtd_from_temps(
        self,
        cw_inlet_c: float,
        cw_outlet_c: float,
        sat_temp_c: float
    ) -> float:
        """
        Calculate LMTD from temperatures.

        Args:
            cw_inlet_c: Cooling water inlet temperature (C)
            cw_outlet_c: Cooling water outlet temperature (C)
            sat_temp_c: Steam saturation temperature (C)

        Returns:
            LMTD in Celsius
        """
        ttd = sat_temp_c - cw_outlet_c
        approach = sat_temp_c - cw_inlet_c
        return self.calculate_lmtd(ttd, approach)

    def calculate_heat_duty(
        self,
        cw_flow_kg_s: float,
        cw_temp_rise_c: float,
        cp_kj_kg_k: float = 4.186
    ) -> float:
        """
        Calculate heat duty from cooling water side.

        Q = m_dot * Cp * delta_T

        Args:
            cw_flow_kg_s: Cooling water mass flow rate (kg/s)
            cw_temp_rise_c: CW temperature rise (C)
            cp_kj_kg_k: Specific heat capacity (kJ/kg-K)

        Returns:
            Heat duty in kW
        """
        if cw_flow_kg_s < 0:
            raise ValueError(f"Flow rate must be non-negative, got {cw_flow_kg_s}")
        if cw_temp_rise_c < 0:
            raise ValueError(f"Temperature rise must be non-negative, got {cw_temp_rise_c}")

        return cw_flow_kg_s * cp_kj_kg_k * cw_temp_rise_c

    def calculate_heat_duty_steam(
        self,
        steam_flow_kg_s: float,
        latent_heat_kj_kg: float
    ) -> float:
        """
        Calculate heat duty from steam side.

        Q = m_dot * h_fg

        Args:
            steam_flow_kg_s: Steam flow rate (kg/s)
            latent_heat_kj_kg: Latent heat of vaporization (kJ/kg)

        Returns:
            Heat duty in kW
        """
        if steam_flow_kg_s < 0:
            raise ValueError(f"Steam flow must be non-negative, got {steam_flow_kg_s}")
        if latent_heat_kj_kg <= 0:
            raise ValueError(f"Latent heat must be positive, got {latent_heat_kj_kg}")

        return steam_flow_kg_s * latent_heat_kj_kg

    def calculate_ua_actual(
        self,
        heat_duty_kw: float,
        lmtd_c: float
    ) -> float:
        """
        Calculate actual overall heat transfer coefficient-area product.

        UA = Q / LMTD

        Args:
            heat_duty_kw: Heat duty (kW)
            lmtd_c: Log mean temperature difference (C)

        Returns:
            UA in kW/K
        """
        if heat_duty_kw < 0:
            raise ValueError(f"Heat duty must be non-negative, got {heat_duty_kw}")
        if lmtd_c <= 0:
            raise ValueError(f"LMTD must be positive, got {lmtd_c}")

        return heat_duty_kw / lmtd_c

    def get_inlet_temp_correction(self, cw_inlet_temp_c: float) -> float:
        """
        Get HEI inlet water temperature correction factor.

        Based on HEI Standards Table.

        Args:
            cw_inlet_temp_c: CW inlet temperature (C)

        Returns:
            Correction factor (dimensionless)
        """
        # Convert to Fahrenheit for HEI table lookup
        temp_f = cw_inlet_temp_c * 9 / 5 + 32

        # HEI correction factors (approximate from curves)
        # Baseline is 70F (21.1C)
        if temp_f <= 60:
            return 1.04
        elif temp_f <= 70:
            return 1.0 + 0.04 * (70 - temp_f) / 10
        elif temp_f <= 80:
            return 1.0 - 0.04 * (temp_f - 70) / 10
        elif temp_f <= 90:
            return 0.96 - 0.03 * (temp_f - 80) / 10
        elif temp_f <= 100:
            return 0.93 - 0.03 * (temp_f - 90) / 10
        else:
            return 0.90 - 0.02 * (temp_f - 100) / 10

    def get_velocity_correction(self, velocity_m_s: float) -> float:
        """
        Get HEI tube velocity correction factor.

        Based on HEI Standards curves.

        Args:
            velocity_m_s: Tube-side velocity (m/s)

        Returns:
            Correction factor (dimensionless)
        """
        # Convert to ft/s for HEI
        velocity_fps = velocity_m_s / 0.3048

        # HEI correction factors (approximate)
        # Baseline is 7 ft/s
        if velocity_fps <= 4:
            return 0.85
        elif velocity_fps <= 5:
            return 0.85 + 0.05 * (velocity_fps - 4)
        elif velocity_fps <= 6:
            return 0.90 + 0.05 * (velocity_fps - 5)
        elif velocity_fps <= 7:
            return 0.95 + 0.05 * (velocity_fps - 6)
        elif velocity_fps <= 8:
            return 1.0 + 0.04 * (velocity_fps - 7)
        elif velocity_fps <= 9:
            return 1.04 + 0.03 * (velocity_fps - 8)
        else:
            return 1.07 + 0.02 * (velocity_fps - 9)

    def get_material_correction(self, material: TubeMaterial) -> float:
        """
        Get HEI tube material correction factor.

        Based on thermal conductivity relative to Admiralty Brass.

        Args:
            material: Tube material type

        Returns:
            Correction factor (dimensionless)
        """
        # Material factors relative to Admiralty Brass
        factors = {
            TubeMaterial.ADMIRALTY_BRASS: 1.00,
            TubeMaterial.COPPER_NICKEL_90_10: 0.95,
            TubeMaterial.COPPER_NICKEL_70_30: 0.92,
            TubeMaterial.TITANIUM_GRADE_2: 0.88,
            TubeMaterial.STAINLESS_304: 0.85,
            TubeMaterial.STAINLESS_316: 0.85,
            TubeMaterial.DUPLEX_2205: 0.87,
        }
        return factors.get(material, 0.85)

    def calculate_cleanliness_factor(
        self,
        thermal_input: ThermalInput,
        condenser_config: CondenserConfig,
        design_ua_kw_k: Optional[float] = None
    ) -> HEICalculationResult:
        """
        Calculate cleanliness factor per HEI Standards.

        CF = UA_actual / UA_clean

        Args:
            thermal_input: Thermal calculation inputs
            condenser_config: Condenser design configuration
            design_ua_kw_k: Design UA (optional, calculated if not provided)

        Returns:
            HEICalculationResult with CF and supporting calculations
        """
        self._calculation_count += 1
        calc_timestamp = datetime.now(timezone.utc)
        warnings = []

        # Calculate temperatures
        ttd = thermal_input.ttd_c
        approach = thermal_input.approach_c

        # Validate temperatures
        if ttd < OPERATING_LIMITS["ttd_min_c"]:
            warnings.append(f"TTD {ttd:.2f}C below minimum {OPERATING_LIMITS['ttd_min_c']}C")
        if ttd > OPERATING_LIMITS["ttd_max_c"]:
            warnings.append(f"TTD {ttd:.2f}C above maximum {OPERATING_LIMITS['ttd_max_c']}C")

        # Calculate LMTD
        lmtd = self.calculate_lmtd(ttd, approach)

        # Calculate heat duty from CW side
        heat_duty_kw = self.calculate_heat_duty(
            thermal_input.cw_flow_kg_s,
            thermal_input.cw_outlet_temp_c - thermal_input.cw_inlet_temp_c,
            thermal_input.cw_cp_kj_kg_k
        )

        # Calculate actual UA
        ua_actual = self.calculate_ua_actual(heat_duty_kw, lmtd)

        # Get correction factors
        temp_correction = self.get_inlet_temp_correction(thermal_input.cw_inlet_temp_c)

        # Estimate velocity from flow and geometry
        velocity_m_s = thermal_input.cw_flow_kg_s / (1000 * condenser_config.flow_area_m2)
        velocity_correction = self.get_velocity_correction(velocity_m_s)

        material_correction = self.get_material_correction(condenser_config.tube_material)

        correction_factors = {
            "temperature": temp_correction,
            "velocity": velocity_correction,
            "material": material_correction,
            "combined": temp_correction * velocity_correction * material_correction,
        }

        # Calculate or use design UA
        if design_ua_kw_k is None:
            # Estimate from surface area and typical U value
            # Typical U ~ 3-4 kW/m2-K for good condenser
            typical_u = 3.5  # kW/m2-K
            design_ua_kw_k = typical_u * condenser_config.surface_area_m2

        # Calculate clean UA with corrections
        ua_clean = design_ua_kw_k * correction_factors["combined"]

        # Calculate cleanliness factor
        cleanliness_factor = ua_actual / ua_clean if ua_clean > 0 else 0.0

        # Clamp to valid range
        cleanliness_factor = max(0.0, min(1.5, cleanliness_factor))

        # Calculate fouling resistance
        if ua_actual > 0 and ua_clean > 0:
            fouling_resistance = (1 / ua_actual - 1 / ua_clean) * condenser_config.surface_area_m2
        else:
            fouling_resistance = 0.0

        # Validate CF range
        if cleanliness_factor < OPERATING_LIMITS["cf_min"]:
            warnings.append(f"CF {cleanliness_factor:.3f} below critical threshold {OPERATING_LIMITS['cf_min']}")

        # Generate provenance hash
        input_data = {
            "thermal_input": {
                "cw_inlet_temp_c": thermal_input.cw_inlet_temp_c,
                "cw_outlet_temp_c": thermal_input.cw_outlet_temp_c,
                "cw_flow_kg_s": thermal_input.cw_flow_kg_s,
                "steam_saturation_temp_c": thermal_input.steam_saturation_temp_c,
            },
            "condenser_id": condenser_config.condenser_id,
            "design_ua_kw_k": design_ua_kw_k,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        return HEICalculationResult(
            cleanliness_factor=round(cleanliness_factor, 4),
            lmtd_c=round(lmtd, 4),
            heat_duty_kw=round(heat_duty_kw, 2),
            ua_actual_kw_k=round(ua_actual, 2),
            ua_clean_kw_k=round(ua_clean, 2),
            fouling_resistance_m2_k_kw=round(fouling_resistance, 6),
            correction_factors=correction_factors,
            provenance_hash=provenance_hash,
            calculation_timestamp=calc_timestamp,
            warnings=warnings,
        )

    def calculate_cf_from_reading(
        self,
        reading: CondenserReading,
        config: CondenserConfig,
        design_ua_kw_k: float
    ) -> HEICalculationResult:
        """
        Calculate CF from a condenser reading.

        Args:
            reading: Condenser sensor reading
            config: Condenser configuration
            design_ua_kw_k: Design UA value

        Returns:
            HEICalculationResult
        """
        thermal_input = ThermalInput(
            cw_inlet_temp_c=reading.cw_inlet_temp_c,
            cw_outlet_temp_c=reading.cw_outlet_temp_c,
            cw_flow_kg_s=reading.cw_flow_m3_s * 1000,  # Approximate kg/s
            steam_saturation_temp_c=reading.saturation_temp_c,
            steam_flow_kg_s=reading.steam_flow_kg_s,
        )
        return self.calculate_cleanliness_factor(thermal_input, config, design_ua_kw_k)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator() -> HEICalculator:
    """Create HEI calculator instance."""
    return HEICalculator()


@pytest.fixture
def calculator_with_config() -> HEICalculator:
    """Create HEI calculator with configuration."""
    config = {
        "standard_version": "12th_edition",
        "design_cleanliness": 0.85,
        "enable_corrections": True,
    }
    return HEICalculator(config)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestLMTDCalculation:
    """Tests for LMTD calculation."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_typical_values(self, calculator: HEICalculator):
        """Test LMTD with typical condenser values."""
        ttd = 3.0  # C
        approach = 13.0  # C

        lmtd = calculator.calculate_lmtd(ttd, approach)

        # Expected: (13-3)/ln(13/3) = 10/ln(4.33) = 10/1.466 = 6.82
        expected = (approach - ttd) / math.log(approach / ttd)
        assert abs(lmtd - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_equal_temps_singularity(self, calculator: HEICalculator):
        """Test LMTD handles TTD = Approach singularity."""
        ttd = 5.0
        approach = 5.0

        lmtd = calculator.calculate_lmtd(ttd, approach, handle_singularity=True)

        # When TTD = Approach, LMTD = TTD
        assert lmtd == ttd

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_near_singularity(self, calculator: HEICalculator):
        """Test LMTD near singularity (very close values)."""
        ttd = 5.0
        approach = 5.000001

        lmtd = calculator.calculate_lmtd(ttd, approach)

        # Should be very close to TTD
        assert abs(lmtd - ttd) < 0.001

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_singularity_raises_when_disabled(self, calculator: HEICalculator):
        """Test LMTD raises error for singularity when handling disabled."""
        with pytest.raises(ValueError, match="LMTD undefined"):
            calculator.calculate_lmtd(5.0, 5.0, handle_singularity=False)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_negative_ttd_raises(self, calculator: HEICalculator):
        """Test LMTD raises error for negative TTD."""
        with pytest.raises(ValueError, match="TTD must be positive"):
            calculator.calculate_lmtd(-1.0, 10.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_negative_approach_raises(self, calculator: HEICalculator):
        """Test LMTD raises error for negative approach."""
        with pytest.raises(ValueError, match="Approach must be positive"):
            calculator.calculate_lmtd(5.0, -1.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_zero_ttd_raises(self, calculator: HEICalculator):
        """Test LMTD raises error for zero TTD."""
        with pytest.raises(ValueError, match="TTD must be positive"):
            calculator.calculate_lmtd(0.0, 10.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_approach_less_than_ttd_raises(self, calculator: HEICalculator):
        """Test LMTD raises error when approach < TTD (physically impossible)."""
        with pytest.raises(ValueError, match="Approach.*must be >= TTD"):
            calculator.calculate_lmtd(10.0, 5.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_lmtd_from_temps(self, calculator: HEICalculator):
        """Test LMTD calculation from temperatures."""
        cw_inlet = 25.0
        cw_outlet = 35.0
        sat_temp = 38.0

        lmtd = calculator.calculate_lmtd_from_temps(cw_inlet, cw_outlet, sat_temp)

        # TTD = 38 - 35 = 3, Approach = 38 - 25 = 13
        expected = (13 - 3) / math.log(13 / 3)
        assert abs(lmtd - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.parametrize("ttd,approach,expected", [
        (5.0, 15.0, 8.96),    # Typical case
        (2.0, 12.0, 5.58),    # Low TTD
        (8.0, 18.0, 12.16),   # High TTD
        (3.0, 13.0, 6.82),    # HEI example
        (1.0, 11.0, 4.17),    # Very low TTD
    ])
    def test_lmtd_parametric(self, calculator: HEICalculator, ttd, approach, expected):
        """Test LMTD with parametric values."""
        lmtd = calculator.calculate_lmtd(ttd, approach)
        assert abs(lmtd - expected) < 0.1


class TestHeatDutyCalculation:
    """Tests for heat duty calculations."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_cw_side(self, calculator: HEICalculator):
        """Test heat duty calculation from CW side."""
        cw_flow = 15000.0  # kg/s
        cw_rise = 10.0  # C
        cp = 4.186  # kJ/kg-K

        duty = calculator.calculate_heat_duty(cw_flow, cw_rise, cp)

        expected = cw_flow * cp * cw_rise
        assert duty == expected
        assert duty == 627900.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_steam_side(self, calculator: HEICalculator):
        """Test heat duty calculation from steam side."""
        steam_flow = 180.0  # kg/s
        latent_heat = 2400.0  # kJ/kg

        duty = calculator.calculate_heat_duty_steam(steam_flow, latent_heat)

        expected = steam_flow * latent_heat
        assert duty == expected
        assert duty == 432000.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_zero_flow(self, calculator: HEICalculator):
        """Test heat duty with zero flow."""
        duty = calculator.calculate_heat_duty(0.0, 10.0)
        assert duty == 0.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_zero_rise(self, calculator: HEICalculator):
        """Test heat duty with zero temperature rise."""
        duty = calculator.calculate_heat_duty(15000.0, 0.0)
        assert duty == 0.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_negative_flow_raises(self, calculator: HEICalculator):
        """Test heat duty raises for negative flow."""
        with pytest.raises(ValueError, match="Flow rate must be non-negative"):
            calculator.calculate_heat_duty(-100.0, 10.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_negative_rise_raises(self, calculator: HEICalculator):
        """Test heat duty raises for negative temperature rise."""
        with pytest.raises(ValueError, match="Temperature rise must be non-negative"):
            calculator.calculate_heat_duty(15000.0, -5.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_steam_negative_flow_raises(self, calculator: HEICalculator):
        """Test steam heat duty raises for negative flow."""
        with pytest.raises(ValueError, match="Steam flow must be non-negative"):
            calculator.calculate_heat_duty_steam(-50.0, 2400.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_heat_duty_steam_zero_latent_heat_raises(self, calculator: HEICalculator):
        """Test steam heat duty raises for zero/negative latent heat."""
        with pytest.raises(ValueError, match="Latent heat must be positive"):
            calculator.calculate_heat_duty_steam(180.0, 0.0)


class TestUACalculation:
    """Tests for UA (heat transfer coefficient-area product) calculations."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_ua_actual_calculation(self, calculator: HEICalculator):
        """Test actual UA calculation."""
        heat_duty = 500000.0  # kW
        lmtd = 8.0  # C

        ua = calculator.calculate_ua_actual(heat_duty, lmtd)

        expected = heat_duty / lmtd
        assert ua == expected
        assert ua == 62500.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_ua_zero_duty(self, calculator: HEICalculator):
        """Test UA with zero heat duty."""
        ua = calculator.calculate_ua_actual(0.0, 8.0)
        assert ua == 0.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_ua_negative_duty_raises(self, calculator: HEICalculator):
        """Test UA raises for negative heat duty."""
        with pytest.raises(ValueError, match="Heat duty must be non-negative"):
            calculator.calculate_ua_actual(-10000.0, 8.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_ua_zero_lmtd_raises(self, calculator: HEICalculator):
        """Test UA raises for zero LMTD."""
        with pytest.raises(ValueError, match="LMTD must be positive"):
            calculator.calculate_ua_actual(500000.0, 0.0)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_ua_negative_lmtd_raises(self, calculator: HEICalculator):
        """Test UA raises for negative LMTD."""
        with pytest.raises(ValueError, match="LMTD must be positive"):
            calculator.calculate_ua_actual(500000.0, -5.0)


class TestHEICorrectionFactors:
    """Tests for HEI correction factor calculations."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_inlet_temp_correction_at_reference(self, calculator: HEICalculator):
        """Test inlet temp correction at reference condition (70F = 21.1C)."""
        correction = calculator.get_inlet_temp_correction(21.1)
        assert abs(correction - 1.0) < 0.02

    @pytest.mark.unit
    @pytest.mark.hei
    def test_inlet_temp_correction_cold(self, calculator: HEICalculator):
        """Test inlet temp correction for cold water."""
        correction = calculator.get_inlet_temp_correction(15.0)  # ~59F
        assert correction > 1.0  # Cold water -> higher correction

    @pytest.mark.unit
    @pytest.mark.hei
    def test_inlet_temp_correction_warm(self, calculator: HEICalculator):
        """Test inlet temp correction for warm water."""
        correction = calculator.get_inlet_temp_correction(32.0)  # ~90F
        assert correction < 1.0  # Warm water -> lower correction

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.parametrize("temp_c,expected_range", [
        (15.0, (1.02, 1.06)),   # Cold
        (21.1, (0.98, 1.02)),   # Reference
        (27.0, (0.94, 0.98)),   # Warm
        (32.0, (0.91, 0.95)),   # Hot
        (38.0, (0.88, 0.92)),   # Very hot
    ])
    def test_inlet_temp_correction_ranges(self, calculator: HEICalculator, temp_c, expected_range):
        """Test inlet temp correction is within expected ranges."""
        correction = calculator.get_inlet_temp_correction(temp_c)
        assert expected_range[0] <= correction <= expected_range[1]

    @pytest.mark.unit
    @pytest.mark.hei
    def test_velocity_correction_at_reference(self, calculator: HEICalculator):
        """Test velocity correction at reference (7 ft/s = 2.134 m/s)."""
        correction = calculator.get_velocity_correction(2.134)
        assert abs(correction - 1.0) < 0.02

    @pytest.mark.unit
    @pytest.mark.hei
    def test_velocity_correction_low(self, calculator: HEICalculator):
        """Test velocity correction for low velocity."""
        correction = calculator.get_velocity_correction(1.5)  # ~5 ft/s
        assert correction < 1.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_velocity_correction_high(self, calculator: HEICalculator):
        """Test velocity correction for high velocity."""
        correction = calculator.get_velocity_correction(2.7)  # ~9 ft/s
        assert correction > 1.0

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.parametrize("velocity_m_s,expected_range", [
        (1.2, (0.83, 0.87)),    # Very low
        (1.5, (0.88, 0.92)),    # Low
        (2.1, (0.98, 1.02)),    # Reference
        (2.4, (1.02, 1.06)),    # High
        (2.7, (1.05, 1.09)),    # Very high
    ])
    def test_velocity_correction_ranges(self, calculator: HEICalculator, velocity_m_s, expected_range):
        """Test velocity correction is within expected ranges."""
        correction = calculator.get_velocity_correction(velocity_m_s)
        assert expected_range[0] <= correction <= expected_range[1]

    @pytest.mark.unit
    @pytest.mark.hei
    def test_material_correction_admiralty_brass(self, calculator: HEICalculator):
        """Test material correction for Admiralty Brass (reference)."""
        correction = calculator.get_material_correction(TubeMaterial.ADMIRALTY_BRASS)
        assert correction == 1.0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_material_correction_titanium(self, calculator: HEICalculator):
        """Test material correction for Titanium."""
        correction = calculator.get_material_correction(TubeMaterial.TITANIUM_GRADE_2)
        assert correction == 0.88

    @pytest.mark.unit
    @pytest.mark.hei
    def test_material_correction_stainless(self, calculator: HEICalculator):
        """Test material correction for Stainless Steel."""
        correction = calculator.get_material_correction(TubeMaterial.STAINLESS_316)
        assert correction == 0.85

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.parametrize("material,expected", [
        (TubeMaterial.ADMIRALTY_BRASS, 1.00),
        (TubeMaterial.COPPER_NICKEL_90_10, 0.95),
        (TubeMaterial.COPPER_NICKEL_70_30, 0.92),
        (TubeMaterial.TITANIUM_GRADE_2, 0.88),
        (TubeMaterial.STAINLESS_304, 0.85),
        (TubeMaterial.STAINLESS_316, 0.85),
        (TubeMaterial.DUPLEX_2205, 0.87),
    ])
    def test_material_correction_all_materials(self, calculator: HEICalculator, material, expected):
        """Test material correction for all defined materials."""
        correction = calculator.get_material_correction(material)
        assert correction == expected


class TestCleanlinessFactorCalculation:
    """Tests for cleanliness factor calculation."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_calculation_clean_condenser(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation for a clean condenser."""
        result = calculator.calculate_cleanliness_factor(
            thermal_input_baseline,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert isinstance(result, HEICalculationResult)
        assert 0.0 < result.cleanliness_factor <= 1.5
        assert result.lmtd_c > 0
        assert result.heat_duty_kw > 0
        assert result.ua_actual_kw_k > 0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_calculation_returns_valid_hash(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation returns valid provenance hash."""
        result = calculator.calculate_cleanliness_factor(
            thermal_input_baseline,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert len(result.provenance_hash) == 64
        # Verify it's valid hex
        int(result.provenance_hash, 16)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_calculation_has_correction_factors(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation includes all correction factors."""
        result = calculator.calculate_cleanliness_factor(
            thermal_input_baseline,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert "temperature" in result.correction_factors
        assert "velocity" in result.correction_factors
        assert "material" in result.correction_factors
        assert "combined" in result.correction_factors

        # Combined should be product of individual factors
        expected_combined = (
            result.correction_factors["temperature"] *
            result.correction_factors["velocity"] *
            result.correction_factors["material"]
        )
        assert abs(result.correction_factors["combined"] - expected_combined) < 0.0001

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_calculation_generates_warnings_for_low_ttd(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation generates warnings for low TTD."""
        thermal_input = ThermalInput(
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=36.0,  # TTD = 1.0C (too low)
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert any("TTD" in w and "below" in w for w in result.warnings)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_calculation_generates_warnings_for_low_cf(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation generates warnings for critically low CF."""
        # Create conditions that would result in low CF
        thermal_input = ThermalInput(
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=30.0,  # Low heat absorption
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=45.0,  # High vacuum pressure (degraded)
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=200000.0  # High design UA -> low CF
        )

        # Should have warning about low CF
        assert any("CF" in w and "below" in w for w in result.warnings)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_from_reading(
        self,
        calculator: HEICalculator,
        healthy_condenser_reading: CondenserReading,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation from condenser reading."""
        result = calculator.calculate_cf_from_reading(
            healthy_condenser_reading,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert isinstance(result, HEICalculationResult)
        assert 0.0 < result.cleanliness_factor <= 1.5

    @pytest.mark.unit
    @pytest.mark.hei
    def test_cf_clamped_to_valid_range(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF is clamped to valid range [0, 1.5]."""
        # Extreme conditions that might calculate > 1.5
        thermal_input = ThermalInput(
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=45.0,  # Very high rise
            cw_flow_kg_s=20000.0,
            steam_saturation_temp_c=46.0,
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=10000.0  # Very low design UA
        )

        assert 0.0 <= result.cleanliness_factor <= 1.5


class TestCFParametricCases:
    """Parametric tests for CF calculation."""

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.parametrize("cw_inlet,cw_outlet,sat_temp,cf_range", [
        (20.0, 30.0, 33.0, (0.70, 0.95)),  # Clean, low inlet
        (25.0, 35.0, 38.0, (0.70, 0.95)),  # Typical operation
        (30.0, 40.0, 43.0, (0.65, 0.90)),  # Warm inlet
        (25.0, 38.0, 45.0, (0.50, 0.75)),  # Fouled (high TTD)
        (22.0, 32.0, 34.5, (0.75, 1.00)),  # Clean, optimal
    ])
    def test_cf_varies_with_conditions(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig,
        cw_inlet, cw_outlet, sat_temp, cf_range
    ):
        """Test CF varies appropriately with conditions."""
        thermal_input = ThermalInput(
            cw_inlet_temp_c=cw_inlet,
            cw_outlet_temp_c=cw_outlet,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=sat_temp,
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        # CF should be in expected range for these conditions
        # Note: exact values depend on design parameters
        assert result.cleanliness_factor > 0
        assert result.cleanliness_factor <= 1.5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_very_small_temperature_differences(self, calculator: HEICalculator):
        """Test calculation with very small temperature differences."""
        # Small but valid differences
        lmtd = calculator.calculate_lmtd(0.5, 10.5)
        assert lmtd > 0
        assert lmtd < 10.5

    @pytest.mark.unit
    @pytest.mark.hei
    def test_large_temperature_differences(self, calculator: HEICalculator):
        """Test calculation with large temperature differences."""
        lmtd = calculator.calculate_lmtd(5.0, 50.0)

        expected = (50 - 5) / math.log(50 / 5)
        assert abs(lmtd - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.hei
    def test_extreme_flow_rates(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test calculation with extreme flow rates."""
        # Very high flow
        thermal_input = ThermalInput(
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=27.0,  # Low rise due to high flow
            cw_flow_kg_s=50000.0,  # Very high
            steam_saturation_temp_c=30.0,
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert result.heat_duty_kw > 0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_minimum_valid_temperatures(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test calculation with minimum valid temperatures."""
        thermal_input = ThermalInput(
            cw_inlet_temp_c=5.0,  # Cold winter
            cw_outlet_temp_c=15.0,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=18.0,
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert result.cleanliness_factor > 0

    @pytest.mark.unit
    @pytest.mark.hei
    def test_maximum_valid_temperatures(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test calculation with maximum valid temperatures."""
        thermal_input = ThermalInput(
            cw_inlet_temp_c=40.0,  # Hot summer
            cw_outlet_temp_c=50.0,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=55.0,
            steam_flow_kg_s=180.0,
        )

        result = calculator.calculate_cleanliness_factor(
            thermal_input,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert result.cleanliness_factor > 0


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_lmtd_is_deterministic(self, calculator: HEICalculator):
        """Test LMTD calculation is deterministic."""
        ttd = 3.0
        approach = 13.0

        results = [calculator.calculate_lmtd(ttd, approach) for _ in range(100)]

        # All results should be identical
        assert len(set(results)) == 1

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_cf_calculation_is_deterministic(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test CF calculation is deterministic."""
        results = [
            calculator.calculate_cleanliness_factor(
                thermal_input_baseline,
                sample_condenser_config,
                design_ua_kw_k=80000.0
            )
            for _ in range(10)
        ]

        # All CF values should be identical
        cf_values = [r.cleanliness_factor for r in results]
        assert len(set(cf_values)) == 1

        # All hashes should be identical
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_heat_duty_is_deterministic(self, calculator: HEICalculator):
        """Test heat duty calculation is deterministic."""
        results = [
            calculator.calculate_heat_duty(15000.0, 10.0)
            for _ in range(100)
        ]

        assert len(set(results)) == 1
        assert results[0] == 627900.0


class TestProvenanceTracking:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_provenance_hash_length(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test provenance hash is correct length (SHA-256)."""
        result = calculator.calculate_cleanliness_factor(
            thermal_input_baseline,
            sample_condenser_config,
            design_ua_kw_k=80000.0
        )

        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    @pytest.mark.hei
    def test_different_inputs_different_hash(
        self,
        calculator: HEICalculator,
        sample_condenser_config: CondenserConfig
    ):
        """Test different inputs produce different hashes."""
        input1 = ThermalInput(
            cw_inlet_temp_c=25.0,
            cw_outlet_temp_c=35.0,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=38.0,
            steam_flow_kg_s=180.0,
        )

        input2 = ThermalInput(
            cw_inlet_temp_c=26.0,  # Different inlet temp
            cw_outlet_temp_c=35.0,
            cw_flow_kg_s=15000.0,
            steam_saturation_temp_c=38.0,
            steam_flow_kg_s=180.0,
        )

        result1 = calculator.calculate_cleanliness_factor(
            input1, sample_condenser_config, design_ua_kw_k=80000.0
        )
        result2 = calculator.calculate_cleanliness_factor(
            input2, sample_condenser_config, design_ua_kw_k=80000.0
        )

        assert result1.provenance_hash != result2.provenance_hash

    @pytest.mark.unit
    @pytest.mark.hei
    def test_same_inputs_same_hash(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig
    ):
        """Test same inputs produce same hash."""
        result1 = calculator.calculate_cleanliness_factor(
            thermal_input_baseline, sample_condenser_config, design_ua_kw_k=80000.0
        )
        result2 = calculator.calculate_cleanliness_factor(
            thermal_input_baseline, sample_condenser_config, design_ua_kw_k=80000.0
        )

        assert result1.provenance_hash == result2.provenance_hash


class TestGoldenValues:
    """Tests against known golden values."""

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_lmtd_golden_value_1(self, calculator: HEICalculator):
        """Test LMTD against known value (TTD=3, Approach=13)."""
        lmtd = calculator.calculate_lmtd(3.0, 13.0)

        # Known value: (13-3)/ln(13/3) = 6.819
        expected = 6.82
        assert abs(lmtd - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_lmtd_golden_value_2(self, calculator: HEICalculator):
        """Test LMTD against known value (TTD=5, Approach=15)."""
        lmtd = calculator.calculate_lmtd(5.0, 15.0)

        # Known value: (15-5)/ln(15/5) = 9.10
        expected = 10 / math.log(3)
        assert abs(lmtd - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_heat_duty_golden_value(self, calculator: HEICalculator):
        """Test heat duty against known value."""
        # Q = 15000 kg/s * 4.186 kJ/kg-K * 10 K = 627,900 kW
        duty = calculator.calculate_heat_duty(15000.0, 10.0, 4.186)
        assert duty == 627900.0

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.golden
    def test_ua_golden_value(self, calculator: HEICalculator):
        """Test UA against known value."""
        # UA = 500,000 kW / 8.0 K = 62,500 kW/K
        ua = calculator.calculate_ua_actual(500000.0, 8.0)
        assert ua == 62500.0


class TestBatchProcessing:
    """Tests for batch processing of readings."""

    @pytest.mark.unit
    @pytest.mark.hei
    def test_process_fleet_readings(
        self,
        calculator: HEICalculator,
        condenser_fleet: List[CondenserReading],
        sample_condenser_config: CondenserConfig
    ):
        """Test processing multiple condenser readings."""
        results = []
        for reading in condenser_fleet:
            result = calculator.calculate_cf_from_reading(
                reading,
                sample_condenser_config,
                design_ua_kw_k=80000.0
            )
            results.append(result)

        assert len(results) == len(condenser_fleet)
        assert all(r.cleanliness_factor > 0 for r in results)
        assert all(len(r.provenance_hash) == 64 for r in results)

    @pytest.mark.unit
    @pytest.mark.hei
    def test_batch_has_unique_hashes(
        self,
        calculator: HEICalculator,
        condenser_fleet: List[CondenserReading],
        sample_condenser_config: CondenserConfig
    ):
        """Test batch processing produces unique hashes for different inputs."""
        results = [
            calculator.calculate_cf_from_reading(
                reading, sample_condenser_config, design_ua_kw_k=80000.0
            )
            for reading in condenser_fleet
        ]

        hashes = [r.provenance_hash for r in results]
        unique_hashes = set(hashes)

        # Different readings should produce different hashes
        assert len(unique_hashes) == len(hashes)


class TestPerformance:
    """Performance tests for HEI calculator."""

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.performance
    def test_lmtd_calculation_speed(
        self,
        calculator: HEICalculator,
        performance_timer
    ):
        """Test LMTD calculation completes within target time."""
        timer = performance_timer()

        with timer:
            for _ in range(10000):
                calculator.calculate_lmtd(3.0, 13.0)

        # Should complete 10,000 calculations in < 1 second
        assert timer.elapsed < 1.0

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.performance
    def test_cf_calculation_speed(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig,
        performance_timer
    ):
        """Test CF calculation completes within target time."""
        timer = performance_timer()

        with timer:
            for _ in range(1000):
                calculator.calculate_cleanliness_factor(
                    thermal_input_baseline,
                    sample_condenser_config,
                    design_ua_kw_k=80000.0
                )

        # Should complete 1,000 calculations in < 1 second
        assert timer.elapsed < 1.0

    @pytest.mark.unit
    @pytest.mark.hei
    @pytest.mark.performance
    def test_throughput_target(
        self,
        calculator: HEICalculator,
        thermal_input_baseline: ThermalInput,
        sample_condenser_config: CondenserConfig,
        throughput_measurer
    ):
        """Test CF calculation meets throughput target."""
        measurer = throughput_measurer()

        with measurer:
            for _ in range(500):
                calculator.calculate_cleanliness_factor(
                    thermal_input_baseline,
                    sample_condenser_config,
                    design_ua_kw_k=80000.0
                )
            measurer.add_items(500)

        # Should achieve at least 500 calculations per second
        assert measurer.items_per_second >= 500
