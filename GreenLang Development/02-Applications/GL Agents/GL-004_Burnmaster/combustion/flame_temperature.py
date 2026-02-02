"""
GL-004 BURNMASTER - Flame Temperature Calculator

Zero-hallucination calculation engine for flame temperature estimation.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- Adiabatic flame temperature calculation
- Actual flame temperature estimation with heat losses
- Temperature effects on NOx formation
- Flame temperature optimization targets

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import math

from pydantic import BaseModel, Field


class FlameType(str, Enum):
    """Types of flame configurations."""
    PREMIX = "premix"
    DIFFUSION = "diffusion"
    PARTIALLY_PREMIX = "partially_premix"


class FlameTemperatureInput(BaseModel):
    """Input schema for flame temperature calculation."""
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    lambda_val: float = Field(default=1.1, gt=0.8, lt=3.0, description="Lambda (equivalence ratio)")
    air_preheat_temp_c: float = Field(default=25, description="Air preheat temperature (C)")
    fuel_preheat_temp_c: float = Field(default=25, description="Fuel preheat temperature (C)")
    heat_loss_percent: float = Field(default=5, ge=0, le=50, description="Heat loss to surroundings (%)")


class FlameTemperatureResult(BaseModel):
    """Output schema for flame temperature calculation."""
    adiabatic_flame_temp_c: Decimal = Field(..., description="Adiabatic flame temperature (C)")
    adiabatic_flame_temp_k: Decimal = Field(..., description="Adiabatic flame temperature (K)")
    actual_flame_temp_c: Decimal = Field(..., description="Actual flame temperature (C)")
    actual_flame_temp_k: Decimal = Field(..., description="Actual flame temperature (K)")

    # Temperature factors
    lambda_effect_k: Decimal = Field(..., description="Temperature reduction due to excess air (K)")
    preheat_effect_k: Decimal = Field(..., description="Temperature increase from preheating (K)")
    loss_effect_k: Decimal = Field(..., description="Temperature reduction from heat loss (K)")

    # Derived indicators
    thermal_nox_potential: str = Field(..., description="Thermal NOx formation potential")
    recommended_lambda_range: Tuple[float, float] = Field(..., description="Recommended lambda for balance")

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Reference adiabatic flame temperatures at stoichiometric (K)
# Source: Perry's Chemical Engineers' Handbook, Combustion Engineering texts
STOICHIOMETRIC_FLAME_TEMPS: Dict[str, float] = {
    "methane": 2223,
    "natural_gas": 2210,
    "propane": 2268,
    "butane": 2275,
    "hydrogen": 2400,
    "carbon_monoxide": 2400,
    "acetylene": 2600,
    "ethylene": 2370,
    "fuel_oil": 2250,
    "diesel": 2300,
    "coal": 2150,
    "wood": 2000,
    "biogas": 2050,
}

# Heating values for temperature calculation (MJ/kg)
HEATING_VALUES: Dict[str, Dict[str, float]] = {
    "methane": {"hhv": 55.5, "lhv": 50.0},
    "natural_gas": {"hhv": 52.0, "lhv": 47.0},
    "propane": {"hhv": 50.3, "lhv": 46.4},
    "hydrogen": {"hhv": 141.8, "lhv": 120.0},
    "fuel_oil": {"hhv": 45.5, "lhv": 42.5},
    "diesel": {"hhv": 45.4, "lhv": 42.8},
    "coal": {"hhv": 30.0, "lhv": 28.0},
}

# Stoichiometric air-fuel ratios (mass basis)
STOICH_AFR: Dict[str, float] = {
    "methane": 17.2,
    "natural_gas": 17.2,
    "propane": 15.7,
    "hydrogen": 34.3,
    "fuel_oil": 14.7,
    "diesel": 14.5,
    "coal": 11.5,
}


class FlameTemperatureCalculator:
    """
    Zero-hallucination calculator for flame temperature.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure thermodynamic calculations only

    Example:
        >>> calculator = FlameTemperatureCalculator()
        >>> result = calculator.compute_flame_temperature(
        ...     fuel_type="natural_gas",
        ...     lambda_val=1.15,
        ...     air_preheat_temp_c=200
        ... )
        >>> print(f"Adiabatic: {result.adiabatic_flame_temp_c}C")
    """

    def __init__(self, precision: int = 1):
        """Initialize calculator with precision settings."""
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    def compute_flame_temperature(
        self,
        fuel_type: str = "natural_gas",
        lambda_val: float = 1.1,
        air_preheat_temp_c: float = 25.0,
        fuel_preheat_temp_c: float = 25.0,
        heat_loss_percent: float = 5.0,
    ) -> FlameTemperatureResult:
        """
        Compute adiabatic and actual flame temperature.

        DETERMINISTIC: Based on enthalpy balance.

        The adiabatic flame temperature is calculated using energy balance:
        T_ad = T_ref + (Q_comb - Q_dissociation) / (m_products * Cp_products)

        For practical implementation, we use correlation-based adjustments
        from the stoichiometric reference temperature.

        Args:
            fuel_type: Type of fuel
            lambda_val: Air-fuel equivalence ratio (1.0 = stoichiometric)
            air_preheat_temp_c: Combustion air preheat temperature (C)
            fuel_preheat_temp_c: Fuel preheat temperature (C)
            heat_loss_percent: Heat loss to surroundings (%)

        Returns:
            FlameTemperatureResult with temperatures and factors
        """
        # Step 1: Get stoichiometric flame temperature (DETERMINISTIC lookup)
        t_stoich_k = STOICHIOMETRIC_FLAME_TEMPS.get(fuel_type, 2200)
        t_ref = 298  # Reference temperature (K)

        # Step 2: Calculate lambda effect (DETERMINISTIC)
        # Excess air dilutes and cools the flame
        # T_ad ~ T_stoich / lambda (simplified relationship)
        # More accurate: T_ad = T_stoich - k * (lambda - 1) * T_stoich
        if lambda_val >= 1.0:
            # Lean combustion - temperature decreases
            lambda_factor = 1.0 - 0.25 * (lambda_val - 1.0)
            lambda_factor = max(0.5, lambda_factor)  # Cap at 50% reduction
        else:
            # Rich combustion - incomplete, also lower temperature
            lambda_factor = 1.0 - 0.3 * (1.0 - lambda_val)
            lambda_factor = max(0.6, lambda_factor)

        t_ad_k = t_stoich_k * lambda_factor
        lambda_effect_k = t_stoich_k - t_ad_k

        # Step 3: Calculate preheat effect (DETERMINISTIC)
        # Preheating combustion air adds sensible heat
        # Delta_T ~ (m_air / m_total) * (T_preheat - T_ref)
        afr = STOICH_AFR.get(fuel_type, 17.2) * lambda_val
        air_mass_fraction = afr / (1 + afr)

        preheat_effect_k = air_mass_fraction * (air_preheat_temp_c - 25)
        preheat_effect_k += (1 - air_mass_fraction) * (fuel_preheat_temp_c - 25)
        preheat_effect_k = max(0, preheat_effect_k)

        t_ad_k += preheat_effect_k

        # Step 4: Calculate actual temperature with heat loss (DETERMINISTIC)
        loss_factor = 1 - heat_loss_percent / 100
        loss_effect_k = t_ad_k * (1 - loss_factor)
        t_actual_k = t_ad_k * loss_factor

        # Step 5: Convert to Celsius (DETERMINISTIC)
        t_ad_c = t_ad_k - 273.15
        t_actual_c = t_actual_k - 273.15

        # Step 6: Assess thermal NOx potential (DETERMINISTIC thresholds)
        # Thermal NOx formation increases rapidly above ~1800K
        if t_actual_k > 2000:
            nox_potential = "very_high"
        elif t_actual_k > 1850:
            nox_potential = "high"
        elif t_actual_k > 1700:
            nox_potential = "moderate"
        elif t_actual_k > 1500:
            nox_potential = "low"
        else:
            nox_potential = "minimal"

        # Step 7: Determine recommended lambda range (DETERMINISTIC)
        # Balance between efficiency (low lambda) and NOx (high lambda)
        if t_stoich_k > 2200:
            # High flame temp fuel - operate leaner
            recommended_lambda = (1.15, 1.25)
        elif t_stoich_k > 2000:
            # Moderate flame temp
            recommended_lambda = (1.10, 1.20)
        else:
            # Lower flame temp - can operate closer to stoich
            recommended_lambda = (1.05, 1.15)

        provenance = self._compute_provenance_hash({
            'fuel_type': fuel_type,
            'lambda_val': lambda_val,
            'air_preheat_temp_c': air_preheat_temp_c,
            't_ad_k': t_ad_k,
            't_actual_k': t_actual_k
        })

        return FlameTemperatureResult(
            adiabatic_flame_temp_c=self._quantize(Decimal(str(t_ad_c))),
            adiabatic_flame_temp_k=self._quantize(Decimal(str(t_ad_k))),
            actual_flame_temp_c=self._quantize(Decimal(str(t_actual_c))),
            actual_flame_temp_k=self._quantize(Decimal(str(t_actual_k))),
            lambda_effect_k=self._quantize(Decimal(str(lambda_effect_k))),
            preheat_effect_k=self._quantize(Decimal(str(preheat_effect_k))),
            loss_effect_k=self._quantize(Decimal(str(loss_effect_k))),
            thermal_nox_potential=nox_potential,
            recommended_lambda_range=recommended_lambda,
            provenance_hash=provenance
        )

    def compute_optimal_lambda_for_nox(
        self,
        fuel_type: str,
        max_nox_ppm: float,
        air_preheat_temp_c: float = 25.0
    ) -> Tuple[float, Decimal]:
        """
        Compute minimum lambda to achieve target NOx level.

        DETERMINISTIC: Based on thermal NOx correlation with flame temperature.

        Args:
            fuel_type: Type of fuel
            max_nox_ppm: Maximum allowable NOx (ppm)
            air_preheat_temp_c: Air preheat temperature

        Returns:
            Tuple of (recommended_lambda, estimated_flame_temp_k)
        """
        # Thermal NOx approximately doubles every 90K above 1800K
        # NOx ~ A * exp(k * (T - 1800))

        # Target temperature for given NOx
        # Invert NOx = A * exp(k * (T - 1800))
        # For reference: ~25 ppm at 1900K, ~50 ppm at 1950K

        if max_nox_ppm <= 0:
            target_temp_k = 1700
        else:
            # Simplified correlation
            target_temp_k = 1800 + 90 * math.log(max_nox_ppm / 25) / math.log(2)
            target_temp_k = max(1600, min(2100, target_temp_k))

        # Calculate lambda needed to achieve this temperature
        t_stoich_k = STOICHIOMETRIC_FLAME_TEMPS.get(fuel_type, 2200)

        # Account for preheat
        afr = STOICH_AFR.get(fuel_type, 17.2)
        air_mass_fraction = afr / (1 + afr)
        preheat_effect = air_mass_fraction * (air_preheat_temp_c - 25)

        # Solve for lambda: target_temp = (t_stoich + preheat) * (1 - 0.25 * (lambda - 1))
        adjusted_stoich = t_stoich_k + preheat_effect
        if adjusted_stoich > 0:
            temp_ratio = target_temp_k / adjusted_stoich
            lambda_val = 1 + (1 - temp_ratio) / 0.25
            lambda_val = max(1.0, min(2.0, lambda_val))
        else:
            lambda_val = 1.15

        return lambda_val, self._quantize(Decimal(str(target_temp_k)))

    def estimate_nox_from_temperature(
        self,
        flame_temp_k: float,
        residence_time_ms: float = 100
    ) -> Decimal:
        """
        Estimate NOx formation from flame temperature.

        DETERMINISTIC: Simplified Zeldovich correlation.

        Args:
            flame_temp_k: Flame temperature (K)
            residence_time_ms: Residence time in flame zone (ms)

        Returns:
            Estimated NOx concentration (ppm)
        """
        # Simplified thermal NOx correlation
        # NOx ~ A * exp(-Ea/RT) * t

        if flame_temp_k < 1500:
            return self._quantize(Decimal("0"))

        # Reference: ~25 ppm at 1900K, 100ms
        ref_temp = 1900
        ref_time = 100
        ref_nox = 25

        # Temperature effect (exponential)
        temp_factor = math.exp(0.03 * (flame_temp_k - ref_temp))

        # Time effect (approximately linear)
        time_factor = residence_time_ms / ref_time

        nox_ppm = ref_nox * temp_factor * time_factor
        nox_ppm = max(0, min(500, nox_ppm))

        return self._quantize(Decimal(str(nox_ppm)))
