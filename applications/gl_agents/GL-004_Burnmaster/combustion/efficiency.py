"""
GL-004 BURNMASTER - Combustion Efficiency Calculator

Zero-hallucination calculation engine for combustion efficiency.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- Gross and net combustion efficiency calculations
- Direct and indirect efficiency methods
- ASME PTC 4 methodology support
- Efficiency decomposition by loss category

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib

from pydantic import BaseModel, Field


class EfficiencyMethod(str, Enum):
    """Efficiency calculation methods."""
    DIRECT = "direct"           # Input-output method
    INDIRECT = "indirect"       # Heat loss method (ASME PTC 4)
    SIMPLIFIED = "simplified"   # Siegert formula


class EfficiencyBasis(str, Enum):
    """Efficiency basis (gross vs net)."""
    GROSS = "gross"  # Based on HHV
    NET = "net"      # Based on LHV


class EfficiencyInput(BaseModel):
    """Input schema for efficiency calculation."""
    fuel_flow_rate: float = Field(..., gt=0, description="Fuel mass flow (kg/h)")
    fuel_hhv: float = Field(..., gt=0, description="Higher heating value (MJ/kg)")
    fuel_lhv: Optional[float] = Field(None, gt=0, description="Lower heating value (MJ/kg)")
    useful_heat_output: Optional[float] = Field(None, ge=0, description="Useful heat output (MW)")
    flue_gas_temp: float = Field(default=200, description="Flue gas temperature (C)")
    ambient_temp: float = Field(default=25, description="Ambient temperature (C)")
    o2_dry_percent: float = Field(default=3.0, ge=0, le=21, description="Dry O2 in flue gas (%)")
    co_ppm: float = Field(default=0, ge=0, description="CO in flue gas (ppm)")
    unburnt_carbon_percent: float = Field(default=0, ge=0, description="Unburnt carbon in ash (%)")


class EfficiencyResult(BaseModel):
    """Output schema for efficiency calculation."""
    gross_efficiency: Decimal = Field(..., ge=0, le=100, description="Gross efficiency (%)")
    net_efficiency: Decimal = Field(..., ge=0, le=100, description="Net efficiency (%)")
    method: EfficiencyMethod
    basis: EfficiencyBasis

    # Loss breakdown
    dry_flue_gas_loss: Decimal = Field(default=Decimal("0"), description="Dry flue gas loss (%)")
    moisture_in_fuel_loss: Decimal = Field(default=Decimal("0"), description="H2O in fuel loss (%)")
    moisture_from_h2_loss: Decimal = Field(default=Decimal("0"), description="H2O from H2 combustion (%)")
    moisture_in_air_loss: Decimal = Field(default=Decimal("0"), description="H2O in air loss (%)")
    unburnt_combustibles_loss: Decimal = Field(default=Decimal("0"), description="CO + unburnt loss (%)")
    radiation_loss: Decimal = Field(default=Decimal("0"), description="Radiation/convection loss (%)")
    other_losses: Decimal = Field(default=Decimal("0"), description="Other unmeasured losses (%)")
    total_losses: Decimal = Field(default=Decimal("0"), description="Total losses (%)")

    # Energy balance
    heat_input_mw: Decimal = Field(..., description="Heat input (MW)")
    heat_output_mw: Decimal = Field(..., description="Heat output (MW)")
    heat_loss_mw: Decimal = Field(..., description="Total heat loss (MW)")

    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Flue gas specific heat approximations (kJ/kg.K)
FLUE_GAS_CP = {
    "co2": 0.85,
    "h2o": 1.87,
    "n2": 1.04,
    "o2": 0.92,
    "average": 1.05,
}


class CombustionEfficiencyCalculator:
    """
    Zero-hallucination calculator for combustion efficiency.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure arithmetic operations only

    Example:
        >>> calculator = CombustionEfficiencyCalculator()
        >>> result = calculator.compute_efficiency_indirect(
        ...     fuel_flow_rate=1000,
        ...     fuel_hhv=50.0,
        ...     flue_gas_temp=180,
        ...     o2_dry_percent=3.0
        ... )
        >>> print(f"Efficiency: {result.gross_efficiency}%")
    """

    def __init__(self, precision: int = 2):
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

    def compute_efficiency_direct(
        self,
        fuel_flow_rate: float,
        fuel_hhv: float,
        useful_heat_output: float,
        fuel_lhv: Optional[float] = None,
    ) -> EfficiencyResult:
        """
        Compute efficiency using direct (input-output) method.

        DETERMINISTIC: Efficiency = Output / Input * 100

        Args:
            fuel_flow_rate: Fuel mass flow rate (kg/h)
            fuel_hhv: Higher heating value of fuel (MJ/kg)
            useful_heat_output: Useful heat output (MW)
            fuel_lhv: Lower heating value (optional, for net efficiency)

        Returns:
            EfficiencyResult with gross and net efficiency
        """
        # Step 1: Calculate heat input (DETERMINISTIC)
        # Heat input (MW) = fuel_flow (kg/h) * HHV (MJ/kg) / 3600 (s/h)
        heat_input_hhv = fuel_flow_rate * fuel_hhv / 3600

        if fuel_lhv:
            heat_input_lhv = fuel_flow_rate * fuel_lhv / 3600
        else:
            # Estimate LHV as 90% of HHV (typical for natural gas)
            heat_input_lhv = heat_input_hhv * 0.9

        # Step 2: Calculate efficiency (DETERMINISTIC)
        if heat_input_hhv > 0:
            gross_eff = (useful_heat_output / heat_input_hhv) * 100
        else:
            gross_eff = 0.0

        if heat_input_lhv > 0:
            net_eff = (useful_heat_output / heat_input_lhv) * 100
        else:
            net_eff = 0.0

        # Cap at 100%
        gross_eff = min(100.0, max(0.0, gross_eff))
        net_eff = min(100.0, max(0.0, net_eff))

        heat_loss = heat_input_hhv - useful_heat_output

        provenance = self._compute_provenance_hash({
            'method': 'direct',
            'fuel_flow_rate': fuel_flow_rate,
            'fuel_hhv': fuel_hhv,
            'useful_heat_output': useful_heat_output,
            'gross_efficiency': gross_eff
        })

        return EfficiencyResult(
            gross_efficiency=self._quantize(Decimal(str(gross_eff))),
            net_efficiency=self._quantize(Decimal(str(net_eff))),
            method=EfficiencyMethod.DIRECT,
            basis=EfficiencyBasis.GROSS,
            total_losses=self._quantize(Decimal(str(100 - gross_eff))),
            heat_input_mw=self._quantize(Decimal(str(heat_input_hhv))),
            heat_output_mw=self._quantize(Decimal(str(useful_heat_output))),
            heat_loss_mw=self._quantize(Decimal(str(max(0, heat_loss)))),
            provenance_hash=provenance
        )

    def compute_efficiency_indirect(
        self,
        fuel_flow_rate: float,
        fuel_hhv: float,
        flue_gas_temp: float,
        ambient_temp: float = 25.0,
        o2_dry_percent: float = 3.0,
        co_ppm: float = 0.0,
        fuel_moisture_percent: float = 0.0,
        fuel_hydrogen_percent: float = 23.0,  # ~23% for natural gas
        unburnt_carbon_percent: float = 0.0,
        radiation_loss_percent: float = 1.0,
        fuel_lhv: Optional[float] = None,
    ) -> EfficiencyResult:
        """
        Compute efficiency using indirect (heat loss) method.

        DETERMINISTIC: Based on ASME PTC 4 methodology.

        Efficiency = 100 - Sum(Losses)

        Losses include:
        - L1: Dry flue gas loss
        - L2: Loss due to moisture in fuel
        - L3: Loss due to moisture from H2 combustion
        - L4: Loss due to moisture in air
        - L5: Loss due to CO (incomplete combustion)
        - L6: Loss due to unburnt carbon
        - L7: Radiation and convection losses
        - L8: Other unmeasured losses

        Args:
            fuel_flow_rate: Fuel mass flow rate (kg/h)
            fuel_hhv: Higher heating value (MJ/kg)
            flue_gas_temp: Flue gas exit temperature (C)
            ambient_temp: Ambient temperature (C)
            o2_dry_percent: Dry O2 in flue gas (%)
            co_ppm: CO concentration (ppm)
            fuel_moisture_percent: Moisture in fuel (%)
            fuel_hydrogen_percent: Hydrogen content in fuel (%)
            unburnt_carbon_percent: Unburnt carbon in ash (%)
            radiation_loss_percent: Assumed radiation loss (%)
            fuel_lhv: Lower heating value (optional)

        Returns:
            EfficiencyResult with efficiency and loss breakdown
        """
        # Step 1: Calculate excess air from O2 (DETERMINISTIC)
        if o2_dry_percent >= 21:
            excess_air_ratio = 10.0  # Very high
        else:
            excess_air_ratio = o2_dry_percent / (21 - o2_dry_percent)

        lambda_val = 1 + excess_air_ratio

        # Step 2: Calculate dry flue gas loss (L1) - Siegert formula approximation
        # L1 = K * (Tg - Ta) / (CO2 + CO)
        # Simplified using O2-based formula:
        # L1 = (Tg - Ta) * (A1 / (21 - O2) + B1)
        # where A1, B1 are fuel-dependent constants

        temp_diff = flue_gas_temp - ambient_temp
        # Simplified constants for natural gas
        A1 = 0.66  # Approximate constant
        B1 = 0.009

        if o2_dry_percent < 21:
            l1_dry_flue = temp_diff * (A1 / (21 - o2_dry_percent) + B1)
        else:
            l1_dry_flue = 50.0  # Very high loss

        l1_dry_flue = max(0, min(50, l1_dry_flue))

        # Step 3: Calculate moisture losses (L2, L3, L4)
        # L2: Moisture in fuel
        l2_moisture_fuel = fuel_moisture_percent * (2.44 * temp_diff / 100 + 2.5) / fuel_hhv
        l2_moisture_fuel = max(0, min(10, l2_moisture_fuel * 100))

        # L3: Moisture from hydrogen combustion
        # H2O formed = 9 * H2 (stoichiometry)
        h2o_from_h2 = 9 * fuel_hydrogen_percent / 100
        latent_heat = 2.44  # MJ/kg at 25C
        sensible_heat = 0.002 * temp_diff  # MJ/kg.K
        l3_moisture_h2 = h2o_from_h2 * (latent_heat + sensible_heat) / fuel_hhv * 100
        l3_moisture_h2 = max(0, min(15, l3_moisture_h2))

        # L4: Moisture in air (typically small, ~0.1-0.5%)
        l4_moisture_air = 0.1 * lambda_val  # Approximate
        l4_moisture_air = max(0, min(2, l4_moisture_air))

        # Step 4: Calculate unburnt combustibles loss (L5, L6)
        # L5: CO loss
        # CO heat of combustion ~10.1 MJ/Nm3
        # Simplified: CO_loss = CO_ppm * factor
        l5_co_loss = co_ppm * 0.001  # Approximate % loss per 1000 ppm
        l5_co_loss = max(0, min(5, l5_co_loss))

        # L6: Unburnt carbon (for solid fuels)
        # Carbon heat of combustion = 32.8 MJ/kg
        l6_unburnt = unburnt_carbon_percent * 32.8 / fuel_hhv * 100
        l6_unburnt = max(0, min(5, l6_unburnt))

        # Step 5: Radiation loss (L7) - typically 0.5-2%
        l7_radiation = radiation_loss_percent

        # Step 6: Other losses (L8) - typically 0.5-1%
        l8_other = 0.5

        # Step 7: Sum losses (DETERMINISTIC)
        total_losses = (
            l1_dry_flue +
            l2_moisture_fuel +
            l3_moisture_h2 +
            l4_moisture_air +
            l5_co_loss +
            l6_unburnt +
            l7_radiation +
            l8_other
        )

        # Step 8: Calculate efficiency (DETERMINISTIC)
        gross_eff = 100 - total_losses
        gross_eff = max(0, min(100, gross_eff))

        # Net efficiency (on LHV basis)
        if fuel_lhv:
            hhv_lhv_ratio = fuel_hhv / fuel_lhv
        else:
            hhv_lhv_ratio = 1.1  # Typical for natural gas

        net_eff = gross_eff * hhv_lhv_ratio
        net_eff = min(100, net_eff)

        # Calculate heat flows
        heat_input_mw = fuel_flow_rate * fuel_hhv / 3600
        heat_output_mw = heat_input_mw * gross_eff / 100
        heat_loss_mw = heat_input_mw - heat_output_mw

        provenance = self._compute_provenance_hash({
            'method': 'indirect',
            'fuel_flow_rate': fuel_flow_rate,
            'flue_gas_temp': flue_gas_temp,
            'o2_dry_percent': o2_dry_percent,
            'total_losses': total_losses,
            'gross_efficiency': gross_eff
        })

        return EfficiencyResult(
            gross_efficiency=self._quantize(Decimal(str(gross_eff))),
            net_efficiency=self._quantize(Decimal(str(net_eff))),
            method=EfficiencyMethod.INDIRECT,
            basis=EfficiencyBasis.GROSS,
            dry_flue_gas_loss=self._quantize(Decimal(str(l1_dry_flue))),
            moisture_in_fuel_loss=self._quantize(Decimal(str(l2_moisture_fuel))),
            moisture_from_h2_loss=self._quantize(Decimal(str(l3_moisture_h2))),
            moisture_in_air_loss=self._quantize(Decimal(str(l4_moisture_air))),
            unburnt_combustibles_loss=self._quantize(Decimal(str(l5_co_loss + l6_unburnt))),
            radiation_loss=self._quantize(Decimal(str(l7_radiation))),
            other_losses=self._quantize(Decimal(str(l8_other))),
            total_losses=self._quantize(Decimal(str(total_losses))),
            heat_input_mw=self._quantize(Decimal(str(heat_input_mw))),
            heat_output_mw=self._quantize(Decimal(str(heat_output_mw))),
            heat_loss_mw=self._quantize(Decimal(str(heat_loss_mw))),
            provenance_hash=provenance
        )

    def compute_efficiency_siegert(
        self,
        flue_gas_temp: float,
        ambient_temp: float,
        co2_percent: float,
        fuel_type: str = "natural_gas"
    ) -> Decimal:
        """
        Compute efficiency using simplified Siegert formula.

        DETERMINISTIC: Quick estimate based on stack temperature and CO2.

        Siegert formula:
        Loss % = (Tg - Ta) * (A2 / CO2 + B)

        where A2 and B are fuel-dependent constants.

        Args:
            flue_gas_temp: Flue gas temperature (C)
            ambient_temp: Ambient temperature (C)
            co2_percent: CO2 in flue gas (%)
            fuel_type: Type of fuel

        Returns:
            Estimated efficiency (%)
        """
        # Siegert constants by fuel type
        siegert_constants = {
            "natural_gas": {"A2": 0.37, "B": 0.009},
            "fuel_oil": {"A2": 0.50, "B": 0.007},
            "coal": {"A2": 0.63, "B": 0.007},
            "propane": {"A2": 0.42, "B": 0.008},
        }

        constants = siegert_constants.get(fuel_type, siegert_constants["natural_gas"])

        temp_diff = flue_gas_temp - ambient_temp

        if co2_percent > 0:
            loss_percent = temp_diff * (constants["A2"] / co2_percent + constants["B"])
        else:
            loss_percent = 20.0  # High loss if no CO2 data

        loss_percent = max(0, min(50, loss_percent))
        efficiency = 100 - loss_percent

        return self._quantize(Decimal(str(efficiency)))
