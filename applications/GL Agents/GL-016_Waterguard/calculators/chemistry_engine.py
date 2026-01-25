# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD Deterministic Chemistry Engine
Zero-Hallucination Calculation Service for Boiler Water Treatment

All numeric outputs from deterministic formulas with SHA-256 provenance tracking.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Constants (ASME/ABMA Standards)
# =============================================================================

# Water properties at standard conditions
WATER_DENSITY_LB_GAL = 8.34  # lb/gal at 60°F
WATER_SPECIFIC_HEAT_BTU_LB_F = 1.0  # BTU/(lb·°F)

# Conductivity to TDS conversion factors
CONDUCTIVITY_TO_TDS_FACTOR = 0.65  # Typical for boiler water
CONDUCTIVITY_TEMP_COEFFICIENT = 0.02  # 2% per °C

# Steam/Water enthalpy references (BTU/lb)
LATENT_HEAT_STEAM_BTU_LB = 970.0  # Approximate at 100 psig

# Silica solubility limits based on pressure (ppm)
SILICA_SOLUBILITY_LIMITS = {
    15: 300,    # psig: max silica ppm
    50: 200,
    100: 175,
    150: 150,
    200: 125,
    300: 100,
    400: 75,
    600: 50,
    900: 35,
    1500: 20,
    2000: 10,
}


# =============================================================================
# Provenance Tracking
# =============================================================================

@dataclass(frozen=True)
class CalculationProvenance:
    """Immutable provenance record for audit trail."""
    calculation_id: UUID
    timestamp: datetime
    formula_id: str
    formula_version: str
    inputs_hash: str
    outputs_hash: str
    combined_hash: str

    def to_dict(self) -> dict:
        return {
            "calculation_id": str(self.calculation_id),
            "timestamp": self.timestamp.isoformat(),
            "formula_id": self.formula_id,
            "formula_version": self.formula_version,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "combined_hash": self.combined_hash,
        }


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of data string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def create_provenance(
    formula_id: str,
    formula_version: str,
    inputs: dict,
    outputs: dict,
) -> CalculationProvenance:
    """Create provenance record for a calculation."""
    calc_id = uuid4()
    timestamp = datetime.utcnow()

    inputs_str = str(sorted(inputs.items()))
    outputs_str = str(sorted(outputs.items()))

    inputs_hash = compute_sha256(inputs_str)
    outputs_hash = compute_sha256(outputs_str)
    combined_hash = compute_sha256(f"{inputs_hash}:{outputs_hash}:{formula_id}:{formula_version}")

    return CalculationProvenance(
        calculation_id=calc_id,
        timestamp=timestamp,
        formula_id=formula_id,
        formula_version=formula_version,
        inputs_hash=inputs_hash,
        outputs_hash=outputs_hash,
        combined_hash=combined_hash,
    )


# =============================================================================
# Cycles of Concentration Calculations
# =============================================================================

class CoCResult(BaseModel):
    """Result of Cycles of Concentration calculation."""
    coc_conductivity: float = Field(..., ge=1.0, description="CoC from conductivity ratio")
    coc_silica: Optional[float] = Field(None, ge=1.0, description="CoC from silica ratio")
    coc_recommended: float = Field(..., ge=1.0, description="Recommended CoC")
    calculation_method: str
    provenance: dict


def calculate_coc_from_conductivity(
    makeup_conductivity_umho: float,
    blowdown_conductivity_umho: float,
) -> Tuple[float, CalculationProvenance]:
    """
    Calculate Cycles of Concentration from conductivity ratio.

    Formula: CoC = Blowdown_Conductivity / Makeup_Conductivity
    Source: ASME Guidelines for Water Quality

    Args:
        makeup_conductivity_umho: Makeup water conductivity (µS/cm)
        blowdown_conductivity_umho: Blowdown water conductivity (µS/cm)

    Returns:
        Tuple of (CoC value, provenance record)
    """
    if makeup_conductivity_umho <= 0:
        raise ValueError("Makeup conductivity must be positive")
    if blowdown_conductivity_umho <= 0:
        raise ValueError("Blowdown conductivity must be positive")

    coc = blowdown_conductivity_umho / makeup_conductivity_umho

    # Create provenance
    provenance = create_provenance(
        formula_id="COC_CONDUCTIVITY_RATIO",
        formula_version="1.0.0",
        inputs={
            "makeup_conductivity_umho": makeup_conductivity_umho,
            "blowdown_conductivity_umho": blowdown_conductivity_umho,
        },
        outputs={"coc": coc},
    )

    return coc, provenance


def calculate_coc_from_silica(
    makeup_silica_ppm: float,
    blowdown_silica_ppm: float,
) -> Tuple[float, CalculationProvenance]:
    """
    Calculate Cycles of Concentration from silica ratio.

    Formula: CoC = Blowdown_Silica / Makeup_Silica
    Source: ASME Guidelines for Water Quality

    Args:
        makeup_silica_ppm: Makeup water silica (ppm as SiO2)
        blowdown_silica_ppm: Blowdown water silica (ppm as SiO2)

    Returns:
        Tuple of (CoC value, provenance record)
    """
    if makeup_silica_ppm <= 0:
        raise ValueError("Makeup silica must be positive")
    if blowdown_silica_ppm <= 0:
        raise ValueError("Blowdown silica must be positive")

    coc = blowdown_silica_ppm / makeup_silica_ppm

    provenance = create_provenance(
        formula_id="COC_SILICA_RATIO",
        formula_version="1.0.0",
        inputs={
            "makeup_silica_ppm": makeup_silica_ppm,
            "blowdown_silica_ppm": blowdown_silica_ppm,
        },
        outputs={"coc": coc},
    )

    return coc, provenance


# =============================================================================
# Water Balance Calculations
# =============================================================================

class WaterBalanceResult(BaseModel):
    """Result of water balance calculation."""
    makeup_rate_klb_hr: float = Field(..., ge=0.0)
    blowdown_rate_klb_hr: float = Field(..., ge=0.0)
    blowdown_percent: float = Field(..., ge=0.0, le=100.0)
    evaporation_rate_klb_hr: float = Field(..., ge=0.0)
    provenance: dict


def calculate_water_balance(
    steam_rate_klb_hr: float,
    coc: float,
    condensate_return_percent: float = 80.0,
) -> Tuple[WaterBalanceResult, CalculationProvenance]:
    """
    Calculate boiler water balance.

    Formulas:
        Blowdown % = 1 / (CoC - 1) × 100
        Makeup = Steam / (1 - Blowdown%) + Losses
        Blowdown = Makeup - Steam - Losses

    Args:
        steam_rate_klb_hr: Steam production rate (klb/hr)
        coc: Cycles of Concentration
        condensate_return_percent: Condensate return percentage

    Returns:
        Tuple of (WaterBalanceResult, provenance record)
    """
    if steam_rate_klb_hr < 0:
        raise ValueError("Steam rate must be non-negative")
    if coc <= 1:
        raise ValueError("CoC must be greater than 1")
    if not 0 <= condensate_return_percent <= 100:
        raise ValueError("Condensate return must be between 0 and 100")

    # Calculate blowdown percentage
    blowdown_percent = 100.0 / (coc - 1)

    # Steam losses (non-returned condensate)
    steam_losses_klb_hr = steam_rate_klb_hr * (1 - condensate_return_percent / 100)

    # Makeup water requirement
    makeup_rate_klb_hr = steam_losses_klb_hr * (1 + blowdown_percent / 100)

    # Blowdown rate
    blowdown_rate_klb_hr = makeup_rate_klb_hr * (blowdown_percent / 100)

    # Evaporation (steam production minus condensate return)
    evaporation_rate_klb_hr = steam_losses_klb_hr

    result = WaterBalanceResult(
        makeup_rate_klb_hr=round(makeup_rate_klb_hr, 3),
        blowdown_rate_klb_hr=round(blowdown_rate_klb_hr, 3),
        blowdown_percent=round(blowdown_percent, 2),
        evaporation_rate_klb_hr=round(evaporation_rate_klb_hr, 3),
        provenance={},
    )

    provenance = create_provenance(
        formula_id="WATER_BALANCE",
        formula_version="1.0.0",
        inputs={
            "steam_rate_klb_hr": steam_rate_klb_hr,
            "coc": coc,
            "condensate_return_percent": condensate_return_percent,
        },
        outputs={
            "makeup_rate_klb_hr": result.makeup_rate_klb_hr,
            "blowdown_rate_klb_hr": result.blowdown_rate_klb_hr,
            "blowdown_percent": result.blowdown_percent,
            "evaporation_rate_klb_hr": result.evaporation_rate_klb_hr,
        },
    )

    result.provenance = provenance.to_dict()
    return result, provenance


# =============================================================================
# Heat Loss Calculations
# =============================================================================

class HeatLossResult(BaseModel):
    """Result of blowdown heat loss calculation."""
    heat_loss_mmbtu_hr: float = Field(..., ge=0.0)
    heat_loss_percent: float = Field(..., ge=0.0)
    annual_heat_loss_mmbtu: float = Field(..., ge=0.0)
    annual_fuel_cost_usd: Optional[float] = None
    provenance: dict


def calculate_blowdown_heat_loss(
    blowdown_rate_klb_hr: float,
    blowdown_temp_f: float,
    makeup_temp_f: float = 60.0,
    boiler_efficiency: float = 0.80,
    fuel_cost_usd_mmbtu: float = 8.0,
    operating_hours_per_year: int = 8760,
) -> Tuple[HeatLossResult, CalculationProvenance]:
    """
    Calculate heat loss from blowdown.

    Formula:
        Q = m × Cp × ΔT
        Heat Loss = Blowdown_Flow × (Blowdown_Enthalpy - Makeup_Enthalpy)

    Args:
        blowdown_rate_klb_hr: Blowdown flow rate (klb/hr)
        blowdown_temp_f: Blowdown temperature (°F)
        makeup_temp_f: Makeup water temperature (°F)
        boiler_efficiency: Boiler thermal efficiency
        fuel_cost_usd_mmbtu: Fuel cost ($/MMBtu)
        operating_hours_per_year: Annual operating hours

    Returns:
        Tuple of (HeatLossResult, provenance record)
    """
    if blowdown_rate_klb_hr < 0:
        raise ValueError("Blowdown rate must be non-negative")
    if blowdown_temp_f <= makeup_temp_f:
        raise ValueError("Blowdown temp must be greater than makeup temp")
    if not 0 < boiler_efficiency <= 1:
        raise ValueError("Boiler efficiency must be between 0 and 1")

    # Heat content in blowdown (BTU/hr)
    # Q = m × Cp × ΔT
    delta_t = blowdown_temp_f - makeup_temp_f
    heat_loss_btu_hr = blowdown_rate_klb_hr * 1000 * WATER_SPECIFIC_HEAT_BTU_LB_F * delta_t

    # Convert to MMBtu/hr
    heat_loss_mmbtu_hr = heat_loss_btu_hr / 1_000_000

    # Account for boiler efficiency (fuel required to generate lost heat)
    fuel_equivalent_mmbtu_hr = heat_loss_mmbtu_hr / boiler_efficiency

    # Annual values
    annual_heat_loss_mmbtu = fuel_equivalent_mmbtu_hr * operating_hours_per_year
    annual_fuel_cost_usd = annual_heat_loss_mmbtu * fuel_cost_usd_mmbtu

    # Estimate heat loss as percentage (assuming typical boiler heat input)
    # This is approximate - actual percentage depends on total heat input
    heat_loss_percent = (blowdown_rate_klb_hr / 100) * 100  # Rough estimate

    result = HeatLossResult(
        heat_loss_mmbtu_hr=round(heat_loss_mmbtu_hr, 4),
        heat_loss_percent=round(heat_loss_percent, 2),
        annual_heat_loss_mmbtu=round(annual_heat_loss_mmbtu, 1),
        annual_fuel_cost_usd=round(annual_fuel_cost_usd, 2),
        provenance={},
    )

    provenance = create_provenance(
        formula_id="BLOWDOWN_HEAT_LOSS",
        formula_version="1.0.0",
        inputs={
            "blowdown_rate_klb_hr": blowdown_rate_klb_hr,
            "blowdown_temp_f": blowdown_temp_f,
            "makeup_temp_f": makeup_temp_f,
            "boiler_efficiency": boiler_efficiency,
            "fuel_cost_usd_mmbtu": fuel_cost_usd_mmbtu,
            "operating_hours_per_year": operating_hours_per_year,
        },
        outputs={
            "heat_loss_mmbtu_hr": result.heat_loss_mmbtu_hr,
            "heat_loss_percent": result.heat_loss_percent,
            "annual_heat_loss_mmbtu": result.annual_heat_loss_mmbtu,
            "annual_fuel_cost_usd": result.annual_fuel_cost_usd,
        },
    )

    result.provenance = provenance.to_dict()
    return result, provenance


# =============================================================================
# Chemical Dosing Calculations
# =============================================================================

class DosingResult(BaseModel):
    """Result of chemical dosing calculation."""
    dose_rate_gph: float = Field(..., ge=0.0)
    dose_rate_lb_hr: float = Field(..., ge=0.0)
    chemical_consumption_lb_day: float = Field(..., ge=0.0)
    pump_speed_percent: float = Field(..., ge=0.0, le=100.0)
    provenance: dict


def calculate_phosphate_dose(
    makeup_rate_klb_hr: float,
    target_phosphate_ppm: float,
    current_phosphate_ppm: float,
    phosphate_product_concentration: float = 0.30,  # 30% active
    product_density_lb_gal: float = 10.5,
) -> Tuple[DosingResult, CalculationProvenance]:
    """
    Calculate phosphate dosing rate for coordinated phosphate treatment.

    Formula:
        Dose Rate = (Makeup × Target_ppm × 8.34) / (Product_Concentration × 1,000,000)

    Args:
        makeup_rate_klb_hr: Makeup water flow (klb/hr)
        target_phosphate_ppm: Target phosphate residual (ppm)
        current_phosphate_ppm: Current phosphate residual (ppm)
        phosphate_product_concentration: Active ingredient concentration
        product_density_lb_gal: Product density (lb/gal)

    Returns:
        Tuple of (DosingResult, provenance record)
    """
    if makeup_rate_klb_hr < 0:
        raise ValueError("Makeup rate must be non-negative")
    if target_phosphate_ppm < 0:
        raise ValueError("Target phosphate must be non-negative")
    if not 0 < phosphate_product_concentration <= 1:
        raise ValueError("Product concentration must be between 0 and 1")

    # Required phosphate addition (ppm deficit)
    phosphate_deficit_ppm = max(0, target_phosphate_ppm - current_phosphate_ppm)

    # Convert makeup rate to gpm (assuming water density)
    makeup_gpm = (makeup_rate_klb_hr * 1000) / (WATER_DENSITY_LB_GAL * 60)

    # Phosphate required (lb/hr)
    phosphate_lb_hr = (makeup_rate_klb_hr * 1000 * phosphate_deficit_ppm) / 1_000_000

    # Product required (lb/hr) accounting for concentration
    product_lb_hr = phosphate_lb_hr / phosphate_product_concentration

    # Product volume (gph)
    dose_rate_gph = (product_lb_hr / product_density_lb_gal) if product_lb_hr > 0 else 0

    # Daily consumption
    chemical_consumption_lb_day = product_lb_hr * 24

    # Pump speed (assuming max pump capacity of 10 gph)
    max_pump_gph = 10.0
    pump_speed_percent = min(100.0, (dose_rate_gph / max_pump_gph) * 100)

    result = DosingResult(
        dose_rate_gph=round(dose_rate_gph, 4),
        dose_rate_lb_hr=round(product_lb_hr, 4),
        chemical_consumption_lb_day=round(chemical_consumption_lb_day, 2),
        pump_speed_percent=round(pump_speed_percent, 1),
        provenance={},
    )

    provenance = create_provenance(
        formula_id="PHOSPHATE_DOSE",
        formula_version="1.0.0",
        inputs={
            "makeup_rate_klb_hr": makeup_rate_klb_hr,
            "target_phosphate_ppm": target_phosphate_ppm,
            "current_phosphate_ppm": current_phosphate_ppm,
            "phosphate_product_concentration": phosphate_product_concentration,
            "product_density_lb_gal": product_density_lb_gal,
        },
        outputs={
            "dose_rate_gph": result.dose_rate_gph,
            "dose_rate_lb_hr": result.dose_rate_lb_hr,
            "chemical_consumption_lb_day": result.chemical_consumption_lb_day,
            "pump_speed_percent": result.pump_speed_percent,
        },
    )

    result.provenance = provenance.to_dict()
    return result, provenance


def calculate_oxygen_scavenger_dose(
    feedwater_flow_klb_hr: float,
    dissolved_oxygen_ppb: float,
    excess_sulfite_ppm: float = 30.0,  # Target excess
    product_concentration: float = 0.30,  # 30% sodium sulfite
    stoichiometric_ratio: float = 8.0,  # 8 ppm sulfite per 1 ppm O2
) -> Tuple[DosingResult, CalculationProvenance]:
    """
    Calculate oxygen scavenger (sulfite) dosing rate.

    Formula:
        Sulfite Demand = DO × Stoich_Ratio + Excess_Residual
        Dose Rate = (Flow × Sulfite_ppm × 8.34) / (Concentration × 1,000,000)

    Args:
        feedwater_flow_klb_hr: Feedwater flow rate (klb/hr)
        dissolved_oxygen_ppb: Feedwater dissolved oxygen (ppb)
        excess_sulfite_ppm: Target excess sulfite residual (ppm)
        product_concentration: Sulfite product concentration
        stoichiometric_ratio: Sulfite:O2 stoichiometric ratio

    Returns:
        Tuple of (DosingResult, provenance record)
    """
    if feedwater_flow_klb_hr < 0:
        raise ValueError("Feedwater flow must be non-negative")
    if dissolved_oxygen_ppb < 0:
        raise ValueError("Dissolved oxygen must be non-negative")

    # Convert ppb to ppm
    do_ppm = dissolved_oxygen_ppb / 1000

    # Calculate sulfite demand
    sulfite_stoich_ppm = do_ppm * stoichiometric_ratio
    total_sulfite_ppm = sulfite_stoich_ppm + excess_sulfite_ppm

    # Product required (lb/hr)
    product_lb_hr = (feedwater_flow_klb_hr * 1000 * total_sulfite_ppm) / (
        1_000_000 * product_concentration
    )

    # Product volume (gph) assuming ~10 lb/gal
    product_density_lb_gal = 10.0
    dose_rate_gph = product_lb_hr / product_density_lb_gal

    # Daily consumption
    chemical_consumption_lb_day = product_lb_hr * 24

    # Pump speed
    max_pump_gph = 5.0
    pump_speed_percent = min(100.0, (dose_rate_gph / max_pump_gph) * 100)

    result = DosingResult(
        dose_rate_gph=round(dose_rate_gph, 4),
        dose_rate_lb_hr=round(product_lb_hr, 4),
        chemical_consumption_lb_day=round(chemical_consumption_lb_day, 2),
        pump_speed_percent=round(pump_speed_percent, 1),
        provenance={},
    )

    provenance = create_provenance(
        formula_id="OXYGEN_SCAVENGER_DOSE",
        formula_version="1.0.0",
        inputs={
            "feedwater_flow_klb_hr": feedwater_flow_klb_hr,
            "dissolved_oxygen_ppb": dissolved_oxygen_ppb,
            "excess_sulfite_ppm": excess_sulfite_ppm,
            "product_concentration": product_concentration,
            "stoichiometric_ratio": stoichiometric_ratio,
        },
        outputs={
            "dose_rate_gph": result.dose_rate_gph,
            "dose_rate_lb_hr": result.dose_rate_lb_hr,
            "chemical_consumption_lb_day": result.chemical_consumption_lb_day,
            "pump_speed_percent": result.pump_speed_percent,
        },
    )

    result.provenance = provenance.to_dict()
    return result, provenance


# =============================================================================
# Conductivity Temperature Compensation
# =============================================================================

def compensate_conductivity_for_temperature(
    measured_conductivity_umho: float,
    measured_temp_c: float,
    reference_temp_c: float = 25.0,
    temp_coefficient: float = CONDUCTIVITY_TEMP_COEFFICIENT,
) -> Tuple[float, CalculationProvenance]:
    """
    Temperature-compensate conductivity to reference temperature.

    Formula:
        Cond_ref = Cond_meas / (1 + α × (T_meas - T_ref))

    Args:
        measured_conductivity_umho: Measured conductivity (µS/cm)
        measured_temp_c: Measured temperature (°C)
        reference_temp_c: Reference temperature (°C)
        temp_coefficient: Temperature coefficient (%/°C as decimal)

    Returns:
        Tuple of (compensated conductivity, provenance record)
    """
    if measured_conductivity_umho < 0:
        raise ValueError("Conductivity must be non-negative")

    temp_diff = measured_temp_c - reference_temp_c
    correction_factor = 1 + (temp_coefficient * temp_diff)

    compensated = measured_conductivity_umho / correction_factor

    provenance = create_provenance(
        formula_id="CONDUCTIVITY_TEMP_COMPENSATION",
        formula_version="1.0.0",
        inputs={
            "measured_conductivity_umho": measured_conductivity_umho,
            "measured_temp_c": measured_temp_c,
            "reference_temp_c": reference_temp_c,
            "temp_coefficient": temp_coefficient,
        },
        outputs={"compensated_conductivity_umho": compensated},
    )

    return round(compensated, 2), provenance


# =============================================================================
# Silica Solubility Check
# =============================================================================

def get_max_silica_limit(pressure_psig: float) -> float:
    """
    Get maximum silica limit based on boiler pressure.

    Based on ASME/ABMA guidelines for silica in boiler water.

    Args:
        pressure_psig: Boiler pressure (psig)

    Returns:
        Maximum silica limit (ppm as SiO2)
    """
    # Find applicable pressure range
    sorted_pressures = sorted(SILICA_SOLUBILITY_LIMITS.keys())

    for i, p in enumerate(sorted_pressures):
        if pressure_psig <= p:
            return SILICA_SOLUBILITY_LIMITS[p]

    # Above highest pressure, use most conservative limit
    return SILICA_SOLUBILITY_LIMITS[sorted_pressures[-1]]


def check_silica_carryover_risk(
    boiler_silica_ppm: float,
    pressure_psig: float,
    steam_purity_ppb_limit: float = 20.0,
) -> Tuple[bool, float, str]:
    """
    Check risk of silica carryover into steam.

    Args:
        boiler_silica_ppm: Boiler water silica (ppm as SiO2)
        pressure_psig: Boiler operating pressure (psig)
        steam_purity_ppb_limit: Steam silica limit (ppb)

    Returns:
        Tuple of (is_at_risk, headroom_percent, risk_level)
    """
    max_silica = get_max_silica_limit(pressure_psig)
    headroom_percent = ((max_silica - boiler_silica_ppm) / max_silica) * 100

    if boiler_silica_ppm >= max_silica:
        return True, 0.0, "CRITICAL"
    elif headroom_percent < 10:
        return True, headroom_percent, "HIGH"
    elif headroom_percent < 25:
        return False, headroom_percent, "MEDIUM"
    else:
        return False, headroom_percent, "LOW"


# =============================================================================
# Main Chemistry Engine Class
# =============================================================================

class ChemistryEngine:
    """
    Deterministic chemistry calculation engine.

    Provides zero-hallucination calculations with full provenance tracking
    for boiler water treatment optimization.
    """

    def __init__(self, boiler_id: str):
        self.boiler_id = boiler_id
        self._calculation_log: list[CalculationProvenance] = []

    def calculate_coc(
        self,
        makeup_conductivity_umho: float,
        blowdown_conductivity_umho: float,
        makeup_silica_ppm: Optional[float] = None,
        blowdown_silica_ppm: Optional[float] = None,
    ) -> CoCResult:
        """Calculate Cycles of Concentration with multiple methods."""

        # Conductivity method
        coc_cond, prov_cond = calculate_coc_from_conductivity(
            makeup_conductivity_umho, blowdown_conductivity_umho
        )
        self._calculation_log.append(prov_cond)

        # Silica method (if available)
        coc_silica = None
        if makeup_silica_ppm and blowdown_silica_ppm:
            coc_silica, prov_silica = calculate_coc_from_silica(
                makeup_silica_ppm, blowdown_silica_ppm
            )
            self._calculation_log.append(prov_silica)

        # Use lower of the two for conservative estimate
        if coc_silica:
            coc_recommended = min(coc_cond, coc_silica)
            method = "min_of_conductivity_and_silica"
        else:
            coc_recommended = coc_cond
            method = "conductivity_only"

        return CoCResult(
            coc_conductivity=round(coc_cond, 2),
            coc_silica=round(coc_silica, 2) if coc_silica else None,
            coc_recommended=round(coc_recommended, 2),
            calculation_method=method,
            provenance=prov_cond.to_dict(),
        )

    def calculate_water_balance(
        self,
        steam_rate_klb_hr: float,
        coc: float,
        condensate_return_percent: float = 80.0,
    ) -> WaterBalanceResult:
        """Calculate boiler water balance."""
        result, provenance = calculate_water_balance(
            steam_rate_klb_hr, coc, condensate_return_percent
        )
        self._calculation_log.append(provenance)
        return result

    def calculate_heat_loss(
        self,
        blowdown_rate_klb_hr: float,
        blowdown_temp_f: float,
        makeup_temp_f: float = 60.0,
        boiler_efficiency: float = 0.80,
        fuel_cost_usd_mmbtu: float = 8.0,
    ) -> HeatLossResult:
        """Calculate blowdown heat loss."""
        result, provenance = calculate_blowdown_heat_loss(
            blowdown_rate_klb_hr,
            blowdown_temp_f,
            makeup_temp_f,
            boiler_efficiency,
            fuel_cost_usd_mmbtu,
        )
        self._calculation_log.append(provenance)
        return result

    def get_calculation_log(self) -> list[dict]:
        """Get all calculation provenance records."""
        return [p.to_dict() for p in self._calculation_log]

    def clear_calculation_log(self) -> None:
        """Clear calculation log."""
        self._calculation_log.clear()
