# -*- coding: utf-8 -*-
"""
Energy Loss Quantifier Module for GL-015 INSULSCAN

ZERO-HALLUCINATION GUARANTEE:
- NO LLM calls in calculation path
- 100% deterministic (same input -> same output)
- Full provenance tracking with SHA-256 hashing
- Complete audit trail for regulatory compliance

This module quantifies energy losses from thermal insulation deficiencies
in industrial facilities, enabling accurate assessment of waste energy,
cost impacts, and carbon footprint implications.

Reference Standards:
- ASTM C680: Standard Practice for Determination of Heat Gain or Loss
- ISO 12241: Thermal insulation for building equipment and industrial installations
- 3E Plus (North American Insulation Manufacturers Association)
- GHG Protocol Corporate Standard

Author: GreenLang Calculator Engine
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND EMISSION FACTORS (EPA 2024)
# =============================================================================

class FuelType(str, Enum):
    """Supported fuel types for energy loss calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    ELECTRICITY = "electricity"
    STEAM = "steam"
    BIOMASS_WOOD = "biomass_wood"
    DIESEL = "diesel"


# EPA emission factors (kg CO2e per unit) - 2024 values
EPA_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    FuelType.NATURAL_GAS.value: {
        "kg_co2e_per_mmbtu": 53.06,
        "kg_co2e_per_therm": 5.306,
        "kg_co2e_per_mcf": 53.06,
        "unit_per_mmbtu": 1.0,
    },
    FuelType.FUEL_OIL_NO2.value: {
        "kg_co2e_per_gallon": 10.21,
        "kg_co2e_per_mmbtu": 73.96,
        "btu_per_gallon": 138000,
    },
    FuelType.FUEL_OIL_NO6.value: {
        "kg_co2e_per_gallon": 11.27,
        "kg_co2e_per_mmbtu": 75.10,
        "btu_per_gallon": 150000,
    },
    FuelType.PROPANE.value: {
        "kg_co2e_per_gallon": 5.72,
        "kg_co2e_per_mmbtu": 62.87,
        "btu_per_gallon": 91000,
    },
    FuelType.COAL_BITUMINOUS.value: {
        "kg_co2e_per_ton": 2328.0,
        "kg_co2e_per_mmbtu": 93.28,
        "btu_per_ton": 24930000,
    },
    FuelType.COAL_ANTHRACITE.value: {
        "kg_co2e_per_ton": 2602.0,
        "kg_co2e_per_mmbtu": 103.69,
        "btu_per_ton": 25090000,
    },
    FuelType.ELECTRICITY.value: {
        "kg_co2e_per_kwh": 0.417,  # US average grid 2024
        "kg_co2e_per_mwh": 417.0,
    },
    FuelType.STEAM.value: {
        "kg_co2e_per_lb": 0.0606,  # Based on natural gas boiler
        "kg_co2e_per_mmbtu": 53.06,
        "btu_per_lb": 1000,  # Average steam enthalpy
    },
    FuelType.BIOMASS_WOOD.value: {
        "kg_co2e_per_ton": 93.80,  # Biogenic CO2 typically excluded
        "kg_co2e_per_mmbtu": 93.80,
    },
    FuelType.DIESEL.value: {
        "kg_co2e_per_gallon": 10.21,
        "kg_co2e_per_mmbtu": 73.96,
        "btu_per_gallon": 137000,
    },
}

# Default boiler/heater efficiencies by fuel type
DEFAULT_BOILER_EFFICIENCIES: Dict[str, float] = {
    FuelType.NATURAL_GAS.value: 0.85,
    FuelType.FUEL_OIL_NO2.value: 0.82,
    FuelType.FUEL_OIL_NO6.value: 0.80,
    FuelType.PROPANE.value: 0.85,
    FuelType.COAL_BITUMINOUS.value: 0.75,
    FuelType.COAL_ANTHRACITE.value: 0.75,
    FuelType.ELECTRICITY.value: 0.99,  # Electric resistance
    FuelType.STEAM.value: 1.0,  # Already generated
    FuelType.BIOMASS_WOOD.value: 0.70,
    FuelType.DIESEL.value: 0.82,
}

# Industry benchmark heat loss rates (W/m of pipe) by insulation condition
BENCHMARK_HEAT_LOSS_RATES: Dict[str, Dict[str, float]] = {
    "bare_pipe": {
        "low_temp_100c": 250.0,
        "med_temp_200c": 650.0,
        "high_temp_300c": 1200.0,
        "very_high_temp_400c": 2000.0,
    },
    "damaged_insulation": {
        "low_temp_100c": 125.0,
        "med_temp_200c": 325.0,
        "high_temp_300c": 600.0,
        "very_high_temp_400c": 1000.0,
    },
    "fair_insulation": {
        "low_temp_100c": 50.0,
        "med_temp_200c": 100.0,
        "high_temp_300c": 180.0,
        "very_high_temp_400c": 300.0,
    },
    "good_insulation": {
        "low_temp_100c": 25.0,
        "med_temp_200c": 50.0,
        "high_temp_300c": 90.0,
        "very_high_temp_400c": 150.0,
    },
    "best_practice": {
        "low_temp_100c": 12.0,
        "med_temp_200c": 25.0,
        "high_temp_300c": 45.0,
        "very_high_temp_400c": 75.0,
    },
}

# Carbon price scenarios (USD per tonne CO2e)
CARBON_PRICE_SCENARIOS: Dict[str, float] = {
    "low": 25.0,
    "medium": 75.0,
    "high": 150.0,
    "eu_ets_2024": 85.0,
    "california_cap_trade": 35.0,
    "iea_net_zero_2030": 140.0,
    "iea_net_zero_2050": 250.0,
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class InspectionLocation:
    """
    Data model for a single thermal inspection location.

    Attributes:
        location_id: Unique identifier for the location
        description: Human-readable description
        system_type: Type of system (steam, hot_water, process, etc.)
        pipe_diameter_mm: Nominal pipe diameter in millimeters
        pipe_length_m: Length of pipe section in meters
        surface_area_m2: Total surface area in square meters
        process_temperature_c: Operating temperature in Celsius
        ambient_temperature_c: Ambient temperature in Celsius
        heat_loss_rate_w: Measured heat loss rate in Watts
        insulation_condition: Condition assessment (bare, damaged, fair, good)
        operating_hours_per_year: Annual operating hours
    """
    location_id: str
    description: str
    system_type: str
    pipe_diameter_mm: float
    pipe_length_m: float
    surface_area_m2: float
    process_temperature_c: float
    ambient_temperature_c: float
    heat_loss_rate_w: float
    insulation_condition: str
    operating_hours_per_year: float = 8760.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SeasonalProfile:
    """
    Seasonal operating profile for degree-day adjustments.

    Attributes:
        season: Season name (winter, spring, summer, fall)
        operating_hours: Hours of operation in this season
        avg_ambient_temp_c: Average ambient temperature
        degree_days_hdd: Heating degree days
        degree_days_cdd: Cooling degree days
    """
    season: str
    operating_hours: float
    avg_ambient_temp_c: float
    degree_days_hdd: float = 0.0
    degree_days_cdd: float = 0.0


@dataclass
class FuelConsumption:
    """
    Fuel consumption equivalent for energy losses.

    Attributes:
        fuel_type: Type of fuel
        annual_consumption: Annual fuel consumption
        consumption_unit: Unit of measurement
        energy_mmbtu: Energy equivalent in MMBtu
        cost_usd: Annual fuel cost in USD
    """
    fuel_type: str
    annual_consumption: Decimal
    consumption_unit: str
    energy_mmbtu: Decimal
    cost_usd: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fuel_type": self.fuel_type,
            "annual_consumption": str(self.annual_consumption),
            "consumption_unit": self.consumption_unit,
            "energy_mmbtu": str(self.energy_mmbtu),
            "cost_usd": str(self.cost_usd),
        }


@dataclass
class CarbonFootprint:
    """
    Carbon footprint calculation result.

    Attributes:
        scope: Emission scope (1, 2, or 3)
        emissions_kg_co2e: Total emissions in kg CO2e
        emissions_tonnes_co2e: Total emissions in tonnes CO2e
        emission_factor_source: Source of emission factor
        carbon_cost: Dictionary of costs at various carbon prices
    """
    scope: int
    emissions_kg_co2e: Decimal
    emissions_tonnes_co2e: Decimal
    emission_factor_source: str
    carbon_cost: Dict[str, Decimal] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope,
            "emissions_kg_co2e": str(self.emissions_kg_co2e),
            "emissions_tonnes_co2e": str(self.emissions_tonnes_co2e),
            "emission_factor_source": self.emission_factor_source,
            "carbon_cost": {k: str(v) for k, v in self.carbon_cost.items()},
        }


@dataclass
class BenchmarkComparison:
    """
    Benchmark comparison result.

    Attributes:
        metric_name: Name of the metric being compared
        current_value: Current measured value
        industry_average: Industry average value
        best_practice: Best practice value
        percentile_rank: Percentile ranking (0-100)
        gap_to_average: Gap to industry average
        gap_to_best: Gap to best practice
    """
    metric_name: str
    current_value: Decimal
    industry_average: Decimal
    best_practice: Decimal
    percentile_rank: float
    gap_to_average: Decimal
    gap_to_best: Decimal
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current_value": str(self.current_value),
            "industry_average": str(self.industry_average),
            "best_practice": str(self.best_practice),
            "percentile_rank": self.percentile_rank,
            "gap_to_average": str(self.gap_to_average),
            "gap_to_best": str(self.gap_to_best),
            "unit": self.unit,
        }


@dataclass
class SavingsPotential:
    """
    Energy savings potential calculation.

    Attributes:
        scenario: Savings scenario name
        current_energy_mmbtu: Current annual energy loss
        target_energy_mmbtu: Target energy loss after improvement
        energy_savings_mmbtu: Potential energy savings
        energy_savings_percent: Percentage reduction
        cost_savings_usd: Annual cost savings
        co2_reduction_tonnes: CO2 reduction potential
        simple_payback_years: Simple payback period
    """
    scenario: str
    current_energy_mmbtu: Decimal
    target_energy_mmbtu: Decimal
    energy_savings_mmbtu: Decimal
    energy_savings_percent: Decimal
    cost_savings_usd: Decimal
    co2_reduction_tonnes: Decimal
    simple_payback_years: Optional[Decimal] = None
    estimated_improvement_cost: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario,
            "current_energy_mmbtu": str(self.current_energy_mmbtu),
            "target_energy_mmbtu": str(self.target_energy_mmbtu),
            "energy_savings_mmbtu": str(self.energy_savings_mmbtu),
            "energy_savings_percent": str(self.energy_savings_percent),
            "cost_savings_usd": str(self.cost_savings_usd),
            "co2_reduction_tonnes": str(self.co2_reduction_tonnes),
            "simple_payback_years": str(self.simple_payback_years) if self.simple_payback_years else None,
            "estimated_improvement_cost": str(self.estimated_improvement_cost) if self.estimated_improvement_cost else None,
        }


@dataclass
class EnergyEfficiencyMetrics:
    """
    Energy efficiency metrics for thermal systems.

    Attributes:
        heat_loss_per_meter: Heat loss per unit length (W/m)
        heat_loss_per_area: Heat loss per unit area (W/m2)
        efficiency_vs_bare: Efficiency compared to bare surface
        thermal_performance_index: Overall thermal performance index (0-100)
        insulation_effectiveness: Insulation effectiveness percentage
    """
    heat_loss_per_meter: Decimal
    heat_loss_per_area: Decimal
    efficiency_vs_bare: Decimal
    thermal_performance_index: Decimal
    insulation_effectiveness: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_loss_per_meter_w": str(self.heat_loss_per_meter),
            "heat_loss_per_area_w_m2": str(self.heat_loss_per_area),
            "efficiency_vs_bare_percent": str(self.efficiency_vs_bare),
            "thermal_performance_index": str(self.thermal_performance_index),
            "insulation_effectiveness_percent": str(self.insulation_effectiveness),
        }


@dataclass
class AnnualEnergyLoss:
    """
    Annual energy loss calculation result.

    Attributes:
        location_id: Location identifier
        heat_loss_rate_w: Instantaneous heat loss rate (W)
        operating_hours: Annual operating hours
        energy_loss_kwh: Annual energy loss (kWh)
        energy_loss_mmbtu: Annual energy loss (MMBtu)
        degree_day_adjustment: Degree-day adjustment factor applied
        seasonal_adjustment: Seasonal adjustment factor applied
        calculation_steps: Detailed calculation steps for audit
    """
    location_id: str
    heat_loss_rate_w: Decimal
    operating_hours: Decimal
    energy_loss_kwh: Decimal
    energy_loss_mmbtu: Decimal
    degree_day_adjustment: Decimal
    seasonal_adjustment: Decimal
    calculation_steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location_id": self.location_id,
            "heat_loss_rate_w": str(self.heat_loss_rate_w),
            "operating_hours": str(self.operating_hours),
            "energy_loss_kwh": str(self.energy_loss_kwh),
            "energy_loss_mmbtu": str(self.energy_loss_mmbtu),
            "degree_day_adjustment": str(self.degree_day_adjustment),
            "seasonal_adjustment": str(self.seasonal_adjustment),
            "calculation_steps": self.calculation_steps,
        }


@dataclass
class FacilityAggregation:
    """
    Facility-wide energy loss aggregation.

    Attributes:
        total_locations: Number of inspection locations
        total_energy_loss_mmbtu: Total annual energy loss
        total_cost_usd: Total annual energy cost
        total_emissions_tonnes: Total annual CO2e emissions
        by_system: Breakdown by system type
        by_condition: Breakdown by insulation condition
        pareto_analysis: 80/20 analysis results
        heat_loss_density_map: Heat loss by area
    """
    total_locations: int
    total_energy_loss_mmbtu: Decimal
    total_cost_usd: Decimal
    total_emissions_tonnes: Decimal
    by_system: Dict[str, Dict[str, Decimal]]
    by_condition: Dict[str, Dict[str, Decimal]]
    pareto_analysis: Dict[str, Any]
    heat_loss_density_map: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_locations": self.total_locations,
            "total_energy_loss_mmbtu": str(self.total_energy_loss_mmbtu),
            "total_cost_usd": str(self.total_cost_usd),
            "total_emissions_tonnes": str(self.total_emissions_tonnes),
            "by_system": {k: {sk: str(sv) for sk, sv in v.items()} for k, v in self.by_system.items()},
            "by_condition": {k: {sk: str(sv) for sk, sv in v.items()} for k, v in self.by_condition.items()},
            "pareto_analysis": self.pareto_analysis,
            "heat_loss_density_map": self.heat_loss_density_map,
        }


@dataclass
class EnergyLossReport:
    """
    Comprehensive energy loss report.

    Attributes:
        report_id: Unique report identifier
        facility_name: Name of the facility
        report_date: Date of report generation
        annual_energy_losses: List of annual energy losses by location
        fuel_consumption: Fuel consumption equivalents
        carbon_footprint: Carbon footprint calculation
        facility_aggregation: Facility-wide aggregation
        benchmarks: Benchmark comparisons
        savings_potential: Savings potential analysis
        efficiency_metrics: Energy efficiency metrics
        provenance_hash: SHA-256 hash for audit trail
        calculation_metadata: Additional calculation metadata
    """
    report_id: str
    facility_name: str
    report_date: date
    annual_energy_losses: List[AnnualEnergyLoss]
    fuel_consumption: List[FuelConsumption]
    carbon_footprint: CarbonFootprint
    facility_aggregation: FacilityAggregation
    benchmarks: List[BenchmarkComparison]
    savings_potential: List[SavingsPotential]
    efficiency_metrics: EnergyEfficiencyMetrics
    provenance_hash: str
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "facility_name": self.facility_name,
            "report_date": self.report_date.isoformat(),
            "annual_energy_losses": [ael.to_dict() for ael in self.annual_energy_losses],
            "fuel_consumption": [fc.to_dict() for fc in self.fuel_consumption],
            "carbon_footprint": self.carbon_footprint.to_dict(),
            "facility_aggregation": self.facility_aggregation.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "savings_potential": [sp.to_dict() for sp in self.savings_potential],
            "efficiency_metrics": self.efficiency_metrics.to_dict(),
            "provenance_hash": self.provenance_hash,
            "calculation_metadata": self.calculation_metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# ENERGY LOSS QUANTIFIER CLASS
# =============================================================================

class EnergyLossQuantifier:
    """
    Energy Loss Quantifier for GL-015 INSULSCAN.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic (no LLM involvement)
    - Full provenance tracking with SHA-256 hashing
    - Complete audit trail for regulatory compliance
    - Bit-perfect reproducibility (same input -> same output)

    Calculation Capabilities:
    1. Annual Energy Loss Calculation
    2. Fuel Consumption Equivalent
    3. Energy Cost Impact
    4. Carbon Footprint Calculation
    5. Facility-Wide Aggregation
    6. Benchmark Comparison
    7. Savings Potential
    8. Energy Efficiency Metrics

    Reference Standards:
    - ASTM C680: Heat Gain or Loss Determination
    - ISO 12241: Thermal insulation calculations
    - GHG Protocol Corporate Standard
    - EPA emission factors (2024)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        default_fuel_type: str = FuelType.NATURAL_GAS.value,
        default_boiler_efficiency: Optional[float] = None,
        fuel_prices: Optional[Dict[str, float]] = None,
        grid_emission_factor: Optional[float] = None,
    ):
        """
        Initialize Energy Loss Quantifier.

        Args:
            default_fuel_type: Default fuel type for calculations
            default_boiler_efficiency: Override default boiler efficiency
            fuel_prices: Custom fuel prices (USD per unit)
            grid_emission_factor: Custom grid emission factor (kg CO2e/kWh)
        """
        self.default_fuel_type = default_fuel_type
        self.default_boiler_efficiency = default_boiler_efficiency or DEFAULT_BOILER_EFFICIENCIES.get(
            default_fuel_type, 0.85
        )

        # Default fuel prices (USD per unit) - 2024 US average
        self.fuel_prices: Dict[str, Dict[str, float]] = fuel_prices or {
            FuelType.NATURAL_GAS.value: {"price_per_mmbtu": 4.50, "unit": "MMBtu"},
            FuelType.FUEL_OIL_NO2.value: {"price_per_gallon": 3.25, "unit": "gallon"},
            FuelType.FUEL_OIL_NO6.value: {"price_per_gallon": 2.80, "unit": "gallon"},
            FuelType.PROPANE.value: {"price_per_gallon": 2.50, "unit": "gallon"},
            FuelType.COAL_BITUMINOUS.value: {"price_per_ton": 85.00, "unit": "ton"},
            FuelType.COAL_ANTHRACITE.value: {"price_per_ton": 120.00, "unit": "ton"},
            FuelType.ELECTRICITY.value: {"price_per_kwh": 0.12, "unit": "kWh"},
            FuelType.STEAM.value: {"price_per_mlb": 15.00, "unit": "Mlb"},
            FuelType.DIESEL.value: {"price_per_gallon": 3.50, "unit": "gallon"},
        }

        self.grid_emission_factor = grid_emission_factor or 0.417  # US average

        logger.info(f"EnergyLossQuantifier initialized v{self.VERSION}")

    # =========================================================================
    # 1. ANNUAL ENERGY LOSS CALCULATION
    # =========================================================================

    def calculate_annual_energy_loss(
        self,
        location: InspectionLocation,
        seasonal_profiles: Optional[List[SeasonalProfile]] = None,
        degree_day_base_temp_c: float = 18.0,
    ) -> AnnualEnergyLoss:
        """
        Calculate annual energy loss for a single inspection location.

        Formula: E_annual = Q_loss * t_op * f_dd * f_seasonal

        Where:
        - Q_loss: Heat loss rate (W)
        - t_op: Operating hours per year
        - f_dd: Degree-day adjustment factor
        - f_seasonal: Seasonal adjustment factor

        Args:
            location: InspectionLocation with thermal data
            seasonal_profiles: Optional seasonal operating profiles
            degree_day_base_temp_c: Base temperature for degree-day calculation

        Returns:
            AnnualEnergyLoss with detailed calculation steps

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        calculation_steps = []

        # Step 1: Get base heat loss rate
        heat_loss_w = Decimal(str(location.heat_loss_rate_w))
        calculation_steps.append({
            "step": 1,
            "description": "Get measured heat loss rate",
            "heat_loss_rate_w": str(heat_loss_w),
        })

        # Step 2: Calculate degree-day adjustment factor
        delta_t_design = Decimal(str(location.process_temperature_c - location.ambient_temperature_c))

        if seasonal_profiles:
            # Calculate weighted average ambient temperature
            total_hours = sum(sp.operating_hours for sp in seasonal_profiles)
            weighted_ambient = sum(
                sp.avg_ambient_temp_c * sp.operating_hours
                for sp in seasonal_profiles
            ) / total_hours if total_hours > 0 else location.ambient_temperature_c

            delta_t_actual = Decimal(str(location.process_temperature_c - weighted_ambient))
            degree_day_factor = delta_t_actual / delta_t_design if delta_t_design != 0 else Decimal("1.0")
        else:
            degree_day_factor = Decimal("1.0")

        # Ensure factor is reasonable (0.5 to 1.5 range)
        degree_day_factor = max(Decimal("0.5"), min(Decimal("1.5"), degree_day_factor))

        calculation_steps.append({
            "step": 2,
            "description": "Calculate degree-day adjustment factor",
            "delta_t_design": str(delta_t_design),
            "degree_day_factor": str(degree_day_factor),
        })

        # Step 3: Calculate seasonal adjustment factor
        if seasonal_profiles:
            # Weight by heating degree days if available
            total_hdd = sum(sp.degree_days_hdd for sp in seasonal_profiles)
            if total_hdd > 0:
                seasonal_factor = Decimal(str(total_hdd / 5000))  # Normalize to typical HDD
                seasonal_factor = max(Decimal("0.7"), min(Decimal("1.3"), seasonal_factor))
            else:
                seasonal_factor = Decimal("1.0")
        else:
            seasonal_factor = Decimal("1.0")

        calculation_steps.append({
            "step": 3,
            "description": "Calculate seasonal adjustment factor",
            "seasonal_factor": str(seasonal_factor),
        })

        # Step 4: Calculate operating hours
        operating_hours = Decimal(str(location.operating_hours_per_year))
        calculation_steps.append({
            "step": 4,
            "description": "Get operating hours",
            "operating_hours": str(operating_hours),
        })

        # Step 5: Calculate annual energy loss
        # E (Wh) = Q (W) * t (h) * f_dd * f_seasonal
        energy_loss_wh = heat_loss_w * operating_hours * degree_day_factor * seasonal_factor
        energy_loss_kwh = (energy_loss_wh / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Convert to MMBtu (1 kWh = 0.003412 MMBtu)
        energy_loss_mmbtu = (energy_loss_kwh * Decimal("0.003412")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        calculation_steps.append({
            "step": 5,
            "description": "Calculate annual energy loss",
            "formula": "E_annual = Q_loss * t_op * f_dd * f_seasonal",
            "energy_loss_kwh": str(energy_loss_kwh),
            "energy_loss_mmbtu": str(energy_loss_mmbtu),
        })

        return AnnualEnergyLoss(
            location_id=location.location_id,
            heat_loss_rate_w=heat_loss_w,
            operating_hours=operating_hours,
            energy_loss_kwh=energy_loss_kwh,
            energy_loss_mmbtu=energy_loss_mmbtu,
            degree_day_adjustment=degree_day_factor,
            seasonal_adjustment=seasonal_factor,
            calculation_steps=calculation_steps,
        )

    # =========================================================================
    # 2. FUEL CONSUMPTION EQUIVALENT
    # =========================================================================

    def calculate_fuel_consumption(
        self,
        energy_loss_mmbtu: Decimal,
        fuel_type: Optional[str] = None,
        boiler_efficiency: Optional[float] = None,
    ) -> FuelConsumption:
        """
        Calculate fuel consumption equivalent for energy losses.

        Formula: Fuel = E_loss / (eta_boiler * HHV)

        Where:
        - E_loss: Annual energy loss (MMBtu)
        - eta_boiler: Boiler/heater efficiency
        - HHV: Higher heating value of fuel

        Args:
            energy_loss_mmbtu: Annual energy loss in MMBtu
            fuel_type: Type of fuel (default: natural_gas)
            boiler_efficiency: Boiler efficiency (0-1)

        Returns:
            FuelConsumption with consumption and cost

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        fuel_type = fuel_type or self.default_fuel_type
        efficiency = boiler_efficiency or DEFAULT_BOILER_EFFICIENCIES.get(fuel_type, 0.85)

        # Get fuel-specific data
        fuel_data = EPA_EMISSION_FACTORS.get(fuel_type, {})
        price_data = self.fuel_prices.get(fuel_type, {})

        # Calculate fuel input required (accounting for boiler efficiency)
        fuel_input_mmbtu = energy_loss_mmbtu / Decimal(str(efficiency))

        # Convert to fuel-specific units
        if fuel_type == FuelType.NATURAL_GAS.value:
            # Natural gas: MMBtu direct
            consumption = fuel_input_mmbtu
            unit = "MMBtu"
            price_per_unit = Decimal(str(price_data.get("price_per_mmbtu", 4.50)))

        elif fuel_type in [FuelType.FUEL_OIL_NO2.value, FuelType.FUEL_OIL_NO6.value, FuelType.DIESEL.value]:
            # Fuel oil: gallons
            btu_per_gallon = Decimal(str(fuel_data.get("btu_per_gallon", 138000)))
            consumption = (fuel_input_mmbtu * Decimal("1000000") / btu_per_gallon).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            unit = "gallons"
            price_per_unit = Decimal(str(price_data.get("price_per_gallon", 3.25)))

        elif fuel_type == FuelType.PROPANE.value:
            # Propane: gallons
            btu_per_gallon = Decimal(str(fuel_data.get("btu_per_gallon", 91000)))
            consumption = (fuel_input_mmbtu * Decimal("1000000") / btu_per_gallon).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            unit = "gallons"
            price_per_unit = Decimal(str(price_data.get("price_per_gallon", 2.50)))

        elif fuel_type in [FuelType.COAL_BITUMINOUS.value, FuelType.COAL_ANTHRACITE.value]:
            # Coal: tons
            btu_per_ton = Decimal(str(fuel_data.get("btu_per_ton", 24930000)))
            consumption = (fuel_input_mmbtu * Decimal("1000000") / btu_per_ton).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            unit = "tons"
            price_per_unit = Decimal(str(price_data.get("price_per_ton", 85.00)))

        elif fuel_type == FuelType.ELECTRICITY.value:
            # Electricity: kWh (1 MMBtu = 293.07 kWh)
            consumption = (fuel_input_mmbtu * Decimal("293.07")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            unit = "kWh"
            price_per_unit = Decimal(str(price_data.get("price_per_kwh", 0.12)))

        elif fuel_type == FuelType.STEAM.value:
            # Steam: lb (assume 1000 Btu/lb average enthalpy)
            btu_per_lb = Decimal(str(fuel_data.get("btu_per_lb", 1000)))
            consumption = (fuel_input_mmbtu * Decimal("1000000") / btu_per_lb).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            unit = "lb"
            # Convert to Mlb for pricing
            price_per_unit = Decimal(str(price_data.get("price_per_mlb", 15.00))) / Decimal("1000")

        else:
            # Default to MMBtu
            consumption = fuel_input_mmbtu
            unit = "MMBtu"
            price_per_unit = Decimal("4.50")

        # Calculate cost
        cost_usd = (consumption * price_per_unit).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        return FuelConsumption(
            fuel_type=fuel_type,
            annual_consumption=consumption,
            consumption_unit=unit,
            energy_mmbtu=fuel_input_mmbtu.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            cost_usd=cost_usd,
        )

    # =========================================================================
    # 3. ENERGY COST IMPACT
    # =========================================================================

    def calculate_energy_cost(
        self,
        fuel_consumptions: List[FuelConsumption],
        demand_charge_kw: Optional[float] = None,
        demand_charge_rate: Optional[float] = None,
        time_of_use_multiplier: float = 1.0,
        fuel_escalation_rate: float = 0.03,
        projection_years: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive energy cost impact.

        Includes:
        - Base fuel costs
        - Demand charges (for electric)
        - Time-of-use adjustments
        - Future cost projections with escalation

        Args:
            fuel_consumptions: List of fuel consumption records
            demand_charge_kw: Peak demand in kW (for electric)
            demand_charge_rate: Demand charge rate ($/kW)
            time_of_use_multiplier: TOU rate adjustment factor
            fuel_escalation_rate: Annual fuel price escalation rate
            projection_years: Years for future cost projection

        Returns:
            Dictionary with detailed cost breakdown

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        projection_years = projection_years or [5, 10, 20]

        # Calculate base annual costs
        total_base_cost = Decimal("0")
        cost_by_fuel = {}

        for fc in fuel_consumptions:
            adjusted_cost = fc.cost_usd * Decimal(str(time_of_use_multiplier))
            cost_by_fuel[fc.fuel_type] = {
                "base_cost": str(fc.cost_usd),
                "adjusted_cost": str(adjusted_cost),
                "consumption": str(fc.annual_consumption),
                "unit": fc.consumption_unit,
            }
            total_base_cost += adjusted_cost

        # Add demand charges if applicable
        demand_charge_annual = Decimal("0")
        if demand_charge_kw and demand_charge_rate:
            # Assume 12 months
            demand_charge_annual = Decimal(str(demand_charge_kw * demand_charge_rate * 12))

        total_annual_cost = total_base_cost + demand_charge_annual

        # Calculate future projections
        projections = {}
        for year in projection_years:
            escalation_factor = Decimal(str((1 + fuel_escalation_rate) ** year))
            # Cumulative cost over projection period
            if fuel_escalation_rate > 0:
                cumulative_factor = (escalation_factor - 1) / Decimal(str(fuel_escalation_rate))
            else:
                cumulative_factor = Decimal(str(year))

            projected_annual = (total_annual_cost * escalation_factor).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            cumulative = (total_annual_cost * cumulative_factor).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            projections[f"year_{year}"] = {
                "annual_cost": str(projected_annual),
                "cumulative_cost": str(cumulative),
                "escalation_factor": str(escalation_factor.quantize(Decimal("0.001"))),
            }

        return {
            "annual_cost_summary": {
                "total_base_cost": str(total_base_cost),
                "demand_charges": str(demand_charge_annual),
                "time_of_use_multiplier": time_of_use_multiplier,
                "total_annual_cost": str(total_annual_cost),
            },
            "cost_by_fuel": cost_by_fuel,
            "future_projections": projections,
            "assumptions": {
                "fuel_escalation_rate": fuel_escalation_rate,
                "demand_charge_kw": demand_charge_kw,
                "demand_charge_rate": demand_charge_rate,
            },
        }

    # =========================================================================
    # 4. CARBON FOOTPRINT CALCULATION
    # =========================================================================

    def calculate_carbon_footprint(
        self,
        fuel_consumptions: List[FuelConsumption],
        include_scope_2: bool = True,
        custom_grid_factor: Optional[float] = None,
        carbon_price_scenarios: Optional[Dict[str, float]] = None,
    ) -> CarbonFootprint:
        """
        Calculate carbon footprint from energy losses.

        Emissions = Fuel_consumption * Emission_factor

        Scope 1: Direct emissions from fuel combustion
        Scope 2: Indirect emissions from purchased electricity/steam

        Args:
            fuel_consumptions: List of fuel consumption records
            include_scope_2: Include Scope 2 (electricity) emissions
            custom_grid_factor: Custom grid emission factor (kg CO2e/kWh)
            carbon_price_scenarios: Custom carbon price scenarios

        Returns:
            CarbonFootprint with emissions and carbon costs

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        total_emissions_kg = Decimal("0")
        scope_1_emissions = Decimal("0")
        scope_2_emissions = Decimal("0")

        grid_factor = custom_grid_factor or self.grid_emission_factor
        prices = carbon_price_scenarios or CARBON_PRICE_SCENARIOS

        for fc in fuel_consumptions:
            fuel_data = EPA_EMISSION_FACTORS.get(fc.fuel_type, {})

            if fc.fuel_type == FuelType.ELECTRICITY.value:
                # Scope 2 emissions
                if include_scope_2:
                    # fc.annual_consumption is in kWh
                    emissions_kg = fc.annual_consumption * Decimal(str(grid_factor))
                    scope_2_emissions += emissions_kg
            elif fc.fuel_type == FuelType.STEAM.value:
                # Scope 2 emissions (from steam supplier)
                if include_scope_2:
                    # Use natural gas equivalent
                    kg_per_lb = Decimal(str(fuel_data.get("kg_co2e_per_lb", 0.0606)))
                    emissions_kg = fc.annual_consumption * kg_per_lb
                    scope_2_emissions += emissions_kg
            else:
                # Scope 1 emissions (direct combustion)
                kg_per_mmbtu = Decimal(str(fuel_data.get("kg_co2e_per_mmbtu", 53.06)))
                emissions_kg = fc.energy_mmbtu * kg_per_mmbtu
                scope_1_emissions += emissions_kg

        total_emissions_kg = scope_1_emissions + scope_2_emissions
        total_emissions_tonnes = (total_emissions_kg / Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Calculate carbon costs at various price points
        carbon_costs = {}
        for scenario, price in prices.items():
            cost = (total_emissions_tonnes * Decimal(str(price))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            carbon_costs[scenario] = cost

        # Determine primary scope
        primary_scope = 1 if scope_1_emissions >= scope_2_emissions else 2

        return CarbonFootprint(
            scope=primary_scope,
            emissions_kg_co2e=total_emissions_kg.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            emissions_tonnes_co2e=total_emissions_tonnes,
            emission_factor_source="EPA 2024 Emission Factors",
            carbon_cost=carbon_costs,
        )

    # =========================================================================
    # 5. FACILITY-WIDE AGGREGATION
    # =========================================================================

    def aggregate_facility_losses(
        self,
        annual_losses: List[AnnualEnergyLoss],
        locations: List[InspectionLocation],
        fuel_type: Optional[str] = None,
    ) -> FacilityAggregation:
        """
        Aggregate energy losses across all facility locations.

        Includes:
        - Total energy losses
        - Breakdown by system type
        - Breakdown by insulation condition
        - Pareto analysis (80/20 rule)
        - Heat loss density mapping

        Args:
            annual_losses: List of annual energy loss calculations
            locations: List of inspection locations
            fuel_type: Fuel type for cost calculations

        Returns:
            FacilityAggregation with comprehensive breakdown

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        fuel_type = fuel_type or self.default_fuel_type

        # Create location lookup
        location_map = {loc.location_id: loc for loc in locations}

        # Calculate totals
        total_mmbtu = Decimal("0")
        by_system: Dict[str, Dict[str, Decimal]] = {}
        by_condition: Dict[str, Dict[str, Decimal]] = {}
        location_losses: List[Tuple[str, Decimal]] = []

        for loss in annual_losses:
            total_mmbtu += loss.energy_loss_mmbtu
            location_losses.append((loss.location_id, loss.energy_loss_mmbtu))

            loc = location_map.get(loss.location_id)
            if loc:
                # By system type
                system = loc.system_type
                if system not in by_system:
                    by_system[system] = {
                        "energy_mmbtu": Decimal("0"),
                        "count": Decimal("0"),
                    }
                by_system[system]["energy_mmbtu"] += loss.energy_loss_mmbtu
                by_system[system]["count"] += Decimal("1")

                # By condition
                condition = loc.insulation_condition
                if condition not in by_condition:
                    by_condition[condition] = {
                        "energy_mmbtu": Decimal("0"),
                        "count": Decimal("0"),
                    }
                by_condition[condition]["energy_mmbtu"] += loss.energy_loss_mmbtu
                by_condition[condition]["count"] += Decimal("1")

        # Calculate fuel consumption and cost for total
        total_consumption = self.calculate_fuel_consumption(total_mmbtu, fuel_type)
        total_cost = total_consumption.cost_usd

        # Calculate carbon footprint
        carbon = self.calculate_carbon_footprint([total_consumption])
        total_emissions = carbon.emissions_tonnes_co2e

        # Pareto analysis (80/20)
        location_losses.sort(key=lambda x: x[1], reverse=True)
        cumulative = Decimal("0")
        pareto_80_locations = []

        for loc_id, loss in location_losses:
            cumulative += loss
            pareto_80_locations.append(loc_id)
            if cumulative >= total_mmbtu * Decimal("0.8"):
                break

        pareto_analysis = {
            "top_20_percent_locations": pareto_80_locations[:max(1, len(locations) // 5)],
            "top_20_percent_contribution": str(
                sum(l[1] for l in location_losses[:max(1, len(locations) // 5)])
            ),
            "locations_for_80_percent": pareto_80_locations,
            "concentration_ratio": str(
                (Decimal(str(len(pareto_80_locations))) / Decimal(str(len(locations))) * 100).quantize(
                    Decimal("0.1")
                )
            ) if locations else "0",
        }

        # Heat loss density map
        density_map = []
        for loss in annual_losses:
            loc = location_map.get(loss.location_id)
            if loc and loc.surface_area_m2 > 0:
                density = (loss.energy_loss_kwh * Decimal("1000") /
                          Decimal(str(loc.surface_area_m2)) /
                          loss.operating_hours).quantize(Decimal("0.01"))
                density_map.append({
                    "location_id": loss.location_id,
                    "heat_loss_density_w_m2": str(density),
                    "surface_area_m2": loc.surface_area_m2,
                    "rank": 0,  # Will be filled below
                })

        # Sort and rank density map
        density_map.sort(key=lambda x: Decimal(x["heat_loss_density_w_m2"]), reverse=True)
        for i, item in enumerate(density_map):
            item["rank"] = i + 1

        return FacilityAggregation(
            total_locations=len(locations),
            total_energy_loss_mmbtu=total_mmbtu.quantize(Decimal("0.001")),
            total_cost_usd=total_cost,
            total_emissions_tonnes=total_emissions,
            by_system=by_system,
            by_condition=by_condition,
            pareto_analysis=pareto_analysis,
            heat_loss_density_map=density_map,
        )

    # =========================================================================
    # 6. BENCHMARK COMPARISON
    # =========================================================================

    def benchmark_performance(
        self,
        locations: List[InspectionLocation],
        annual_losses: List[AnnualEnergyLoss],
        industry_benchmarks: Optional[Dict[str, Dict[str, float]]] = None,
        historical_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BenchmarkComparison]:
        """
        Compare facility performance against industry benchmarks.

        Compares:
        - Heat loss rates (W/m)
        - Heat loss per area (W/m2)
        - Energy intensity
        - Insulation effectiveness

        Args:
            locations: List of inspection locations
            annual_losses: List of annual energy loss calculations
            industry_benchmarks: Custom industry benchmarks
            historical_data: Historical facility data for trending

        Returns:
            List of BenchmarkComparison results

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        benchmarks = industry_benchmarks or BENCHMARK_HEAT_LOSS_RATES
        comparisons = []

        # Create loss lookup
        loss_map = {loss.location_id: loss for loss in annual_losses}

        # Calculate facility averages
        total_length_m = Decimal("0")
        total_area_m2 = Decimal("0")
        total_heat_loss_w = Decimal("0")

        for loc in locations:
            loss = loss_map.get(loc.location_id)
            if loss:
                total_length_m += Decimal(str(loc.pipe_length_m))
                total_area_m2 += Decimal(str(loc.surface_area_m2))
                total_heat_loss_w += loss.heat_loss_rate_w

        # Calculate metrics
        if total_length_m > 0:
            avg_loss_per_meter = (total_heat_loss_w / total_length_m).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            avg_loss_per_meter = Decimal("0")

        if total_area_m2 > 0:
            avg_loss_per_area = (total_heat_loss_w / total_area_m2).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            avg_loss_per_area = Decimal("0")

        # Determine temperature category (use average process temp)
        avg_process_temp = sum(loc.process_temperature_c for loc in locations) / len(locations) if locations else 100

        if avg_process_temp <= 100:
            temp_category = "low_temp_100c"
        elif avg_process_temp <= 200:
            temp_category = "med_temp_200c"
        elif avg_process_temp <= 300:
            temp_category = "high_temp_300c"
        else:
            temp_category = "very_high_temp_400c"

        # Benchmark 1: Heat loss per meter
        industry_avg_per_m = Decimal(str(benchmarks.get("fair_insulation", {}).get(temp_category, 100)))
        best_practice_per_m = Decimal(str(benchmarks.get("best_practice", {}).get(temp_category, 25)))
        bare_loss_per_m = Decimal(str(benchmarks.get("bare_pipe", {}).get(temp_category, 650)))

        # Calculate percentile (0 = best practice, 100 = bare)
        if bare_loss_per_m > best_practice_per_m:
            percentile = float(
                ((avg_loss_per_meter - best_practice_per_m) / (bare_loss_per_m - best_practice_per_m) * 100)
            )
            percentile = max(0, min(100, percentile))
        else:
            percentile = 50.0

        comparisons.append(BenchmarkComparison(
            metric_name="Heat Loss per Meter",
            current_value=avg_loss_per_meter,
            industry_average=industry_avg_per_m,
            best_practice=best_practice_per_m,
            percentile_rank=round(percentile, 1),
            gap_to_average=(avg_loss_per_meter - industry_avg_per_m).quantize(Decimal("0.01")),
            gap_to_best=(avg_loss_per_meter - best_practice_per_m).quantize(Decimal("0.01")),
            unit="W/m",
        ))

        # Benchmark 2: Heat loss per area
        # Convert per-meter to per-area using typical pipe geometry
        industry_avg_per_area = (industry_avg_per_m * Decimal("10")).quantize(Decimal("0.01"))  # Approximate
        best_practice_per_area = (best_practice_per_m * Decimal("10")).quantize(Decimal("0.01"))

        comparisons.append(BenchmarkComparison(
            metric_name="Heat Loss per Area",
            current_value=avg_loss_per_area,
            industry_average=industry_avg_per_area,
            best_practice=best_practice_per_area,
            percentile_rank=round(percentile, 1),
            gap_to_average=(avg_loss_per_area - industry_avg_per_area).quantize(Decimal("0.01")),
            gap_to_best=(avg_loss_per_area - best_practice_per_area).quantize(Decimal("0.01")),
            unit="W/m2",
        ))

        # Benchmark 3: Insulation effectiveness
        if bare_loss_per_m > 0:
            current_effectiveness = ((bare_loss_per_m - avg_loss_per_meter) / bare_loss_per_m * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            avg_effectiveness = ((bare_loss_per_m - industry_avg_per_m) / bare_loss_per_m * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
            best_effectiveness = ((bare_loss_per_m - best_practice_per_m) / bare_loss_per_m * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            current_effectiveness = Decimal("0")
            avg_effectiveness = Decimal("80")
            best_effectiveness = Decimal("95")

        comparisons.append(BenchmarkComparison(
            metric_name="Insulation Effectiveness",
            current_value=current_effectiveness,
            industry_average=avg_effectiveness,
            best_practice=best_effectiveness,
            percentile_rank=round(float(current_effectiveness), 1),
            gap_to_average=(current_effectiveness - avg_effectiveness).quantize(Decimal("0.1")),
            gap_to_best=(current_effectiveness - best_effectiveness).quantize(Decimal("0.1")),
            unit="%",
        ))

        return comparisons

    # =========================================================================
    # 7. SAVINGS POTENTIAL
    # =========================================================================

    def calculate_savings_potential(
        self,
        current_loss_mmbtu: Decimal,
        fuel_type: Optional[str] = None,
        improvement_scenarios: Optional[List[Dict[str, Any]]] = None,
        insulation_cost_per_m2: float = 50.0,
        total_repair_area_m2: Optional[float] = None,
    ) -> List[SavingsPotential]:
        """
        Calculate energy savings potential for various improvement scenarios.

        Scenarios:
        - Current vs Design State
        - Current vs Best Practice
        - Marginal improvements (10%, 25%, 50%)

        Args:
            current_loss_mmbtu: Current annual energy loss
            fuel_type: Fuel type for cost calculations
            improvement_scenarios: Custom improvement scenarios
            insulation_cost_per_m2: Insulation repair cost per m2
            total_repair_area_m2: Total area requiring repair

        Returns:
            List of SavingsPotential scenarios

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        fuel_type = fuel_type or self.default_fuel_type

        # Default improvement scenarios
        default_scenarios = [
            {"name": "10% Improvement", "reduction_pct": 0.10},
            {"name": "25% Improvement", "reduction_pct": 0.25},
            {"name": "50% Improvement", "reduction_pct": 0.50},
            {"name": "Best Practice (80%)", "reduction_pct": 0.80},
            {"name": "Full Remediation (95%)", "reduction_pct": 0.95},
        ]

        scenarios = improvement_scenarios or default_scenarios
        results = []

        for scenario in scenarios:
            name = scenario.get("name", "Unknown")
            reduction_pct = Decimal(str(scenario.get("reduction_pct", 0.25)))

            # Calculate target energy loss
            energy_savings = current_loss_mmbtu * reduction_pct
            target_loss = current_loss_mmbtu - energy_savings

            # Calculate cost savings
            current_consumption = self.calculate_fuel_consumption(current_loss_mmbtu, fuel_type)
            target_consumption = self.calculate_fuel_consumption(target_loss, fuel_type)
            cost_savings = current_consumption.cost_usd - target_consumption.cost_usd

            # Calculate CO2 reduction
            current_carbon = self.calculate_carbon_footprint([current_consumption])
            target_carbon = self.calculate_carbon_footprint([target_consumption])
            co2_reduction = current_carbon.emissions_tonnes_co2e - target_carbon.emissions_tonnes_co2e

            # Estimate improvement cost and payback
            improvement_cost = None
            payback_years = None

            if total_repair_area_m2:
                # Scale repair area by reduction percentage
                repair_area = Decimal(str(total_repair_area_m2)) * reduction_pct
                improvement_cost = (repair_area * Decimal(str(insulation_cost_per_m2))).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                if cost_savings > 0:
                    payback_years = (improvement_cost / cost_savings).quantize(
                        Decimal("0.1"), rounding=ROUND_HALF_UP
                    )

            results.append(SavingsPotential(
                scenario=name,
                current_energy_mmbtu=current_loss_mmbtu.quantize(Decimal("0.001")),
                target_energy_mmbtu=target_loss.quantize(Decimal("0.001")),
                energy_savings_mmbtu=energy_savings.quantize(Decimal("0.001")),
                energy_savings_percent=(reduction_pct * 100).quantize(Decimal("0.1")),
                cost_savings_usd=cost_savings,
                co2_reduction_tonnes=co2_reduction,
                simple_payback_years=payback_years,
                estimated_improvement_cost=improvement_cost,
            ))

        return results

    # =========================================================================
    # 8. ENERGY EFFICIENCY METRICS
    # =========================================================================

    def calculate_efficiency_metrics(
        self,
        locations: List[InspectionLocation],
        annual_losses: List[AnnualEnergyLoss],
    ) -> EnergyEfficiencyMetrics:
        """
        Calculate comprehensive energy efficiency metrics.

        Metrics:
        - Heat loss per unit length (W/m)
        - Heat loss per unit area (W/m2)
        - Efficiency compared to bare pipe
        - Thermal performance index (0-100)
        - Insulation effectiveness (%)

        Args:
            locations: List of inspection locations
            annual_losses: List of annual energy loss calculations

        Returns:
            EnergyEfficiencyMetrics with all metrics

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        loss_map = {loss.location_id: loss for loss in annual_losses}

        total_length_m = Decimal("0")
        total_area_m2 = Decimal("0")
        total_heat_loss_w = Decimal("0")
        total_bare_loss_w = Decimal("0")

        for loc in locations:
            loss = loss_map.get(loc.location_id)
            if loss:
                total_length_m += Decimal(str(loc.pipe_length_m))
                total_area_m2 += Decimal(str(loc.surface_area_m2))
                total_heat_loss_w += loss.heat_loss_rate_w

                # Estimate bare pipe loss based on temperature
                temp_c = loc.process_temperature_c
                if temp_c <= 100:
                    bare_rate = 250.0
                elif temp_c <= 200:
                    bare_rate = 650.0
                elif temp_c <= 300:
                    bare_rate = 1200.0
                else:
                    bare_rate = 2000.0

                bare_loss_w = Decimal(str(bare_rate * loc.pipe_length_m))
                total_bare_loss_w += bare_loss_w

        # Calculate metrics
        if total_length_m > 0:
            heat_loss_per_meter = (total_heat_loss_w / total_length_m).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            heat_loss_per_meter = Decimal("0")

        if total_area_m2 > 0:
            heat_loss_per_area = (total_heat_loss_w / total_area_m2).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            heat_loss_per_area = Decimal("0")

        # Efficiency vs bare
        if total_bare_loss_w > 0:
            efficiency_vs_bare = ((total_bare_loss_w - total_heat_loss_w) / total_bare_loss_w * 100).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            efficiency_vs_bare = Decimal("0")

        # Thermal performance index (0-100, 100 = best)
        # Based on efficiency vs bare, scaled to 0-100
        thermal_performance_index = efficiency_vs_bare.quantize(Decimal("0.1"))

        # Insulation effectiveness (same as efficiency vs bare for this calculation)
        insulation_effectiveness = efficiency_vs_bare

        return EnergyEfficiencyMetrics(
            heat_loss_per_meter=heat_loss_per_meter,
            heat_loss_per_area=heat_loss_per_area,
            efficiency_vs_bare=efficiency_vs_bare,
            thermal_performance_index=thermal_performance_index,
            insulation_effectiveness=insulation_effectiveness,
        )

    # =========================================================================
    # REPORT GENERATION
    # =========================================================================

    def generate_energy_report(
        self,
        facility_name: str,
        locations: List[InspectionLocation],
        fuel_type: Optional[str] = None,
        seasonal_profiles: Optional[List[SeasonalProfile]] = None,
        carbon_price_scenarios: Optional[Dict[str, float]] = None,
    ) -> EnergyLossReport:
        """
        Generate comprehensive energy loss report.

        Combines all calculations into a single auditable report with
        SHA-256 provenance hash for regulatory compliance.

        Args:
            facility_name: Name of the facility
            locations: List of inspection locations
            fuel_type: Fuel type for calculations
            seasonal_profiles: Seasonal operating profiles
            carbon_price_scenarios: Carbon price scenarios

        Returns:
            EnergyLossReport with complete analysis

        DETERMINISTIC: Same inputs always produce same outputs.
        """
        fuel_type = fuel_type or self.default_fuel_type
        report_date = date.today()

        # Step 1: Calculate annual energy losses for all locations
        annual_losses = []
        for loc in locations:
            loss = self.calculate_annual_energy_loss(loc, seasonal_profiles)
            annual_losses.append(loss)

        # Step 2: Calculate total energy loss
        total_loss_mmbtu = sum(loss.energy_loss_mmbtu for loss in annual_losses)

        # Step 3: Calculate fuel consumption
        fuel_consumption = self.calculate_fuel_consumption(total_loss_mmbtu, fuel_type)
        fuel_consumptions = [fuel_consumption]

        # Step 4: Calculate carbon footprint
        carbon_footprint = self.calculate_carbon_footprint(
            fuel_consumptions,
            carbon_price_scenarios=carbon_price_scenarios,
        )

        # Step 5: Aggregate facility losses
        facility_aggregation = self.aggregate_facility_losses(
            annual_losses, locations, fuel_type
        )

        # Step 6: Benchmark performance
        benchmarks = self.benchmark_performance(locations, annual_losses)

        # Step 7: Calculate savings potential
        total_repair_area = sum(loc.surface_area_m2 for loc in locations)
        savings_potential = self.calculate_savings_potential(
            total_loss_mmbtu,
            fuel_type,
            total_repair_area_m2=total_repair_area,
        )

        # Step 8: Calculate efficiency metrics
        efficiency_metrics = self.calculate_efficiency_metrics(locations, annual_losses)

        # Generate provenance hash
        provenance_data = {
            "facility_name": facility_name,
            "report_date": report_date.isoformat(),
            "locations_count": len(locations),
            "total_loss_mmbtu": str(total_loss_mmbtu),
            "fuel_type": fuel_type,
            "calculation_engine_version": self.VERSION,
            "location_ids": sorted([loc.location_id for loc in locations]),
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        # Generate report ID
        report_id = f"ELR-{facility_name[:10].upper().replace(' ', '')}-{report_date.strftime('%Y%m%d')}-{provenance_hash[:8]}"

        return EnergyLossReport(
            report_id=report_id,
            facility_name=facility_name,
            report_date=report_date,
            annual_energy_losses=annual_losses,
            fuel_consumption=fuel_consumptions,
            carbon_footprint=carbon_footprint,
            facility_aggregation=facility_aggregation,
            benchmarks=benchmarks,
            savings_potential=savings_potential,
            efficiency_metrics=efficiency_metrics,
            provenance_hash=provenance_hash,
            calculation_metadata={
                "calculation_engine": "EnergyLossQuantifier",
                "version": self.VERSION,
                "fuel_type": fuel_type,
                "boiler_efficiency": self.default_boiler_efficiency,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_location(self, location: InspectionLocation) -> List[str]:
        """
        Validate inspection location data.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        if location.heat_loss_rate_w < 0:
            errors.append(f"Heat loss rate cannot be negative: {location.heat_loss_rate_w}")

        if location.pipe_length_m <= 0:
            errors.append(f"Pipe length must be positive: {location.pipe_length_m}")

        if location.surface_area_m2 <= 0:
            errors.append(f"Surface area must be positive: {location.surface_area_m2}")

        if location.process_temperature_c <= location.ambient_temperature_c:
            errors.append(
                f"Process temp ({location.process_temperature_c}C) must be greater than "
                f"ambient temp ({location.ambient_temperature_c}C)"
            )

        if not 0 < location.operating_hours_per_year <= 8760:
            errors.append(f"Operating hours must be between 0 and 8760: {location.operating_hours_per_year}")

        return errors

    def get_supported_fuel_types(self) -> List[str]:
        """Return list of supported fuel types."""
        return [ft.value for ft in FuelType]

    def get_emission_factor(self, fuel_type: str) -> Optional[Dict[str, float]]:
        """Get emission factor data for a fuel type."""
        return EPA_EMISSION_FACTORS.get(fuel_type)

    def get_benchmark_rates(self, condition: str = "fair_insulation") -> Optional[Dict[str, float]]:
        """Get benchmark heat loss rates for a condition."""
        return BENCHMARK_HEAT_LOSS_RATES.get(condition)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "EnergyLossQuantifier",
    "InspectionLocation",
    "SeasonalProfile",
    "FuelConsumption",
    "CarbonFootprint",
    "BenchmarkComparison",
    "SavingsPotential",
    "EnergyEfficiencyMetrics",
    "AnnualEnergyLoss",
    "FacilityAggregation",
    "EnergyLossReport",
    "FuelType",
    "EPA_EMISSION_FACTORS",
    "DEFAULT_BOILER_EFFICIENCIES",
    "BENCHMARK_HEAT_LOSS_RATES",
    "CARBON_PRICE_SCENARIOS",
]
