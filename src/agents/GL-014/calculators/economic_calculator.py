# -*- coding: utf-8 -*-
"""
Economic Impact Calculator for GL-014 EXCHANGER-PRO.

Comprehensive economic analysis for heat exchanger fouling including:
- Energy loss calculations
- Production impact analysis
- Maintenance cost modeling
- Total Cost of Ownership (TCO)
- ROI and payback analysis
- Sensitivity analysis with Monte Carlo support
- Carbon cost integration

Zero-hallucination design: All calculations are deterministic using
standard financial formulas and engineering economics principles.

References:
- Engineering Economics and Financial Accounting, Park & Sharp-Bette
- ASME PTC 12.5 Heat Exchanger Performance Test Codes
- ASHRAE Energy Cost Savings Guide
- IRS MACRS Depreciation Tables
- IPCC AR6 Global Warming Potentials
- Carbon Border Adjustment Mechanism (CBAM) Regulations

Author: GreenLang AI Agent Factory - GL-CalculatorEngineer
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, FrozenSet
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Emission Factors
# =============================================================================

# IPCC AR6 Global Warming Potentials (100-year horizon)
GWP_AR6: Dict[str, float] = {
    'CO2': 1.0,
    'CH4': 29.8,
    'N2O': 273.0,
}

# CO2 emission factors by fuel type (kg CO2 per kWh thermal)
CO2_EMISSION_FACTORS: Dict[str, float] = {
    'natural_gas': 0.185,
    'fuel_oil_light': 0.265,
    'fuel_oil_heavy': 0.280,
    'coal_bituminous': 0.340,
    'coal_anthracite': 0.355,
    'lpg': 0.215,
    'biomass': 0.015,  # Net biogenic (near-zero fossil)
    'electricity_grid_us': 0.420,
    'electricity_grid_eu': 0.295,
    'electricity_grid_china': 0.555,
    'steam_from_boiler': 0.195,
}

# MACRS depreciation schedules (IRS Publication 946)
MACRS_5_YEAR: Tuple[float, ...] = (0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576)
MACRS_7_YEAR: Tuple[float, ...] = (0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446)
MACRS_10_YEAR: Tuple[float, ...] = (0.1000, 0.1800, 0.1440, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328)
MACRS_15_YEAR: Tuple[float, ...] = (0.0500, 0.0950, 0.0855, 0.0770, 0.0693, 0.0623, 0.0590, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0295)


# =============================================================================
# Enumerations
# =============================================================================

class FuelType(str, Enum):
    """Fuel types for energy calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_LIGHT = "fuel_oil_light"
    FUEL_OIL_HEAVY = "fuel_oil_heavy"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    LPG = "lpg"
    BIOMASS = "biomass"
    ELECTRICITY_GRID_US = "electricity_grid_us"
    ELECTRICITY_GRID_EU = "electricity_grid_eu"
    ELECTRICITY_GRID_CHINA = "electricity_grid_china"
    STEAM_FROM_BOILER = "steam_from_boiler"


class CleaningMethod(str, Enum):
    """Heat exchanger cleaning methods."""
    CHEMICAL_CLEANING = "chemical_cleaning"
    MECHANICAL_CLEANING = "mechanical_cleaning"
    HYDRO_BLASTING = "hydro_blasting"
    THERMAL_SHOCKING = "thermal_shocking"
    ULTRASONIC_CLEANING = "ultrasonic_cleaning"
    ONLINE_CLEANING = "online_cleaning"


class MaintenanceType(str, Enum):
    """Maintenance categories."""
    PLANNED = "planned"
    UNPLANNED = "unplanned"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"


class DepreciationMethod(str, Enum):
    """Depreciation methods for capital assets."""
    STRAIGHT_LINE = "straight_line"
    MACRS_5_YEAR = "macrs_5"
    MACRS_7_YEAR = "macrs_7"
    MACRS_10_YEAR = "macrs_10"
    MACRS_15_YEAR = "macrs_15"
    DOUBLE_DECLINING = "double_declining"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Value chain emissions


# =============================================================================
# Input Data Classes (Immutable)
# =============================================================================

@dataclass(frozen=True)
class EnergyLossInput:
    """Input parameters for energy loss calculation."""
    design_duty_kw: Decimal
    actual_duty_kw: Decimal
    fuel_type: FuelType
    fuel_cost_per_kwh: Decimal
    operating_hours_per_year: Decimal
    system_efficiency: Decimal = Decimal("0.85")
    include_carbon_cost: bool = True
    carbon_price_per_tonne: Decimal = Decimal("50.00")


@dataclass(frozen=True)
class ProductionImpactInput:
    """Input parameters for production impact calculation."""
    design_capacity_units_per_hour: Decimal
    actual_capacity_units_per_hour: Decimal
    product_value_per_unit: Decimal
    operating_hours_per_year: Decimal
    quality_rejection_rate_percent: Decimal = Decimal("0.0")
    quality_penalty_per_rejected_unit: Decimal = Decimal("0.0")
    downtime_hours_per_incident: Decimal = Decimal("8.0")
    incidents_per_year: int = 0
    labor_cost_per_hour: Decimal = Decimal("75.00")
    workers_per_incident: int = 4


@dataclass(frozen=True)
class MaintenanceCostInput:
    """Input parameters for maintenance cost calculation."""
    cleaning_method: CleaningMethod
    cleanings_per_year: int
    chemical_cost_per_cleaning: Decimal = Decimal("5000.00")
    labor_hours_per_cleaning: Decimal = Decimal("24.0")
    labor_rate_per_hour: Decimal = Decimal("85.00")
    equipment_rental_per_cleaning: Decimal = Decimal("2000.00")
    inspection_cost_per_year: Decimal = Decimal("3000.00")
    spare_parts_cost_per_year: Decimal = Decimal("5000.00")
    unplanned_maintenance_multiplier: Decimal = Decimal("2.5")
    unplanned_incidents_per_year: int = 0


@dataclass(frozen=True)
class TCOInput:
    """Input parameters for Total Cost of Ownership calculation."""
    equipment_purchase_cost: Decimal
    installation_cost: Decimal
    commissioning_cost: Decimal
    useful_life_years: int
    residual_value_percent: Decimal = Decimal("10.0")
    depreciation_method: DepreciationMethod = DepreciationMethod.MACRS_7_YEAR
    annual_insurance_rate_percent: Decimal = Decimal("0.5")
    annual_property_tax_rate_percent: Decimal = Decimal("1.0")
    decommissioning_cost: Decimal = Decimal("0.0")
    environmental_compliance_cost_per_year: Decimal = Decimal("0.0")


@dataclass(frozen=True)
class ROIInput:
    """Input parameters for ROI analysis."""
    investment_cost: Decimal
    annual_savings: Decimal
    discount_rate_percent: Decimal = Decimal("10.0")
    analysis_period_years: int = 15
    inflation_rate_percent: Decimal = Decimal("2.5")
    tax_rate_percent: Decimal = Decimal("25.0")
    energy_cost_escalation_percent: Decimal = Decimal("3.0")


@dataclass(frozen=True)
class SensitivityInput:
    """Input parameters for sensitivity analysis."""
    base_npv: Decimal
    base_roi: Decimal
    base_payback: Decimal
    parameter_variations: Dict[str, Tuple[Decimal, Decimal, Decimal]]  # name: (min, base, max)
    monte_carlo_iterations: int = 1000
    confidence_level_percent: Decimal = Decimal("95.0")


@dataclass(frozen=True)
class CarbonImpactInput:
    """Input parameters for carbon cost calculation."""
    energy_loss_kwh_per_year: Decimal
    fuel_type: FuelType
    carbon_price_per_tonne: Decimal = Decimal("50.00")
    include_upstream_emissions: bool = False
    upstream_emission_factor: Decimal = Decimal("0.015")  # kg CO2e/kWh
    emission_scope: EmissionScope = EmissionScope.SCOPE_1


# =============================================================================
# Output Data Classes (Immutable)
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Represents a single step in a calculation chain."""
    step_number: int
    operation: str
    description: str
    inputs: Tuple[Tuple[str, str], ...]  # Immutable pairs
    output_value: str
    output_name: str
    formula: Optional[str] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class EnergyLossResult:
    """Result of energy loss calculation."""
    heat_transfer_loss_kw: Decimal
    heat_transfer_loss_percent: Decimal
    additional_fuel_kwh_per_year: Decimal
    energy_cost_per_year_usd: Decimal
    carbon_emissions_kg_per_year: Decimal
    carbon_cost_per_year_usd: Decimal
    total_energy_penalty_per_year_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class ProductionImpactResult:
    """Result of production impact calculation."""
    throughput_loss_units_per_year: Decimal
    throughput_loss_percent: Decimal
    throughput_loss_value_usd: Decimal
    quality_rejection_cost_usd: Decimal
    downtime_cost_usd: Decimal
    total_production_impact_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class MaintenanceCostResult:
    """Result of maintenance cost calculation."""
    planned_cleaning_cost_usd: Decimal
    chemical_cost_usd: Decimal
    labor_cost_usd: Decimal
    equipment_cost_usd: Decimal
    inspection_cost_usd: Decimal
    spare_parts_cost_usd: Decimal
    unplanned_maintenance_cost_usd: Decimal
    total_maintenance_cost_usd: Decimal
    cost_per_cleaning_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class TCOResult:
    """Result of Total Cost of Ownership calculation."""
    total_capital_cost_usd: Decimal
    total_operating_cost_usd: Decimal
    total_maintenance_cost_usd: Decimal
    end_of_life_cost_usd: Decimal
    total_cost_of_ownership_usd: Decimal
    annualized_cost_usd: Decimal
    npv_of_costs_usd: Decimal
    depreciation_schedule: Tuple[Tuple[int, Decimal], ...]
    tax_benefit_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class ROIResult:
    """Result of ROI analysis."""
    net_present_value_usd: Decimal
    internal_rate_of_return_percent: Decimal
    simple_payback_years: Decimal
    discounted_payback_years: Decimal
    profitability_index: Decimal
    annual_cash_flows: Tuple[Tuple[int, Decimal], ...]
    cumulative_npv_by_year: Tuple[Tuple[int, Decimal], ...]
    break_even_utilization_percent: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class SensitivityResult:
    """Result of sensitivity analysis."""
    parameter_name: str
    base_value: Decimal
    min_value: Decimal
    max_value: Decimal
    npv_at_min: Decimal
    npv_at_base: Decimal
    npv_at_max: Decimal
    npv_sensitivity_percent: Decimal
    tornado_rank: int
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    mean_npv: Decimal
    std_dev_npv: Decimal
    percentile_5_npv: Decimal
    percentile_50_npv: Decimal
    percentile_95_npv: Decimal
    probability_positive_npv_percent: Decimal
    value_at_risk_5_percent: Decimal
    iterations: int
    provenance_hash: str


@dataclass(frozen=True)
class CarbonImpactResult:
    """Result of carbon impact calculation."""
    direct_emissions_kg_co2e: Decimal
    upstream_emissions_kg_co2e: Decimal
    total_emissions_kg_co2e: Decimal
    total_emissions_tonnes_co2e: Decimal
    carbon_cost_usd: Decimal
    carbon_intensity_kg_per_kwh: Decimal
    scope_1_emissions_kg: Decimal
    scope_2_emissions_kg: Decimal
    emission_breakdown: Dict[str, Decimal]
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str


@dataclass(frozen=True)
class EconomicReport:
    """Comprehensive economic analysis report."""
    report_id: str
    generated_at: str
    energy_loss: Optional[EnergyLossResult]
    production_impact: Optional[ProductionImpactResult]
    maintenance_costs: Optional[MaintenanceCostResult]
    total_cost_of_ownership: Optional[TCOResult]
    roi_analysis: Optional[ROIResult]
    carbon_impact: Optional[CarbonImpactResult]
    sensitivity_results: Tuple[SensitivityResult, ...]
    monte_carlo_result: Optional[MonteCarloResult]
    total_annual_cost_usd: Decimal
    total_annual_savings_potential_usd: Decimal
    executive_summary: Dict[str, Any]
    provenance_hash: str


# =============================================================================
# Calculator Implementation
# =============================================================================

class EconomicCalculator:
    """
    Comprehensive economic impact calculator for heat exchanger fouling.

    Zero-hallucination guarantee:
    - All calculations use deterministic formulas
    - NO LLM involvement in calculation path
    - Complete provenance tracking with SHA-256 hashes
    - Bit-perfect reproducibility (same input -> same output)
    - Financial calculations use Decimal for precision

    Performance target: <5ms per calculation

    Example:
        >>> calculator = EconomicCalculator()
        >>> energy_input = EnergyLossInput(
        ...     design_duty_kw=Decimal("1000"),
        ...     actual_duty_kw=Decimal("850"),
        ...     fuel_type=FuelType.NATURAL_GAS,
        ...     fuel_cost_per_kwh=Decimal("0.05"),
        ...     operating_hours_per_year=Decimal("8000")
        ... )
        >>> result = calculator.calculate_energy_loss_cost(energy_input)
        >>> print(f"Annual energy penalty: ${result.total_energy_penalty_per_year_usd}")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the economic calculator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.calculation_count = 0
        self._precision = 6  # Decimal places for financial calculations
        logger.info(f"EconomicCalculator initialized (version {self.VERSION})")

    # =========================================================================
    # Energy Loss Calculation
    # =========================================================================

    def calculate_energy_loss_cost(self, input_data: EnergyLossInput) -> EnergyLossResult:
        """
        Calculate cost of energy loss from heat exchanger fouling.

        Formula: C_energy = delta_Q * C_fuel * t_operation / eta_system

        Args:
            input_data: Energy loss input parameters

        Returns:
            EnergyLossResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        # Step 1: Calculate heat transfer loss
        heat_loss_kw = input_data.design_duty_kw - input_data.actual_duty_kw
        heat_loss_percent = (heat_loss_kw / input_data.design_duty_kw * Decimal("100")
                            if input_data.design_duty_kw > 0 else Decimal("0"))
        steps.append(CalculationStep(
            step_number=1,
            operation="subtract",
            description="Calculate heat transfer loss",
            inputs=(
                ("design_duty_kw", str(input_data.design_duty_kw)),
                ("actual_duty_kw", str(input_data.actual_duty_kw)),
            ),
            output_value=str(heat_loss_kw),
            output_name="heat_loss_kw",
            formula="heat_loss_kw = design_duty_kw - actual_duty_kw",
            units="kW"
        ))

        # Step 2: Calculate additional fuel consumption
        # Additional fuel = heat_loss * operating_hours / system_efficiency
        additional_fuel_kwh = (heat_loss_kw * input_data.operating_hours_per_year
                              / input_data.system_efficiency
                              if input_data.system_efficiency > 0 else Decimal("0"))
        steps.append(CalculationStep(
            step_number=2,
            operation="multiply_divide",
            description="Calculate additional fuel consumption",
            inputs=(
                ("heat_loss_kw", str(heat_loss_kw)),
                ("operating_hours_per_year", str(input_data.operating_hours_per_year)),
                ("system_efficiency", str(input_data.system_efficiency)),
            ),
            output_value=str(additional_fuel_kwh),
            output_name="additional_fuel_kwh",
            formula="additional_fuel_kwh = heat_loss_kw * operating_hours / system_efficiency",
            units="kWh/year"
        ))

        # Step 3: Calculate energy cost
        energy_cost = additional_fuel_kwh * input_data.fuel_cost_per_kwh
        steps.append(CalculationStep(
            step_number=3,
            operation="multiply",
            description="Calculate annual energy cost",
            inputs=(
                ("additional_fuel_kwh", str(additional_fuel_kwh)),
                ("fuel_cost_per_kwh", str(input_data.fuel_cost_per_kwh)),
            ),
            output_value=str(energy_cost),
            output_name="energy_cost_usd",
            formula="energy_cost_usd = additional_fuel_kwh * fuel_cost_per_kwh",
            units="USD/year"
        ))

        # Step 4: Calculate carbon emissions
        emission_factor = Decimal(str(CO2_EMISSION_FACTORS.get(
            input_data.fuel_type.value, 0.185)))
        carbon_emissions_kg = additional_fuel_kwh * emission_factor
        steps.append(CalculationStep(
            step_number=4,
            operation="multiply",
            description="Calculate carbon emissions",
            inputs=(
                ("additional_fuel_kwh", str(additional_fuel_kwh)),
                ("emission_factor_kg_per_kwh", str(emission_factor)),
            ),
            output_value=str(carbon_emissions_kg),
            output_name="carbon_emissions_kg",
            formula="carbon_emissions_kg = additional_fuel_kwh * emission_factor",
            units="kg CO2/year"
        ))

        # Step 5: Calculate carbon cost
        carbon_cost = Decimal("0")
        if input_data.include_carbon_cost:
            carbon_tonnes = carbon_emissions_kg / Decimal("1000")
            carbon_cost = carbon_tonnes * input_data.carbon_price_per_tonne
        steps.append(CalculationStep(
            step_number=5,
            operation="multiply",
            description="Calculate carbon cost",
            inputs=(
                ("carbon_emissions_tonnes", str(carbon_emissions_kg / Decimal("1000"))),
                ("carbon_price_per_tonne", str(input_data.carbon_price_per_tonne)),
            ),
            output_value=str(carbon_cost),
            output_name="carbon_cost_usd",
            formula="carbon_cost_usd = (carbon_emissions_kg / 1000) * carbon_price_per_tonne",
            units="USD/year"
        ))

        # Step 6: Calculate total energy penalty
        total_penalty = energy_cost + carbon_cost
        steps.append(CalculationStep(
            step_number=6,
            operation="add",
            description="Calculate total energy penalty",
            inputs=(
                ("energy_cost_usd", str(energy_cost)),
                ("carbon_cost_usd", str(carbon_cost)),
            ),
            output_value=str(total_penalty),
            output_name="total_energy_penalty_usd",
            formula="total_energy_penalty = energy_cost + carbon_cost",
            units="USD/year"
        ))

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            "energy_loss_calculation",
            input_data,
            total_penalty
        )

        return EnergyLossResult(
            heat_transfer_loss_kw=self._round_decimal(heat_loss_kw),
            heat_transfer_loss_percent=self._round_decimal(heat_loss_percent),
            additional_fuel_kwh_per_year=self._round_decimal(additional_fuel_kwh),
            energy_cost_per_year_usd=self._round_decimal(energy_cost),
            carbon_emissions_kg_per_year=self._round_decimal(carbon_emissions_kg),
            carbon_cost_per_year_usd=self._round_decimal(carbon_cost),
            total_energy_penalty_per_year_usd=self._round_decimal(total_penalty),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Production Impact Calculation
    # =========================================================================

    def calculate_production_impact(
        self, input_data: ProductionImpactInput
    ) -> ProductionImpactResult:
        """
        Calculate production impact from heat exchanger degradation.

        Includes:
        - Throughput reduction costs
        - Quality rejection costs
        - Downtime costs

        Args:
            input_data: Production impact input parameters

        Returns:
            ProductionImpactResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        # Step 1: Calculate throughput loss
        capacity_loss = (input_data.design_capacity_units_per_hour
                        - input_data.actual_capacity_units_per_hour)
        throughput_loss_units = capacity_loss * input_data.operating_hours_per_year
        throughput_loss_percent = (
            capacity_loss / input_data.design_capacity_units_per_hour * Decimal("100")
            if input_data.design_capacity_units_per_hour > 0 else Decimal("0")
        )
        steps.append(CalculationStep(
            step_number=1,
            operation="subtract_multiply",
            description="Calculate annual throughput loss",
            inputs=(
                ("design_capacity", str(input_data.design_capacity_units_per_hour)),
                ("actual_capacity", str(input_data.actual_capacity_units_per_hour)),
                ("operating_hours", str(input_data.operating_hours_per_year)),
            ),
            output_value=str(throughput_loss_units),
            output_name="throughput_loss_units",
            formula="throughput_loss = (design_capacity - actual_capacity) * operating_hours",
            units="units/year"
        ))

        # Step 2: Calculate throughput loss value
        throughput_loss_value = throughput_loss_units * input_data.product_value_per_unit
        steps.append(CalculationStep(
            step_number=2,
            operation="multiply",
            description="Calculate throughput loss value",
            inputs=(
                ("throughput_loss_units", str(throughput_loss_units)),
                ("product_value_per_unit", str(input_data.product_value_per_unit)),
            ),
            output_value=str(throughput_loss_value),
            output_name="throughput_loss_value_usd",
            formula="throughput_loss_value = throughput_loss_units * product_value_per_unit",
            units="USD/year"
        ))

        # Step 3: Calculate quality rejection cost
        actual_production = (input_data.actual_capacity_units_per_hour
                            * input_data.operating_hours_per_year)
        rejected_units = actual_production * input_data.quality_rejection_rate_percent / Decimal("100")
        quality_cost = rejected_units * input_data.quality_penalty_per_rejected_unit
        steps.append(CalculationStep(
            step_number=3,
            operation="multiply",
            description="Calculate quality rejection cost",
            inputs=(
                ("actual_production", str(actual_production)),
                ("rejection_rate_percent", str(input_data.quality_rejection_rate_percent)),
                ("penalty_per_unit", str(input_data.quality_penalty_per_rejected_unit)),
            ),
            output_value=str(quality_cost),
            output_name="quality_rejection_cost_usd",
            formula="quality_cost = actual_production * rejection_rate * penalty_per_unit",
            units="USD/year"
        ))

        # Step 4: Calculate downtime cost
        total_downtime_hours = (input_data.downtime_hours_per_incident
                               * input_data.incidents_per_year)
        lost_production_value = (total_downtime_hours
                                * input_data.actual_capacity_units_per_hour
                                * input_data.product_value_per_unit)
        labor_cost = (total_downtime_hours
                     * input_data.labor_cost_per_hour
                     * input_data.workers_per_incident)
        downtime_cost = lost_production_value + labor_cost
        steps.append(CalculationStep(
            step_number=4,
            operation="multiply_add",
            description="Calculate downtime cost",
            inputs=(
                ("downtime_hours_per_incident", str(input_data.downtime_hours_per_incident)),
                ("incidents_per_year", str(input_data.incidents_per_year)),
                ("actual_capacity", str(input_data.actual_capacity_units_per_hour)),
                ("product_value", str(input_data.product_value_per_unit)),
                ("labor_cost_per_hour", str(input_data.labor_cost_per_hour)),
                ("workers_per_incident", str(input_data.workers_per_incident)),
            ),
            output_value=str(downtime_cost),
            output_name="downtime_cost_usd",
            formula="downtime_cost = lost_production + labor_cost",
            units="USD/year"
        ))

        # Step 5: Calculate total production impact
        total_impact = throughput_loss_value + quality_cost + downtime_cost
        steps.append(CalculationStep(
            step_number=5,
            operation="add",
            description="Calculate total production impact",
            inputs=(
                ("throughput_loss_value", str(throughput_loss_value)),
                ("quality_cost", str(quality_cost)),
                ("downtime_cost", str(downtime_cost)),
            ),
            output_value=str(total_impact),
            output_name="total_production_impact_usd",
            formula="total_impact = throughput_loss + quality_cost + downtime_cost",
            units="USD/year"
        ))

        provenance_hash = self._calculate_provenance_hash(
            "production_impact_calculation",
            input_data,
            total_impact
        )

        return ProductionImpactResult(
            throughput_loss_units_per_year=self._round_decimal(throughput_loss_units),
            throughput_loss_percent=self._round_decimal(throughput_loss_percent),
            throughput_loss_value_usd=self._round_decimal(throughput_loss_value),
            quality_rejection_cost_usd=self._round_decimal(quality_cost),
            downtime_cost_usd=self._round_decimal(downtime_cost),
            total_production_impact_usd=self._round_decimal(total_impact),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Maintenance Cost Calculation
    # =========================================================================

    def calculate_maintenance_costs(
        self, input_data: MaintenanceCostInput
    ) -> MaintenanceCostResult:
        """
        Calculate maintenance costs for heat exchanger cleaning and upkeep.

        Includes:
        - Planned cleaning costs (chemical, labor, equipment)
        - Inspection costs
        - Spare parts costs
        - Unplanned maintenance premium

        Args:
            input_data: Maintenance cost input parameters

        Returns:
            MaintenanceCostResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        # Step 1: Calculate chemical cost per year
        annual_chemical_cost = (input_data.chemical_cost_per_cleaning
                               * input_data.cleanings_per_year)
        steps.append(CalculationStep(
            step_number=1,
            operation="multiply",
            description="Calculate annual chemical cost",
            inputs=(
                ("chemical_cost_per_cleaning", str(input_data.chemical_cost_per_cleaning)),
                ("cleanings_per_year", str(input_data.cleanings_per_year)),
            ),
            output_value=str(annual_chemical_cost),
            output_name="annual_chemical_cost_usd",
            formula="annual_chemical_cost = chemical_cost_per_cleaning * cleanings_per_year",
            units="USD/year"
        ))

        # Step 2: Calculate labor cost per year
        hours_per_year = input_data.labor_hours_per_cleaning * input_data.cleanings_per_year
        annual_labor_cost = hours_per_year * input_data.labor_rate_per_hour
        steps.append(CalculationStep(
            step_number=2,
            operation="multiply",
            description="Calculate annual labor cost",
            inputs=(
                ("labor_hours_per_cleaning", str(input_data.labor_hours_per_cleaning)),
                ("cleanings_per_year", str(input_data.cleanings_per_year)),
                ("labor_rate_per_hour", str(input_data.labor_rate_per_hour)),
            ),
            output_value=str(annual_labor_cost),
            output_name="annual_labor_cost_usd",
            formula="annual_labor_cost = labor_hours * cleanings * labor_rate",
            units="USD/year"
        ))

        # Step 3: Calculate equipment rental cost per year
        annual_equipment_cost = (input_data.equipment_rental_per_cleaning
                                * input_data.cleanings_per_year)
        steps.append(CalculationStep(
            step_number=3,
            operation="multiply",
            description="Calculate annual equipment rental cost",
            inputs=(
                ("equipment_rental_per_cleaning", str(input_data.equipment_rental_per_cleaning)),
                ("cleanings_per_year", str(input_data.cleanings_per_year)),
            ),
            output_value=str(annual_equipment_cost),
            output_name="annual_equipment_cost_usd",
            formula="annual_equipment_cost = equipment_rental * cleanings_per_year",
            units="USD/year"
        ))

        # Step 4: Calculate planned cleaning total
        planned_cleaning_cost = annual_chemical_cost + annual_labor_cost + annual_equipment_cost
        steps.append(CalculationStep(
            step_number=4,
            operation="add",
            description="Calculate total planned cleaning cost",
            inputs=(
                ("annual_chemical_cost", str(annual_chemical_cost)),
                ("annual_labor_cost", str(annual_labor_cost)),
                ("annual_equipment_cost", str(annual_equipment_cost)),
            ),
            output_value=str(planned_cleaning_cost),
            output_name="planned_cleaning_cost_usd",
            formula="planned_cost = chemical + labor + equipment",
            units="USD/year"
        ))

        # Step 5: Calculate unplanned maintenance cost
        cost_per_cleaning = (planned_cleaning_cost / input_data.cleanings_per_year
                            if input_data.cleanings_per_year > 0 else Decimal("0"))
        unplanned_cost = (cost_per_cleaning
                         * input_data.unplanned_maintenance_multiplier
                         * input_data.unplanned_incidents_per_year)
        steps.append(CalculationStep(
            step_number=5,
            operation="multiply",
            description="Calculate unplanned maintenance cost",
            inputs=(
                ("cost_per_cleaning", str(cost_per_cleaning)),
                ("unplanned_multiplier", str(input_data.unplanned_maintenance_multiplier)),
                ("unplanned_incidents", str(input_data.unplanned_incidents_per_year)),
            ),
            output_value=str(unplanned_cost),
            output_name="unplanned_maintenance_cost_usd",
            formula="unplanned_cost = cost_per_cleaning * multiplier * incidents",
            units="USD/year"
        ))

        # Step 6: Calculate total maintenance cost
        total_maintenance = (planned_cleaning_cost
                            + input_data.inspection_cost_per_year
                            + input_data.spare_parts_cost_per_year
                            + unplanned_cost)
        steps.append(CalculationStep(
            step_number=6,
            operation="add",
            description="Calculate total maintenance cost",
            inputs=(
                ("planned_cleaning_cost", str(planned_cleaning_cost)),
                ("inspection_cost", str(input_data.inspection_cost_per_year)),
                ("spare_parts_cost", str(input_data.spare_parts_cost_per_year)),
                ("unplanned_cost", str(unplanned_cost)),
            ),
            output_value=str(total_maintenance),
            output_name="total_maintenance_cost_usd",
            formula="total = planned + inspection + spare_parts + unplanned",
            units="USD/year"
        ))

        provenance_hash = self._calculate_provenance_hash(
            "maintenance_cost_calculation",
            input_data,
            total_maintenance
        )

        return MaintenanceCostResult(
            planned_cleaning_cost_usd=self._round_decimal(planned_cleaning_cost),
            chemical_cost_usd=self._round_decimal(annual_chemical_cost),
            labor_cost_usd=self._round_decimal(annual_labor_cost),
            equipment_cost_usd=self._round_decimal(annual_equipment_cost),
            inspection_cost_usd=self._round_decimal(input_data.inspection_cost_per_year),
            spare_parts_cost_usd=self._round_decimal(input_data.spare_parts_cost_per_year),
            unplanned_maintenance_cost_usd=self._round_decimal(unplanned_cost),
            total_maintenance_cost_usd=self._round_decimal(total_maintenance),
            cost_per_cleaning_usd=self._round_decimal(cost_per_cleaning),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Total Cost of Ownership Calculation
    # =========================================================================

    def calculate_total_cost_of_ownership(
        self,
        tco_input: TCOInput,
        annual_operating_cost: Decimal,
        annual_maintenance_cost: Decimal,
        discount_rate_percent: Decimal = Decimal("10.0"),
        tax_rate_percent: Decimal = Decimal("25.0")
    ) -> TCOResult:
        """
        Calculate Total Cost of Ownership over equipment lifetime.

        Includes:
        - Capital costs (CAPEX)
        - Operating costs (OPEX)
        - Maintenance costs
        - End-of-life costs
        - NPV analysis
        - Tax benefits from depreciation

        Args:
            tco_input: TCO input parameters
            annual_operating_cost: Annual operating cost
            annual_maintenance_cost: Annual maintenance cost
            discount_rate_percent: Discount rate for NPV calculation
            tax_rate_percent: Corporate tax rate

        Returns:
            TCOResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        # Step 1: Calculate total capital cost
        total_capex = (tco_input.equipment_purchase_cost
                      + tco_input.installation_cost
                      + tco_input.commissioning_cost)
        steps.append(CalculationStep(
            step_number=1,
            operation="add",
            description="Calculate total capital cost (CAPEX)",
            inputs=(
                ("equipment_purchase_cost", str(tco_input.equipment_purchase_cost)),
                ("installation_cost", str(tco_input.installation_cost)),
                ("commissioning_cost", str(tco_input.commissioning_cost)),
            ),
            output_value=str(total_capex),
            output_name="total_capex_usd",
            formula="total_capex = equipment + installation + commissioning",
            units="USD"
        ))

        # Step 2: Calculate depreciation schedule
        depreciation_schedule = self._calculate_depreciation(
            total_capex,
            tco_input.useful_life_years,
            tco_input.depreciation_method,
            tco_input.residual_value_percent
        )
        total_depreciation = sum(d[1] for d in depreciation_schedule)
        steps.append(CalculationStep(
            step_number=2,
            operation="depreciation",
            description="Calculate depreciation schedule",
            inputs=(
                ("total_capex", str(total_capex)),
                ("useful_life_years", str(tco_input.useful_life_years)),
                ("depreciation_method", tco_input.depreciation_method.value),
                ("residual_value_percent", str(tco_input.residual_value_percent)),
            ),
            output_value=str(total_depreciation),
            output_name="total_depreciation_usd",
            formula=f"depreciation using {tco_input.depreciation_method.value}",
            units="USD"
        ))

        # Step 3: Calculate tax benefit from depreciation
        tax_rate = tax_rate_percent / Decimal("100")
        tax_benefit = total_depreciation * tax_rate
        steps.append(CalculationStep(
            step_number=3,
            operation="multiply",
            description="Calculate tax benefit from depreciation",
            inputs=(
                ("total_depreciation", str(total_depreciation)),
                ("tax_rate", str(tax_rate)),
            ),
            output_value=str(tax_benefit),
            output_name="tax_benefit_usd",
            formula="tax_benefit = total_depreciation * tax_rate",
            units="USD"
        ))

        # Step 4: Calculate total operating costs over lifetime
        # Include insurance and property tax
        annual_insurance = total_capex * tco_input.annual_insurance_rate_percent / Decimal("100")
        annual_property_tax = total_capex * tco_input.annual_property_tax_rate_percent / Decimal("100")
        total_annual_opex = (annual_operating_cost + annual_insurance
                           + annual_property_tax + tco_input.environmental_compliance_cost_per_year)
        total_opex = total_annual_opex * tco_input.useful_life_years
        steps.append(CalculationStep(
            step_number=4,
            operation="multiply",
            description="Calculate total operating costs (OPEX) over lifetime",
            inputs=(
                ("annual_operating_cost", str(annual_operating_cost)),
                ("annual_insurance", str(annual_insurance)),
                ("annual_property_tax", str(annual_property_tax)),
                ("environmental_compliance", str(tco_input.environmental_compliance_cost_per_year)),
                ("useful_life_years", str(tco_input.useful_life_years)),
            ),
            output_value=str(total_opex),
            output_name="total_opex_usd",
            formula="total_opex = annual_opex * useful_life_years",
            units="USD"
        ))

        # Step 5: Calculate total maintenance costs over lifetime
        total_maintenance = annual_maintenance_cost * tco_input.useful_life_years
        steps.append(CalculationStep(
            step_number=5,
            operation="multiply",
            description="Calculate total maintenance costs over lifetime",
            inputs=(
                ("annual_maintenance_cost", str(annual_maintenance_cost)),
                ("useful_life_years", str(tco_input.useful_life_years)),
            ),
            output_value=str(total_maintenance),
            output_name="total_maintenance_usd",
            formula="total_maintenance = annual_maintenance * useful_life_years",
            units="USD"
        ))

        # Step 6: Calculate end-of-life costs
        residual_value = total_capex * tco_input.residual_value_percent / Decimal("100")
        end_of_life_cost = tco_input.decommissioning_cost - residual_value
        steps.append(CalculationStep(
            step_number=6,
            operation="subtract",
            description="Calculate end-of-life net cost",
            inputs=(
                ("decommissioning_cost", str(tco_input.decommissioning_cost)),
                ("residual_value", str(residual_value)),
            ),
            output_value=str(end_of_life_cost),
            output_name="end_of_life_cost_usd",
            formula="end_of_life = decommissioning - residual_value",
            units="USD"
        ))

        # Step 7: Calculate total cost of ownership (nominal)
        total_tco = total_capex + total_opex + total_maintenance + end_of_life_cost
        steps.append(CalculationStep(
            step_number=7,
            operation="add",
            description="Calculate total cost of ownership (nominal)",
            inputs=(
                ("total_capex", str(total_capex)),
                ("total_opex", str(total_opex)),
                ("total_maintenance", str(total_maintenance)),
                ("end_of_life_cost", str(end_of_life_cost)),
            ),
            output_value=str(total_tco),
            output_name="total_tco_usd",
            formula="total_tco = capex + opex + maintenance + end_of_life",
            units="USD"
        ))

        # Step 8: Calculate NPV of all costs
        discount_rate = discount_rate_percent / Decimal("100")
        npv_costs = self._calculate_npv_of_costs(
            total_capex,
            total_annual_opex + annual_maintenance_cost,
            end_of_life_cost,
            tco_input.useful_life_years,
            discount_rate
        )
        steps.append(CalculationStep(
            step_number=8,
            operation="npv",
            description="Calculate NPV of all costs",
            inputs=(
                ("discount_rate", str(discount_rate)),
                ("useful_life_years", str(tco_input.useful_life_years)),
            ),
            output_value=str(npv_costs),
            output_name="npv_of_costs_usd",
            formula="NPV = sum(cash_flows / (1 + r)^t)",
            units="USD"
        ))

        # Calculate annualized cost
        annualized_cost = total_tco / tco_input.useful_life_years

        provenance_hash = self._calculate_provenance_hash(
            "tco_calculation",
            {
                "tco_input": str(tco_input),
                "annual_operating_cost": str(annual_operating_cost),
                "annual_maintenance_cost": str(annual_maintenance_cost),
            },
            total_tco
        )

        return TCOResult(
            total_capital_cost_usd=self._round_decimal(total_capex),
            total_operating_cost_usd=self._round_decimal(total_opex),
            total_maintenance_cost_usd=self._round_decimal(total_maintenance),
            end_of_life_cost_usd=self._round_decimal(end_of_life_cost),
            total_cost_of_ownership_usd=self._round_decimal(total_tco),
            annualized_cost_usd=self._round_decimal(annualized_cost),
            npv_of_costs_usd=self._round_decimal(npv_costs),
            depreciation_schedule=tuple(depreciation_schedule),
            tax_benefit_usd=self._round_decimal(tax_benefit),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # ROI Analysis
    # =========================================================================

    def perform_roi_analysis(self, input_data: ROIInput) -> ROIResult:
        """
        Perform comprehensive ROI analysis for investment.

        Calculates:
        - Net Present Value (NPV)
        - Internal Rate of Return (IRR)
        - Simple Payback Period
        - Discounted Payback Period
        - Profitability Index

        Args:
            input_data: ROI analysis input parameters

        Returns:
            ROIResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        discount_rate = input_data.discount_rate_percent / Decimal("100")
        escalation_rate = input_data.energy_cost_escalation_percent / Decimal("100")
        tax_rate = input_data.tax_rate_percent / Decimal("100")

        # Step 1: Calculate NPV
        cash_flows: List[Tuple[int, Decimal]] = []
        npv = -input_data.investment_cost
        cumulative_npv: List[Tuple[int, Decimal]] = [(0, npv)]

        for year in range(1, input_data.analysis_period_years + 1):
            # Escalate savings
            escalated_savings = input_data.annual_savings * (
                (Decimal("1") + escalation_rate) ** (year - 1)
            )
            # After-tax cash flow
            after_tax_savings = escalated_savings * (Decimal("1") - tax_rate)
            cash_flows.append((year, after_tax_savings))

            # Discount to present value
            pv = after_tax_savings / ((Decimal("1") + discount_rate) ** year)
            npv += pv
            cumulative_npv.append((year, npv))

        steps.append(CalculationStep(
            step_number=1,
            operation="npv",
            description="Calculate Net Present Value",
            inputs=(
                ("investment_cost", str(input_data.investment_cost)),
                ("annual_savings", str(input_data.annual_savings)),
                ("discount_rate", str(discount_rate)),
                ("analysis_period_years", str(input_data.analysis_period_years)),
            ),
            output_value=str(npv),
            output_name="npv_usd",
            formula="NPV = -Investment + sum(CF_t / (1 + r)^t)",
            units="USD"
        ))

        # Step 2: Calculate IRR using Newton-Raphson method
        irr = self._calculate_irr(
            input_data.investment_cost,
            input_data.annual_savings,
            input_data.analysis_period_years,
            escalation_rate,
            tax_rate
        )
        steps.append(CalculationStep(
            step_number=2,
            operation="irr",
            description="Calculate Internal Rate of Return",
            inputs=(
                ("investment_cost", str(input_data.investment_cost)),
                ("annual_savings", str(input_data.annual_savings)),
            ),
            output_value=str(irr),
            output_name="irr_percent",
            formula="IRR is the rate where NPV = 0",
            units="%"
        ))

        # Step 3: Calculate Simple Payback
        simple_payback = (input_data.investment_cost / input_data.annual_savings
                         if input_data.annual_savings > 0 else Decimal("999"))
        steps.append(CalculationStep(
            step_number=3,
            operation="divide",
            description="Calculate Simple Payback Period",
            inputs=(
                ("investment_cost", str(input_data.investment_cost)),
                ("annual_savings", str(input_data.annual_savings)),
            ),
            output_value=str(simple_payback),
            output_name="simple_payback_years",
            formula="simple_payback = investment / annual_savings",
            units="years"
        ))

        # Step 4: Calculate Discounted Payback
        discounted_payback = self._calculate_discounted_payback(
            input_data.investment_cost,
            input_data.annual_savings,
            discount_rate,
            escalation_rate,
            tax_rate
        )
        steps.append(CalculationStep(
            step_number=4,
            operation="discounted_payback",
            description="Calculate Discounted Payback Period",
            inputs=(
                ("investment_cost", str(input_data.investment_cost)),
                ("discount_rate", str(discount_rate)),
            ),
            output_value=str(discounted_payback),
            output_name="discounted_payback_years",
            formula="Time until cumulative discounted cash flow >= investment",
            units="years"
        ))

        # Step 5: Calculate Profitability Index
        pv_of_cash_flows = npv + input_data.investment_cost
        profitability_index = (pv_of_cash_flows / input_data.investment_cost
                              if input_data.investment_cost > 0 else Decimal("0"))
        steps.append(CalculationStep(
            step_number=5,
            operation="divide",
            description="Calculate Profitability Index",
            inputs=(
                ("pv_of_cash_flows", str(pv_of_cash_flows)),
                ("investment_cost", str(input_data.investment_cost)),
            ),
            output_value=str(profitability_index),
            output_name="profitability_index",
            formula="PI = PV(cash flows) / Investment",
            units="ratio"
        ))

        # Step 6: Calculate Break-even Utilization
        # At what utilization % does NPV = 0?
        break_even_utilization = self._calculate_break_even_utilization(
            input_data.investment_cost,
            input_data.annual_savings,
            discount_rate,
            input_data.analysis_period_years
        )
        steps.append(CalculationStep(
            step_number=6,
            operation="break_even",
            description="Calculate Break-even Utilization",
            inputs=(
                ("investment_cost", str(input_data.investment_cost)),
                ("annual_savings", str(input_data.annual_savings)),
            ),
            output_value=str(break_even_utilization),
            output_name="break_even_utilization_percent",
            formula="Utilization % where NPV = 0",
            units="%"
        ))

        provenance_hash = self._calculate_provenance_hash(
            "roi_analysis",
            input_data,
            npv
        )

        return ROIResult(
            net_present_value_usd=self._round_decimal(npv),
            internal_rate_of_return_percent=self._round_decimal(irr * Decimal("100")),
            simple_payback_years=self._round_decimal(simple_payback),
            discounted_payback_years=self._round_decimal(discounted_payback),
            profitability_index=self._round_decimal(profitability_index, 3),
            annual_cash_flows=tuple(cash_flows),
            cumulative_npv_by_year=tuple(cumulative_npv),
            break_even_utilization_percent=self._round_decimal(break_even_utilization),
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Sensitivity Analysis
    # =========================================================================

    def perform_sensitivity_analysis(
        self,
        base_roi_input: ROIInput,
        parameter_variations: Dict[str, Tuple[Decimal, Decimal, Decimal]]
    ) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis on key parameters.

        Calculates how NPV changes with parameter variations.

        Args:
            base_roi_input: Base case ROI input
            parameter_variations: Dict of parameter name -> (min, base, max) values

        Returns:
            List of SensitivityResult sorted by impact (tornado chart order)
        """
        self.calculation_count += 1
        results: List[SensitivityResult] = []

        # Calculate base case NPV
        base_result = self.perform_roi_analysis(base_roi_input)
        base_npv = base_result.net_present_value_usd

        for param_name, (min_val, base_val, max_val) in parameter_variations.items():
            steps: List[CalculationStep] = []

            # Calculate NPV at minimum value
            modified_input_min = self._modify_roi_input(base_roi_input, param_name, min_val)
            result_min = self.perform_roi_analysis(modified_input_min)
            npv_at_min = result_min.net_present_value_usd

            # Calculate NPV at maximum value
            modified_input_max = self._modify_roi_input(base_roi_input, param_name, max_val)
            result_max = self.perform_roi_analysis(modified_input_max)
            npv_at_max = result_max.net_present_value_usd

            # Calculate sensitivity (% change in NPV per % change in parameter)
            npv_range = abs(npv_at_max - npv_at_min)
            sensitivity = (npv_range / abs(base_npv) * Decimal("100")
                          if base_npv != 0 else Decimal("0"))

            steps.append(CalculationStep(
                step_number=1,
                operation="sensitivity",
                description=f"Calculate NPV sensitivity to {param_name}",
                inputs=(
                    ("min_value", str(min_val)),
                    ("base_value", str(base_val)),
                    ("max_value", str(max_val)),
                    ("npv_at_min", str(npv_at_min)),
                    ("npv_at_max", str(npv_at_max)),
                ),
                output_value=str(sensitivity),
                output_name="npv_sensitivity_percent",
                formula="sensitivity = |NPV_max - NPV_min| / |NPV_base| * 100",
                units="%"
            ))

            provenance_hash = self._calculate_provenance_hash(
                f"sensitivity_{param_name}",
                {"min": str(min_val), "base": str(base_val), "max": str(max_val)},
                sensitivity
            )

            results.append(SensitivityResult(
                parameter_name=param_name,
                base_value=base_val,
                min_value=min_val,
                max_value=max_val,
                npv_at_min=self._round_decimal(npv_at_min),
                npv_at_base=self._round_decimal(base_npv),
                npv_at_max=self._round_decimal(npv_at_max),
                npv_sensitivity_percent=self._round_decimal(sensitivity),
                tornado_rank=0,  # Will be set after sorting
                calculation_steps=tuple(steps),
                provenance_hash=provenance_hash
            ))

        # Sort by sensitivity (descending) for tornado chart
        results.sort(key=lambda x: x.npv_sensitivity_percent, reverse=True)

        # Assign tornado ranks
        ranked_results = []
        for rank, result in enumerate(results, 1):
            # Create new result with rank (since frozen)
            ranked_results.append(SensitivityResult(
                parameter_name=result.parameter_name,
                base_value=result.base_value,
                min_value=result.min_value,
                max_value=result.max_value,
                npv_at_min=result.npv_at_min,
                npv_at_base=result.npv_at_base,
                npv_at_max=result.npv_at_max,
                npv_sensitivity_percent=result.npv_sensitivity_percent,
                tornado_rank=rank,
                calculation_steps=result.calculation_steps,
                provenance_hash=result.provenance_hash
            ))

        return ranked_results

    def perform_monte_carlo_simulation(
        self,
        base_roi_input: ROIInput,
        parameter_distributions: Dict[str, Tuple[Decimal, Decimal]],
        iterations: int = 1000,
        seed: int = 42
    ) -> MonteCarloResult:
        """
        Perform Monte Carlo simulation for probabilistic NPV analysis.

        Uses deterministic pseudo-random number generation for reproducibility.

        Args:
            base_roi_input: Base case ROI input
            parameter_distributions: Dict of parameter name -> (mean, std_dev)
            iterations: Number of simulation iterations
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult with distribution statistics
        """
        self.calculation_count += 1

        # Use deterministic PRNG for reproducibility
        import random
        rng = random.Random(seed)

        npv_results: List[Decimal] = []

        for i in range(iterations):
            # Generate random parameter values
            modified_input = base_roi_input
            for param_name, (mean, std_dev) in parameter_distributions.items():
                # Use Box-Muller transform for normal distribution
                u1 = Decimal(str(rng.random()))
                u2 = Decimal(str(rng.random()))

                # Avoid log(0)
                if u1 == 0:
                    u1 = Decimal("0.0001")

                # Box-Muller transform
                z = Decimal(str(math.sqrt(-2 * float(u1.ln())))) * Decimal(
                    str(math.cos(2 * math.pi * float(u2)))
                )
                random_value = mean + z * std_dev

                # Ensure non-negative for certain parameters
                if param_name in ["investment_cost", "annual_savings"]:
                    random_value = max(Decimal("0"), random_value)

                modified_input = self._modify_roi_input(modified_input, param_name, random_value)

            # Calculate NPV for this iteration
            result = self.perform_roi_analysis(modified_input)
            npv_results.append(result.net_present_value_usd)

        # Calculate statistics
        npv_results.sort()
        n = len(npv_results)

        mean_npv = sum(npv_results) / n
        variance = sum((x - mean_npv) ** 2 for x in npv_results) / n
        std_dev_npv = variance.sqrt()

        # Percentiles
        p5_idx = int(0.05 * n)
        p50_idx = int(0.50 * n)
        p95_idx = int(0.95 * n)

        percentile_5 = npv_results[p5_idx]
        percentile_50 = npv_results[p50_idx]
        percentile_95 = npv_results[p95_idx]

        # Probability of positive NPV
        positive_count = sum(1 for x in npv_results if x > 0)
        prob_positive = Decimal(str(positive_count / n * 100))

        # Value at Risk (5th percentile loss)
        var_5 = -percentile_5 if percentile_5 < 0 else Decimal("0")

        provenance_hash = self._calculate_provenance_hash(
            "monte_carlo_simulation",
            {"iterations": iterations, "seed": seed},
            mean_npv
        )

        return MonteCarloResult(
            mean_npv=self._round_decimal(mean_npv),
            std_dev_npv=self._round_decimal(std_dev_npv),
            percentile_5_npv=self._round_decimal(percentile_5),
            percentile_50_npv=self._round_decimal(percentile_50),
            percentile_95_npv=self._round_decimal(percentile_95),
            probability_positive_npv_percent=self._round_decimal(prob_positive),
            value_at_risk_5_percent=self._round_decimal(var_5),
            iterations=iterations,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Carbon Impact Calculation
    # =========================================================================

    def calculate_carbon_impact(self, input_data: CarbonImpactInput) -> CarbonImpactResult:
        """
        Calculate carbon emissions and costs from energy loss.

        Implements GHG Protocol methodology for Scope 1/2/3 emissions.

        Args:
            input_data: Carbon impact input parameters

        Returns:
            CarbonImpactResult with complete provenance
        """
        self.calculation_count += 1
        steps: List[CalculationStep] = []

        # Step 1: Get emission factor for fuel type
        emission_factor = Decimal(str(CO2_EMISSION_FACTORS.get(
            input_data.fuel_type.value, 0.185
        )))
        steps.append(CalculationStep(
            step_number=1,
            operation="lookup",
            description="Lookup emission factor for fuel type",
            inputs=(
                ("fuel_type", input_data.fuel_type.value),
            ),
            output_value=str(emission_factor),
            output_name="emission_factor_kg_per_kwh",
            formula="IPCC/GHG Protocol emission factor database",
            units="kg CO2e/kWh"
        ))

        # Step 2: Calculate direct emissions
        direct_emissions = input_data.energy_loss_kwh_per_year * emission_factor
        steps.append(CalculationStep(
            step_number=2,
            operation="multiply",
            description="Calculate direct CO2e emissions",
            inputs=(
                ("energy_loss_kwh", str(input_data.energy_loss_kwh_per_year)),
                ("emission_factor", str(emission_factor)),
            ),
            output_value=str(direct_emissions),
            output_name="direct_emissions_kg",
            formula="direct_emissions = energy_loss * emission_factor",
            units="kg CO2e/year"
        ))

        # Step 3: Calculate upstream emissions (if included)
        upstream_emissions = Decimal("0")
        if input_data.include_upstream_emissions:
            upstream_emissions = (input_data.energy_loss_kwh_per_year
                                 * input_data.upstream_emission_factor)
        steps.append(CalculationStep(
            step_number=3,
            operation="multiply",
            description="Calculate upstream emissions",
            inputs=(
                ("energy_loss_kwh", str(input_data.energy_loss_kwh_per_year)),
                ("upstream_factor", str(input_data.upstream_emission_factor)),
                ("include_upstream", str(input_data.include_upstream_emissions)),
            ),
            output_value=str(upstream_emissions),
            output_name="upstream_emissions_kg",
            formula="upstream_emissions = energy_loss * upstream_factor",
            units="kg CO2e/year"
        ))

        # Step 4: Calculate total emissions
        total_emissions_kg = direct_emissions + upstream_emissions
        total_emissions_tonnes = total_emissions_kg / Decimal("1000")
        steps.append(CalculationStep(
            step_number=4,
            operation="add",
            description="Calculate total emissions",
            inputs=(
                ("direct_emissions", str(direct_emissions)),
                ("upstream_emissions", str(upstream_emissions)),
            ),
            output_value=str(total_emissions_kg),
            output_name="total_emissions_kg",
            formula="total_emissions = direct + upstream",
            units="kg CO2e/year"
        ))

        # Step 5: Calculate carbon cost
        carbon_cost = total_emissions_tonnes * input_data.carbon_price_per_tonne
        steps.append(CalculationStep(
            step_number=5,
            operation="multiply",
            description="Calculate carbon cost",
            inputs=(
                ("total_emissions_tonnes", str(total_emissions_tonnes)),
                ("carbon_price_per_tonne", str(input_data.carbon_price_per_tonne)),
            ),
            output_value=str(carbon_cost),
            output_name="carbon_cost_usd",
            formula="carbon_cost = emissions_tonnes * carbon_price",
            units="USD/year"
        ))

        # Step 6: Calculate carbon intensity
        carbon_intensity = (total_emissions_kg / input_data.energy_loss_kwh_per_year
                           if input_data.energy_loss_kwh_per_year > 0 else Decimal("0"))
        steps.append(CalculationStep(
            step_number=6,
            operation="divide",
            description="Calculate carbon intensity",
            inputs=(
                ("total_emissions_kg", str(total_emissions_kg)),
                ("energy_kwh", str(input_data.energy_loss_kwh_per_year)),
            ),
            output_value=str(carbon_intensity),
            output_name="carbon_intensity_kg_per_kwh",
            formula="carbon_intensity = total_emissions / energy",
            units="kg CO2e/kWh"
        ))

        # Scope attribution based on fuel type
        scope_1_emissions = Decimal("0")
        scope_2_emissions = Decimal("0")

        if input_data.fuel_type.value.startswith("electricity"):
            scope_2_emissions = direct_emissions
        else:
            scope_1_emissions = direct_emissions

        emission_breakdown = {
            "CO2": total_emissions_kg * Decimal("0.99"),  # Assume 99% CO2
            "CH4": total_emissions_kg * Decimal("0.008"),  # 0.8% CH4
            "N2O": total_emissions_kg * Decimal("0.002"),  # 0.2% N2O
        }

        provenance_hash = self._calculate_provenance_hash(
            "carbon_impact_calculation",
            input_data,
            total_emissions_kg
        )

        return CarbonImpactResult(
            direct_emissions_kg_co2e=self._round_decimal(direct_emissions),
            upstream_emissions_kg_co2e=self._round_decimal(upstream_emissions),
            total_emissions_kg_co2e=self._round_decimal(total_emissions_kg),
            total_emissions_tonnes_co2e=self._round_decimal(total_emissions_tonnes, 3),
            carbon_cost_usd=self._round_decimal(carbon_cost),
            carbon_intensity_kg_per_kwh=self._round_decimal(carbon_intensity, 4),
            scope_1_emissions_kg=self._round_decimal(scope_1_emissions),
            scope_2_emissions_kg=self._round_decimal(scope_2_emissions),
            emission_breakdown={k: self._round_decimal(v) for k, v in emission_breakdown.items()},
            calculation_steps=tuple(steps),
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Economic Report Generation
    # =========================================================================

    def generate_economic_report(
        self,
        energy_loss_input: Optional[EnergyLossInput] = None,
        production_impact_input: Optional[ProductionImpactInput] = None,
        maintenance_cost_input: Optional[MaintenanceCostInput] = None,
        tco_input: Optional[TCOInput] = None,
        roi_input: Optional[ROIInput] = None,
        carbon_impact_input: Optional[CarbonImpactInput] = None,
        sensitivity_variations: Optional[Dict[str, Tuple[Decimal, Decimal, Decimal]]] = None,
        run_monte_carlo: bool = False,
        monte_carlo_iterations: int = 1000
    ) -> EconomicReport:
        """
        Generate comprehensive economic analysis report.

        Combines all economic calculations into a single report with
        executive summary and complete provenance.

        Args:
            energy_loss_input: Energy loss calculation input
            production_impact_input: Production impact calculation input
            maintenance_cost_input: Maintenance cost calculation input
            tco_input: Total Cost of Ownership input
            roi_input: ROI analysis input
            carbon_impact_input: Carbon impact calculation input
            sensitivity_variations: Parameter variations for sensitivity analysis
            run_monte_carlo: Whether to run Monte Carlo simulation
            monte_carlo_iterations: Number of Monte Carlo iterations

        Returns:
            EconomicReport with all analyses and executive summary
        """
        report_id = str(uuid.uuid4())
        generated_at = datetime.utcnow().isoformat() + "Z"

        # Run individual calculations
        energy_loss_result = None
        production_impact_result = None
        maintenance_cost_result = None
        tco_result = None
        roi_result = None
        carbon_impact_result = None
        sensitivity_results: List[SensitivityResult] = []
        monte_carlo_result = None

        total_annual_cost = Decimal("0")
        total_savings_potential = Decimal("0")

        if energy_loss_input:
            energy_loss_result = self.calculate_energy_loss_cost(energy_loss_input)
            total_annual_cost += energy_loss_result.total_energy_penalty_per_year_usd

        if production_impact_input:
            production_impact_result = self.calculate_production_impact(production_impact_input)
            total_annual_cost += production_impact_result.total_production_impact_usd

        if maintenance_cost_input:
            maintenance_cost_result = self.calculate_maintenance_costs(maintenance_cost_input)
            total_annual_cost += maintenance_cost_result.total_maintenance_cost_usd

        if tco_input and maintenance_cost_result:
            annual_opex = (energy_loss_result.total_energy_penalty_per_year_usd
                          if energy_loss_result else Decimal("0"))
            tco_result = self.calculate_total_cost_of_ownership(
                tco_input,
                annual_opex,
                maintenance_cost_result.total_maintenance_cost_usd
            )

        if roi_input:
            roi_result = self.perform_roi_analysis(roi_input)
            total_savings_potential = roi_input.annual_savings

            if sensitivity_variations:
                sensitivity_results = self.perform_sensitivity_analysis(
                    roi_input, sensitivity_variations
                )

            if run_monte_carlo:
                # Default parameter distributions
                distributions = {
                    "annual_savings": (
                        roi_input.annual_savings,
                        roi_input.annual_savings * Decimal("0.15")
                    ),
                    "investment_cost": (
                        roi_input.investment_cost,
                        roi_input.investment_cost * Decimal("0.10")
                    ),
                }
                monte_carlo_result = self.perform_monte_carlo_simulation(
                    roi_input, distributions, monte_carlo_iterations
                )

        if carbon_impact_input:
            carbon_impact_result = self.calculate_carbon_impact(carbon_impact_input)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            energy_loss_result,
            production_impact_result,
            maintenance_cost_result,
            tco_result,
            roi_result,
            carbon_impact_result,
            total_annual_cost
        )

        # Calculate overall provenance hash
        provenance_data = {
            "report_id": report_id,
            "generated_at": generated_at,
            "total_annual_cost": str(total_annual_cost),
            "energy_hash": energy_loss_result.provenance_hash if energy_loss_result else None,
            "production_hash": production_impact_result.provenance_hash if production_impact_result else None,
            "maintenance_hash": maintenance_cost_result.provenance_hash if maintenance_cost_result else None,
            "tco_hash": tco_result.provenance_hash if tco_result else None,
            "roi_hash": roi_result.provenance_hash if roi_result else None,
            "carbon_hash": carbon_impact_result.provenance_hash if carbon_impact_result else None,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return EconomicReport(
            report_id=report_id,
            generated_at=generated_at,
            energy_loss=energy_loss_result,
            production_impact=production_impact_result,
            maintenance_costs=maintenance_cost_result,
            total_cost_of_ownership=tco_result,
            roi_analysis=roi_result,
            carbon_impact=carbon_impact_result,
            sensitivity_results=tuple(sensitivity_results),
            monte_carlo_result=monte_carlo_result,
            total_annual_cost_usd=self._round_decimal(total_annual_cost),
            total_annual_savings_potential_usd=self._round_decimal(total_savings_potential),
            executive_summary=executive_summary,
            provenance_hash=provenance_hash
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _round_decimal(
        self, value: Decimal, precision: Optional[int] = None
    ) -> Decimal:
        """Round decimal to specified precision using ROUND_HALF_UP."""
        prec = precision if precision is not None else self._precision
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance_hash(
        self,
        calculation_type: str,
        input_data: Any,
        result: Any
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        # Serialize input data
        if hasattr(input_data, "__dict__"):
            input_dict = {k: str(v) for k, v in input_data.__dict__.items()}
        elif isinstance(input_data, dict):
            input_dict = {k: str(v) for k, v in input_data.items()}
        else:
            input_dict = {"value": str(input_data)}

        provenance_data = {
            "calculation_type": calculation_type,
            "version": self.VERSION,
            "inputs": input_dict,
            "result": str(result),
        }

        canonical_json = json.dumps(provenance_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def _calculate_depreciation(
        self,
        capital_cost: Decimal,
        useful_life: int,
        method: DepreciationMethod,
        residual_percent: Decimal
    ) -> List[Tuple[int, Decimal]]:
        """Calculate depreciation schedule."""
        schedule: List[Tuple[int, Decimal]] = []
        depreciable_base = capital_cost * (Decimal("1") - residual_percent / Decimal("100"))

        if method == DepreciationMethod.STRAIGHT_LINE:
            annual_depreciation = depreciable_base / useful_life
            for year in range(1, useful_life + 1):
                schedule.append((year, self._round_decimal(annual_depreciation)))

        elif method == DepreciationMethod.MACRS_5_YEAR:
            for year, rate in enumerate(MACRS_5_YEAR, 1):
                depreciation = capital_cost * Decimal(str(rate))
                schedule.append((year, self._round_decimal(depreciation)))

        elif method == DepreciationMethod.MACRS_7_YEAR:
            for year, rate in enumerate(MACRS_7_YEAR, 1):
                depreciation = capital_cost * Decimal(str(rate))
                schedule.append((year, self._round_decimal(depreciation)))

        elif method == DepreciationMethod.MACRS_10_YEAR:
            for year, rate in enumerate(MACRS_10_YEAR, 1):
                depreciation = capital_cost * Decimal(str(rate))
                schedule.append((year, self._round_decimal(depreciation)))

        elif method == DepreciationMethod.MACRS_15_YEAR:
            for year, rate in enumerate(MACRS_15_YEAR, 1):
                depreciation = capital_cost * Decimal(str(rate))
                schedule.append((year, self._round_decimal(depreciation)))

        elif method == DepreciationMethod.DOUBLE_DECLINING:
            book_value = capital_cost
            rate = Decimal("2") / useful_life
            residual_value = capital_cost * residual_percent / Decimal("100")

            for year in range(1, useful_life + 1):
                depreciation = min(book_value * rate, book_value - residual_value)
                depreciation = max(Decimal("0"), depreciation)
                schedule.append((year, self._round_decimal(depreciation)))
                book_value -= depreciation

        return schedule

    def _calculate_npv_of_costs(
        self,
        capital_cost: Decimal,
        annual_cost: Decimal,
        end_of_life_cost: Decimal,
        years: int,
        discount_rate: Decimal
    ) -> Decimal:
        """Calculate NPV of all costs."""
        npv = capital_cost

        for year in range(1, years + 1):
            pv = annual_cost / ((Decimal("1") + discount_rate) ** year)
            npv += pv

        # Add end-of-life cost
        eol_pv = end_of_life_cost / ((Decimal("1") + discount_rate) ** years)
        npv += eol_pv

        return npv

    def _calculate_irr(
        self,
        investment: Decimal,
        annual_savings: Decimal,
        years: int,
        escalation_rate: Decimal,
        tax_rate: Decimal
    ) -> Decimal:
        """Calculate IRR using Newton-Raphson method."""
        # Initial guess
        irr = Decimal("0.10")

        for iteration in range(100):
            npv = -investment
            npv_derivative = Decimal("0")

            for year in range(1, years + 1):
                escalated_savings = annual_savings * ((Decimal("1") + escalation_rate) ** (year - 1))
                after_tax = escalated_savings * (Decimal("1") - tax_rate)

                discount_factor = (Decimal("1") + irr) ** year
                npv += after_tax / discount_factor
                npv_derivative -= year * after_tax / ((Decimal("1") + irr) ** (year + 1))

            # Check convergence
            if abs(npv) < Decimal("0.01"):
                break

            # Newton-Raphson update
            if npv_derivative != 0:
                irr = irr - npv / npv_derivative
            else:
                break

            # Bounds check
            if irr < Decimal("-0.99"):
                irr = Decimal("-0.99")
            elif irr > Decimal("10.0"):
                irr = Decimal("10.0")
                break

        return irr

    def _calculate_discounted_payback(
        self,
        investment: Decimal,
        annual_savings: Decimal,
        discount_rate: Decimal,
        escalation_rate: Decimal,
        tax_rate: Decimal
    ) -> Decimal:
        """Calculate discounted payback period."""
        cumulative_pv = Decimal("0")

        for year in range(1, 51):  # Max 50 years
            escalated_savings = annual_savings * ((Decimal("1") + escalation_rate) ** (year - 1))
            after_tax = escalated_savings * (Decimal("1") - tax_rate)
            pv = after_tax / ((Decimal("1") + discount_rate) ** year)
            cumulative_pv += pv

            if cumulative_pv >= investment:
                # Interpolate for fractional year
                previous_cumulative = cumulative_pv - pv
                fraction = (investment - previous_cumulative) / pv
                return Decimal(str(year - 1)) + fraction

        return Decimal("999")  # No payback within 50 years

    def _calculate_break_even_utilization(
        self,
        investment: Decimal,
        annual_savings: Decimal,
        discount_rate: Decimal,
        years: int
    ) -> Decimal:
        """Calculate break-even utilization percentage."""
        # Calculate annuity factor
        annuity_factor = Decimal("0")
        for year in range(1, years + 1):
            annuity_factor += Decimal("1") / ((Decimal("1") + discount_rate) ** year)

        # Required annual savings for NPV = 0
        required_savings = investment / annuity_factor if annuity_factor > 0 else Decimal("0")

        # Break-even utilization
        utilization = (required_savings / annual_savings * Decimal("100")
                      if annual_savings > 0 else Decimal("0"))

        return utilization

    def _modify_roi_input(
        self, base_input: ROIInput, param_name: str, new_value: Decimal
    ) -> ROIInput:
        """Create modified ROI input with changed parameter."""
        params = {
            "investment_cost": base_input.investment_cost,
            "annual_savings": base_input.annual_savings,
            "discount_rate_percent": base_input.discount_rate_percent,
            "analysis_period_years": base_input.analysis_period_years,
            "inflation_rate_percent": base_input.inflation_rate_percent,
            "tax_rate_percent": base_input.tax_rate_percent,
            "energy_cost_escalation_percent": base_input.energy_cost_escalation_percent,
        }

        if param_name in params:
            params[param_name] = new_value

        return ROIInput(**params)

    def _generate_executive_summary(
        self,
        energy_loss: Optional[EnergyLossResult],
        production_impact: Optional[ProductionImpactResult],
        maintenance_costs: Optional[MaintenanceCostResult],
        tco: Optional[TCOResult],
        roi: Optional[ROIResult],
        carbon_impact: Optional[CarbonImpactResult],
        total_annual_cost: Decimal
    ) -> Dict[str, Any]:
        """Generate executive summary for economic report."""
        summary = {
            "total_annual_economic_impact_usd": str(total_annual_cost),
            "key_findings": [],
            "recommendations": [],
            "risk_factors": [],
        }

        if energy_loss:
            summary["energy_loss_usd_per_year"] = str(energy_loss.total_energy_penalty_per_year_usd)
            summary["energy_loss_percent"] = str(energy_loss.heat_transfer_loss_percent)
            if energy_loss.heat_transfer_loss_percent > Decimal("15"):
                summary["key_findings"].append(
                    f"High energy loss of {energy_loss.heat_transfer_loss_percent}% "
                    "indicates significant fouling"
                )
                summary["recommendations"].append(
                    "Implement optimized cleaning schedule to reduce energy loss"
                )

        if production_impact:
            summary["production_impact_usd_per_year"] = str(
                production_impact.total_production_impact_usd
            )
            if production_impact.throughput_loss_percent > Decimal("10"):
                summary["key_findings"].append(
                    f"Throughput reduction of {production_impact.throughput_loss_percent}% "
                    "significantly impacts production"
                )

        if maintenance_costs:
            summary["maintenance_cost_usd_per_year"] = str(
                maintenance_costs.total_maintenance_cost_usd
            )
            if maintenance_costs.unplanned_maintenance_cost_usd > maintenance_costs.planned_cleaning_cost_usd:
                summary["risk_factors"].append(
                    "Unplanned maintenance costs exceed planned costs - "
                    "consider predictive maintenance"
                )

        if roi:
            summary["npv_usd"] = str(roi.net_present_value_usd)
            summary["irr_percent"] = str(roi.internal_rate_of_return_percent)
            summary["payback_years"] = str(roi.simple_payback_years)

            if roi.net_present_value_usd > 0:
                summary["key_findings"].append(
                    f"Positive NPV of ${roi.net_present_value_usd} indicates "
                    "financially viable investment"
                )

            if roi.simple_payback_years < Decimal("3"):
                summary["recommendations"].append(
                    f"Short payback period of {roi.simple_payback_years} years - "
                    "recommend proceeding with investment"
                )

        if carbon_impact:
            summary["carbon_emissions_tonnes_per_year"] = str(
                carbon_impact.total_emissions_tonnes_co2e
            )
            summary["carbon_cost_usd_per_year"] = str(carbon_impact.carbon_cost_usd)

            if carbon_impact.total_emissions_tonnes_co2e > Decimal("100"):
                summary["key_findings"].append(
                    f"Significant carbon impact of {carbon_impact.total_emissions_tonnes_co2e} "
                    "tonnes CO2e/year from energy loss"
                )
                summary["recommendations"].append(
                    "Include carbon costs in economic decision-making"
                )

        return summary


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create calculator instance
    calculator = EconomicCalculator()

    # Example 1: Energy Loss Calculation
    energy_input = EnergyLossInput(
        design_duty_kw=Decimal("1000"),
        actual_duty_kw=Decimal("850"),
        fuel_type=FuelType.NATURAL_GAS,
        fuel_cost_per_kwh=Decimal("0.05"),
        operating_hours_per_year=Decimal("8000"),
        system_efficiency=Decimal("0.85"),
        include_carbon_cost=True,
        carbon_price_per_tonne=Decimal("50.00")
    )
    energy_result = calculator.calculate_energy_loss_cost(energy_input)
    print(f"Energy Loss Analysis:")
    print(f"  Heat Transfer Loss: {energy_result.heat_transfer_loss_kw} kW ({energy_result.heat_transfer_loss_percent}%)")
    print(f"  Annual Energy Penalty: ${energy_result.total_energy_penalty_per_year_usd}")
    print(f"  Carbon Emissions: {energy_result.carbon_emissions_kg_per_year} kg CO2/year")
    print(f"  Provenance Hash: {energy_result.provenance_hash[:16]}...")
    print()

    # Example 2: ROI Analysis
    roi_input = ROIInput(
        investment_cost=Decimal("50000"),
        annual_savings=Decimal("25000"),
        discount_rate_percent=Decimal("10.0"),
        analysis_period_years=10,
        tax_rate_percent=Decimal("25.0")
    )
    roi_result = calculator.perform_roi_analysis(roi_input)
    print(f"ROI Analysis:")
    print(f"  NPV: ${roi_result.net_present_value_usd}")
    print(f"  IRR: {roi_result.internal_rate_of_return_percent}%")
    print(f"  Simple Payback: {roi_result.simple_payback_years} years")
    print(f"  Profitability Index: {roi_result.profitability_index}")
    print()

    # Example 3: Carbon Impact
    carbon_input = CarbonImpactInput(
        energy_loss_kwh_per_year=Decimal("1411765"),
        fuel_type=FuelType.NATURAL_GAS,
        carbon_price_per_tonne=Decimal("75.00"),
        include_upstream_emissions=True
    )
    carbon_result = calculator.calculate_carbon_impact(carbon_input)
    print(f"Carbon Impact:")
    print(f"  Total Emissions: {carbon_result.total_emissions_tonnes_co2e} tonnes CO2e/year")
    print(f"  Carbon Cost: ${carbon_result.carbon_cost_usd}/year")
    print(f"  Carbon Intensity: {carbon_result.carbon_intensity_kg_per_kwh} kg/kWh")
